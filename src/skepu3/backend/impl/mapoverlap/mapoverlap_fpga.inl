/*! \file mapoverlap_fpga.inl
*  \brief Contains the definitions of FPGA specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_FPGA

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  Argument startIdx tell from where to perform a partial MapOverlap. Set startIdx=0 for MapOverlap of entire input.
		 *  The function uses only \em one device which is decided by a parameter. Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::mapOverlapSingle_FPGA(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			// Setup parameters
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			size_t numElem = arg.size() - startIdx;
			size_t overlap = this->m_overlap;
			size_t n = numElem + std::min(startIdx, overlap);
			size_t out_offset = std::min(startIdx, overlap);
			size_t out_numelements = numElem;
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap);
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHostAndInvalidateDevice();
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = arg.end()(i - overlap);
					wrap[overlap+i] = arg(i);
				}
			}
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrap_mem_p(&wrap[0], wrap.size(), device);
			wrap_mem_p.copyHostToDevice();
			
			const size_t numThreads = std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
			const size_t numBlocks = std::max<size_t>(1, std::min(n / numThreads + (n % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks()));
			const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
			
			// Copy elements to device and allocate output memory.
			auto in_mem_p = arg.updateDevice_CL(arg.getAddress() + startIdx - out_offset, n, device, true);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...)
					.getAddress(), get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
			auto outMemP    = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + startIdx, numElem, device, false)...);
			
			size_t threads = std::min<size_t>(numElem, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(numElem, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapOverlapVector(
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				in_mem_p,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				&wrap_mem_p,
				n, overlap, out_offset, out_numelements, _poly, _pad,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
		 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
		 *  on multiple devices, calling mapOverlapNumDevices_CL.
		 *  Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::vector_FPGA(size_t startIdx, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			if (numDevices <= 1)
				return this->mapOverlapSingle_FPGA(0, startIdx, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
				SKEPU_ERROR("Multiple devices not supported on FPGA backend");
		}
		
		
		/*!
		 * For Matrix overlap, we need to check whether overlap configuration is runnable considering total size of shared memory available on that system.
		 * This method is a helper funtion doing that. It is called by another helper \p getThreadNumber_CL() method.
		 *
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<typename T>
		bool MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::sharedMemAvailable_FPGA(size_t &numThreads, size_t deviceID)
		{
			size_t overlap = this->m_overlap;
			size_t maxShMem = this->m_environment->m_devices_CL.at(deviceID)->getSharedMemPerBlock() / sizeof(T) - SHMEM_SAFITY_BUFFER; // little buffer for other usage
			size_t orgThreads = numThreads;
			
			numThreads = (numThreads + 2 * overlap < maxShMem) ? numThreads : maxShMem - 2 * overlap;
			
			if (orgThreads == numThreads) // return true when nothing changed because of overlap constraint
				return true;
			
			if (numThreads < 8) // not original numThreads then atleast 8 threads should be there
				SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted");
				
			return false;
		}
		
		
		/*!
		 * Helper method used for calculating optimal thread count. For row- or column-wise overlap,
		 * it determines a thread block size with perfect division of problem size.
		 *
		 * \param width The problem size.
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<typename T>
		int MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::getThreadNumber_FPGA(size_t width, size_t numThreads, size_t deviceID)
		{
			// first check whether shared memory would be ok for this numThreads. Changes numThreads accordingly
			if (!sharedMemAvailable_FPGA<T>(numThreads, deviceID) && numThreads < 1)
				SKEPU_ERROR("Too low overlap size to continue.");
			
			if (width % numThreads == 0)
				return width / numThreads;
			
			for (size_t i = numThreads - 1; i >= 1; i--) // decreament numThreads and see which one is a perfect divisor
			{
				if (width % numThreads == 0)
					return width / numThreads;
			}
			return -1;
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Performs MapOverlap on the first rows of the matrix, specified by the numrows argument.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::mapOverlapSingle_FPGA_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t n = arg.total_cols()*numrows;
			const size_t overlap = this->m_overlap;
			const size_t out_offset = 0;
			const size_t out_numelements = n;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t rowWidth = arg.total_cols(); // same as numcols
			size_t trdsize = rowWidth;
			size_t blocksPerRow = 1;
			
			if (rowWidth > this->m_selected_spec->GPUThreads())
			{
				int tmp = getThreadNumber_CL<T>(rowWidth, maxThreads, deviceID);
				if (tmp == -1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("Cannot execute overlap with current overlap width");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap * numrows);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t row = 0; row < numrows; row++)
				{
					auto inputEndTemp = inputBeginTemp + rowWidth;
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + row * overlap * 2] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[i + row * overlap * 2 + overlap] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
			}
			/*    else
         wrap.resize(1); // not used so minimize overhead;*/
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			const size_t numThreads = trdsize; // std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numrows, this->m_selected_spec->GPUBlocks()));
			const size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP  = arg.updateDevice_CL(arg.getAddress(), numrows, rowWidth, device, true);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), numrows, rowWidth, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
				
			size_t threads = std::min<size_t>(n, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(n, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapOverlapMatrixRowWise(
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				inMemP,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().row_size_info(),
				&wrapMemP, n,
				overlap, out_offset, out_numelements, _poly, _pad, blocksPerRow, rowWidth,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::rowwise_FPGA(size_t numrows, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			if (numDevices <= 1)
				return this->mapOverlapSingle_FPGA_Row(0, numrows, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else 
				SKEPU_ERROR("Multiple devices not supported on FPGA backend");
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::mapOverlapSingle_FPGA_Col(size_t deviceID, size_t _numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t n = arg.size();
			const size_t overlap = this->m_overlap;
			const size_t out_offset = 0;
			const size_t out_numelements = n;
			const size_t colWidth = arg.total_rows();
			const size_t numcols = arg.total_cols();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t trdsize = colWidth;
			size_t blocksPerCol = 1;
			
			if (colWidth > maxThreads)
			{
				int tmp = getThreadNumber_CL<T>(colWidth, maxThreads, deviceID);
				if (tmp == -1 || blocksPerCol * numcols > maxBlocks)
					SKEPU_ERROR("Col width is larger than maximum thread size: " << colWidth << " " << maxThreads);
				
				blocksPerCol = tmp;
				trdsize = colWidth / blocksPerCol;
				if (trdsize < overlap)
					SKEPU_ERROR("Thread size should be larger than overlap width");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(overlap * 2 * (n / colWidth));
			
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t col=0; col< numcols; col++)
				{
					auto inputEndTemp = inputBeginTemp + numcols * (colWidth - 1);
					auto inputEndMinusOverlap = (inputEndTemp - (overlap - 1) * numcols);
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + col * overlap * 2] = inputEndMinusOverlap(i * numcols);
						wrap[overlap + i + col * overlap * 2] = inputBeginTemp(i * numcols);
					}
					inputBeginTemp++;
				}
			}
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			const size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol * numcols, maxBlocks));
			const size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP  = arg.updateDevice_CL(arg.getAddress(), colWidth, numcols, device, true);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), colWidth, numcols, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
					
			size_t threads = std::min<size_t>(n, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(n, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapOverlapMatrixColWise(
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				inMemP,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().col_size_info(),
				&wrapMemP,
				n, overlap, out_offset, out_numelements, _poly, _pad,
				blocksPerCol, numcols, colWidth,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
		}
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
		 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
		 *  on multiple devices, calling mapOverlapNumDevices_CL.
		 *  Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::colwise_FPGA(size_t numcols, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
					
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL_Col(0, numcols, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
				SKEPU_ERROR("Multiple devices not supported on FPGA backend");
		}	
		
		
		
		/*!
		 *  Performs the 2D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::mapOverlapSingleThread_FPGA(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_rows  = arg.total_rows();
			const size_t in_cols  = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), in_rows,  in_cols,  device, true)...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getParent().getAddress(), out_rows, out_cols, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
				
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			size_t numThreads[2], numBlocks[2];
			numThreads[0] = std::min<size_t>(out_cols, 16);
			numThreads[1] = std::min(out_rows, maxThreads / 16);
			numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_rows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			
			const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
			const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
			const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 2D: device = " << deviceID << ", numThreads = "
				<< "[" << numThreads[0] << " x " << numThreads[1] << "]" 
				<< ", numBlocks = " << "[" << numBlocks[0] << " x " << numBlocks[1] << "], shmem = " << sharedMemSize); 
			
			size_t threads = std::min<size_t>(out_rows * out_cols, numBlocks[0] * numBlocks[1] * numThreads[0] * numThreads[1]);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(out_rows * out_cols, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapOverlap2D(
				deviceID, numThreads, numBlocks,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_rows, out_cols,
				this->m_overlap_y, this->m_overlap_x,
				in_rows, in_cols, sharedRows, sharedCols,
				edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::helper_FPGA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 2D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CL(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
				SKEPU_ERROR("Multiple devices not supported on FPGA backend");

		}
		
		
		
		
		
		
		
		
		
		
		
		/*!
		 *  Performs the 3D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::mapOverlapSingleThread_FPGA(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_size_i  = arg.size_i();
			const size_t in_size_j  = arg.size_j();
			const size_t in_size_k  = arg.size_k();
			const size_t out_size_i = res.size_i();
			const size_t out_size_j = res.size_j();
			const size_t out_size_k = res.size_k();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), in_size_i * in_size_j * in_size_k, device, true)...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getParent().getAddress(), out_size_i * out_size_j * out_size_k, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			size_t numThreads[3], numBlocks[3];
			size_t sizeLength = (size_t)std::cbrt(maxThreads);
			numThreads[0] = std::min<size_t>(out_size_k, sizeLength);
			numThreads[1] = std::min<size_t>(out_size_j, std::min<size_t>(maxThreads / numThreads[0], sizeLength));
			numThreads[2] = std::min(out_size_i, maxThreads / (numThreads[0] * numThreads[1]));
			numBlocks[0] = (size_t)((out_size_k + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_size_j + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			numBlocks[2] = (size_t)((out_size_i + numThreads[2] - 1) / numThreads[2]) * numThreads[2];
			
			const size_t sharedK = numThreads[0] + this->m_overlap_k * 2;
			const size_t sharedJ = numThreads[1] + this->m_overlap_j * 2;
			const size_t sharedI = numThreads[2] + this->m_overlap_i * 2;
			const size_t sharedMemSize =  sharedI * sharedJ * sharedK * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 3D: device = " << deviceID << ", numThreads = "
				<< "[" << numThreads[0] << " x " << numThreads[1] << " x " << numThreads[2] << "]" 
				<< ", numBlocks = " << "[" << numBlocks[0] << " x " << numBlocks[1] << " x " << numBlocks[2] << "], shmem = " << sharedMemSize);
					
			size_t threads = std::min<size_t>(out_size_i * out_size_j * out_size_k, numBlocks[0] * numBlocks[1] * numBlocks[2] * numThreads[0] * numThreads[1] * numThreads[2]); // handle division factor
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapOverlap3D(
				deviceID, numThreads, numBlocks,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_size_i, out_size_j, out_size_k, 
				this->m_overlap_i, this->m_overlap_j, this->m_overlap_k,
				in_size_i, in_size_j, in_size_k, 
				sharedI, sharedJ, sharedK,
				edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::helper_FPGA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 3D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_FPGA(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
				SKEPU_ERROR("Multiple devices not supported on FPGA backend");
		}
		
		
		
		
		
		
		
		
		
		
		/*!
		 *  Performs the 4D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::mapOverlapSingleThread_FPGA(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_size_i  = arg.size_i();
			const size_t in_size_j  = arg.size_j();
			const size_t in_size_k  = arg.size_k();
			const size_t in_size_l  = arg.size_l();
			const size_t out_size_i = res.size_i();
			const size_t out_size_j = res.size_j();
			const size_t out_size_k = res.size_k();
			const size_t out_size_l = res.size_l();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), in_size_i * in_size_j * in_size_k * in_size_l, device, true)...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getParent().getAddress(), out_size_i * out_size_j * out_size_k * out_size_l, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			size_t numThreads[3], numBlocks[3];
			size_t sizeLength = (size_t)std::sqrt(std::sqrt(maxThreads));
			size_t numThreads_0a = std::min<size_t>(out_size_l, sizeLength);
			numThreads[0] = std::min<size_t>(out_size_k, std::min<size_t>(maxThreads / numThreads_0a, sizeLength)) * numThreads_0a;
			numThreads[1] = std::min<size_t>(out_size_j, std::min<size_t>(maxThreads / numThreads[0], sizeLength));
			numThreads[2] = std::min(out_size_i, maxThreads / (numThreads[0] * numThreads[1]));
			
			size_t numBlocks_0a = (size_t)((out_size_l + numThreads_0a - 1) / numThreads_0a) * numThreads_0a;
			numBlocks[0] = (size_t)((out_size_k * out_size_l + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_size_j + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			numBlocks[2] = (size_t)((out_size_i + numThreads[2] - 1) / numThreads[2]) * numThreads[2];
			
			const size_t sharedL = numThreads_0a + this->m_overlap_l * 2;
			const size_t sharedK = (numThreads[0] / numThreads_0a) + this->m_overlap_k * 2;
			const size_t sharedJ = numThreads[1] + this->m_overlap_j * 2;
			const size_t sharedI = numThreads[2] + this->m_overlap_i * 2;
			const size_t sharedMemSize =  sharedI * sharedJ * sharedK * sharedL * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 4D: device = " << deviceID << ", numThreads = "
				<< "[(" << numThreads_0a << ") " << numThreads[0] << " x " << numThreads[1] << " x " << numThreads[2] << "]" 
				<< ", numBlocks = " << "[(" << numBlocks_0a << ") " << numBlocks[0] << " x " << numBlocks[1] << " x " << numBlocks[2] << "], shmem = " << sharedMemSize);
				
			size_t threads = std::min<size_t>(out_size_i * out_size_j * out_size_k * out_size_l, numBlocks[0] * numBlocks[1] * numBlocks[2] * numThreads[0] * numThreads[1] * numThreads[2]); // handle division factor
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k * out_size_l, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapOverlap4D(
				deviceID, numThreads, numBlocks,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_size_i, out_size_j, out_size_k, out_size_l, 
				this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l,
				in_size_i, in_size_j, in_size_k, in_size_l, 
				sharedI, sharedJ, sharedK, sharedL,
				//numThreads_0a, numBlocks_0a,
				edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::helper_FPGA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 4D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
					
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_FPGA(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
				SKEPU_ERROR("Multiple devices not supported on FPGA backend");
		}
	}
}

#endif
