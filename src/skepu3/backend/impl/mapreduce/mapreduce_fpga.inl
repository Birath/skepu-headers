/*! \file mapreduce_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_FPGA

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel, FPGAKernel>
		::mapReduceSingle_FPGA(size_t deviceID, size_t startIdx, size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			Ret startValue = res;
			auto aArgs  = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const size_t numBlocks = std::max<size_t>(1, std::min(size / numThreads, this->m_selected_spec->GPUBlocks()));
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			
			DEBUG_TEXT_LEVEL1("OpenCL MapReduce: numThreads = " << numThreads << ", numBlocks = " << numBlocks);
			
			// Copies the elements to the device
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + startIdx, size * abs(this->m_strides[EI]), device, true)...);
			auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).size(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
			
			// Create the output memory
			DeviceMemPointer_CL<Ret> outMemP(&res, numBlocks, device);
			
			size_t threads = std::min<size_t>(size, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapFunc::randomCount>(size, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			FPGAKernel::mapReduce (
				deviceID, numThreads, numBlocks * numThreads,
				randomMemP,
				device_mem_pointer_const_cast(std::get<EI>(elwiseMemP))...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				&outMemP,
				size_info(defaultDim{}, this->default_size_i, this->default_size_j, this->default_size_k, this->default_size_l, get<EI>(std::forward<CallArgs>(args)...)...),
				size, startIdx, this->m_strides,
				sizeof(Ret) * numThreads
			);
			
			// Copy back result
			pack_expand((std::get<AI-arity>(anyMemP)->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			outMemP.changeDeviceData();
			outMemP.copyDeviceToHost(1);
			
			return ReduceFunc::CPU(res, startValue);
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel, FPGAKernel>
		::mapReduceNumDevices_FPGA(size_t numDevices, size_t startIdx, size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			auto aArgs  = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			Ret result[numDevices];
			std::vector<DeviceMemPointer_CL<Ret>> outMemP;
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + (i == numDevices - 1 ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::max<size_t>(1, std::min(numElem / numThreads, this->m_selected_spec->GPUBlocks()));
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				
				DEBUG_TEXT_LEVEL1("OpenCL MapReduce: device = " << i << ", numThreads = " << numThreads << ", numBlocks = " << numBlocks);
				
				// Copies the elements to the device
				auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + baseIndex, numElem * abs(this->m_strides[EI]), device, true)...);
				auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).size(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
			
				// Create the output memory
				outMemP.emplace_back(&result[i], numBlocks, device);
			
				size_t threads = std::min<size_t>(size, numBlocks * numThreads);
				auto random = this->template prepareRandom_CL<MapFunc::randomCount>(size, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				FPGAKernel::mapReduce (
					i, numThreads, numBlocks * numThreads,
					randomMemP,
					device_mem_pointer_const_cast(std::get<EI>(elwiseMemP))...,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					&outMemP[i],
					size_info(defaultDim{}, this->default_size_i, this->default_size_j, this->default_size_k, this->default_size_l, get<EI>(std::forward<CallArgs>(args)...)...),
					numElem, baseIndex, this->m_strides,
					sizeof(Ret) * numThreads
				);
				
				// Copy back result
				pack_expand((std::get<AI-arity>(anyMemP)->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				outMemP[i].changeDeviceData();
			}
			
			// Reduces results from each device on the CPU to yield the total result.
			for (size_t i = 0; i < numDevices; ++i)
			{
				outMemP[i].copyDeviceToHost(1);
				res = ReduceFunc::CPU(res, result[i]);
			}
			
			return res;
		}
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		typename MapFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel, FPGAKernel>
		::FPGA(size_t startIdx, size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapReduce: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return mapReduceSingle_FPGA(0, startIdx, size, elwise_indices, ai, ci, res, std::forward<CallArgs>(args)...);
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
			return mapReduceNumDevices_FPGA(numDevices, startIdx, size, elwise_indices, ai, ci, res, std::forward<CallArgs>(args)...);
		}
		
	} // end namespace backend
} // end namespace skepu

#endif
