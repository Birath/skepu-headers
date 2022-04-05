/*! \file scan_fpga.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_FPGA

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Scans a Vector using the same recursive algorithm as NVIDIA SDK. First the vector is scanned producing partial results for each block.
		 *  Then the function is called recursively to scan these partial results, which in turn can produce partial results and so on.
		 *  This continues until only one block with partial results is left.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel, typename FPGAKernel>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel, FPGAKernel>
		::scanLargeVectorRecursively_FPGA(
			DeviceMemPointer_CL<T>* res, size_t deviceID,
			DeviceMemPointer_CL<T>* input, DeviceMemPointer_CL<T>* output, const std::vector<DeviceMemPointer_CL<T>*>& blockSums,
			size_t size, ScanMode mode, T initial, size_t level
		)
		{
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const size_t numBlocks = std::min(size / numThreads + (size % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());
			const int isInclusive = (mode == ScanMode::Inclusive || level > 0) ? 1 : 0;
			
			DEBUG_TEXT_LEVEL1("OpenCL Scan: level = " << level << ", size = " << size << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
			
			FPGAKernel::scan(
				deviceID, 
				input, output, size,
				isInclusive
			);
		}
		
		
		/*!
		 *  Performs the Scan on an input range using \em OpenCL with a separate output range. Only one device is used for the scan.
		 *  Allocates space for intermediate results from each block, and then calls scanLargeVectorRecursively_CL.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel, typename FPGAKernel>
		template <typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel, FPGAKernel>
		::scanSingle_FPGA(size_t deviceID, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			// Setup parameters
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			
			typename InIterator::device_pointer_type_cl inMemP = arg.getParent().updateDevice_CL(arg.getAddress(), size, device, true);
			typename OutIterator::device_pointer_type_cl outMemP = res.getParent().updateDevice_CL(res.getAddress(), size, device, false);
			
			const int isInclusive = mode == ScanMode::Inclusive ? 1 : 0;
			
			DEBUG_TEXT_LEVEL1("OpenCL Scan: size = " << size);
			
			FPGAKernel::scan(
				deviceID, 
				inMemP, outMemP, size,
				isInclusive
			);
			outMemP->changeDeviceData();
		}

		/*!
		 *  Performs the Scan on an input range using \em OpenCL with a separate output range. One or more devices can be
		 *  used in the scan. The range is divided evenly among the participating devices which scans their part producing partial device results.
		 *  The device results are scanned on the CPU before they are applied to each devices part.
		 *  Allocates space for intermediate results from each block, and then calls scanLargeVectorRecursively_CL for each device.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel, typename FPGAKernel>
		template <typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel, FPGAKernel>
		::scanNumDevices_FPGA(size_t numDevices, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			typename InIterator::device_pointer_type_cl inMemP[MAX_GPU_DEVICES];
			typename OutIterator::device_pointer_type_cl outMemP[MAX_GPU_DEVICES];
			
			std::vector<DeviceMemPointer_CL<T>*> blockSums[MAX_GPU_DEVICES];
			Vector<T> deviceSums(numDevices);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElements = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				blockSums[i] = allocateBlockSums<T>(numElements, numThreads, device);
				
				inMemP[i]  = arg.getParent().updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numElements, device, false);
				outMemP[i] = res.getParent().updateDevice_CL(res.getAddress() + i * numElemPerSlice, numElements, device, false);
			}
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElements = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				inMemP[i] = arg.getParent().updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numElements, device, true);
				
				T ret;
				DeviceMemPointer_CL<T> retMemP(&ret, 1, device);
				this->scanLargeVectorRecursively_FPGA(&retMemP, i, inMemP[i], outMemP[i], blockSums[i], numElements, mode, initial);
				retMemP.changeDeviceData();
				retMemP.copyDeviceToHost();
			
				deviceSums[i] = ret;
				outMemP[i]->changeDeviceData();
			}
			
			CPU(deviceSums.size(), deviceSums.begin(), deviceSums.begin(), ScanMode::Inclusive, T{});
			
			// Add device sums to each devices data
			for (size_t i = 1; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElements = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::min(numElements / numThreads + (numElements % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());
				
				DEBUG_TEXT_LEVEL1("OpenCL Scan (add device sums): device " << i << ", numElem = " << numElements << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				typename OutIterator::device_pointer_type_cl outMemP = res.getParent().updateDevice_CL((res + i * numElemPerSlice).getAddress(), numElements, device, false);
				
				CLKernel::scanAdd(
					i, numThreads, numBlocks * numThreads,
					outMemP, deviceSums[i - 1], numElements
				);
			}
			
			// Clean up
			for (size_t i = 0; i < numDevices; ++i)
				for (size_t j = 0; j < blockSums[i].size(); ++j)
					delete blockSums[i][j];
		}
		
		
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel, typename FPGAKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel, FPGAKernel>
		::FPGA(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Scan: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				this->scanSingle_FPGA(0, size, res, arg, mode, initial);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				this->scanNumDevices_FPGA(numDevices, size, res, arg, mode, initial);
		}
	}
}

#endif
