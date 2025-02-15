/*! \file mapoverlap_hy.inl
*  \brief Contains the definitions of Hybrid execution specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_HYBRID

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Performs the MapOverlap on a range of elements using \em Hybrid backend and a seperate output range.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::vector_Hybrid(skepu::Parity, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const int overlap = this->m_overlap;
			const size_t size = arg.size();
			const size_t stride = 1;
			
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			const size_t nthr = this->m_selected_spec->CPUThreads();
			const size_t numCPUThreads = nthr-1;

			DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize < 32) { // Not smaller than a warp (=32 threads)
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small GPU size, fall back to CPU-only.");
				this->vector_OpenMP(oi, ei, ai, ci, std::forward<CallArgs>(args)...);
				return;
			}
			else if(cpuSize < numCPUThreads) {
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				this->vector_CUDA(0, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#else
				this->vector_OpenCL(0, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#endif
				return;
			}
			
			T start[3*overlap], end[3*overlap];
			
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			
			omp_set_nested(true);
			
#pragma omp parallel num_threads(2)
			{
				if(omp_get_thread_num() == 0) {
					// Let first thread handle GPU
#ifdef SKEPU_HYBRID_USE_CUDA
					this->vector_CUDA(cpuSize, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#else
					this->vector_OpenCL(cpuSize, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#endif
				}
				else {
					// CPU threads
#pragma omp parallel for num_threads(numCPUThreads)
					for (size_t i = 0; i < overlap; ++i)
					{
						switch (this->m_edge)
						{
						case Edge::Cyclic:
							start[i] = arg(size + i  - overlap);
							end[3*overlap-1 - i] = arg(overlap-i-1);
							break;
						case Edge::Duplicate:
							start[i] = arg(0);
							end[3*overlap-1 - i] = arg(size-1);
							break;
						case Edge::Pad:
							start[i] = this->m_pad;
							end[3*overlap-1 - i] = this->m_pad;
						}
					}
					
					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = arg(j);
					
					for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
						end[i] = arg(j + size - 2*overlap);
					
					for (size_t i = 0; i < overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &start[i + overlap]},
								get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
					}
						
#pragma omp parallel for num_threads(numCPUThreads)
					for (size_t i = overlap; i < size - overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &arg(i)},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
					}
						
					for (size_t i = size - overlap; i < size; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &end[i + 2 * overlap - size]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
					}
				} // end else
				
			} // end omp parallel num_threads(2)
			
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em Hybrid backend with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::rowwise_Hybrid(skepu::Parity, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const size_t rowWidth = arg.total_cols();
			const size_t stride = 1;
			
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuRows = cpuPartitionSize*arg.total_rows();
			const size_t gpuRows = arg.total_rows()-cpuRows;
			const size_t nthr = this->m_selected_spec->CPUThreads();
			const size_t numCPUThreads = nthr-1;
			
			DEBUG_TEXT_LEVEL1("Hybrid row-wise MapOverlap: rows = " << arg.total_rows() << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuRows == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small GPU size, fall back to CPU-only.");
				this->rowwise_OpenMP(oi, ei, ai, ci, std::forward<CallArgs>(args)...);
				return;
			}
			else if(cpuRows < numCPUThreads) {
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				this->rowwise_CUDA(arg.total_rows(), oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#else
				this->rowwise_OpenCL(arg.total_rows(), oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#endif
				return;
			}
			
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			
			const int overlap = this->m_overlap;
			T start[3*overlap], end[3*overlap];
			
			const T *inputBegin = arg.getAddress() + gpuRows*rowWidth;
			const T *inputEnd = inputBegin + cpuRows*rowWidth;
			
			omp_set_nested(true);
			
			// Let GPU take the _first_ gpuRows of the matrix, and the CPU the _last_ cpuRows.
#pragma omp parallel num_threads(2)
			{
				if(omp_get_thread_num() == 1) {
					// Let last thread handle GPU
#ifdef SKEPU_HYBRID_USE_CUDA
					this->rowwise_CUDA(gpuRows, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#else
					this->rowwise_OpenCL(gpuRows, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#endif
				}
				else {
					// CPU Part
					for (size_t row = gpuRows; row < arg.total_rows(); ++row)
					{
						inputEnd = inputBegin + rowWidth;
						
#pragma omp parallel for num_threads(numCPUThreads)
						for (size_t i = 0; i < overlap; ++i)
						{
							switch (this->m_edge)
							{
							case Edge::Cyclic:
								start[i] = inputEnd[i  - overlap];
								end[3*overlap-1 - i] = inputBegin[overlap-i-1];
								break;
							case Edge::Duplicate:
								start[i] = inputBegin[0];
								end[3*overlap-1 - i] = inputEnd[-1];
								break;
							case Edge::Pad:
								start[i] = this->m_pad;
								end[3*overlap-1 - i] = this->m_pad;
								break;
							}
						}
						
						for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
							start[i] = inputBegin[j];
						
						for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
							end[i] = inputEnd[j - 2*overlap];
						
						for (size_t i = 0; i < overlap; ++i)
						{
							auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &start[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
						}
							
#pragma omp parallel for num_threads(numCPUThreads)
						for (size_t i = overlap; i < rowWidth - overlap; ++i)
						{
							auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &inputBegin[i]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
						}
						
						for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
						{
							auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, Region1D<T>{overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
						}
						
						inputBegin += rowWidth;
					}
				} // end CPU part
			} // END omp parallel
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em Hybrid backend with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel, FPGAKernel>
		::colwise_Hybrid(skepu::Parity, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			std::cout << "WARNING: colwise_Hybrid is not implemented for Hybrid exection yet. Will run OpenMP version." << std::endl;
			
			this->colwise_OpenMP(oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel, FPGAKernel>
		::helper_Hybrid(skepu::Parity, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			std::cout << "WARNING: helper_Hybrid is not implemented for Hybrid exection yet. Will run OpenMP version." << std::endl;
			
			this->helper_OpenMP(oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
	} // namespace backend
} // namespace skepu

#endif
