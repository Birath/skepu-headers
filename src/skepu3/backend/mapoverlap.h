/*! \file mapoverlap.h
 *  \brief Contains a class declaration for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_H
#define MAPOVERLAP_H

#include "skepu3/impl/region.hpp"

namespace skepu
{
	
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
		 */
		/*!
		 *  \class MapOverlap
		 *
		 *  \brief A class representing the MapOverlap skeleton.
		 *
		 *  This class defines the MapOverlap skeleton which is similar to a Map, but each element of the result (vecor/matrix) is a function
		 *  of \em several adjacent elements of one input (vecor/matrix) that reside at a certain constant maximum distance from each other.
		 *  This class can be used to apply (1) overlap to a vector and (2) separable-overlap to a matrix (row-wise, column-wise). For
		 *  non-separable matrix overlap which considers diagonal neighbours as well besides row- and column-wise neighbours, please see \p src/MapOverlap2D.
		 *  MapOverlap2D class can be used by including same header file (i.e., mapoverlap.h) but class name is different (MapOverlap2D).
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel, typename FPGAKernel>
		class MapOverlap1D: public SkeletonBase
		{
			using Ret = typename MapOverlapFunc::Ret;
			using T = typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type;
			using F = ConditionalIndexForwarder<MapOverlapFunc::indexed, MapOverlapFunc::usesPRNG, decltype(&MapOverlapFunc::CPU)>;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapOverlap1D;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
			using UniformArgs = typename MapOverlapFunc::UniformArgs;
			static constexpr bool prefers_matrix = false;
			
			static constexpr size_t arity = 1;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t numArgs = MapOverlapFunc::totalArity - (MapOverlapFunc::indexed ? 1 : 0) - (MapOverlapFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
			
			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity + outArity, outArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity + outArity, arity + outArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity + outArity>::type const_indices{};
			
			MapOverlap1D(CUDAKernel kernel, C2 k2, C3 k3, C4 k4)
			: m_cuda_kernel(kernel), m_cuda_rowwise_kernel(k2), m_cuda_colwise_kernel(k3), m_cuda_colwise_multi_kernel(k4)
			{
#ifdef SKEPU_FPGA
				FPGAKernel::initialize();
#endif

#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setOverlapMode(Overlap mode)
			{
				this->m_overlapPolicy = mode;
			}
			
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}
			
			void setPad(T pad)
			{
				this->m_pad = pad;
			}
			
			void setUpdateMode(UpdateMode mode)
			{
				this->m_updateMode = mode;
			}
			
			void setOverlap(size_t o)
			{
				this->m_overlap = o;
			}
			
			size_t getOverlap() const
			{
				return this->m_overlap;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAKernel m_cuda_kernel;
			C2 m_cuda_rowwise_kernel;
			C3 m_cuda_colwise_kernel;
			C4 m_cuda_colwise_multi_kernel;
			
			Overlap m_overlapPolicy = Overlap::RowWise;
			Edge m_edge = Edge::Duplicate;
			UpdateMode m_updateMode = skepu::UpdateMode::Normal;
			T m_pad {};
			
			size_t m_overlap = 1;
		
		public:
		   
		
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
		   
		
#ifdef SKEPU_OPENMP
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
		
#endif
		
#ifdef SKEPU_CUDA
		public:
		   
		
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapNumDevices_CU(size_t numDevices, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_CUDA(size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU_Col(size_t deviceID, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultiThread_CU_Col(size_t numDevices, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_CUDA(size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultiThread_CU_Row(size_t numDevices, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_CUDA(size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template <typename T>
			size_t getThreadNumber_CU(size_t width, size_t &numThreads, size_t deviceID);
			
			template <typename T>
			bool sharedMemAvailable_CU(size_t &numThreads, size_t deviceID);
			
#endif
		
#ifdef SKEPU_OPENCL
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_OpenCL(size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_OpenCL(size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_OpenCL(size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
		
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapNumDevices_CL(size_t numDevices, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_RowMulti(size_t numDevices, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_Col(size_t deviceID, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_ColMulti(size_t numDevices, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			
			template<typename T>
			int getThreadNumber_CL(size_t width, size_t numThreads, size_t deviceID);
			
			template<typename T>
			bool sharedMemAvailable_CL(size_t &numThreads, size_t deviceID);
			
#endif

#ifdef SKEPU_FPGA
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_FPGA(size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_FPGA(size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_FPGA(size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
		
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_FPGA(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_FPGA_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_FPGA_Col(size_t deviceID, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			
			template<typename T>
			int getThreadNumber_FPGA(size_t width, size_t numThreads, size_t deviceID);
			
			template<typename T>
			bool sharedMemAvailable_FPGA(size_t &numThreads, size_t deviceID);
			
#endif
			
			
		
#ifdef SKEPU_HYBRID
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
		
#endif
		
		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs,
				REQUIRES(is_skepu_vector<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto size = get<outArity>(std::forward<CallArgs>(args)...).size();
				
				// Verify overlap radius is valid
				if (this->m_edge != Edge::None && size < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");
				
				if (disjunction(get<OI>(std::forward<CallArgs>(args)...).size() < size...))
					SKEPU_ERROR("Non-matching output container sizes");
					
				if (disjunction(get<EI>(std::forward<CallArgs>(args)...).size() != size...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				
				this->selectBackend(size);
					
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->vector_Hybrid(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->vector_CUDA(0, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->vector_OpenCL(0, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::FPGA:
#ifdef SKEPU_FPGA
					this->vector_FPGA(0, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif

				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->vector_OpenMP(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->vector_CPU(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
				}
				
				return get<0>(std::forward<CallArgs>(args)...);
			}
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs,
				REQUIRES(is_skepu_matrix<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				
				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");
					
				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
					
				// Verify overlap radius is valid
				if (this->m_overlapPolicy == Overlap::RowWise && this->m_edge != Edge::None && size_j < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");
				
				// Verify overlap radius is valid
				if (this->m_overlapPolicy == Overlap::ColWise && this->m_edge != Edge::None && size_i < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");
				
				
				this->selectBackend(get<outArity>(std::forward<CallArgs>(args)...).size());
				
				switch (this->m_overlapPolicy)
				{	
					case Overlap::ColWise:
						switch (this->m_selected_spec->activateBackend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->colwise_Hybrid(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->colwise_CUDA(size_j, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->colwise_OpenCL(size_j, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::FPGA:
#ifdef SKEPU_FPGA
							this->colwise_FPGA(size_j, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->colwise_OpenMP(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->colwise_CPU(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;
					
					case Overlap::RowWise:
						switch (this->m_selected_spec->activateBackend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->rowwise_Hybrid(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->rowwise_CUDA(size_i, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->rowwise_OpenCL(size_i, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::FPGA:
#ifdef SKEPU_FPGA
							this->rowwise_FPGA(size_i, p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->rowwise_OpenMP(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->rowwise_CPU(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;
						
					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap mode");
				}
				
				return get<0>(std::forward<CallArgs>(args)...);
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->backendDispatch(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->backendDispatch(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->backendDispatch(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}
		};
		
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		class MapOverlap2D: public SkeletonBase
		{
			using Ret = typename MapOverlapFunc::Ret;
			using T = typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type;
			using F = ConditionalIndexForwarder<MapOverlapFunc::indexed, MapOverlapFunc::usesPRNG, decltype(&MapOverlapFunc::CPU)>;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapOverlap2D;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
			using UniformArgs = typename MapOverlapFunc::UniformArgs;
			static constexpr bool prefers_matrix = false;
			
			static constexpr size_t arity = 1;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t numArgs = MapOverlapFunc::totalArity - (MapOverlapFunc::indexed ? 1 : 0) - (MapOverlapFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
			
			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity + outArity, outArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity + outArity, arity + outArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity + outArity>::type const_indices{};
			
			MapOverlap2D(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_FPGA
				FPGAKernel::initialize();
#endif

#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}
			
			void setPad(T pad)
			{
				this->m_pad = pad;
			}
			
			void setUpdateMode(UpdateMode mode)
			{
				this->m_updateMode = mode;
			}
			
			void setOverlap(size_t o)
			{
				this->m_overlap_x = o;
				this->m_overlap_y = o;
			}
			
			void setOverlap(size_t y, size_t x)
			{
				this->m_overlap_x = x;
				this->m_overlap_y = y;
			}
			
			std::pair<size_t, size_t> getOverlap() const
			{
				return std::make_pair(this->m_overlap_x, this->m_overlap_y);
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAKernel m_cuda_kernel;
			
			Edge m_edge = Edge::None;
			UpdateMode m_updateMode = skepu::UpdateMode::Normal;
			T m_pad {};
			
			int m_overlap_x, m_overlap_y;
			
			
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#ifdef SKEPU_OPENMP
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif
		
#ifdef SKEPU_OPENCL
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenCL(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif

#ifdef SKEPU_FPGA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_FPGA(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_FPGA(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);			
#endif
		
#ifdef SKEPU_CUDA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CUDA(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif
			
#ifdef SKEPU_HYBRID
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif
			
		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				
				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching container sizes");
				
				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				// Remove later
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
				// End remove
				
				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());
					
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->helper_Hybrid(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->helper_CUDA(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->helper_OpenCL(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::FPGA:
#ifdef SKEPU_FPGA
					this->helper_FPGA(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->helper_OpenMP(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->helper_CPU(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
				}
				
				return get<0>(std::forward<CallArgs>(args)...);
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->backendDispatch(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->backendDispatch(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->backendDispatch(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}
		};
		
		
		
		
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		class MapOverlap3D: public SkeletonBase
		{
			using Ret = typename MapOverlapFunc::Ret;
			using T = typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type;
			using F = ConditionalIndexForwarder<MapOverlapFunc::indexed, MapOverlapFunc::usesPRNG, decltype(&MapOverlapFunc::CPU)>;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapOverlap3D;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
			using UniformArgs = typename MapOverlapFunc::UniformArgs;
			static constexpr bool prefers_matrix = false;
			
			static constexpr size_t arity = 1;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t numArgs = MapOverlapFunc::totalArity - (MapOverlapFunc::indexed ? 1 : 0) - (MapOverlapFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
			
			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity + outArity, outArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity + outArity, arity + outArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity + outArity>::type const_indices{};
			
			MapOverlap3D(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}
			
			void setPad(T pad)
			{
				this->m_pad = pad;
			}
			
			void setUpdateMode(UpdateMode mode)
			{
				this->m_updateMode = mode;
			}
			
			void setOverlap(int o)
			{
				this->m_overlap_i = o;
				this->m_overlap_j = o;
				this->m_overlap_k = o;
			}
			
			void setOverlap(int oi, int oj, int ok)
			{
				this->m_overlap_i = oi;
				this->m_overlap_j = oj;
				this->m_overlap_k = ok;
			}
			
			std::tuple<int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap_i, this->m_overlap_j, this->m_overlap_k);
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAKernel m_cuda_kernel;
			
			Edge m_edge = Edge::None;
			UpdateMode m_updateMode = skepu::UpdateMode::Normal;
			T m_pad {};
			
			int m_overlap_i, m_overlap_j, m_overlap_k;
			
			
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#ifdef SKEPU_OPENMP
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif
		
#ifdef SKEPU_OPENCL
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenCL(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif

#ifdef SKEPU_FPGA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_FPGA(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_FPGA(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
						
#endif
		
#ifdef SKEPU_CUDA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CUDA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
#endif
			
#ifdef SKEPU_HYBRID
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void helper_Hybrid(Tensor3<Ret>& res, Tensor3<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>,  CallArgs&&... args);
			
#endif
			
		public:
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				
				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() < size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() < size_j) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_k() < size_k) ...))
					SKEPU_ERROR("Non-matching output container sizes");
				
				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != size_j) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_k() != size_k) ...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				// Remove later
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
				// End remove
				
				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());
					
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
				//	this->helper_Hybrid(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->helper_CUDA(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->helper_OpenCL(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif

				case Backend::Type::FPGA:
#ifdef SKEPU_FPGA
					this->helper_FPGA(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif

				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->helper_OpenMP(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->helper_CPU(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
				}
				
				return get<0>(std::forward<CallArgs>(args)...);
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->backendDispatch(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->backendDispatch(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->backendDispatch(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}
		};
		
		
		
		
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		class MapOverlap4D: public SkeletonBase
		{
			using Ret = typename MapOverlapFunc::Ret;
			using T = typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type;
			using F = ConditionalIndexForwarder<MapOverlapFunc::indexed, MapOverlapFunc::usesPRNG, decltype(&MapOverlapFunc::CPU)>;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::MapOverlap4D;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
			using UniformArgs = typename MapOverlapFunc::UniformArgs;
			static constexpr bool prefers_matrix = false;
			
			static constexpr size_t arity = 1;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t numArgs = MapOverlapFunc::totalArity - (MapOverlapFunc::indexed ? 1 : 0) - (MapOverlapFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;
			
			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity + outArity, outArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity + outArity, arity + outArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity + outArity>::type const_indices{};
			
			MapOverlap4D(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}
			
			void setPad(T pad)
			{
				this->m_pad = pad;
			}
			
			void setUpdateMode(UpdateMode mode)
			{
				this->m_updateMode = mode;
			}
			
			void setOverlap(int o)
			{
				this->m_overlap_i = o;
				this->m_overlap_j = o;
				this->m_overlap_k = o;
				this->m_overlap_l = o;
			}
			
			void setOverlap(int oi, int oj, int ok, int ol)
			{
				this->m_overlap_i = oi;
				this->m_overlap_j = oj;
				this->m_overlap_k = ok;
				this->m_overlap_l = ol;
			}
			
			std::tuple<int, int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l);
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAKernel m_cuda_kernel;
			
			Edge m_edge = Edge::None;
			UpdateMode m_updateMode = skepu::UpdateMode::Normal;
			T m_pad {};
			
			int m_overlap_i, m_overlap_j, m_overlap_k, m_overlap_l;
			
			
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#ifdef SKEPU_OPENMP
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif
		
#ifdef SKEPU_OPENCL
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenCL(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif

#ifdef SKEPU_FPGA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_FPGA(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_FPGA(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif
	
#ifdef SKEPU_CUDA
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CUDA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);
			
#endif
			
#ifdef SKEPU_HYBRID
			
			template<size_t... AnyIndx, size_t... ConstIndx, typename... CallArgs>
			void helper_Hybrid(Tensor4<Ret>& res, Tensor4<T>& arg, pack_indices<AnyIndx...>, pack_indices<ConstIndx...>,  CallArgs&&... args);
			
#endif
			
		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();
				
				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() < size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() < size_j) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_k() < size_k) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_l() < size_l)...))
					SKEPU_ERROR("Non-matching output container sizes");
				
				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != size_j) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_k() != size_k) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_l() != size_l)...))
					SKEPU_ERROR("Non-matching input container sizes");
				
				// Remove later
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
				// End remove
				
				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());
					
				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					//this->helper_Hybrid(res, arg, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->helper_CUDA(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->helper_OpenCL(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::FPGA:
#ifdef SKEPU_FPGA
					this->helper_FPGA(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->helper_OpenMP(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->helper_CPU(p, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
					break;
				}
				
				return get<0>(std::forward<CallArgs>(args)...);
			}
			
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->backendDispatch(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->backendDispatch(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->backendDispatch(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}
		};
		
		
		
		
		
	} // namespace backend

#ifdef SKEPU_MERCURIUM

template<typename Ret, typename Arg1, typename... Args>
class MapOverlapImpl: public SeqSkeletonBase
{
protected:
	using T =
		typename std::remove_const<typename std::remove_pointer<Arg1>::type>::type;
	using MapFunc1D = std::function<Ret(int, size_t, Arg1, Args...)>;
	using MapFunc2D = std::function<Ret(int, int, size_t, Arg1, Args...)>;

public:
	void setOverlapMode(Overlap mode);
	void setEdgeMode(Edge mode);
	void setPad(T pad);

	MapOverlapImpl(MapFunc1D map);
	MapOverlapImpl(MapFunc2D map);

	void setOverlap(size_t o);
	void setOverlap(size_t y, size_t x);
	size_t getOverlap() const;
	std::pair<size_t, size_t> getOverlap() const;

	template<
		template<class> class Container,
		size_t... AI,
		size_t... CI,
		typename... CallArgs>
	Container<Ret> &helper(
		Container<Ret> &res,
		Container<T> &arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<template<class> class Container, typename... CallArgs>
	Container<Ret> &operator()(
		Container<Ret> &res, Container<T>& arg, CallArgs&&... args);

	template<size_t... AI, size_t... CI, typename... CallArgs>
	void apply_colwise(
		skepu::Matrix<Ret>& res,
		skepu::Matrix<T>& arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<size_t... AI, size_t... CI, typename... CallArgs>
	void apply_rowwise(
		skepu::Matrix<Ret>& res,
		skepu::Matrix<T>& arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<typename... CallArgs>
	Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args);

	template<size_t... AI, size_t... CI, typename... CallArgs>
	void apply_helper(
		Matrix<Ret> &res,
		Matrix<T> &arg,
		pack_indices<AI...>,
		pack_indices<CI...>,
		CallArgs&&... args);

	template<typename... CallArgs>
	Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args);

};

template<typename Ret, typename Arg1, typename ... ArgRest>
auto inline
MapOverlap(Ret (*)(int, size_t, Arg1, ArgRest...))
-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

template<typename Ret, typename Arg1, typename ... ArgRest>
auto inline
MapOverlap(std::function<Ret(int, size_t, Arg1, ArgRest...)>)
-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

template<typename Ret, typename Arg1, typename ... ArgRest>
auto inline
MapOverlap(Ret (*)(int, int, size_t, Arg1, ArgRest...))
-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

template<typename Ret, typename Arg1, typename ... ArgRest>
auto inline
MapOverlap(std::function<Ret(int, int, size_t, Arg1, ArgRest...)>)
-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

template<typename T>
auto inline
MapOverlap(T op)
-> decltype(MapOverlap(lambda_cast(op)));

#endif // SKEPU_MERCURIUM

} // namespace skepu


#include "impl/mapoverlap/mapoverlap_cpu.inl"
#include "impl/mapoverlap/mapoverlap_omp.inl"
#include "impl/mapoverlap/mapoverlap_cl.inl"
#include "impl/mapoverlap/mapoverlap_cu.inl"
#include "impl/mapoverlap/mapoverlap_hy.inl"
#include "impl/mapoverlap/mapoverlap_fpga.inl"

#endif // MAPOVERLAP_H
