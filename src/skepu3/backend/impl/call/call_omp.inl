/*! \file call_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Call skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<typename CallFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel, FPGAKernel>
		::OMP(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP Call");
			
			// Sync with device data
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(CallFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
#pragma omp parallel
			CallFunc::OMP(get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
			
		}
	}
}

#endif // SKEPU_OPENMP