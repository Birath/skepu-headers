/*! \file map_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapPairs skeleton.
 */

namespace skepu
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel, typename FPGAKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel, FPGAKernel> 
		::CPU(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU MapPairs: hsize = " << Hsize << ", vsize = " << Vsize);
			
			// Sync with device data
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			pack_expand((get<HEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			
			auto random = this->template prepareRandom<MapPairsFunc::randomCount>(Vsize * Hsize);
			
			for (size_t i = 0; i < Vsize; ++i)
			{
				for (size_t j = 0; j < Hsize; ++j)
				{
					Index2D index{ i, j };
					auto res = F::forward(MapPairsFunc::CPU, index, random,
						get<VEI>(std::forward<CallArgs>(args)...)(i)...,
						get<HEI>(std::forward<CallArgs>(args)...)(j)...,
						get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-Varity-Harity-outArity>(typename MapPairsFunc::ProxyTags{}), index)...,
						get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN_INPLACE(this->m_in_place, get<OI>(std::forward<CallArgs>(args)...)(i, j)..., res);
				}
			}
		}
	}
}

