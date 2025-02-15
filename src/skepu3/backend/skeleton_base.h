/*! \file scan.h
 *  \brief Contains a class declaration for the skeleton base class.
 */

#ifndef SKELETON_BASE_H
#define SKELETON_BASE_H

#include "skepu3/backend/environment.h"

namespace skepu
{
	namespace backend
	{
		class SkeletonBase
		{
		public:
			void finishAll()
			{
				this->m_environment->finishAll();
			}
			
			// Transfers ownership of ´plan´
			void setExecPlan(ExecPlan *plan)
			{
				// Clean up old plan
				if (this->m_execPlan != nullptr)
					delete this->m_execPlan;
				
				this->m_execPlan = plan;
			}
			
			void setBackend(BackendSpec const& spec)
			{
				this->m_user_spec = new BackendSpec(spec);
			}
			
			void resetBackend()
			{
				if (this->m_user_spec)
					delete this->m_user_spec;
				this->m_user_spec = nullptr;
			}
			
			const BackendSpec& selectBackend(size_t size = 0)
			{
				if (this->m_user_spec)
					this->m_selected_spec = this->m_user_spec;
				else if (this->m_execPlan)
					this->m_selected_spec = &this->m_execPlan->find(size);
				else
					this->m_selected_spec = &internalGlobalBackendSpecAccessor();
				
			//	this->m_selected_spec = (this->m_user_spec != nullptr)
			//		? this->m_user_spec
			//		: &this->m_execPlan->find(size);
				return *this->m_selected_spec;
			}
			
			void setPRNG(PRNG &prng, size_t iterations = 1)
			{
				this->m_prng = &prng;
				this->m_prng->registerInstance(this, iterations);
			}
			
			constexpr int setFPGAUnroll(size_t unrollFactor)
			{
				return 0;
			}
			
		protected:
			SkeletonBase()
			{
				this->m_environment = Environment<int>::getInstance();
			/*
#if defined(SKEPU_OPENCL)
				BackendSpec bspec(Backend::Type::OpenCL);
				bspec.setDevices(this->m_environment->m_devices_CL.size());
				bspec.setGPUThreads(this->m_environment->m_devices_CL.at(0)->getMaxThreads());
				bspec.setGPUBlocks(this->m_environment->m_devices_CL.at(0)->getMaxBlocks());
			
#elif defined(SKEPU_CUDA)
				BackendSpec bspec(Backend::Type::CUDA);
				bspec.setDevices(this->m_environment->m_devices_CU.size());
				bspec.setGPUThreads(this->m_environment->m_devices_CU.at(0)->getMaxThreads());
				bspec.setGPUBlocks(this->m_environment->m_devices_CU.at(0)->getMaxBlocks());
			
#elif defined(SKEPU_OPENMP)
				BackendSpec bspec(Backend::Type::OpenMP);
				bspec.setCPUThreads(omp_get_max_threads());
				
#else
				BackendSpec bspec(Backend::Type::CPU);
#endif
				ExecPlan *plan = new ExecPlan();
				plan->setCalibrated();
				plan->add(1, MAX_SIZE, bspec);
				setExecPlan(plan);
			*/
			}
			
			
			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			Random<randomCount> prepareRandom(size_t size)
			{
				if (this->m_prng == nullptr)
					SKEPU_ERROR("No random stream set in skeleton instance");
				return this->m_prng->template asRandom<randomCount>(size);
			}
			
			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			skepu::Vector<Random<randomCount>> prepareRandom(size_t size, size_t copies, size_t atomic_size = 1)
			{
				if (this->m_prng == nullptr)
					SKEPU_ERROR("No random stream set in skeleton instance");
				return this->m_prng->template asRandom<randomCount>(size, copies, atomic_size);
			}
			
			template<size_t randomCount, REQUIRES(randomCount == SKEPU_NO_RANDOM)>
			PRNG::Placeholder prepareRandom(size_t size = 0, size_t copies = 0, size_t atomic_size = 0)
			{
				return {};
			}
			
#ifdef SKEPU_OPENCL

			template<size_t randomCount, REQUIRES(randomCount != SKEPU_NO_RANDOM)>
			skepu::Vector<RandomForCL> prepareRandom_CL(size_t size, size_t copies, size_t atomic_size = 1)
			{
				if (this->m_prng == nullptr)
					SKEPU_ERROR("No random stream set in skeleton instance");
				return this->m_prng->template asRandom_CL<randomCount>(size, copies, atomic_size);
			}
			
			template<size_t randomCount, REQUIRES(randomCount == SKEPU_NO_RANDOM)>
			PRNG::Placeholder prepareRandom_CL(size_t size = 0, size_t copies = 0, size_t atomic_size = 0)
			{
				return {};
			}
			
#endif
			
			
			Environment<int>* m_environment;
			
			/*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
			ExecPlan *m_execPlan = nullptr;
			
			const BackendSpec *m_user_spec = nullptr;
			
			const BackendSpec *m_selected_spec = nullptr;
			
			PRNG *m_prng = nullptr;
			
		}; // class SkeletonBase
		
	} // namespace backend
	
} // namespace skepu


#endif // SKELETON_BASE_H
