/*! \file debug.h
 *  \brief Defines a few macros that includes macros to output text when debugging. The macros use std::cerr.
 */

#ifndef DEBUG_H
#define DEBUG_H

#include <assert.h>
#include <sstream>
#include <iomanip>
#include <chrono>

#ifndef SKEPU_DEBUG
#define SKEPU_DEBUG 0
#endif


#if SKEPU_TUNING_DEBUG > 0
#include <iostream>
#endif

#if SKEPU_TUNING_DEBUG > 1
#define DEBUG_TUNING_LEVEL2(skepu_macro_text) std::cerr << "[SKEPU_TUNING_L1 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TUNING_LEVEL2(skepu_macro_text)
#endif

#if SKEPU_TUNING_DEBUG > 2
#define DEBUG_TUNING_LEVEL3(skepu_macro_text) std::cerr << "[SKEPU_TUNING_L2 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TUNING_LEVEL3(skepu_macro_text)
#endif

#if SKEPU_DEBUG > 0
#include <iostream>
#endif

#if SKEPU_DEBUG > 0
#define DEBUG_TEXT_LEVEL1(skepu_macro_text) std::cerr << "[" << debug_timestamp() << "][SKEPU_DEBUG_L1 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TEXT_LEVEL1(skepu_macro_text)
#endif

#if SKEPU_DEBUG > 1
#define DEBUG_TEXT_LEVEL2(skepu_macro_text) std::cerr << "[" << debug_timestamp() << "][SKEPU_DEBUG_L2 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TEXT_LEVEL2(skepu_macro_text)
#endif

#if SKEPU_DEBUG > 2
#define DEBUG_TEXT_LEVEL3(skepu_macro_text) std::cerr << "[" << debug_timestamp() << "][SKEPU_DEBUG_L3 " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n";
#else
#define DEBUG_TEXT_LEVEL3(skepu_macro_text)
#endif


#ifndef SKEPU_ASSERT
#define SKEPU_ASSERT(expr) assert(expr)
#endif // SKEPU_ASSERT

#ifdef SKEPU_ENABLE_EXCEPTIONS
#define SKEPU_ERROR(skepu_macro_text) { std::stringstream skepu_macro_msg; skepu_macro_msg << skepu_macro_text; throw(skepu_macro_msg.str()); }
#else
#define SKEPU_ERROR(skepu_macro_text) { std::cerr << "[SKEPU_ERROR " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n"; exit(1); }
#endif // SKEPU_ENABLE_EXCEPTIONS

#define SKEPU_WARNING(skepu_macro_text) { std::cerr << "[SKEPU_WARNING " << __FILE__ << ":" << __LINE__ << "] " << skepu_macro_text << "\n"; }

#define SKEPU_EXIT() exit(0)

#ifdef __GNUC__
#define SKEPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#define SKEPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))
#define SKEPU_ATTRIBUTE_UNUSED        __attribute__((unused))
#define SKEPU_ATTRIBUTE_INTERNAL      __attribute__ ((visibility ("internal")))
#else
#define SKEPU_UNLIKELY(expr)          (expr)
#define SKEPU_LIKELY(expr)            (expr)
#define SKEPU_ATTRIBUTE_UNUSED
#define SKEPU_ATTRIBUTE_INTERNAL
#endif

#ifndef SKEPU_NO_FORCE_INLINE
// Force inline in GCC and Clang (should also apply to NVCC?)
#if defined(__GNUC__) || defined(__clang__)
#define SKEPU_ATTRIBUTE_FORCE_INLINE __attribute__((always_inline))
// Force inline in MS VC
#elif defined(_MSC_VER)
#define SKEPU_ATTRIBUTE_FORCE_INLINE __forceinline
#else
// Intel compiler?
#define SKEPU_ATTRIBUTE_FORCE_INLINE
#endif
#else
#define SKEPU_ATTRIBUTE_FORCE_INLINE
#endif

#ifdef SKEPU_OPENCL
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/*!
* A helper function for OpenCL backends. It takes an OpenCL error code and prints the corresponding error message
*
* \param Err OpenCL error
* \param s Optional text string that may give more information on the error source
*/
inline void printCLError(cl_int Err, std::string s)
{
	std::string msg;
	if (Err != CL_SUCCESS)
	{
		switch(Err)
		{
		case CL_DEVICE_NOT_FOUND:
			msg = "Device not found"; break;
		case CL_DEVICE_NOT_AVAILABLE:
			msg = "Device not available"; break;
		case CL_COMPILER_NOT_AVAILABLE:
			msg = "Compiler not available"; break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			msg = "Memory object allocation failure"; break;
		case CL_OUT_OF_RESOURCES:
			msg = "Out of resources"; break;
		case CL_OUT_OF_HOST_MEMORY:
			msg = "Out of host memory"; break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			msg = "Profiling info not available"; break;
		case CL_MEM_COPY_OVERLAP:
			msg = "Memory copy overlap"; break;
		case CL_IMAGE_FORMAT_MISMATCH:
			msg = "Image format mismatch"; break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			msg = "Image format not supported"; break;
		case CL_BUILD_PROGRAM_FAILURE:
			msg = "Build program failure"; break;
		case CL_MAP_FAILURE:
			msg = "Map failure"; break;
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			msg = "Misaligned sub buffer offset"; break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			msg = "Exec status error for events in wait list"; break;
		case CL_INVALID_VALUE:
			msg = "Invalid value"; break;
		case CL_INVALID_DEVICE_TYPE:
			msg = "Invalid device type"; break;
		case CL_INVALID_PLATFORM:
			msg = "Invalid platform"; break;
		case CL_INVALID_DEVICE:
			msg = "Invalid device"; break;
		case CL_INVALID_CONTEXT:
			msg = "Invalid context"; break;
		case CL_INVALID_QUEUE_PROPERTIES:
			msg = "Invalid queue properties"; break;
		case CL_INVALID_COMMAND_QUEUE:
			msg = "Invalid command queue"; break;
		case CL_INVALID_HOST_PTR:
			msg = "Invalid host pointer"; break;
		case CL_INVALID_MEM_OBJECT:
			msg = "Invalid memory object"; break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			msg = "Invalid image format descriptor"; break;
		case CL_INVALID_IMAGE_SIZE:
			msg = "Invalid image size"; break;
		case CL_INVALID_SAMPLER:
			msg = "Invalid sampler"; break;
		case CL_INVALID_BINARY:
			msg = "Invalid binary"; break;
		case CL_INVALID_BUILD_OPTIONS:
			msg = "Invalid build options"; break;
		case CL_INVALID_PROGRAM:
			msg = "Invalid program"; break;
		case CL_INVALID_PROGRAM_EXECUTABLE:
			msg = "Invalid program executable"; break;
		case CL_INVALID_KERNEL_NAME:
			msg = "Invalid kernel name"; break;
		case CL_INVALID_KERNEL_DEFINITION:
			msg = "Invalid kernel definition"; break;
		case CL_INVALID_KERNEL:
			msg = "Invalid kernel"; break;
		case CL_INVALID_ARG_INDEX:
			msg = "Invalid argument index"; break;
		case CL_INVALID_ARG_VALUE:
			msg = "Invalid argument value"; break;
		case CL_INVALID_ARG_SIZE:
			msg = "Invalid argument size"; break;
		case CL_INVALID_KERNEL_ARGS:
			msg = "Invalid kernel arguments"; break;
		case CL_INVALID_WORK_DIMENSION:
			msg = "Invalid work dimension"; break;
		case CL_INVALID_WORK_GROUP_SIZE:
			msg = "Invalid work group size"; break;
		case CL_INVALID_WORK_ITEM_SIZE:
			msg = "Invalid work item size"; break;
		case CL_INVALID_GLOBAL_OFFSET:
			msg = "Invalid global offset"; break;
		case CL_INVALID_EVENT_WAIT_LIST:
			msg = "Invalid event wait list"; break;
		case CL_INVALID_EVENT:
			msg = "Invalid event"; break;
		case CL_INVALID_OPERATION:
			msg = "Invalid operation"; break;
		case CL_INVALID_GL_OBJECT:
			msg = "Invalid GL object"; break;
		case CL_INVALID_BUFFER_SIZE:
			msg = "Invalid buffer size"; break;
		case CL_INVALID_MIP_LEVEL:
			msg = "Invalid MIP level"; break;
		case CL_INVALID_GLOBAL_WORK_SIZE:
			msg = "Invalid global work size"; break;
		case CL_INVALID_PROPERTY:
			msg = "Invalid property"; break;
		default:
			msg = "Unknown error"; break;
		}
		std::cerr << s << " OpenCL error code " << Err << " " << msg << "\n";
	}
}
			

template<typename ERROR_T, typename... MESSAGE_T>
auto inline
CL_CHECK_ERROR(ERROR_T const & err, MESSAGE_T const & ... m)
-> void
{
	if(err != CL_SUCCESS)
	{
		/* Unpack the messages and print them. */
		int print[sizeof...(m)] = {(std::cerr << m,0)...};
		std::cerr  << ": " << err << "\n";
		printCLError(err, "");
		// exit(0);
	}
}
#endif

inline std::string debug_timestamp()
{
	std::stringstream s;
/*	auto now = std::chrono::system_clock::now();
	auto now_c = std::chrono::system_clock::to_time_t(now);
	s << std::put_time(std::localtime(&now_c), "%H:%M:%S.");*/
	s << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	return s.str();
}

#endif
