#ifndef _CUDA_HELPERS_H_
#define _CUDA_HELPERS_H_

#include <iostream>

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()  __cudaCheckError(__FILE__, __LINE__)

namespace drosophila {

    inline void __cudaCheckMemory()
    {
        // show memory usage of GPU

        size_t free_byte ;
        size_t total_byte ;

        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte );

        if ( cudaSuccess != cuda_status )
        {
            std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
            exit(1);
        }

        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;

        std::cout <<
            "GPU memory usage: used = " << used_db/1024.0/1024.0 <<
            "MB, free = " << free_db/1024.0/1024.0 <<
            "MB, total = " << total_db/1024.0/1024.0 <<
            "MB\n" << std::endl;
    }

    inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
    {
#ifdef _DEBUG
        if ( cudaSuccess != err )
        {
            std::cerr << "cudaSafeCall() failed at " <<
                file << ':' << line << ' ' <<
                cudaGetErrorString(err) <<
                std::endl;

            exit(EXIT_FAILURE);
        }
#endif
        return;
    }

    inline void __cudaCheckError(const char *file, const int line)
    {
#ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            std::cerr << "cudaCheckError() failed at " <<
                file << ':' << line << ' ' <<
                cudaGetErrorString(err) <<
                std::endl;

            exit(EXIT_FAILURE);
        }

        // More careful checking. However, this will affect performance.
        // Comment away if needed.
        err = cudaDeviceSynchronize();

        if( cudaSuccess != err )
        {
            std::cerr << "cudaCheckError() with sync failed at " <<
                file << ':' << line << ' ' <<
                cudaGetErrorString(err) <<
                std::endl;

            exit(EXIT_FAILURE);
        }

#endif
        return;
    }

    inline void __setup_cuda()
    {
        cudaDeviceReset();

        int deviceCount = 0;
        int gpuDeviceCount = 0;

        struct cudaDeviceProp properties;

        std::cout << "Checking for CUDA Device support... ";
        cudaError_t e = cudaGetDeviceCount(&deviceCount);

        if (e != cudaSuccess)
        {
            deviceCount = 0;
        }

        for (int device = 0; device < deviceCount; ++device)
        {
            cudaGetDeviceProperties(&properties, device);
            if (properties.major != 9999) /* 9999 means emulation only */
                ++gpuDeviceCount;
        }

        if (!gpuDeviceCount)
        {
            std::cerr << "No available CUDA Devices found\n";
        }
        else
        {
            std::cout << gpuDeviceCount << " GPU CUDA device(s) found\n";
            cudaDeviceProp prop;
            CudaSafeCall(cudaGetDeviceProperties(&prop, 0));
            std::cout << "Using " << prop.name;
        }

        // if (!prop.canMapHostMemory)
        //   {
        //     std::cerr << "CUDA Error: Device cannot Map Memory\n" << std::endl;
        //     exit(EXIT_FAILURE);
        //   }
        // else
        //   {
        //     CudaSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
        //     CudaCheckError();
        //     std::cout << "OK!\n" << std::endl;
        //   }
    }
}

#endif /* _CUDA_HELPERS_H_ */
