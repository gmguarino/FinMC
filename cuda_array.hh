#pragma once
#ifndef _CUDA_ARRAY_
#define _CUDA_ARRAY_

#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

template<typename T>
class CudaArray
{
private:
    T* _start;
    T* _end;

    void allocate(size_t size)
    {
        cudaError_t result = cudaMalloc(&_start, size * sizeof(T));
        if (result != cudaSuccess)
        {
            _start = _end = 0;
            throw std::runtime_error("Couldn't allocate memory to device");
        }

        _end = _start + size;
    }

    void free()
    {
        if (_start != 0)
        {
            cudaFree(_start);
            _start = _end = 0;
        }
    }


public:
    CudaArray() : 
        _start(0),
        _end(0)
    {}
    CudaArray(size_t size)
    {
        allocate(size);
    }
    
    ~CudaArray() { free();}

    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    size_t getSize() const
    {
        return _end - _start;
    }

    const T *getData() const
    {
        return _start;
    }

    T* getData()
    {
        return _start;
    }

    void set(T *data, size_t size)
    {
        size_t min = std::min(getSize(), size);
        cudaError_t result = cudaMemcpy(_start, data, min * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            _start = _end = 0;
            throw std::runtime_error("Couldn't copy memory to device");
        }
    }

    void get(T *data_destination, size_t size)
    {
        size_t min = std::min(getSize(), size);
        cudaError_t result = cudaMemcpy(data_destination, _start, min * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            _start = _end = 0;
            throw std::runtime_error("Couldn't copy memory from device");
        }
    }
};


#endif