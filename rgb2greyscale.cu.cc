#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_rgb2greyscale.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
// Define the CUDA kernel.
template <typename T>
__global__ void rgb_greyscaleCudaKernel(const int size, const T* in, T* out) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int N_pixels = size/3;
    if(i<N_pixels) {
        out[i] = in[i]*0.21+in[i+1]*0.71+in[i+2]*0.07;
    //    printf("gpu_output[i]=%d\n",gpu_output[i]);
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void rgb_greyscaleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 256;
  int thread_per_block = 256;
  rgb_greyscaleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct rgb_greyscale_kernelFunctor<GPUDevice, float>;
template struct rgb_greyscale_kernelFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
