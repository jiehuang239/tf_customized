#ifndef KERNEL_RGB2GREYSCALE_H_
#define KERNEL_RGB2GREYSCALE_H_

template <typename Device, typename T>
struct rgb_greyscaleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#ifdef GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Device, typename T>
struct rgb_greyscalekernelFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};
#endif

#endif
