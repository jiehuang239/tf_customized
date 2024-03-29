#include "kernel_rgb2greyscale.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
//step 1. CPU specialization of actual computation
template <typename T>
struct rgb_greyscaleFunctor<CPUDevice,T> {
    void operator()(const CPUDevice& d,int size,const T*in,T*out) {
        int N_pixels = size/3;
        for(int i=0;i<N_pixels;i++) {
            out[i] = in[3*i]*0.21+in[3*i+1]*0.71+in[3*i+2]*0.07;
        }
    }
};

// Inherit OpKernel and override the compute method to implement your own kernel
//template T is the data type of the tensors
template<typename Device,typename T>
class rgb_greyscale : public OpKernel {
    public:
      explicit rgb_greyscale(OpKernelConstruction* context):OpKernel(context) {}
      virtual void compute( OpKernelContext* context) {
          //your can access input and output tensors from context
          //grab the input tensor
          const Tensor& input_tensor = context->input(0);
	  auto input = input_tensor.flat<float>();
          //create an output tensor
          Tensor* output_tensor = NULL; 
          OP_REQUIRES_OK(context,context->allocate_output(0,input_tensor.shape(),&output_tensor));
	  auto output = output_tensor->template flat<float>(); 
          //get in/out pointers and their size
          //const T* input_ptr = input_tensor.flat<T>().data();
          //T*output_ptr = output_tensor->flat<T>().data();
          //int num_elements = input_tensor.NumElements();
          //Do the computation
	  const int N = input.size()/3;
	  for (int i = 1;i< N;i++) {
	  	output(i) = input(3*i)*0.21+input(3*i+1)*0.71+input(3*i+2);
	  }
	  return;
      } 
    
};
// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("rgb2greyscale").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      rgb_greyscale<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in rgb2greyscale.cu.cc. */ \
  extern template rgb_greyscalekernelFunctor<GPUDevice, T>;                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("rgb2greyscale").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      rgb_greyscale<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
