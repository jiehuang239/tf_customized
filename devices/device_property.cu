#include <stdio.h> 

int main() {
  int nDevices;

 cudaError_t err =  cudaGetDeviceCount(&nDevices);
 if(err!=cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 
 for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf(" max grid size: %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    printf("  max threads per block: %d\n",prop.maxThreadsPerBlock); 
    printf("  shared memory per block: %d bytes\n",prop.sharedMemPerBlock);
    printf("  registers per block: %d\n",prop.regsPerBlock);
    printf("  total constant memory available on device: %d bytes\n",prop.totalConstMem);
    printf("  total global memory available on device: %d bytes \n",prop.totalGlobalMem);
    printf("  warp size: %d\n",prop.warpSize);
    printf("  compute capability :%d\n", prop.major);
 
 }
}
