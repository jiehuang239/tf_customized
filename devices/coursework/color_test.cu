#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "png.h"
/* Portable Network Graphic (PNG) format. It contains a bitmap of indexed colors and uses
loseless compression, similar to .gif file but without copyright limitations.
PNG images may include an 8-bit transparency channel, which allows the image colors to fade 
from opaque to transparent.
*/
/* cimg library: http://cimg.eu/reference/structcimg__library_1_1CImg.html
reference: 1. RGB to grey scale using CIMG :http://obsessive-coffee-disorder.com/rgb-to-grayscale-using-cimg/
Measure execution time using std::chrono, https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
2. read pixel value in bmp file: https://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file/38440684
*/
//step 1. read images. CUDA boards do not provide access to the file system. 
//You can only do it via a CPU and then transfer the image to the CUDA board's image
//using namespace cimg_library;
__global__ void rgb2greyscale(unsigned char*base_pointer,unsigned char *gpu_output,int width,int height) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width*height) {
        gpu_output[i] = base_pointer[i]*0.21+base_pointer[i+1]*0.71+base_pointer[i+2]*0.07;
    //    printf("gpu_output[i]=%d\n",gpu_output[i]);
    }
}
unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];
    printf("width=%d\n",width);
    printf("height=%d\n",height); 
    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    for(i = 0; i < size; i += 3)
    {
            unsigned char tmp = data[i];
            data[i] = data[i+2];
            data[i+2] = tmp;
    }

    return data;
}
int main() {
    /*CImg<unsigned char> image("color.png");
    CImg<unsigned char> grey(image.width(),image.height(),1,1,0);
    CImg<unsigned char> grey_gpu(image.width(),image.height(),1,1,0);
    printf("width = %d, height= %d, depth= %d, spectrum= %d\n",image._width,image._height,image._depth,image._spectrum);
    const clock_t start = clock();
    for (int i=0;i<512*384;i++) {    
            //printf("%d ",image._data[3*i+j]);
            grey[i]=(int)(image._data[i+0]*0.21+image._data[i+1]*0.71+image._data[i+2]*0.07);    
        //printf("\n");
    }    
    //image.display();
    const clock_t stop = clock();
    printf("execution time = %f\n",(float)(stop-start)/CLOCKS_PER_SEC);
    */
    //cuda code
    unsigned char *img,*img_grey;
    unsigned char*d_img_rgb,*d_img_grey;
    int N = 3*512*384;//3*image._width*image._height*sizeof(unsigned char);
    img =(unsigned char*) malloc(N*sizeof(unsigned char));
    img_grey =(unsigned char*) malloc(N/3*sizeof(unsigned char));
   // for (int i=0;i<N;i++) {
   // 	img[i]=134;
    //}
    FILE*f = fopen("prepare.txt","rb");
    unsigned int width;
    unsigned int height;
    fread(&width,sizeof(unsigned int),1,f);
    fread(&height,sizeof(unsigned int),1,f);
    fread(img,sizeof(unsigned char),width*height*3,f);
    fclose(f);
    printf("width=%d,height=%d\n",width,height);
    cudaMalloc((void**)&d_img_rgb,N*sizeof(unsigned char));
    cudaMalloc((void**)&d_img_grey,N/3*sizeof(unsigned char));
    cudaMemcpy((void*)d_img_rgb,(void*)img,N,cudaMemcpyHostToDevice);
    rgb2greyscale<<<((N/3+255)/256),256>>>(d_img_rgb,d_img_grey,512,384);
    cudaMemcpy(img_grey,d_img_grey,N/3*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    FILE*f_w = fopen("result.txt","w");
    fwrite(&width,sizeof(unsigned int),1,f_w);
    fwrite(&height,sizeof(unsigned int),1,f_w);
    fwrite(img,sizeof(unsigned char),width*height,f_w);
    fclose(f_w);
    cudaFree(d_img_rgb);
    cudaFree(d_img_grey);
    free(img);
    free(img_grey);
}
/*

==> Next steps:
- Install the Linuxbrew dependencies if you have sudo access:
  Debian, Ubuntu, etc.
    sudo apt-get install build-essential
  Fedora, Red Hat, CentOS, etc.
    sudo yum groupinstall 'Development Tools'
  See http://linuxbrew.sh/#dependencies for more information.
- Add Linuxbrew to your ~/.bash_profile by running
    echo 'export PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"' >>~/.bash_profile
    echo 'export MANPATH="/home/linuxbrew/.linuxbrew/share/man:$MANPATH"' >>~/.bash_profile
    echo 'export INFOPATH="/home/linuxbrew/.linuxbrew/share/info:$INFOPATH"' >>~/.bash_profile
- Add Linuxbrew to your PATH
    PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"
- We recommend that you install GCC by running:
    brew install gcc
- Run `brew help` to get started
- Further documentation: 
    http://docs.brew.sh
Warning: /home/linuxbrew/.linuxbrew/bin is not in your PATH.
/usr/bin/brew: 78: exec: /home/jh4000/.linuxbrew/bin/brew: not found
*/
//sudo apt-get install libpng-dev
/*
isl@0.18 is keg-only, which means it was not symlinked into /home/linuxbrew/.linuxbrew,
because this is an alternate version of another formula.
For compilers to find isl@0.18 you may need to set:
  export LDFLAGS="-L/home/linuxbrew/.linuxbrew/opt/isl@0.18/lib"
  export CPPFLAGS="-I/home/linuxbrew/.linuxbrew/opt/isl@0.18/include"
For pkg-config to find isl@0.18 you may need to set:
*/
