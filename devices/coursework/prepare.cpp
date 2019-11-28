#include "CImg.h"
#include <stdio.h>
#include <stdlib.h>
using namespace cimg_library;
int main() {
	CImg<unsigned char> image("color.png");
	printf("width = %d, height= %d, depth= %d, spectrum= %d\n",image._width,image._height,image._depth,image._spectrum);
	FILE *f = fopen("prepare.txt","w");
	int N;
	N=image._width*image._height*3;
	fwrite(&image._width,sizeof(unsigned int),1,f);
	fwrite(&image._height,sizeof(unsigned int),1,f);
	fwrite(image._data,sizeof(unsigned char),N,f);
	fclose(f);
	FILE *f_w = fopen("result.txt","rb");
	unsigned int width,height;
	unsigned char *grey_data;
	grey_data = (unsigned char*)malloc(sizeof(unsigned char)*N);
	fread(&width,sizeof(unsigned int),1,f_w);
	fread(&height,sizeof(unsigned int),1,f_w);
	fread(grey_data,sizeof(unsigned char),width*height,f_w);
	CImg<unsigned char>grey(width,height,1,1,0);
	grey._data = grey_data;
	grey.display();
	fclose(f_w);
	return 0;

}
