all: prepare.cpp color_test.cu
	g++ prepare.cpp -o prepare.o -Dcimg_use_png -lpng -lz -lX11 -lpthread
	./prepare.o
	nvcc  color_test.cu -o  color.o  -std=c++11 -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO
	./color.o
clean:
	rm *.o
	
