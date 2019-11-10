TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
all: rgb2greyscale.cc rgb2greyscale.cu.cc
	nvcc -std=c++11 -c -o cuda_op_kernel.cu.o rgb2greyscale.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-10.0/lib64/ -I /usr/local/cuda-10.0/include -I /usr/local/lib/python3.6/dist-packages/tensorflow/include
	g++ -std=c++11 -shared rgb2greyscale.cc cuda_op_kernel.cu.o -o rgb2greyscale.so -fPIC -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -I /usr/local/lib/python3.6/dist-packages/tensorflow/include
