TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
all: rgb2greyscale.cc rgb2greyscale.cu
	nvcc -std=c++11 -c -o cuda_op_kernel.cu.o rgb2greyscale.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-10.0/lib64/ -I /usr/local/cuda-10.0/include -I /usr/local/lib/python3.6/dist-packages/tensorflow/include -expt-extended-lambda --expt-relaxed-constexpr -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0
	/usr/bin/g++-4.8 -std=c++11 -shared rgb2greyscale.cc cuda_op_kernel.cu.o -o rgb2greyscale.so -L /usr/local/cuda-10.0/lib64/ -I /usr/local/cuda-10.0/include -L $TF_LIB -fPIC -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I /usr/local/lib/python3.6/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.1
