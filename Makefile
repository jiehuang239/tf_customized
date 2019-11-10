TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
all: rgb2greyscale.cc rgb2greyscale.cu
	nvcc -std=c++11 -c -o cuda_op_kernel.cu.o rgb2greyscale.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-10.0/lib64/ -I /usr/local/cuda-10.0/include -I /usr/local/lib/python3.6/dist-packages/tensorflow/include --expt-relaxed-constexpr -DNDEBUG -L$TF_LIB -ltensorflow_framework
	g++ -std=c++11 -shared rgb2greyscale.cc cuda_op_kernel.cu.o -o rgb2greyscale.so -fPIC -lcublas -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -I /usr/local/lib/python3.6/dist-packages/tensorflow/include -expt-extended-lambda -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework
