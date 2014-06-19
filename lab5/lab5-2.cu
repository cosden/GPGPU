// needs to be compiled with option -arch=sm_20 to work

#include <stdio.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
#define CUDA_CHECK_KERNEL {cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define N 1000000
#define DELTA 0.001f 

__global__ void init_c(float *c){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx==0) *c=0.0f;
}

__global__ void scalarp(float *a, float *b, float *c){
  // get global thread id
  
  float temp;

  if (idx<N) {
    // multiply own elements
	// add to result
  }
}


int main (int argc, char **argv){

  float a_host[N], b_host[N], c_host, d_host=0.0f;
  float *a_device, *b_device, *c_device;
  int i;
  int blocksize=256;
  dim3 dimBlock(blocksize);
  dim3 dimGrid(ceil(N/(float)blocksize));

  for (i=0;i<N;i++) a_host[i]=1.0f*i;
  for (i=0;i<N;i++) b_host[i]=1.0f*i;

  // alloc device memory
  
  // transfer data
  
  // set c to 0
  init_c<<<1,1>>>(c_device);
  CUDA_CHECK_KERNEL

  // invoke scalarp kernel
  CUDA_CHECK_KERNEL

  // transfer data
  
  for (i=0;i<N;i++) d_host+=a_host[i]*b_host[i];
  if ((abs(d_host - c_host) > DELTA*c_host)) printf("Solution invalid. GPU has %g, CPU has %g\n",c_host,d_host);

  // free device
  
  return 0;
}

