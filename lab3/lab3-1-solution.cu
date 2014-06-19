#include <stdio.h>
#include <assert.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
#define CUDA_CHECK_KERNEL {cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define N 1000000


__global__ void vecadd(int *a, int *b, int *c){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<N) c[idx]=a[idx]+b[idx];
}


int main (int argc, char **argv){

  int a_host[N], b_host[N], c_host[N];
  int *a_device, *b_device, *c_device;
  int i;
  int blocksize=256;
  dim3 dimBlock(blocksize);
  dim3 dimGrid(ceil(N/(float)blocksize));

  for (i=0;i<N;i++) a_host[i]=i;
  for (i=0;i<N;i++) b_host[i]=i;

  CUDA_CHECK(cudaMalloc((void**)&a_device,N*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&b_device,N*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&c_device,N*sizeof(int)));
  
  CUDA_CHECK(cudaMemcpy(a_device,a_host,N*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_device,b_host,N*sizeof(int),cudaMemcpyHostToDevice));

  vecadd<<<dimGrid,dimBlock>>>(a_device,b_device,c_device);
  CUDA_CHECK_KERNEL

  CUDA_CHECK(cudaMemcpy(c_host,c_device,N*sizeof(int),cudaMemcpyDeviceToHost));

  for (i=0;i<N;i++) assert (c_host[i] == a_host[i] + b_host[i]);

  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  CUDA_CHECK(cudaFree(c_device));

  return 0;
}

