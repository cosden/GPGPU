#include <stdio.h>
#include <assert.h>


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

  cudaMalloc((void**)&a_device,N*sizeof(int));
  cudaMalloc((void**)&b_device,N*sizeof(int));
  cudaMalloc((void**)&c_device,N*sizeof(int));
  
  cudaMemcpy(a_device,a_host,N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(b_device,b_host,N*sizeof(int),cudaMemcpyHostToDevice);

  vecadd<<<dimGrid,dimBlock>>>(a_device,b_device,c_device);

  cudaMemcpy(c_host,c_device,N*sizeof(int),cudaMemcpyDeviceToHost);

  for (i=0;i<N;i++) assert (c_host[i] == a_host[i] + b_host[i]);

  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);

  return 0;
}

