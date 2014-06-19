#include <stdio.h>
#include <assert.h>

#define N 1000000

int main (int argc, char **argv){

  int a_host[N], b_host[N];
  int *a_device, *b_device;
  int i;

  for (i=0;i<N;i++) a_host[i]=i;

  cudaMalloc((void**)&a_device,N*sizeof(int));
  cudaMalloc((void**)&b_device,N*sizeof(int));

  cudaMemcpy(a_device,a_host,N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(b_device,a_device,N*sizeof(int),cudaMemcpyDeviceToDevice);
  cudaMemcpy(b_host,b_device,N*sizeof(int),cudaMemcpyDeviceToHost);

  for (i=0;i<N;i++) assert (a_host[i]==b_host[i]);

  cudaFree(a_device);
  cudaFree(b_device);

  return 0;
}
