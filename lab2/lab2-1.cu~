#include <stdio.h>
#include <assert.h>

#define N 1000000

int main (int argc, char **argv){

  int a_host[N], b_host[N];
  int *a_device, *b_device;
  int i;
  
  // initialize data
  for (i=0;i<N;i++) {
    a_host[i]=i;
  }
  
  // allocate device memory
  cudaMalloc((void**)&a_device,N*sizeof(int));
  cudaMalloc((void**)&b_device,N*sizeof(int));
  
  // transfer data onto the device, copy on device, transfer back
  cudaMemcpy(a_device,a_host,N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(b_device,a_host,N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(b_host,b_device,N*sizeof(int),cudaMemcpyDeviceToHost);
 
  
  // correctness check
  for (i=0;i<N;i++) {
	assert (a_host[i]==b_host[i]);
  }
  // free GPU memory
  for (i=0;i<N;i++) {
    cudaFree(a_device);
    cudaFree(b_device);
  } 
  
  
  return 0;
}
