#include <stdio.h>
#include <assert.h>


#define N 1000000

static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 

static void HandleKernelError(const char *file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_KERNEL_ERROR (HandleKernelError(__FILE__, __LINE__))
  

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

  HANDLE_ERROR(cudaMalloc((void**)&a_device,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&b_device,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&c_device,N*sizeof(int)));
  
  HANDLE_ERROR(cudaMemcpy(a_device,a_host,N*sizeof(int),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(b_device,b_host,N*sizeof(int),cudaMemcpyHostToDevice));

  vecadd<<<dimGrid,dimBlock>>>(a_device,b_device,c_device);
  HANDLE_KERNEL_ERROR;

  HANDLE_ERROR(cudaMemcpy(c_host,c_device,N*2*sizeof(int),cudaMemcpyDeviceToHost));

  for (i=0;i<N;i++) assert (c_host[i] == a_host[i] + b_host[i]);

  HANDLE_ERROR(cudaFree(a_device));
  HANDLE_ERROR(cudaFree(b_device));
  HANDLE_ERROR(cudaFree(c_device));

  return 0;
}

