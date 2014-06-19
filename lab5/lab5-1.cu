#include <stdio.h>
#include <stdlib.h>

#define SRAND_VALUE 1985

#define CUDA_CHECK_RETURN(value) {				\
  cudaError_t _m_cudaStat = value;				\
  if (_m_cudaStat != cudaSuccess) {				\
    fprintf(stderr, "Error %s at line %d in file %s\n",	\
    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);	\
    exit(1);							\
  }								\
}

#define BLOCK_SIZE 16

#define PRINT 0
#define VERIFY 0
#define SPEEDUP 1

__global__ void calcMatrixGPU(int dim, float *matA, float *matB, float *matC)
{
   //Thread ID
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  
  float result = 0;
  int k;

  for(k=0; k<dim; k++) {
     result = result + (matA[(i*dim)+k] * matB[(k*+dim)+j]);
  }
  
  matC[(i*dim)+j] = result;
}

__host__ void calcMatrixCPU(int dim, float* matA, float* matB, float* matC)
{
  int i, j, k;
  float result = 0;

  #pragma omp parallel for private(i, j, k, result)  
  for(j=0; j<dim; j++) {
    for(i=0; i<dim; i++) {
      result = 0;
      for(k=0; k<dim; k++) {
        result = result + (matA[(i*dim)+k] * matB[(k*+dim)+j]);
      }
      matC[(i*dim)+j] = result;
    }
  }
}

__host__ void printMat(int dim, float* mat)
{
  int i, j;

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      printf("%8.4f ", mat[(i*dim)+j]);
    }
    printf("\n");
  }
}

int main(int argc, char* argv[])
{
  int i, j;

  //Matrix A
  float* h_matA;
  float* d_matA;

  //Matrix B
  float* h_matB;
  float* d_matB;

  //Matrix C
  float* h_matC;
  float* d_matC;
  
  //CUDA-Events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  int dim = 4*1024;

  size_t matBytes = sizeof(float)*dim*dim;

  printf("Memory: %f MByte\n", (double) (3*matBytes)/1024/1024);
  
  // Memory allocation on host
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_matA, matBytes, cudaHostAllocDefault) );
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_matB, matBytes, cudaHostAllocDefault) );
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_matC, matBytes, cudaHostAllocDefault) );

  // Allocate device worlds
  CUDA_CHECK_RETURN( cudaMalloc(&d_matA, matBytes) );
  CUDA_CHECK_RETURN( cudaMalloc(&d_matB, matBytes) );
  CUDA_CHECK_RETURN( cudaMalloc(&d_matC, matBytes) );

  // Assign initial data
  srand(SRAND_VALUE);
  for(i = 0; i<dim; i++) {
    for(j = 0; j<dim; j++) {
      h_matA[(i*dim)+j] = rand() % 10;
      h_matB[(i*dim)+j] = rand() % 10;
    }
  }
  
  CUDA_CHECK_RETURN( cudaEventRecord(start, 0) );
  
  // Copy data from host to device
  CUDA_CHECK_RETURN( cudaMemcpy(d_matA, h_matA, matBytes, cudaMemcpyHostToDevice) );
  CUDA_CHECK_RETURN( cudaMemcpy(d_matB, h_matB, matBytes, cudaMemcpyHostToDevice) );
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  int linGrid = (int)ceil(dim/(float)BLOCK_SIZE);
  dim3 dimGrid(linGrid,linGrid);
  
  calcMatrixGPU<<<dimGrid, dimBlock>>>(dim, d_matA, d_matB, d_matC);
  
  CUDA_CHECK_RETURN( cudaThreadSynchronize());	// Wait for the GPU launched work 
  CUDA_CHECK_RETURN( cudaGetLastError());
  
  // Copy data from device to host
  CUDA_CHECK_RETURN( cudaMemcpy(h_matC, d_matC, matBytes, cudaMemcpyDeviceToHost) );
  
  CUDA_CHECK_RETURN( cudaEventRecord(stop, 0) );
  CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
  
  float runtime_gpu;
  cudaEventElapsedTime(&runtime_gpu, start, stop);
  
  printf("\nElapsed GPU time: %8.2f ms\n", runtime_gpu);
 
#if SPEEDUP
  CUDA_CHECK_RETURN( cudaEventRecord(start, 0) );
    
  calcMatrixCPU(dim, h_matA, h_matB, h_matC);
    
  CUDA_CHECK_RETURN( cudaEventRecord(stop, 0) );
  CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
    
  float runtime_cpu;
  cudaEventElapsedTime(&runtime_cpu, start, stop);
  
  printf("\nElapsed CPU time: %8.2f ms\n", runtime_cpu);
  printf("\nSpeedup: %8.2f\n", runtime_cpu/runtime_gpu);
#endif
  
  
#if VERIFY
  float* h_verify;
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_verify, matBytes, cudaHostAllocDefault) );
  
  calcMatrixCPU(dim, h_matA, h_matB, h_verify);
  
  int correct = 0;
  for(i = 0; i < (dim*dim); i++) {
    if (h_matC[i] != h_verify[i]){
      printf("Error: %8.4f - expected: %8.4f @( %i, %i)\n", h_matC[i], h_verify[i], i/dim, i%dim);
      break;
    } else {
      correct++;
    }
  }
  
  if(correct == (dim*dim))
    printf("Correct results\n");
  
  CUDA_CHECK_RETURN( cudaFreeHost(h_verify) );
  
#endif
  
#if PRINT
  printf("Matrix A:\n");
  printMat(dim, h_matA);

  printf("\nMatrix B:\n");
  printMat(dim, h_matB);

  printf("\nMatrix C:\n");
  printMat(dim, h_matC);
#endif
  
  // Release memory
  CUDA_CHECK_RETURN( cudaFreeHost(h_matA) );
  CUDA_CHECK_RETURN( cudaFreeHost(h_matB) );
  CUDA_CHECK_RETURN( cudaFreeHost(h_matC) );

  CUDA_CHECK_RETURN( cudaFree(d_matA) );
  CUDA_CHECK_RETURN( cudaFree(d_matB) );
  CUDA_CHECK_RETURN( cudaFree(d_matC) );
  
  return 0;
}
