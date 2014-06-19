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

__global__ void calcDyadicGPU(int dim, float *vecA, float *vecB, float *product)
{
   //Thread ID
   // get thread id for access in vecA and vecB
  
  if ( (i<dim) && (j<dim) )
    // compute product for current position
}

__host__ void calcDyadicCPU(int dim, float* vecA, float* vecB, float* product)
{
  int i, j;

  for(j=0; j<dim; j++) {
    for(i=0; i<dim; i++) {
      product[(i*dim)+j] = vecA[i] * vecB[j];
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

__host__ void printVec(int dim, float* vec)
{
  int i;

  for (i=0; i<dim; i++) {
      printf("%8.4f ", vec[i]);
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  int i;

  //Vector A
  float* h_vecA;
  float* d_vecA;

  //Vector B
  float* h_vecB;
  float* d_vecB;

  //Vector C
  float* h_product;
  float* d_product;
  
  //CUDA-Events
  cudaEvent_t start, stop;
  CUDA_CHECK_RETURN( cudaEventCreate(&start) );
  CUDA_CHECK_RETURN( cudaEventCreate(&stop) );

  
  int dim = 1024;

  size_t vecBytes = sizeof(float)*dim;
  size_t matBytes = sizeof(float)*dim*dim;
  
  printf("Memory: %f MByte\n", (double) ((2*vecBytes)+matBytes)/1024/1024);

  // Memory allocation on host
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_vecA, vecBytes, cudaHostAllocDefault) );
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_vecB, vecBytes, cudaHostAllocDefault) );
  CUDA_CHECK_RETURN( cudaHostAlloc((void**)&h_product, matBytes, cudaHostAllocDefault) );

  // Allocate device worlds
  CUDA_CHECK_RETURN(cudaMalloc(&d_vecA, vecBytes));
  CUDA_CHECK_RETURN(cudaMalloc(&d_vecB, vecBytes));
  CUDA_CHECK_RETURN(cudaMalloc(&d_product, matBytes));  
  
  // Assign initial data
  srand(SRAND_VALUE);
  for(i = 0; i<dim; i++) {
    h_vecA[i] = rand();
    h_vecB[i] = rand();
  }
  
  CUDA_CHECK_RETURN( cudaEventRecord(start, 0) );
  
  // Copy data from host to device
   CUDA_CHECK_RETURN(cudaMemcpy(d_vecA,h_vecA,vecBytes,cudaMemcpyHostToDevice));
   CUDA_CHECK_RETURN(cudaMemcpy(d_vecB,h_vecB,vecBytes,cudaMemcpyHostToDevice));

  // design 2D grid
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   int linGrid = (int)ceil(dim/(float)BLOCK_SIZE);
   dim3 dimGrid(linGrid,linGrid);

  // invoke kernel
  
  CUDA_CHECK_RETURN( cudaThreadSynchronize());	// Wait for the GPU launched work 
  CUDA_CHECK_RETURN( cudaGetLastError());
  
  // Copy data from device to host
  
  // timing
  CUDA_CHECK_RETURN( cudaEventRecord(stop, 0) );
  CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
  
  float runtime_gpu;
  cudaEventElapsedTime(&runtime_gpu, start, stop);
  
  printf("\nElapsed GPU time: %8.2f ms\n", runtime_gpu);
 
#if SPEEDUP
  CUDA_CHECK_RETURN( cudaEventRecord(start, 0) );
    
  calcDyadicCPU(dim, h_vecA, h_vecB, h_product);
    
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
  
  calcDyadicCPU(dim, h_vecA, h_vecB, h_verify);
  
  int correct = 0;
  for(i = 0; i < (dim*dim); i++) {
    if (h_product[i] != h_verify[i]){
      printf("Error: %8.4f - expected: %8.4f @( %i, %i)\n", h_product[i], h_verify[i], i/dim, i%dim);
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
  printf("Vector A:\n");
  printVec(dim, h_vecA);

  printf("\nVector B:\n");
  printVec(dim, h_vecB);

  printf("\nMatrix:\n");
  printMat(dim, h_product);
#endif
  
  // Release memory
  CUDA_CHECK_RETURN( cudaFreeHost(h_vecA) );
  CUDA_CHECK_RETURN( cudaFreeHost(h_vecB) );
  CUDA_CHECK_RETURN( cudaFreeHost(h_product) );

  
  return 0;
}
