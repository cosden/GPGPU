#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

const int blocksize=16;
const int N=256;

// only works for squared blocks and grids
__global__ void mm_gpu(float *a, float *b, float *c){
   const int cx=blockIdx.x*blockDim.x+threadIdx.x;
   const int cy=blockIdx.y*blockDim.y+threadIdx.y;
   const int tx=threadIdx.x;
   const int ty=threadIdx.y;
   __shared__ float as[blocksize][blocksize];
   __shared__ float bs[blocksize][blocksize];
   float c_temp=0.0f;

   // loop over blocks
   for (int l=0;l<gridDim.x; l++)
   {
      // copy data to shared mem
      as[ty][tx]=a[cy*N+l*blocksize+tx];
      bs[ty][tx]=b[(l*blocksize+ty)*N+cx];
      __syncthreads();

      // now loop over shared mem
      for (int k=0;k<blocksize;k++)
         c_temp+=as[ty][k]*bs[k][tx];
      __syncthreads();
   }

   c[cy*N+cx]=c_temp;
}

void mm_cpu(float *a, float *b, float *c, int N){
   int i,j,k;
   for (i=0;i<N;i++) 
      for (j=0;j<N;j++)
         {
         c[i*N+j]=0.0f;
         for (k=0;k<N;k++) 
            c[i*N+j]+=a[i*N+k]*b[k*N+j];
         }
}

int main(void)
{
  float *a_h, *b_h, *c_h ,*c2_h; // host data
  float *a_d, *b_d, *c_d;// device data
  float delta = 0.1f;
  int nBytes, i;
  dim3 dimBlock(blocksize,blocksize);
  dim3 dimGrid(ceil(N/(float)blocksize),ceil(N/(float)blocksize));
  struct timeval t1, t2, t3;
  long cgpu, chost;

  nBytes = N*N*sizeof(float);
  a_h = (float *)malloc(nBytes);
  b_h = (float *)malloc(nBytes);
  c_h = (float *)malloc(nBytes);
  c2_h = (float *)malloc(nBytes);
  CUDA_CHECK(cudaMalloc((void **) &a_d, nBytes));
  CUDA_CHECK(cudaMalloc((void **) &b_d, nBytes));
  CUDA_CHECK(cudaMalloc((void **) &c_d, nBytes));


  for (i=0; i<N*N; i++) {
     a_h[i] = 1.0f + 0.001*i;
     b_h[i] = 5.0f + 0.0001*i;
  }

  gettimeofday(&t1,NULL);
  CUDA_CHECK(cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice));
  mm_gpu<<<dimGrid,dimBlock>>>(a_d,b_d,c_d);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c_h, c_d, nBytes, cudaMemcpyDeviceToHost));
  gettimeofday(&t2,NULL);
  
  mm_cpu(a_h,b_h,c2_h,N);

  gettimeofday(&t3,NULL);
//  for (i=0; i< N*N; i++) printf("%d,%d: %f %f\n",i/N,i%N,c_h[i],c2_h[i]); 
  for (i=0; i< N*N; i++) if (abs(c_h[i]-c2_h[i])>delta) printf("Result incorrect! %d,%d: %f %f\n",i/N,i%N,c_h[i],c2_h[i]);

  free(a_h); free(b_h); free(c_h); free(c2_h); cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);

  cgpu=(t2.tv_sec - t1.tv_sec)*1000000 + (t2.tv_usec - t1.tv_usec);
  chost = (t3.tv_sec - t2.tv_sec)*1000000 + (t3.tv_usec - t2.tv_usec);
  printf( "%13ld microseconds on GPU\n", cgpu );
    printf( "%13ld microseconds on host\n", chost );
  return 0;
}
