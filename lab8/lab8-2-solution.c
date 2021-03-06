#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define SIZE 1000

#ifdef _OPENACC

/* 
 * This routine uses the OpenACC kernel construct. Input Matrices are copied
 * to the device and the output matrix is copied back to host. 
 */
void mm_oacc_kernel(float* a, float* b, float* c)
{
  int i,j,k;

  #pragma acc kernels copyin(a[0:SIZE*SIZE],b[0:SIZE*SIZE]) copy(c[0:SIZE*SIZE])
  {
    float ctmp = 0;
    #pragma acc loop gang(32) independent
    for (i = 0; i < SIZE; ++i) {
       #pragma acc loop vector(64) independent
      for (j = 0; j < SIZE; ++j) {
	#pragma acc loop reduction(+:ctmp)
	for (k = 0; k < SIZE; ++k) {
	  ctmp += a[i*SIZE+k] * b[k*SIZE+j];
        }
        c[i*SIZE+j] = ctmp;
      }
    }
  }
}

/* 
 * This routine uses the OpenACC kernel construct. Input Matrices are created 
 * and initialized on the device, but the output matrix is copied back to the host.
 * Independent loops are tagged.
 */
void mm_oacc_kernel_with_init(float* a, float* b, float* c)
{
  int i,j,k;
  
  #pragma acc kernels create(a[0:SIZE*SIZE],b[0:SIZE*SIZE]) copy(c[0:SIZE*SIZE])
  {
    #pragma acc loop independent
    for (i = 0; i < SIZE; ++i) {
      #pragma acc loop independent
      for (j = 0; j < SIZE; ++j) {
	a[i*SIZE + j] = (float)i + j;
	b[i*SIZE + j] = (float)i - j;
	c[i*SIZE + j] = 0.0f;
      }
    }

    #pragma acc loop independent
    for (i = 0; i < SIZE; ++i) {
      #pragma acc loop independent
      for (j = 0; j < SIZE; ++j) {
	//#pragma acc loop seq
	for (k = 0; k < SIZE; ++k) {
	  c[i*SIZE+j] += a[i*SIZE+k] * b[k*SIZE+j];
	}
      }
    }
  }
}

/* 
 * This routine uses the OpenACC parallel construct. Input Matrices are created 
 * and initialized on the device, but the output matrix is copied back to the host.
 * 
 */
void mm_oacc_parallel_with_init(float* a, float* b, float* c)
{
  int i,j,k;
  
  #pragma acc parallel create(a[0:SIZE*SIZE],b[0:SIZE*SIZE]) copy(c[0:SIZE*SIZE])
  {
    #pragma acc loop
    for (i = 0; i < SIZE; ++i) {
      #pragma acc loop
      for (j = 0; j < SIZE; ++j) {
	a[i*SIZE + j] = (float)i + j;
	b[i*SIZE + j] = (float)i - j;
	c[i*SIZE + j] = 0.0f;
      }
    }

    #pragma acc loop
    for (i = 0; i < SIZE; ++i) {
      #pragma acc loop
      for (j = 0; j < SIZE; ++j) {
	#pragma acc loop seq
	for (k = 0; k < SIZE; ++k) {
	  c[i*SIZE+j] += a[i*SIZE+k] * b[k*SIZE+j];
	}
      }
    }
  }
}

#endif /* _OPENACC */

/*
 * Matrix multiplication on the CPU (OpenMP parallel).
 */
void mm_cpu_compute(float* a, float* b, float* c)
{
  int i,j,k, nthreads, tid;
  
  #pragma omp parallel shared(a,b,c,nthreads) private(tid,i,j,k)
  {
    tid = omp_get_thread_num();
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("CPU MM with %d threads\n",nthreads);
    }
    
    #pragma omp for
    for (i = 0; i < SIZE; ++i) 
      for (j = 0; j < SIZE; ++j) 
	for (k = 0; k < SIZE; ++k) 
	  c[i*SIZE+j] += a[i*SIZE+k] * b[k*SIZE+j];
	
  }
}

/*
 * Initialization of the matrices on the CPU  (OpenMP parallel).
 */
void mm_cpu_initialize(float* a, float* b, float* c)
{
  int i, j;
  
  #pragma omp parallel for private(i,j) shared(a,b,c)
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      a[i*SIZE + j] = (float)i + j;
      b[i*SIZE + j] = (float)i - j;
      c[i*SIZE + j] = 0.0f;
    }
  }
}

/*
 * Check the results of two matrices.
 */
void check_results(float *res1, float *res2)
{
  int i, j;
   // check all the OpenACC matrices
  for (i = 0; i < SIZE; ++i)
    for (j = 0; j < SIZE; ++j)
      if(res1[i*SIZE+j] != res2[i*SIZE+j]) {
	printf("Error %d %d %g %g\n", i,j,res1[i*SIZE+j],res2[i*SIZE+j] );
    exit(1);
      }
  printf("OpenACC matrix multiplication test was successful!\n");
}

double gtod() 
{
  struct timeval act_time;
  gettimeofday(&act_time, NULL);
  return (double)act_time.tv_sec + (double)act_time.tv_usec / 1000000.0;
}

int main()
{
  double start, time;
  float *a, *b, *c, *c_cpu;
  int i, j, k;
   
  /* bytes to be allocated for one matrix */
  const size_t nBytes = SIZE * SIZE * sizeof(float);
  
  a = (float*)malloc(nBytes);
  b = (float*)malloc(nBytes);
  c = (float*)malloc(nBytes);
  c_cpu  = (float*)malloc(nBytes);
  
  // Initialize matrices.
  mm_cpu_initialize(a, b, c);
  
  
  time = gtod();
  start = time;
  
  /* Run OpenACC versions of the matrix multiplication */
#ifdef _OPENACC  
  mm_oacc_kernel(a, b, c);
  printf("mm_oacc_kernel(): %lf sec \n", gtod()-time);
  
  time = gtod();
  mm_oacc_kernel_with_init(a, b, c);
  printf("mm_oacc_kernel_with_init(): %lf sec \n", gtod()-time);
  
  time = gtod();
  mm_oacc_parallel_with_init(a, b, c);
  printf("mm_oacc_parallel_with_init(): %lf sec \n", gtod()-time);
#endif  

  /* Initialize the CPU result matrix */
  for(i = 0; i < SIZE; ++i) 
    for(j = 0; j < SIZE; ++j) 
      c_cpu[i*SIZE + j] = 0.0f;
   
  time = gtod();
  
  /* Perform the matrix multiplication on the CPU */
  mm_cpu_compute(a, b, c_cpu);
  
  printf("MM on CPU: %lf sec \n", gtod()-time);
   
  /* not necessary here, but if the async clause is used make sure OpenACC tasks
     are finished */
  #pragma acc wait
  
  printf("Total runtime: %lf sec \n", gtod()-start);
  
  check_results(c, c_cpu);
   
  return 0;
}
