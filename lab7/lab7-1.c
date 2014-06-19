#include <stdio.h>
#include <assert.h>


#define N 1000000


int main (int argc, char **argv){

  int a[N], b[N], c[N];
  int i;

  // initialize a and b
  for (i=0; i<N; i++) {
    a[i]=i;
    b[i]=i;
  }
  
  // add c = a+b using openacc  

#pragma acc parallel loop
  for (i=0; i<N; i++) {
    c[i]=a[i]+b[i];
  }



  for (i=0;i<N;i++) assert (c[i] == a[i] + b[i]);

  
  return 0;
}

