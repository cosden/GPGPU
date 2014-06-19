/*
 *  Copyright 2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include "timer.h"
#include "checkResults.h"

#ifdef _OPENACC
#include "openacc.h"
#endif

#define NN 8192
#define NM 8192

double A[NN][NM];
double Anew[NN][NM];

static void initGPU(int argc, char** argv);

int main(int argc, char** argv)
{
    const int n = NN;
    const int m = NM;
    const int iter_max = 500;

    const double tol = 1.0e-6;
    double error     = 1.0;
    
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));
        
    for (int j = 0; j < n; j++)
    {
        A[j][0]    = 1.0;
        Anew[j][0] = 1.0;
    }
    
#ifdef _OPENACC
  initGPU(argc, argv);
#endif
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    StartTimer();
    int iter = 0;

#pragma acc data copy(A[0:NN][0:NM]), create(Anew[0:NN][0:NM])    
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

#pragma acc kernels present(A,Anew) 
    {
#pragma acc loop gang(512)        
		for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
			+ A[j-1][i] + A[j+1][i]);
                error = fmax( error, fabs(Anew[j][i] - A[j][i]));
            }
        }

#pragma acc loop gang(512)        
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];    
            }
        }
	}

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
/*
    int passed = checkSolution(error, &A[0][0], NN, NM, iter_max);
    if (passed == 0) {
        printf("Error check passed.\n");
    } else if (passed > 0) {
        printf("Error check failed.\n");
    }
*/ 
    printf(" total : %f s\n", runtime / 1000);
    return 0;
}

#ifdef _OPENACC
static void initGPU(int argc, char** argv) {
        // gets the device id (if specified) to run on
        int devId = -1;
        if (argc > 1) {
                devId = atoi(argv[1]);
                int devCount = acc_get_num_devices(acc_device_nvidia);
                if (devId < 0 || devId >= devCount) {
                        printf("The specified device ID is not supported.\n");
                        exit(1);
                }
        }
        if (devId != -1) {
                acc_set_device_num(devId, acc_device_nvidia);
        }
        // creates a context on the GPU just to
        // exclude initialization time from computations
        acc_init(acc_device_nvidia);

        // print device id
        devId = acc_get_device_num(acc_device_nvidia);
        printf("Running on GPU with ID %d.\n\n", devId);

}
#endif
