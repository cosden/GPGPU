#ifndef CHECKRESULTS_H
#define CHECKRESULTS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int checkSolution(double error, double* A, int n, int m, int iter) {
    int j,i;
    double tol = 1.0e-22;
    FILE *solFile;
    char filename[30] = "solution";
    char str_iter[10];
    sprintf(str_iter,"%d",iter);
    strcat(filename,str_iter);
    strcat(filename,".txt");
    solFile = fopen(filename,"r");
    double value;
    if (solFile == NULL) {
        printf("Error: Solution could not be checked as the solution file could not be read in.\n");
	return 2;
    } else {
        fscanf(solFile, "%lf", &value);
        if (fabs(value - error) > tol) {
                printf("Error: Results are not correct: Found error: %24.22f, correct error: %24.22f\n",error, value);
                return 1;
        }
        for(j = 0; j < n; j++) {
	     for (i = 0; i < m; i++) {
                 fscanf(solFile, "%lf", &value);
                 if (fabs(value - A[j*m+i]) > tol) {
                     printf("Error: Results are not correct. See [j,i] = [%d,%d]: found: %24.22f, correct: %24.22f\n",j,i,A[j*m+i],value);
                     return 1;
                 }
            }
        }
        fclose(solFile);
    }
    return 0;
}

void writeSolution(double error, double* A, int n, int m, int iter) {
    int j,i;
    FILE *solFile;
    char filename[30] = "solution";
    char str_iter[10];
    sprintf(str_iter,"%d",iter);
    strcat(filename,str_iter);
    strcat(filename,".txt");
    solFile = fopen(filename,"w");
    if (solFile == NULL) {
        printf("Error: Solution could not be written.\n");
    } else {
        fprintf(solFile, "%24.22f\n", error);
        for(j = 0; j < n; j++) {
            for(i = 0; i < m; i++)  {
                fprintf(solFile, "%24.22f ", A[j*m+i]);
            }
            fprintf(solFile, "\n");
        }
        fclose(solFile);
    }
}

#endif // CHECKRESULTS_H
