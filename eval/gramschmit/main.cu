/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "polybenchUtilFuncts.h"
#include "penguin.h"

#define GPU_DEVICE 0

#define MiB 21812
#define RESERVATION ((MiB*1024UL*1024))

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define M 2048 * 8
#define N 2048 * 8

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
/* #define DIM_THREAD_BLOCK_Y 64 */
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

int do_saby = 0;


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
}


void init_array(DATA_TYPE* A, DATA_TYPE* A_gpu, DATA_TYPE* R_gpu, DATA_TYPE* Q_gpu)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
			A_gpu[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
			Q_gpu[i*N + j] = 0;
			R_gpu[i*N + j] = 0;
		}
	}
}


void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				/* printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]); */
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );	
	return;
}


__global__ void gramschmidt_kernel1(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < M; i++)
		{
			nrm += a[i * N + k] * a[i * N + k];
		}
      		r[k * N + k] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < M)
	{	
		q[i * N + k] = a[i * N + k] / r[k * N + k];
	}
}


__global__ void gramschmidt_kernel3(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < N))
	{
		r[k*N + j] = 0.0;

		int i;
		for (i = 0; i < M; i++)
		{
			r[k*N + j] += q[i*N + k] * a[i*N + j];
		}
		
		for (i = 0; i < M; i++)
		{
			a[i*N + j] -= q[i*N + k] * r[k*N + j];
		}
	}
}


void gramschmidtCuda(DATA_TYPE* A_gpu, DATA_TYPE* R_gpu, DATA_TYPE* Q_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	
  nvml_start();
  t_start = rtclock();
  int k;

  penguinStartStatCollection();
  for (k = 0; k < N; k++)
  {
    gramschmidt_kernel1<<<gridKernel1,block>>>(A_gpu, R_gpu, Q_gpu, k);
    cudaDeviceSynchronize();
    gramschmidt_kernel2<<<gridKernel2,block>>>(A_gpu, R_gpu, Q_gpu, k);
    cudaDeviceSynchronize();
    gramschmidt_kernel3<<<gridKernel3,block>>>(A_gpu, R_gpu, Q_gpu, k);
    cudaDeviceSynchronize();
  }
  t_end = rtclock();
  penguinStopStatCollection();
  fprintf(stdout, "\nGPU.Parser.Time: %0.6lf\n", t_end - t_start);
  nvml_stop();

}


int main(int argc, char *argv[])
{
  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);
  /* cudaMalloc((void**) &reservation, (22*1024UL*1024*1024) ); */

	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* R;
	DATA_TYPE* Q;
	DATA_TYPE *A_gpu;
	DATA_TYPE *R_gpu;
	DATA_TYPE *Q_gpu;
	
	A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMallocManaged(&R_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMallocManaged(&Q_gpu, sizeof(DATA_TYPE) * M * N);
	
	init_array(A, A_gpu, R_gpu, Q_gpu);
	
	GPU_argv_init();
	gramschmidtCuda(A_gpu, R_gpu, Q_gpu);
	
	t_start = rtclock();
	/* gramschmidt(A, R, Q); */
	t_end = rtclock();

	/* fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); */
	
	compareResults(A, A_gpu);	
	free(A);
	free(R);
	free(Q);  
	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);

    	return 0;
}

