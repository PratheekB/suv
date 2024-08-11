/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include "penguin.h"

#include "polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define MiB 21130
#define RESERVATION ((MiB*1024UL*1024))

/* Problem size */
#define N (4096 * 8ULL)

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2,
		DATA_TYPE* a_gpu, DATA_TYPE* x1_gpu, DATA_TYPE* x2_gpu, DATA_TYPE* y_1_gpu, DATA_TYPE* y_2_gpu)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		x1_gpu[i] = ((DATA_TYPE) i) / N;
		x2_gpu[i] = ((DATA_TYPE) i + 1) / N;
		y_1_gpu[i] = ((DATA_TYPE) i + 3) / N;
		y_2_gpu[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
			a_gpu[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
	}
}



void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       			x1[i] = x1[i] + a[i*N + j] * y1[j];
        	}
    	}
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
 		       	x2[i] = x2[i] + a[j*N + i] * y2[j];
      		}
    	}
}


void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
    return;
	int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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
}


__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}

void mvtCuda(DATA_TYPE* a_gpu, DATA_TYPE* x1_gpu, DATA_TYPE* x2_gpu, DATA_TYPE* y_1_gpu, DATA_TYPE* y_2_gpu)
{
	cudaError_t err;
	double t_start, t_end;
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X)), 1);
	t_start = rtclock();

	mvt_kernel1<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
	cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "hi1%s", cudaGetErrorString(err));
    }
	mvt_kernel2<<<grid,block>>>(a_gpu,x2_gpu,y_2_gpu);
	cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "hi2%s", cudaGetErrorString(err));
    }
	t_end = rtclock();
	fprintf(stdout, "GPU.Parser.Time: %0.6lf\n", t_end - t_start);
}


int main()
{
  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);
  /* cudaMalloc((void**) &reservation, (22*1024UL*1024*1024) + (256 *1024UL*1024)); */
  /* cudaMalloc((void**) &reservation, (22*1024UL*1024*1024) ); */
  /* sleep(10); */

	double t_start, t_end;

	cudaError_t err;
	DATA_TYPE* a;
	DATA_TYPE* x1;
	DATA_TYPE* x2;
	DATA_TYPE* y_1;
	DATA_TYPE* y_2;
	DATA_TYPE* a_gpu;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;
	a = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	cudaMallocManaged(&a_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMallocManaged(&x1_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&x2_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&y_1_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&y_2_gpu, sizeof(DATA_TYPE) * N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "hi0%s", cudaGetErrorString(err));
    }
	init_array(a, x1, x2, y_1, y_2, a_gpu, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);

	
	GPU_argv_init();

  nvml_start();
  penguinStartStatCollection();
	mvtCuda(a_gpu, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);
  penguinStopStatCollection();
  nvml_stop();
	
	/* compareResults(x1, x1_gpu, x2, x2_gpu); */

	free(a);
	free(x1);
	free(x2);
	free(y_1);
	free(y_2);
	cudaFree(a_gpu);
	cudaFree(x1_gpu);
	cudaFree(x2_gpu);
	cudaFree(y_1_gpu);
	cudaFree(y_2_gpu);
  	return 0;
}

