#include "IntegrateEzGPU.h"

#include <cuda.h>

__device__ __constant__ float d_gridSizeZ;
__device__ __constant__ float d_ezField;
__device__ __constant__ int d_scanSize;

__global__ void integrationCalculation
(
	float *d_arrayofIntEx,
	float *d_arrayofEx	
)
{
	extern __shared__ float temp[];

	int threadIndex = threadIdx.x;	
	int arrayIndex = blockIdx.x * (d_scanSize + 1);
	
	float first, second, last;
	
	int n = blockDim.x * 2;

	int offset = 1;
	
	// load data from input
	float temp_a = d_arrayofEx[arrayIndex + (2 * threadIndex)];
	float temp_b = d_arrayofEx[arrayIndex + (2 * threadIndex + 1)];

	// load last element from array to first variable
	first = d_arrayofEx[arrayIndex + d_scanSize];
	second = d_arrayofEx[arrayIndex + d_scanSize - 1];

/* odd function */
	// save data to shared memory flipped
	temp[(d_scanSize - 1) - (2 * threadIndex)] = 4 * temp_a;
	temp[(d_scanSize - 1) - (2 * threadIndex + 1)] = 2 * temp_b;
	
	// scan the array
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		
		if (threadIndex < d)
		{
			int ai = offset * (2 * threadIndex + 1) - 1;
			int bi = offset * (2 * threadIndex + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadIndex == 0)
	{
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (threadIndex < d)
		{
			int ai = offset * (2 * threadIndex + 1) - 1;
			int bi = offset * (2 * threadIndex + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	
	// save odd-numbered scan to even-numbered array
	d_arrayofIntEx[arrayIndex + (2 * threadIndex + 1)] = ((1.5 * first) + (0.5 * second) + temp[(d_scanSize - 1) - (2 * threadIndex)] - temp_b) * (d_gridSizeZ / 3.0) / (-1 * d_ezField);

/* even function */
	// save data to shared memory flipped
	temp[(d_scanSize - 1) - (2 * threadIndex)] = 2 * temp_a;
	temp[(d_scanSize - 1) - (2 * threadIndex + 1)] = 4 * temp_b;
	
	// scan the array
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		
		if (threadIndex < d)
		{
			int ai = offset * (2 * threadIndex + 1) - 1;
			int bi = offset * (2 * threadIndex + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadIndex == 0)
	{
		last = temp[n - 1];		
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (threadIndex < d)
		{
			int ai = offset * (2 * threadIndex + 1) - 1;
			int bi = offset * (2 * threadIndex + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadIndex == 0)
	{
		d_arrayofIntEx[arrayIndex + d_scanSize] = 0.0;
		d_arrayofIntEx[arrayIndex] = (first + last - temp_a) * (d_gridSizeZ / 3.0) / (-1 * d_ezField);
	}
	else
	{
		d_arrayofIntEx[arrayIndex + (2 * threadIndex)] = (first + temp[(d_scanSize - 1) - (2 * threadIndex) + 1] - temp_a) * (d_gridSizeZ / 3.0) / (-1 * d_ezField);
	}
}

extern "C" void IntegrateEzGPU 
(
	float *arrayOfIntEx, 
	float *arrayOfEx, 
	const int rows, 
	const int columns,  
	const int phislices, 
	float gridSizeZ, 
	float ezField	
)
{
	// initialize device array
	float *d_arrayofIntEx;
	float *d_arrayofEx;

	// set scan size to columns - 1
	int scanSize = columns - 1;

	std::cout << scanSize << std::endl;

	// set grid size and block size
	dim3 gridSize(rows * phislices);
	dim3 blockSize(scanSize / 2);

	// device memory allocation
	cudaMalloc( &d_arrayofIntEx, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_arrayofEx, rows * columns * phislices * sizeof(float) );

	// copy data from host to device
	cudaMemcpy( d_arrayofEx, arrayOfEx, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );

	// copy constant to device memory
	cudaMemcpyToSymbol( d_gridSizeZ, &gridSizeZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_ezField, &ezField, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_scanSize, &scanSize, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );

	// run the kernel
	integrationCalculation<<< gridSize, blockSize, 2 * scanSize * sizeof(float) >>>( d_arrayofIntEx, d_arrayofEx );

	// copy result from device to host
	cudaMemcpy( arrayOfIntEx, d_arrayofIntEx, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );

	// free device memory
	cudaFree( d_arrayofIntEx );
	cudaFree( d_arrayofEx );
}

