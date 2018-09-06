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

void IntegrateEzDriftLineGPU(float * distDrDz, float * distDPhiRDz, float * distDz, float *corrDrDz, float * corrDPhiRDz, float * corrDz,  const int rows, const int columns, const int phislices, const int symmetry, const float fgkIFCRadius, const float fgkOFCRadius, const float fgkTPCZ0, float * GDistDrDz, float * GDistDPhiRDz, float * GDistDz, float * GCorrDrDz, float * GCorrDPhiRDz, float * GCorrDz, int interpolationType) {

	// initialize device array
	float *d_distDrDz;
	float *d_distDPhiRDz;
	float *d_distDz;
	float *d_corrDrDz;
	float *d_corrDPhiRDz;
	float *d_corrDz;
	float *d_GDistDrDz;
	float *d_GDistDPhiRDz;
	float *d_GDistDz;
	float *d_GCorrDrDz;
	float *d_GCorrDPhiRDz;
	float *d_GCorrDz;
	
	cudaError error;

	cudaMalloc( &d_distDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_distDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_distDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_corrDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_corrDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_corrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GDistDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GDistDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GDistDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GCorrDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GCorrDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GCorrDz, rows * columns * phislices * sizeof(float) );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{    	
		std::cout << "CUDA memory allocation error: " << cudaGetErrorString(error) << '\n';
	}


	// copy from CPU to GPU
	// copy local distortion 
	cudaMemcpy( d_distDrDz, distDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_distDPhiRDz, distDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_distDz, distDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );

	cudaMemcpy( d_corrDrDz, corrDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_corrDPhiRDz, corrDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_corrDz, corrDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy host to device error: " << cudaGetErrorString(error) << '\n';
	}

	// call kernel
	// set grid size and block size
	dim3 gridSize((rows / 32) + 1, (columns / 32) + 1, phislices);
	dim3 blockSize(32, 32);


	
	cudaMemcpy( GDistDrDz, d_GDistDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GDistDPhiRDz, d_GDistDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GDistDz, d_GDistDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GCorrDrDz, d_GCorrDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GCorrDPhiRDz, d_GCorrDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GCorrDz, d_GCorrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy device to host error: " << cudaGetErrorString(error) << '\n';
	}

	cudaFree( d_distDrDz );
	cudaFree( d_distDPhiRDz );
	cudaFree( d_distDz );
	cudaFree( d_corrDrDz );
	cudaFree( d_corrDPhiRDz );
	cudaFree( d_corrDz );
	cudaFree( d_GDistDrDz );
	cudaFree( d_GDistDPhiRDz );
	cudaFree( d_GDistDz );
	cudaFree( d_GCorrDrDz );
	cudaFree( d_GCorrDPhiRDz );
	cudaFree( d_GCorrDz );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA free allocated memory error: " << cudaGetErrorString(error) << '\n';
	}
}






