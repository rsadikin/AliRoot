#include "LocalDistCorrDzGPU.h"
#include <cuda.h>
#include <math.h>

__device__ __constant__ float d_gridSizeZ;
__device__ __constant__ float d_ezField;
__device__ __constant__ float d_fC0;
__device__ __constant__ float d_fC1;
__device__ __constant__ float d_fgkdvdE;

__global__ void localDistCorrDzGPUKernel
(
	float *matEr,
	float *matEz,
	float *matEPhi, 
	float *matDistDrDz,
	float *matDistDPhiRDz,
	float *matDistDz, 
	float *matCorrDrDz,
	float *matCorrDPhiRDz,
	float *matCorrDz, 
	const int rows,
	const int columns,
	const int phislices
)
{
	int index, index_x, index_y, index_z;

	
	float localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz;	
	index = (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	index_x = index / (rows * columns);
	
	if (index_x == 0)
	{
		index_y = index / rows;
	}
	else	
	{
		index_y = (index % (index_x * rows * columns)) / rows;
	}

	index_z = index % columns;
	
	
	if ((index_x >= 0) && (index_x < phislices) && (index_y > 0) && (index_y < rows - 1) && (index_z > 0) && (index_z < columns - 1))
	{
        	localIntErOverEz = (d_gridSizeZ / 2.0) * (matEr[index_x * rows * columns + index_y * columns + index_z ] + matEr[index_x * rows * columns + index_y * columns + index_z + 1]) / (-1 * d_ezField);
        	localIntEPhiOverEz = (d_gridSizeZ / 2.0) * (matEPhi[index_x * rows * columns + index_y * columns + index_z ] + matEPhi[index_x * rows * columns + index_y * columns + index_z + 1]) / (-1 * d_ezField);
        	localIntDeltaEz = (d_gridSizeZ / 2.0) * (matEz[index_x * rows * columns + index_y * columns + index_z ] + matEz[index_x * rows * columns + index_y * columns + index_z + 1]) ;


		matDistDrDz[index_x * rows  *columns + index_y *columns + index_z] = d_fC0 * localIntErOverEz + d_fC1 * localIntEPhiOverEz;
		matDistDPhiRDz[index_x * rows  *columns + index_y *columns + index_z] = d_fC0 * localIntEPhiOverEz - d_fC1 * localIntErOverEz;
		matDistDz[index_x * rows  *columns + index_y *columns + index_z] = d_fgkdvdE * d_fgkdvdE * localIntDeltaEz;

		matCorrDrDz[index_x * rows  *columns + index_y *columns + index_z + 1] = -1 * matDistDrDz[index_x * rows  *columns + index_y *columns + index_z ]; 
		matCorrDPhiRDz[index_x * rows  *columns + index_y *columns + index_z + 1] = -1 * matDistDPhiRDz[index_x * rows  *columns + index_y *columns + index_z ]; 
		matCorrDz[index_x * rows  *columns + index_y *columns + index_z + 1] = -1 * matDistDz[index_x * rows  *columns + index_y *columns + index_z ]; 


	}

}





extern "C" void LocalDistCorrDzGPU (
	float *matEr,
	float *matEz,
	float *matEPhi, 
	float *matDistDrDz,
	float *matDistDPhiRDz,
	float *matDistDz, 
	float *matCorrDrDz,
	float *matCorrDPhiRDz,
	float *matCorrDz, 
	const int rows,
	const int columns,
	const int phislices,
	const float gridSizeZ,
	const float ezField,	
  	const float fC0,
	const float fC1,
	const float fgkdvdE	
)
{
	// device array
	float *d_matEr;
	float *d_matEz;
	float *d_matEPhi;
	float *d_matDistDrDz;
	float *d_matDistDPhiRDz;
	float *d_matDistDz;
	float *d_matCorrDrDz;
	float *d_matCorrDPhiRDz;
	float *d_matCorrDz;

	cudaError error;

	// pre-compute constant

	
	// device memory allocation
	cudaMalloc( &d_matEr, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matEz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matEPhi, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matDistDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matDistDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matDistDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matCorrDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matCorrDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_matCorrDz, rows * columns * phislices * sizeof(float) );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{    	
		std::cout << "CUDA memory allocation error: " << cudaGetErrorString(error) << '\n';
	}

	// copy data from host to device
	cudaMemcpy( d_matEr, matEr, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_matEPhi, matEPhi, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_matEz, matEz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy host to device error: " << cudaGetErrorString(error) << '\n';
	}

	// copy constant from host to device
	cudaMemcpyToSymbol( d_gridSizeZ, &gridSizeZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_ezField, &ezField, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_fC0, &fC0, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_fC1, &fC1, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_fgkdvdE, &fgkdvdE, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy to constant memory host to device error: " << cudaGetErrorString(error) << '\n';
	}

	// set grid size and block size
	dim3 gridSize((rows / 32) + 1, (columns / 32) + 1, phislices);
	dim3 blockSize(32, 32);

	// run the kernel
 	localDistCorrDzGPUKernel<<< gridSize, blockSize >>>( d_matEr, d_matEz, d_matEPhi, d_matDistDrDz, d_matDistDPhiRDz, d_matDistDz, d_matCorrDrDz, d_matCorrDPhiRDz, d_matCorrDz, rows, columns, phislices );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA kernel run error: " << cudaGetErrorString(error) << '\n';
	}

	// copy result from device to host
	cudaMemcpy( matDistDrDz, d_matDistDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( matDistDPhiRDz, d_matDistDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( matDistDz, d_matDistDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( matCorrDrDz, d_matCorrDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( matCorrDPhiRDz, d_matCorrDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( matCorrDz, d_matCorrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy device to host error: " << cudaGetErrorString(error) << '\n';
	}

	// free device memory
	cudaFree( matEr );
	cudaFree( matEPhi );
	cudaFree( matEz );
	
	cudaFree( matDistDrDz );
	cudaFree( matDistDPhiRDz );
	cudaFree( matDistDz );
	cudaFree( matCorrDrDz );
	cudaFree( matCorrDPhiRDz );
	cudaFree( matCorrDz );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA free allocated memory error: " << cudaGetErrorString(error) << '\n';
	}
}

