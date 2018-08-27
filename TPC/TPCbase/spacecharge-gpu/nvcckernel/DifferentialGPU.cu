#include "DifferentialGPU.h"
#include <cuda.h>
#include <math.h>

__device__ __constant__ float d_gridSizeR;
__device__ __constant__ float d_gridSizeZ;
__device__ __constant__ float d_gridSizePhi;
__device__ __constant__ float d_fgkIFCRadius;

__global__ void nonBoundaryDifferentialCalculation
(
	float *arrayofArrayV,
	float *arrayofArrayEr,
	float *arrayofArrayEz,
	float *arrayofArrayEphi, 
	const int rows,
	const int columns,
	const int phislices,
	const int symmetry
)
{
	int index, index_x, index_y, index_z;
	float radius;
	int mplus, mminus, signplus, signminus;

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
	
	// arrayofArrayEr[index] = 0.0;
	// arrayofArrayEz[index] = 0.0;
	// arrayofArrayEphi[index] = 0.0;
	
	if ((index_x >= 0) && (index_x < phislices) && (index_y > 0) && (index_y < rows - 1) && (index_z > 0) && (index_z < columns - 1))
	{
		mplus = index_x + 1;
		mminus = index_x - 1;
		signplus = 1;
		signminus = 1;
		// Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
		if (symmetry == 1)
		{
			if (mplus > phislices - 1)
			{
				mplus  = phislices - 2;
			}
			if (mminus < 0)
			{
				mminus = 1;
			}
		}
		// Anti-symmetry in phi		
		else if (symmetry == -1)
		{
			if (mplus > phislices - 1 )
			{
				mplus = phislices - 2;
				signplus = -1;
			}
			if (mminus < 0)
			{
				mminus = 1;
				signminus = -1;
			}
		}
		// No Symmetries in phi, no boundaries, the calculations is continuous across all phi
		else
		{
			if (mplus > phislices - 1)
			{
				mplus  = index_x + 1 - phislices;
			}
			if (mminus < 0)
			{
				mminus = index_x - 1 + phislices;
			}
		}

		radius = d_fgkIFCRadius + index_y * d_gridSizeR;
		// calculate r direction
		arrayofArrayEr[index] = -1 * (arrayofArrayV[index_x * rows * columns + (index_y + 1) * columns + index_z] - arrayofArrayV[index_x * rows * columns + (index_y - 1) * columns + index_z]) / (2 * d_gridSizeR);
		// calculate z direction			
		arrayofArrayEz[index] = -1 * (arrayofArrayV[index_x * rows * columns + index_y * columns + (index_z + 1)] - arrayofArrayV[index_x * rows * columns + index_y * columns + (index_z - 1)]) / (2 * d_gridSizeZ);
		// calculate phi direction			
		arrayofArrayEphi[index] = -1 * (signplus * arrayofArrayV[mplus * rows * columns + index_y * columns + index_z] - signminus * arrayofArrayV[mminus * rows * columns + index_y * columns + index_z]) / (2 * radius * d_gridSizePhi);

/*
// DEBUG
		arrayofArrayEr[index] = index_x;
		arrayofArrayEz[index] = index_y;
		arrayofArrayEphi[index] = index_z;
*/
	}

}

void nonBoundaryDifferentialCalculationGPU
(
	float *arrayofArrayV,
	float *arrayofArrayEr,
	float *arrayofArrayEz,
	float *arrayofArrayEphi, 
	const int rows,
	const int columns,
	const int phislices,
	const int symmetry,
		const float fgkIFCRadius,
	const float fgkOFCRadius,
	const float fgkTPCZ0
)
{
	// device array
	float *d_arrayofArrayV;
	float *d_arrayofArrayEr;
	float *d_arrayofArrayEz;
	float *d_arrayofArrayEphi;

	cudaError error;

	// pre-compute constant

	const float gridSizeR = (fgkOFCRadius - fgkIFCRadius) / (rows - 1);
 	const float gridSizeZ = fgkTPCZ0 / (columns - 1);
 	const float gridSizePhi = M_PI * 2 / phislices;
	
	// device memory allocation
	cudaMalloc( &d_arrayofArrayV, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_arrayofArrayEr, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_arrayofArrayEz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_arrayofArrayEphi, rows * columns * phislices * sizeof(float) );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{    	
		std::cout << "CUDA memory allocation error: " << cudaGetErrorString(error) << '\n';
	}

	// copy data from host to device
	cudaMemcpy( d_arrayofArrayV, arrayofArrayV, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy host to device error: " << cudaGetErrorString(error) << '\n';
	}

	// copy constant from host to device
	cudaMemcpyToSymbol( d_gridSizeR, &gridSizeR, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_gridSizeZ, &gridSizeZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_gridSizePhi, &gridSizePhi, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_fgkIFCRadius, &fgkIFCRadius, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy to constant memory host to device error: " << cudaGetErrorString(error) << '\n';
	}

	// set grid size and block size
	dim3 gridSize((rows / 32) + 1, (columns / 32) + 1, phislices);
	dim3 blockSize(32, 32);

	// run the kernel
	nonBoundaryDifferentialCalculation<<< gridSize, blockSize >>>( d_arrayofArrayV, d_arrayofArrayEr, d_arrayofArrayEz, d_arrayofArrayEphi, rows, columns, phislices, symmetry );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA kernel run error: " << cudaGetErrorString(error) << '\n';
	}

	// copy result from device to host
	cudaMemcpy( arrayofArrayEr, d_arrayofArrayEr, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( arrayofArrayEz, d_arrayofArrayEz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( arrayofArrayEphi, d_arrayofArrayEphi, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy device to host error: " << cudaGetErrorString(error) << '\n';
	}

	// free device memory
	cudaFree( d_arrayofArrayV );
	cudaFree( d_arrayofArrayEr );
	cudaFree( d_arrayofArrayEz );
	cudaFree( d_arrayofArrayEphi );
	
	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA free allocated memory error: " << cudaGetErrorString(error) << '\n';
	}
}

void boundaryDifferentialCalculationCPU
(
	float *arrayofArrayV,
	float *arrayofArrayEr,
	float *arrayofArrayEz,
	float *arrayofArrayEphi, 
	const int rows,
	const int columns,
	const int phislices,
	const int symmetry,
	const float fgkIFCRadius,
	const float fgkOFCRadius,
	const float fgkTPCZ0
)
{
	const float gridSizeR = (fgkOFCRadius - fgkIFCRadius) / (rows - 1);
 	const float gridSizeZ = fgkTPCZ0 / (columns - 1);
 	const float gridSizePhi = M_PI * 2 / phislices; // TwoPi() / phislices;
	
	float radius;
	int mplus, mminus, signplus, signminus;

	for (int m = 0; m < phislices; m++)
	{
		mplus = m + 1;
		mminus = m - 1;
		signplus = 1;
		signminus = 1;
		// Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
		if (symmetry == 1)
		{
			if (mplus > phislices - 1)
			{
				mplus  = phislices - 2;
			}
			if (mminus < 0)
			{
				mminus = 1;
			}
    	}
		// Anti-symmetry in phi		
		else if (symmetry == -1)
		{
			if (mplus > phislices - 1 )
			{
				mplus = phislices - 2;
				signplus = -1;
			}
			if (mminus < 0)
			{
				mminus = 1;
				signminus = -1;
			}
		}
		// No Symmetries in phi, no boundaries, the calculations is continuous across all phi
   		else
		{
			if (mplus > phislices - 1)
			{
				mplus  = m + 1 - phislices;
			}
			if (mminus < 0)
			{
				mminus = m - 1 + phislices;
			}
		}

		// calculate boundary r
		for (int j = 0; j < columns; j++)
		{
			// forward difference
			arrayofArrayEr[m * rows * columns + 0 * columns + j] = -1 * (-0.5 * arrayofArrayV[m * rows * columns + 2 * columns + j] + 2.0 * arrayofArrayV[m * rows * columns + 1 * columns + j] - 1.5 * arrayofArrayV[m * rows * columns + 0 * columns + j]) / gridSizeR;
			// backward difference
			arrayofArrayEr[m * rows * columns + (rows - 1) * columns + j] = -1 * (1.5 * arrayofArrayV[m * rows * columns + (rows - 1) * columns + j] - 2.0 * arrayofArrayV[m * rows * columns + (rows - 2) * columns + j] + 0.5 * arrayofArrayV[m * rows * columns + (rows - 3) * columns + j]) / gridSizeR;
		}

		for (int i = 0; i < rows; i += rows - 1)
		{
			radius = fgkIFCRadius + i * gridSizeR;
			for (int j = 1; j < columns - 1; j++)
			{
				// z direction
				arrayofArrayEz[m * rows * columns + i * columns + j] = -1 * (arrayofArrayV[m * rows * columns + i * columns + (j + 1)] - arrayofArrayV[m * rows * columns + i * columns + (j - 1)]) / (2 * gridSizeZ);
				// phi direction
				arrayofArrayEphi[m * rows * columns + i * columns + j] = -1 * (signplus * arrayofArrayV[mplus * rows * columns + i * columns + j] - signminus * arrayofArrayV[mminus * rows * columns + i * columns + j]) / (2 * radius * gridSizePhi);
			}
		}
		
		// calculate boundary z
		for (int i = 0; i < rows; i++)
		{
			arrayofArrayEz[m * rows * columns + i * columns + 0] = -1 * (-0.5 * arrayofArrayV[m * rows * columns + i * columns + 2] + 2.0 * arrayofArrayV[m * rows * columns + i * columns + 1] - 1.5 * arrayofArrayV[m * rows * columns + i * columns + 0]) / gridSizeZ;
			arrayofArrayEz[m * rows * columns + i * columns + (columns - 1)] = -1 * (1.5 * arrayofArrayV[m * rows * columns + i * columns + (columns - 1)] - 2.0 * arrayofArrayV[m * rows * columns + i * columns + (columns - 2)] + 0.5 * arrayofArrayV[m * rows * columns + i * columns + (columns - 3)]) / gridSizeZ;			
		}

		for (int i = 1; i < rows - 1; i++)
		{
			radius = fgkIFCRadius + i * gridSizeR;
			for (int j = 0; j < columns; j += columns - 1)
			{
				// r direction
				arrayofArrayEr[m * rows * columns + i * columns + j] = -1 * (arrayofArrayV[m * rows * columns + (i + 1) * columns + j] - arrayofArrayV[m * rows * columns + (i - 1) * columns + j]) / (2 * gridSizeR);
				// phi direction
				arrayofArrayEphi[m * rows * columns + i * columns + j] = -1 * (signplus * arrayofArrayV[mplus * rows * columns + i * columns + j] - signminus * arrayofArrayV[mminus * rows * columns + i * columns + j]) / (2 * radius * gridSizePhi);
			}
		}
		
		// calculate corner points for Ephi
		for ( int i = 0; i < rows; i += rows - 1)
		{
			radius = fgkIFCRadius + i * gridSizeR;
			for (int j = 0; j < columns; j += columns - 1)
			{
				// phi didrection
				arrayofArrayEphi[m * rows * columns + i * columns + j] = -1 * (signplus * arrayofArrayV[mplus * rows * columns + i * columns + j] - signminus * arrayofArrayV[mminus * rows * columns + i * columns + j]) / (2 * radius * gridSizePhi);
			}
		}
	}
}


extern "C" void DifferentialCalculationGPU 
(
	float *arrayofArrayV,
	float *arrayofArrayEr,
	float *arrayofArrayEz,
	float *arrayofArrayEphi, 
	const int rows,
	const int columns,
	const int phislices,
	const int symmetry,
	const float fgkIFCRadius,
	const float fgkOFCRadius,
	const float fgkTPCZ0
)
{
	nonBoundaryDifferentialCalculationGPU(arrayofArrayV, arrayofArrayEr, arrayofArrayEz, arrayofArrayEphi,rows, columns, phislices, symmetry, fgkIFCRadius,fgkOFCRadius, fgkTPCZ0);
	boundaryDifferentialCalculationCPU(arrayofArrayV, arrayofArrayEr, arrayofArrayEz, arrayofArrayEphi, rows, columns, phislices, symmetry, fgkIFCRadius,fgkOFCRadius, fgkTPCZ0);
}
