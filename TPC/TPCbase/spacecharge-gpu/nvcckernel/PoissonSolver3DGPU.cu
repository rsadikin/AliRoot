#include "PoissonSolver3DGPU.h"
#include <cuda.h>
#include <math.h>

// GPU constant variables
__device__ __constant__ int d_coef_StartPos;
__device__ __constant__ int d_grid_StartPos;
__device__ __constant__ float d_h2;
__device__ __constant__ float d_ih2;
__device__ __constant__ float d_tempRatioZ;


/* GPU kernels start */
__global__ void relaxationGaussSeidelRed
(
	float *VPotential,
	float *RhoChargeDensity,
	const int RRow,
	const int ZColumn,
	const int PhiSlice,
	float *coef1, 
	float *coef2,
	float *coef3, 
	float *coef4
)
{
	int index_x, index_y, index, index_left, index_right, index_up, index_down, index_front, index_back, index_coef;

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	index		= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;
	index_left	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (index_x - 1);
	index_right	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (index_x + 1);
	index_up	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + (index_y - 1) * ZColumn + index_x;
	index_down	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + (index_y + 1) * ZColumn + index_x;
	index_front	= d_grid_StartPos + ((blockIdx.z - 1 + PhiSlice) % PhiSlice) * RRow * ZColumn + index_y * ZColumn + index_x;
	index_back	= d_grid_StartPos + ((blockIdx.z + 1) % PhiSlice) * RRow * ZColumn + index_y * ZColumn + index_x;
	index_coef	= d_coef_StartPos + index_y;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		//calculate red			
		if ((blockIdx.z % 2 == 0 && (index_x + index_y) % 2 == 0) || (blockIdx.z % 2 != 0 && (index_x + index_y) % 2 != 0))
		{			
			VPotential[index] = (coef2[index_coef] * VPotential[index_up] + 
								coef1[index_coef] * VPotential[index_down] + 
								d_tempRatioZ * (VPotential[index_left] + VPotential[index_right]) + 
								coef3[index_coef] * (VPotential[index_front] + VPotential[index_back]) + 
								d_h2 * RhoChargeDensity[index]) * coef4[index_coef];
		}
	}
}

__global__ void relaxationGaussSeidelBlack
(
	float *VPotential,
	float *RhoChargeDensity,
	const int RRow,
	const int ZColumn,
	const int PhiSlice,
	float *coef1, 
	float *coef2,
	float *coef3, 
	float *coef4
)
{
	int index_x, index_y, index, index_left, index_right, index_up, index_down, index_front, index_back, index_coef;

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	index		= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;
	index_left	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (index_x - 1);
	index_right	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (index_x + 1);
	index_up	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + (index_y - 1) * ZColumn + index_x;
	index_down	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + (index_y + 1) * ZColumn + index_x;
	index_front	= d_grid_StartPos + ((blockIdx.z - 1 + PhiSlice) % PhiSlice) * RRow * ZColumn + index_y * ZColumn + index_x;
	index_back	= d_grid_StartPos + ((blockIdx.z + 1) % PhiSlice) * RRow * ZColumn + index_y * ZColumn + index_x;
	index_coef	= d_coef_StartPos + index_y;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		//calculate black		
		if ((blockIdx.z % 2 == 0 && (index_x + index_y) % 2 != 0) || (blockIdx.z % 2 != 0 && (index_x + index_y) % 2 == 0))
		{			
			VPotential[index] = (coef2[index_coef] * VPotential[index_up] + 
								coef1[index_coef] * VPotential[index_down] +								
								d_tempRatioZ * (VPotential[index_left] + VPotential[index_right]) + 
								coef3[index_coef] * (VPotential[index_front] + VPotential[index_back]) + 
								d_h2 * RhoChargeDensity[index]) * coef4[index_coef];
		}
	}
}

__global__ void residueCalculation
(
	float *VPotential,
	float *RhoChargeDensity,
	float *DeltaResidue,
	const int RRow,
	const int ZColumn,
	const int PhiSlice,
	float *coef1, 
	float *coef2,
	float *coef3, 
	float *icoef4
)
{
	int index_x, index_y, index, index_left, index_right, index_up, index_down, index_front, index_back, index_coef;

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	index		= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;
	index_left	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (index_x - 1);
	index_right	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (index_x + 1);
	index_up	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + (index_y - 1) * ZColumn + index_x;
	index_down	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + (index_y + 1) * ZColumn + index_x;
	index_front	= d_grid_StartPos + ((blockIdx.z - 1 + PhiSlice) % PhiSlice)  * RRow * ZColumn + index_y * ZColumn + index_x;
	index_back	= d_grid_StartPos + ((blockIdx.z + 1) % PhiSlice)  * RRow * ZColumn + index_y * ZColumn + index_x;
	index_coef	= d_coef_StartPos + index_y;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		DeltaResidue[index] = d_ih2 * (coef2[index_coef] * VPotential[index_up] +
						coef1[index_coef] * VPotential[index_down] +
						d_tempRatioZ * (VPotential[index_left] + VPotential[index_right]) +
						coef3[index_coef] * (VPotential[index_front] + VPotential[index_back]) -
						icoef4[index_coef] * VPotential[index]) + RhoChargeDensity[index];
	}
}

__global__ void restriction2DHalf
(
	float *RhoChargeDensity,
	float *DeltaResidue,	
	const int RRow,
	const int ZColumn,
	const int PhiSlice
)
{
	int index_x, index_y, index;
	int finer_RRow, finer_ZColumn, finer_grid_StartPos;
	int finer_index_x, finer_index_y, finer_index, finer_index_left, finer_index_right, finer_index_up, finer_index_down;
	
	index_x	= blockIdx.x * blockDim.x + threadIdx.x;
	index_y	= blockIdx.y * blockDim.y + threadIdx.y;
	index	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;

	finer_RRow = 2 * RRow - 1;
	finer_ZColumn = 2 * ZColumn - 1;

	finer_grid_StartPos = d_grid_StartPos - (finer_RRow * finer_ZColumn * PhiSlice);

	finer_index_x = index_x * 2;
	finer_index_y = index_y * 2;

	finer_index			= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + finer_index_y * finer_ZColumn + finer_index_x;
	finer_index_left	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + finer_index_y * finer_ZColumn + (finer_index_x - 1);
	finer_index_right	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + finer_index_y * finer_ZColumn + (finer_index_x + 1);
	finer_index_up		= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y - 1) * finer_ZColumn + finer_index_x;
	finer_index_down	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y + 1) * finer_ZColumn + finer_index_x;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		RhoChargeDensity[index] = 0.5 * DeltaResidue[finer_index] + 
								0.125 * (DeltaResidue[finer_index_left] + DeltaResidue[finer_index_right] + DeltaResidue[finer_index_up] + DeltaResidue[finer_index_down]);
	}
}

__global__ void restriction2DFull
(
	float *RhoChargeDensity,
	float *DeltaResidue,	
	const int RRow,
	const int ZColumn,
	const int PhiSlice
)
{
	int index_x, index_y, index;
	int finer_RRow, finer_ZColumn, finer_grid_StartPos;
	int finer_index_x, finer_index_y, finer_index, finer_index_left, finer_index_right, finer_index_up, finer_index_down;
	int finer_index_up_left, finer_index_up_right, finer_index_down_left, finer_index_down_right;
	
	index_x	= blockIdx.x * blockDim.x + threadIdx.x;
	index_y	= blockIdx.y * blockDim.y + threadIdx.y;
	index	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;

	finer_RRow = 2 * RRow - 1;
	finer_ZColumn = 2 * ZColumn - 1;

	finer_grid_StartPos = d_grid_StartPos - (finer_RRow * finer_ZColumn * PhiSlice);

	finer_index_x = index_x * 2;
	finer_index_y = index_y * 2;

	finer_index			= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + finer_index_y * finer_ZColumn + finer_index_x;
	finer_index_left	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + finer_index_y * finer_ZColumn + (finer_index_x - 1);
	finer_index_right	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + finer_index_y * finer_ZColumn + (finer_index_x + 1);
	finer_index_up		= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y - 1) * finer_ZColumn + finer_index_x;
	finer_index_down	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y + 1) * finer_ZColumn + finer_index_x;
	finer_index_up_left		= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y - 1) * finer_ZColumn + (finer_index_x - 1);
	finer_index_up_right	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y - 1) * finer_ZColumn + (finer_index_x + 1);
	finer_index_down_left	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y + 1) * finer_ZColumn + (finer_index_x - 1);
	finer_index_down_right	= finer_grid_StartPos + blockIdx.z * finer_RRow * finer_ZColumn + (finer_index_y + 1) * finer_ZColumn + (finer_index_x + 1);

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		RhoChargeDensity[index] = 0.25 * DeltaResidue[finer_index] +
								0.125 * (DeltaResidue[finer_index_left] + DeltaResidue[finer_index_right] + DeltaResidue[finer_index_up] + DeltaResidue[finer_index_down]) +
								0.0625 * (DeltaResidue[finer_index_up_left] + DeltaResidue[finer_index_up_right] + DeltaResidue[finer_index_down_left] + DeltaResidue[finer_index_down_right]);
	} else {
		RhoChargeDensity[index] =  DeltaResidue[finer_index];
	}
	

}

__global__ void zeroingVPotential
(
	float *VPotential,
	const int RRow,
	const int ZColumn,
	const int PhiSlice
)
{
	int index_x, index_y, index;

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	index		= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		// zeroing V
		VPotential[index] = 0;
	}

	if (index_x == ZColumn - 2) {
		index_x++;
		index			= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;
		VPotential[index] 	= 0;		
	}
}


__global__ void zeroingBoundaryTopBottom
(
	float *VPotential,
	int RRow,
	int ZColumn,
	int PhiSlice
)
{
	int index_x, index_top, index_bottom;
	
	index_x = blockIdx.x * blockDim.x + threadIdx.x;

	index_top = d_grid_StartPos + blockIdx.z * RRow * ZColumn + 0 * ZColumn + index_x;
	index_bottom = d_grid_StartPos + blockIdx.z * RRow * ZColumn + (ZColumn - 1) * ZColumn + index_x;

	if (index_x < RRow)
	{
		VPotential[index_top] = 0.0;
		VPotential[index_bottom] = 0.0;
	}
}

__global__ void zeroingBoundaryLeftRight
(
	float *VPotential,
	int RRow,
	int ZColumn,
	int PhiSlice
)
{
	int index_y, index_left, index_right;
	
	index_y = blockIdx.x * blockDim.x + threadIdx.x;

	index_left = d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + 0;
	index_right = d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + (RRow - 1);

	if (index_y < ZColumn)
	{
		VPotential[index_left] = 0.0;
		VPotential[index_right] = 0.0;
	}
}

__global__ void prolongation2DHalf
(
	float *VPotential,
	const int RRow,
	const int ZColumn,
	const int PhiSlice
)
{
	int index_x, index_y, index;
	
	int coarser_RRow = (RRow >> 1) + 1;
	int coarser_ZColumn = (ZColumn >> 1) + 1;
	int coarser_grid_StartPos = d_grid_StartPos + RRow * ZColumn * PhiSlice;

	int coarser_index_self;
	int coarser_index_up, coarser_index_down, coarser_index_left, coarser_index_right;	
	int coarser_index_up_left, coarser_index_up_right, coarser_index_down_left, coarser_index_down_right;

	index_x	= blockIdx.x * blockDim.x + threadIdx.x;
	index_y	= blockIdx.y * blockDim.y + threadIdx.y;
	index	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		// x odd, y odd
		if ((index_x % 2 != 0) && (index_y % 2 != 0))
		{
			coarser_index_up_left = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2);
			coarser_index_up_right = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2 + 1);
			coarser_index_down_left = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2 + 1) * coarser_ZColumn + (index_x / 2);
			coarser_index_down_right = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2 + 1) * coarser_ZColumn + (index_x / 2 + 1);

			VPotential[index] += 0.25 * (VPotential[coarser_index_up_left] + VPotential[coarser_index_up_right] + VPotential[coarser_index_down_left] + VPotential[coarser_index_down_right]);
		}
		// x even, y odd
		else if ((index_x % 2 == 0) && (index_y % 2 != 0))
		{
			coarser_index_up = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2);
			coarser_index_down = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2 + 1) * coarser_ZColumn + (index_x / 2);

			VPotential[index] += 0.5 * (VPotential[coarser_index_up] + VPotential[coarser_index_down]);
		}
		// x odd, y even
		else if ((index_x % 2 != 0) && (index_y % 2 == 0))
		{
			coarser_index_left = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2);
			coarser_index_right = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2 + 1);

			VPotential[index] += 0.5 * (VPotential[coarser_index_left] + VPotential[coarser_index_right]);
		}
		// x even, y even
		else
		{
			coarser_index_self = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2)	 * coarser_ZColumn + (index_x / 2);

			VPotential[index] += VPotential[coarser_index_self];
		}
	}
}

__global__ void prolongation2DHalfNoAdd
(
	float *VPotential,
	const int RRow,
	const int ZColumn,
	const int PhiSlice
)
{
	int index_x, index_y, index;
	
	int coarser_RRow = (RRow >> 1) + 1;
	int coarser_ZColumn = (ZColumn >> 1) + 1;
	int coarser_grid_StartPos = d_grid_StartPos + RRow * ZColumn * PhiSlice;

	int coarser_index_self;
	int coarser_index_up, coarser_index_down, coarser_index_left, coarser_index_right;	
	int coarser_index_up_left, coarser_index_up_right, coarser_index_down_left, coarser_index_down_right;

	index_x	= blockIdx.x * blockDim.x + threadIdx.x;
	index_y	= blockIdx.y * blockDim.y + threadIdx.y;
	index	= d_grid_StartPos + blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;

	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		// x odd, y odd
		if ((index_x % 2 != 0) && (index_y % 2 != 0))
		{
			coarser_index_up_left = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2);
			coarser_index_up_right = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2 + 1);
			coarser_index_down_left = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2 + 1) * coarser_ZColumn + (index_x / 2);
			coarser_index_down_right = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2 + 1) * coarser_ZColumn + (index_x / 2 + 1);

			VPotential[index] = 0.25 * (VPotential[coarser_index_up_left] + VPotential[coarser_index_up_right] + VPotential[coarser_index_down_left] + VPotential[coarser_index_down_right]);
		}
		// x even, y odd
		else if ((index_x % 2 == 0) && (index_y % 2 != 0))
		{
			coarser_index_up = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2);
			coarser_index_down = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2 + 1) * coarser_ZColumn + (index_x / 2);

			VPotential[index] = 0.5 * (VPotential[coarser_index_up] + VPotential[coarser_index_down]);
		}
		// x odd, y even
		else if ((index_x % 2 != 0) && (index_y % 2 == 0))
		{
			coarser_index_left = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2);
			coarser_index_right = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2) * coarser_ZColumn + (index_x / 2 + 1);

			VPotential[index] = 0.5 * (VPotential[coarser_index_left] + VPotential[coarser_index_right]);
		}
		// x even, y even
		else
		{
			coarser_index_self = coarser_grid_StartPos + blockIdx.z * coarser_RRow * coarser_ZColumn + (index_y / 2)	 * coarser_ZColumn + (index_x / 2);

			VPotential[index] = VPotential[coarser_index_self];
		}
	}
}


__global__ void errorCalculation
(
	float *VPotentialPrev,
	float *VPotential,
	float *EpsilonError,
	const int RRow,
	const int ZColumn,
	const int PhiSlice
)
{
	int index_x, index_y, index;
	float error;
	float sum_error;

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	index =  blockIdx.z * RRow * ZColumn + index_y * ZColumn + index_x;
	
	if (index_x != 0 && index_x < (ZColumn - 1) && index_y != 0 && index_y < (RRow - 1))
	{
		error = VPotential[index] - VPotentialPrev[index];
		sum_error = error * error;
		__syncthreads();

		atomicAdd( EpsilonError, sum_error );

	}
}
/* GPU kernels end */



/* Error related functions start */
float GetErrorNorm2
(
	float * VPotential,
	float * VPotentialPrev,
	const int rows,
	const int cols,
	float weight
) 
{
	float error = 0.0;	
	float sum_error = 0.0;
	for (int i=0;i<rows;i++)
		for (int j=0;j <cols;j++)
			{
				error = (VPotential[i * cols + j] - VPotentialPrev[i * cols + j]) / weight;
				sum_error  += (error * error);
			}
			
	return sum_error / (rows * cols);
}


float GetAbsMax
(
	float *VPotentialExact,
	int size
)
{
	float mymax = 0.0;
	for (int i=0;i< size;i++) 
		if (abs(VPotentialExact[i]) > mymax) mymax = abs(VPotentialExact[i]); 
	return mymax;
}
/* Error related functions end */

/* Restrict Boundary for FCycle start -- just CPU enough */

void Restrict_Boundary
(
	float *VPotential, 
	const int RRow, 
	const int ZColumn, 
	const int PhiSlice, 
	const int Offset
)
{
	int i,ii,j,jj;		
	int finer_RRow = 2 * RRow - 1;
	int finer_ZColumn = 2 * ZColumn - 1;
	
	int finer_Offset = Offset - (finer_RRow * finer_ZColumn * PhiSlice);
	int sliceStart;
	int finer_SliceStart;

	//printf("(%d,%d,%d) -> (%d,%d,%d)\n",RRow,ZColumn,Offset,finer_RRow,finer_ZColumn,finer_Offset); 
	// do for each slice
	for ( int m = 0;m < PhiSlice;m++)
	{	
		sliceStart = m * (RRow * ZColumn);
		finer_SliceStart = m * (finer_RRow * finer_ZColumn);
		// copy boundary
		for ( j = 0, jj =0; j < ZColumn ; j++,jj+=2) 
		{
			VPotential[Offset + sliceStart + (0 * ZColumn) + j] =
				VPotential[finer_Offset + finer_SliceStart + (0 * finer_ZColumn) + jj];

			VPotential[Offset + sliceStart + ((RRow - 1) * ZColumn) + j] =
				VPotential[finer_Offset + finer_SliceStart + ((finer_RRow -1) * finer_ZColumn) + jj];

		}		
		for ( i = 0, ii =0; i < RRow  ; i++,ii+=2) {
			VPotential[Offset + sliceStart + (i * ZColumn)] =
				VPotential[finer_Offset + finer_SliceStart + (ii * finer_ZColumn)];
			
			VPotential[Offset + sliceStart + (i * ZColumn) + (ZColumn - 1)] =
				VPotential[finer_Offset + finer_SliceStart + (ii * finer_ZColumn) + (finer_ZColumn - 1)];

		}
	}
/**
		// top left (0,0)

		// boundary in top and down
		for ( j = 1, jj =2; j < ZColumn-1 ; j++,jj+=2) 
		{
			VPotential[Offset + sliceStart + (0 * ZColumn) + j] =
				0.5 * VPotential[finer_Offset + finer_SliceStart + (0 * finer_ZColumn) + jj] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + (0 * finer_ZColumn) + jj - 1] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + (0 * finer_ZColumn) + jj + 1];
			
			VPotential[Offset + sliceStart + ((RRow - 1) * ZColumn) + j] =
				0.5 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow -1) * finer_ZColumn) + jj] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow -1) * finer_ZColumn) + jj - 1] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow -1) * finer_ZColumn) + jj + 1];

				 
		}
				
		// boundary in left and right
		for ( i = 1, ii =2; i < RRow - 1 ; i++,ii+=2) {
			VPotential[Offset + sliceStart + (i * ZColumn)] =
				0.5 * VPotential[finer_Offset + finer_SliceStart + (ii * finer_ZColumn)] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + ((ii-1) * finer_ZColumn)] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + ((ii + 1) * finer_ZColumn)];
			
			VPotential[Offset + sliceStart + (i * ZColumn) + (ZColumn - 1)] =
				0.5 * VPotential[finer_Offset + finer_SliceStart + (ii * finer_ZColumn) + jj  + (finer_ZColumn - 1)] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + ((ii -1) * finer_ZColumn) + (finer_ZColumn - 1)] +
				0.25 * VPotential[finer_Offset + finer_SliceStart + ((ii +1) * finer_ZColumn) + (finer_ZColumn - 1)];

		}

		// top left (0,0)

		VPotential[Offset + sliceStart + (0 * ZColumn) + 0] =
			0.5 * VPotential[finer_Offset  + finer_SliceStart] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + (0 * finer_ZColumn) + 1] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + (1 * finer_ZColumn)];
		
		// top right
		VPotential[Offset + sliceStart + (0 * ZColumn) + (ZColumn - 1) ] =
			0.5 * VPotential[finer_Offset + finer_SliceStart  + (0 * finer_ZColumn) + (finer_ZColumn -1) ] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + (0 * finer_ZColumn) + (finer_ZColumn - 2)] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + (1 * finer_ZColumn) + (finer_ZColumn - 1)];

		
		// bottom left
		VPotential[Offset + sliceStart + ((RRow - 1) * ZColumn) + 0] =
			0.5 * VPotential[finer_Offset + finer_SliceStart  + ((finer_RRow - 1) * finer_ZColumn) + 0] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow - 1) * finer_ZColumn) + 1] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow - 2) * finer_ZColumn)  + 0];

		// bottom right
		VPotential[Offset + sliceStart + ((RRow - 1) * ZColumn) + (ZColumn - 1)] =
			0.5 * VPotential[finer_Offset + finer_SliceStart  + ((finer_RRow - 1) * finer_ZColumn) + (finer_ZColumn - 1)] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow - 1) * finer_ZColumn) + (finer_ZColumn - 2)] +
			0.25 * VPotential[finer_Offset + finer_SliceStart + ((finer_RRow - 2) * finer_ZColumn)  + (finer_ZColumn - 1)];	

	}
**/
}

/* Restrict Boundary for FCycle end */

/** Print matrix  **/

void PrintMatrix
(
	float *Mat,
	const int Row, 
	const int Column
)
{
	printf("Matrix (%d,%d)\n",Row,Column);
	for (int i=0;i<Row;i++)
	{
		for (int j=0;j<Column;j++)
		{
			printf("%11.4g ",Mat[i*Column + j]);
		}
		printf("\n");
	}

} 



/* Cycle functions start */
void VCycleSemiCoarseningGPU
(
	float *d_VPotential,
	float *d_RhoChargeDensity,
	float *d_DeltaResidue,
	float *d_coef1,
	float *d_coef2,
	float *d_coef3,
	float *d_coef4,
	float *d_icoef4,
	float gridSizeR,
	float ratioZ,
	float ratioPhi,
	int RRow,
	int ZColumn,
	int PhiSlice,
	int gridFrom,
	int gridTo,
	int nPre,
	int nPost	
)
{
	int grid_RRow;
	int grid_ZColumn;
	int grid_PhiSlice = PhiSlice;
	int grid_StartPos;
	int coef_StartPos;
	int iOne, jOne;
	float h, h2, ih2;
	float tempRatioZ;
	float tempRatioPhi;
	float radius;
	
	// V-Cycle => Finest Grid
	iOne = 1 << (gridFrom - 1); 
	jOne = 1 << (gridFrom - 1);

	//grid_RRow		= ((RRow - 1) / iOne) + 1;
	//grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

	// change accordingly to gridFrom
	grid_StartPos = 0;
	coef_StartPos = 0;


	for (int step = 1; step < gridFrom; step++)
	{
		grid_RRow = ((RRow - 1) / (1 << (step - 1))) + 1;
		grid_ZColumn = ((ZColumn - 1) / (1 << (step - 1))) + 1;
		
		grid_StartPos += grid_RRow * grid_ZColumn * grid_PhiSlice;
		coef_StartPos += grid_RRow;
	}

	grid_RRow		= ((RRow - 1) / iOne) + 1;
	grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;



	// pre-compute constant memory
	h 	= gridSizeR * iOne;
	h2	= h * h;
	ih2	= 1.0 / h2;

	tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
	tempRatioPhi = ratioPhi * iOne * iOne;

	// copy constant to device memory
	cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

	// set kernel grid size and block size
	dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
	dim3 grid_ThreadPerBlock(16, 16);

	// red-black gauss seidel relaxation (nPre times)
	for (int i = 0; i < nPre; i++)
	{
		relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
		relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
	}

	// residue calculation
	residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );

	// V-Cycle => from finer to coarsest grid
	for (int step = gridFrom + 1; step <= gridTo; step++)
	{
		iOne = 1 << (step - 1); 
		jOne = 1 << (step - 1);

		grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;
		coef_StartPos += grid_RRow;

		grid_RRow		= ((RRow - 1) / iOne) + 1;
		grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

		// pre-compute constant memory
		h	= gridSizeR * iOne;
		h2	= h * h;
		ih2	= 1.0 / h2;

		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		// copy constant to device memory
		cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

		// set kernel grid size and block size
		dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
		dim3 grid_ThreadPerBlock(16, 16);

		// restriction
		restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice );

		// zeroing V
		zeroingVPotential<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );

		// zeroing boundaries
		dim3 grid_BlockPerGridTopBottom((grid_RRow < 16) ? 1 : ((grid_RRow / 16) + 1), 1, PhiSlice);
		dim3 grid_BlockPerGridLeftRight((grid_ZColumn < 16) ? 1 : ((grid_ZColumn / 16) + 1), 1, PhiSlice);
		dim3 grid_ThreadPerBlockBoundary(16);

		zeroingBoundaryTopBottom<<< grid_BlockPerGridTopBottom, grid_ThreadPerBlockBoundary >>>( d_VPotential, grid_RRow, grid_ZColumn, PhiSlice );
		zeroingBoundaryLeftRight<<< grid_BlockPerGridLeftRight, grid_ThreadPerBlockBoundary >>>( d_VPotential, grid_RRow, grid_ZColumn, PhiSlice );

		// red-black gauss seidel relaxation (nPre times)
		for (int i = 0; i < nPre; i++)
		{
			relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
			relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
		}

		// residue calculation
		if (step < gridTo)
		{
			residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );

		}
	}

	// V-Cycle => from coarser to finer grid
	for (int step = (gridTo - 1); step >= gridFrom; step--)
	{
		iOne = iOne / 2;
		jOne = jOne / 2;
	
		grid_RRow		= ((RRow - 1) / iOne) + 1;
		grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

		grid_StartPos -= grid_RRow * grid_ZColumn * PhiSlice;
		coef_StartPos -= grid_RRow;
	
		h	= gridSizeR * iOne;
		h2	= h * h;
		ih2	= 1.0 / h2;
	
		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		// copy constant to device memory
		cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

		// set kernel grid size and block size
		dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
		dim3 grid_ThreadPerBlock(16, 16);

		// prolongation
		prolongation2DHalf<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );

		// red-black gauss seidel relaxation (nPost times)
		for (int i = 0; i < nPost; i++)
		{
			relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
			relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
		}
	}
}
/* Cycle functions end */





/*extern function */
extern "C" void PoissonMultigrid3DSemiCoarseningGPUError
(
	float *VPotential, 
	float *RhoChargeDensity,
	const int RRow, 
	const int ZColumn,  
	const int PhiSlice,   
	const int Symmetry,
	float *fparam,
	int *iparam,
	bool isExactPresent,
	float *errorConv,
	float *errorExact,
	float *VPotentialExact //allocation in the client
)
{
	// variables for CPU memory
	float *temp_VPotential;
	float *VPotentialPrev;
	float *EpsilonError;	

	// variables for GPU memory	
	float *d_VPotential;
	float *d_RhoChargeDensity;
	float *d_DeltaResidue;
	float *d_VPotentialPrev;
	float *d_EpsilonError;
	
	float *d_coef1;
	float *d_coef2;
	float *d_coef3;
	float *d_coef4;
	float *d_icoef4;

	// variables for coefficent calculations
	float *coef1;
	float *coef2;
	float *coef3;
	float *coef4;
	float *icoef4;
	float tempRatioZ;
	float tempRatioPhi;
	float radius;

	int gridFrom;
	int gridTo; 
	int loops;


	// variables passed from ALIROOT
	float gridSizeR		= fparam[0];
	float gridSizePhi	= fparam[1];
	float gridSizeZ		= fparam[2];
	float ratioPhi		= fparam[3];
	float ratioZ		= fparam[4];
	float convErr		= fparam[5];
	float IFCRadius		= fparam[6];
	int nPre		= iparam[0];
	int nPost		= iparam[1];
	int maxLoop		= iparam[2];
	int nCycle		= iparam[3];

	// variables for calculating GPU memory allocation
	int grid_RRow;
	int grid_ZColumn;
	int grid_PhiSlice = PhiSlice;
	int grid_Size = 0;
	int grid_StartPos;
	int coef_Size = 0;
	int coef_StartPos;
	int iOne, jOne;
	float h, h2, ih2;

	// variables for calculating multigrid maximum depth
	int depth_RRow = 0;
	int depth_ZColumn = 0;
	int temp_RRow = RRow;
	int temp_ZColumn = ZColumn;

	// calculate depth for multigrid
	while (temp_RRow >>= 1) depth_RRow++;  
	while (temp_ZColumn >>= 1) depth_ZColumn++;
  
	loops = (depth_RRow > depth_ZColumn) ? depth_ZColumn : depth_RRow;
	loops = (loops > maxLoop) ? maxLoop : loops;

	gridFrom = 1;
	gridTo = loops;

	// calculate GPU memory allocation for multigrid
	for (int step = gridFrom; step <= gridTo; step++)
	{
		grid_RRow = ((RRow - 1) / (1 << (step - 1))) + 1;
		grid_ZColumn = ((ZColumn - 1) / (1 << (step - 1))) + 1;
		
		grid_Size += grid_RRow * grid_ZColumn * grid_PhiSlice;
		coef_Size += grid_RRow;
	}

	// allocate memory for temporary output
	temp_VPotential 		= (float *) malloc(grid_Size * sizeof(float));
	VPotentialPrev = (float *) malloc(RRow * ZColumn * PhiSlice * sizeof(float));
	EpsilonError = (float *) malloc(1 * sizeof(float));


	// allocate memory for relaxation coefficient
	coef1 = (float *) malloc(coef_Size * sizeof(float));
	coef2 = (float *) malloc(coef_Size * sizeof(float));
	coef3 = (float *) malloc(coef_Size * sizeof(float));
	coef4 = (float *) malloc(coef_Size * sizeof(float));
	icoef4 = (float *) malloc(coef_Size * sizeof(float));

	// pre-compute relaxation coefficient
	coef_StartPos = 0;
	iOne = 1 << (gridFrom - 1); 
	jOne = 1 << (gridFrom - 1);
	
	for (int step = gridFrom; step <= gridTo; step++)
	{
		grid_RRow = ((RRow - 1) / iOne) + 1;

		h = gridSizeR * iOne;
		h2 = h * h;
		ih2 = 1.0 / h2;

		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		for (int i = 1; i < grid_RRow - 1; i++)
		{
			radius = IFCRadius + i * h;
			coef1[coef_StartPos + i] = 1.0 + h / (2 * radius);
			coef2[coef_StartPos + i] = 1.0 - h / (2 * radius);
			coef3[coef_StartPos + i] = tempRatioPhi / (radius * radius);
			coef4[coef_StartPos + i] = 0.5 / (1.0 + tempRatioZ + coef3[coef_StartPos + i]);
			icoef4[coef_StartPos + i] = 1.0 / coef4[coef_StartPos + i];
		}
		coef_StartPos += grid_RRow;
		iOne = 2 * iOne;
		jOne = 2 * jOne;
	}

	// device memory allocation
	cudaMalloc( &d_VPotential, grid_Size * sizeof(float) );
	cudaMalloc( &d_VPotentialPrev, RRow * ZColumn * PhiSlice * sizeof(float) );
	cudaMalloc( &d_EpsilonError, 1 * sizeof(float) );	
	cudaMalloc( &d_DeltaResidue, grid_Size * sizeof(float) );
	cudaMalloc( &d_RhoChargeDensity, grid_Size * sizeof(float) );
	cudaMalloc( &d_coef1, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef2, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef3, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef4, coef_Size * sizeof(float) );
	cudaMalloc( &d_icoef4, coef_Size * sizeof(float) );

	// set memory to zero
	cudaMemset( d_VPotential, 0, grid_Size * sizeof(float) );
	cudaMemset( d_DeltaResidue, 0, grid_Size * sizeof(float) );
	cudaMemset( d_RhoChargeDensity, 0, grid_Size * sizeof(float) );
	cudaMemset( d_VPotentialPrev, 0, RRow * ZColumn * PhiSlice * sizeof(float) );
	cudaMemset( d_EpsilonError, 0, 1 * sizeof(float) );


	// copy data from host to device
	cudaMemcpy( d_VPotential, VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice ); //check
	cudaMemcpy( d_RhoChargeDensity, RhoChargeDensity, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice ); //check
	cudaMemcpy( d_coef1, coef1, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef2, coef2, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef3, coef3, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef4, coef4, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_icoef4, icoef4, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_VPotentialPrev, VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice );
	
	// max exact
	
	// float maxAbsExact = GetAbsMax(VPotentialExact, RRow * PhiSlice * ZColumn);
	float maxAbsExact = 1.0;

	if (isExactPresent == true)
		maxAbsExact = GetAbsMax(VPotentialExact, RRow * PhiSlice * ZColumn);
	dim3 error_BlockPerGrid((RRow < 16) ? 1 : (RRow / 16), (ZColumn < 16) ? 1 : (ZColumn / 16), PhiSlice);
	dim3 error_ThreadPerBlock(16, 16);		

	
	for (int cycle = 0; cycle < nCycle; cycle++)
	{
		cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
		if (isExactPresent == true) errorExact[cycle] = GetErrorNorm2(temp_VPotential, VPotentialExact, RRow * PhiSlice,ZColumn, maxAbsExact); 


		VCycleSemiCoarseningGPU(d_VPotential, d_RhoChargeDensity, d_DeltaResidue, d_coef1, d_coef2, d_coef3, d_coef4, d_icoef4, gridSizeR, ratioZ, ratioPhi, RRow, ZColumn, PhiSlice, gridFrom, gridTo, nPre, nPost);
		

		errorCalculation<<< error_BlockPerGrid, error_ThreadPerBlock >>> ( d_VPotentialPrev, d_VPotential, d_EpsilonError, RRow, ZColumn, PhiSlice);

		cudaMemcpy( EpsilonError, d_EpsilonError, 1 * sizeof(float), cudaMemcpyDeviceToHost );		
		

		errorConv[cycle] = *EpsilonError  / (RRow * ZColumn * PhiSlice);

		if (errorConv[cycle] < convErr)
		{
			//errorConv
			nCycle = cycle;
			break;
		}

		cudaMemcpy( d_VPotentialPrev, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToDevice );
		cudaMemset( d_EpsilonError, 0, 1 * sizeof(float) );

	}
	iparam[3] = nCycle;

//	for (int cycle = 0; cycle < nCycle; cycle++)
//	{
//		cudaMemcpy( temp_VPotentialPrev, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );

	
//		VCycleSemiCoarseningGPU(d_VPotential, d_RhoChargeDensity, d_DeltaResidue, d_coef1, d_coef2, d_coef3, d_coef4, d_icoef4, gridSizeR, ratioZ, ratioPhi, RRow, ZColumn, PhiSlice, gridFrom, gridTo, nPre, nPost);
		
//		cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
//		errorConv[cycle] = GetErrorNorm2(temp_VPotential, temp_VPotentialPrev, RRow * PhiSlice, ZColumn, 1.0); 
//		//errorExact[cycle] = GetErrorNorm2(temp_VPotential, VPotentialExact, RRow * PhiSlice,ZColumn, 1.0); 
//	}


	// copy result from device to host
	cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );

	memcpy(VPotential, temp_VPotential, RRow * ZColumn * PhiSlice * sizeof(float));

	// free device memory
	cudaFree( d_VPotential );
	cudaFree( d_DeltaResidue );
	cudaFree( d_RhoChargeDensity );
	cudaFree( d_VPotentialPrev );
	cudaFree( d_EpsilonError );
	cudaFree( d_coef1 );
	cudaFree( d_coef2 );
	cudaFree( d_coef3 );
	cudaFree( d_coef4 );
	cudaFree( d_icoef4 );

	// free host memory
	free( coef1 );
	free( coef2 );
	free( coef3 );
	free( coef4 );
	free( icoef4 );
	free( temp_VPotential );
	free( VPotentialPrev );
}



extern "C" void PoissonMultigrid3DSemiCoarseningGPUErrorWCycle
(
	float *VPotential, 
	float *RhoChargeDensity,
	const int RRow, 
	const int ZColumn,  
	const int PhiSlice,   
	const int Symmetry,
	float *fparam,
	int *iparam,
	float *errorConv,
	float *errorExact,
	float *VPotentialExact //allocation in the client
)
{
	// variables for CPU memory
	float *temp_VPotential;
	float *VPotentialPrev;
	float *EpsilonError;		

	// variables for GPU memory	
	float *d_VPotential;
	float *d_RhoChargeDensity;
	float *d_DeltaResidue;
	float *d_coef1;
	float *d_coef2;
	float *d_coef3;
	float *d_coef4;
	float *d_icoef4;
	float *d_VPotentialPrev;
	float *d_EpsilonError;
	

	// variables for coefficent calculations
	float *coef1;
	float *coef2;
	float *coef3;
	float *coef4;
	float *icoef4;
	float tempRatioZ;
	float tempRatioPhi;
	float radius;

	int gridFrom;
	int gridTo; 
	int loops;

	// variables passed from ALIROOT
	float gridSizeR		= fparam[0];
	//float gridSizePhi	= fparam[1];
	//float gridSizeZ		= fparam[2];
	float ratioPhi		= fparam[3];
	float ratioZ		= fparam[4];
	float convErr		= fparam[5];
	float IFCRadius		= fparam[6];
	int nPre	= iparam[0];
	int nPost	= iparam[1];
	int maxLoop	= iparam[2];
	int nCycle	= iparam[3];

	// variables for calculating GPU memory allocation
	int grid_RRow;
	int grid_ZColumn;
	int grid_PhiSlice = PhiSlice;
	int grid_Size = 0;
	int grid_StartPos;
	int coef_Size = 0;
	int coef_StartPos;
	int iOne, jOne;
	float h, h2, ih2;

	// variables for calculating multigrid maximum depth
	int depth_RRow = 0;
	int depth_ZColumn = 0;
	int temp_RRow = RRow;
	int temp_ZColumn = ZColumn;

	// calculate depth for multigrid
	while (temp_RRow >>= 1) depth_RRow++;  
	while (temp_ZColumn >>= 1) depth_ZColumn++;
  
	loops = (depth_RRow > depth_ZColumn) ? depth_ZColumn : depth_RRow;
	loops = (loops > maxLoop) ? maxLoop : loops;

	gridFrom = 1;
	gridTo = loops;

	// calculate GPU memory allocation for multigrid
	for (int step = gridFrom; step <= gridTo; step++)
	{
		grid_RRow = ((RRow - 1) / (1 << (step - 1))) + 1;
		grid_ZColumn = ((ZColumn - 1) / (1 << (step - 1))) + 1;
		
		grid_Size += grid_RRow * grid_ZColumn * grid_PhiSlice;
		coef_Size += grid_RRow;
	}

	// allocate memory for temporary output
	temp_VPotential 		= (float *) malloc(grid_Size * sizeof(float));
	VPotentialPrev = (float *) malloc(RRow * ZColumn * PhiSlice * sizeof(float));
	EpsilonError = (float *) malloc(1 * sizeof(float));


	// allocate memory for relaxation coefficient
	coef1 = (float *) malloc(coef_Size * sizeof(float));
	coef2 = (float *) malloc(coef_Size * sizeof(float));
	coef3 = (float *) malloc(coef_Size * sizeof(float));
	coef4 = (float *) malloc(coef_Size * sizeof(float));
	icoef4 = (float *) malloc(coef_Size * sizeof(float));

	// pre-compute relaxation coefficient
	coef_StartPos = 0;
	iOne = 1 << (gridFrom - 1); 
	jOne = 1 << (gridFrom - 1);
	
	for (int step = gridFrom; step <= gridTo; step++)
	{
		grid_RRow = ((RRow - 1) / iOne) + 1;

		h = gridSizeR * iOne;
		h2 = h * h;
		ih2 = 1.0 / h2;

		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		for (int i = 1; i < grid_RRow - 1; i++)
		{
			radius = IFCRadius + i * h;
			coef1[coef_StartPos + i] = 1.0 + h / (2 * radius);
			coef2[coef_StartPos + i] = 1.0 - h / (2 * radius);
			coef3[coef_StartPos + i] = tempRatioPhi / (radius * radius);
			coef4[coef_StartPos + i] = 0.5 / (1.0 + tempRatioZ + coef3[coef_StartPos + i]);
			icoef4[coef_StartPos + i] = 1.0 / coef4[coef_StartPos + i];
		}
		coef_StartPos += grid_RRow;
		iOne = 2 * iOne;
		jOne = 2 * jOne;
	}

	// device memory allocation
	cudaMalloc( &d_VPotential, grid_Size * sizeof(float) );
	cudaMalloc( &d_DeltaResidue, grid_Size * sizeof(float) );
	cudaMalloc( &d_VPotentialPrev, RRow * ZColumn * PhiSlice * sizeof(float) );
	cudaMalloc( &d_EpsilonError, 1 * sizeof(float) );	
		
	cudaMalloc( &d_RhoChargeDensity, grid_Size * sizeof(float) );
	cudaMalloc( &d_coef1, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef2, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef3, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef4, coef_Size * sizeof(float) );
	cudaMalloc( &d_icoef4, coef_Size * sizeof(float) );

	// set memory to zero
	cudaMemset( d_VPotential, 0, grid_Size * sizeof(float) );
	cudaMemset( d_DeltaResidue, 0, grid_Size * sizeof(float) );
	cudaMemset( d_RhoChargeDensity, 0, grid_Size * sizeof(float) );
	cudaMemset( d_VPotentialPrev, 0, RRow * ZColumn * PhiSlice * sizeof(float) );
	cudaMemset( d_EpsilonError, 0, 1 * sizeof(float) );


	// copy data from host to device
	cudaMemcpy( d_VPotential, VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice ); //check
	cudaMemcpy( d_RhoChargeDensity, RhoChargeDensity, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice ); //check
	cudaMemcpy( d_coef1, coef1, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef2, coef2, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef3, coef3, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef4, coef4, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_icoef4, icoef4, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_VPotentialPrev, VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice );
	
	// max exact	float maxAbsExact = GetAbsMax(VPotentialExact,RRow * PhiSlice * ZColumn);
	float maxAbsExact = GetAbsMax(VPotentialExact, RRow * PhiSlice * ZColumn);
	dim3 error_BlockPerGrid((RRow < 16) ? 1 : (RRow / 16), (ZColumn < 16) ? 1 : (ZColumn / 16), PhiSlice);
	dim3 error_ThreadPerBlock(16, 16);		


	for (int cycle = 0; cycle < nCycle; cycle++)
	{
	/*V-Cycle starts*/

		// error conv		
		//	cudaMemcpy( temp_VPotentialPrev, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
		
		cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
		errorExact[cycle] = GetErrorNorm2(temp_VPotential,VPotentialExact,RRow * PhiSlice,ZColumn,maxAbsExact); 


		// V-Cycle => Finest Grid
		iOne = 1 << (gridFrom - 1); 
		jOne = 1 << (gridFrom - 1);

		grid_RRow		= ((RRow - 1) / iOne) + 1;
		grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

		grid_StartPos = 0;
		coef_StartPos = 0;

		// pre-compute constant memory
		h 	= gridSizeR * iOne;
		h2	= h * h;
		ih2	= 1.0 / h2;

		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		// copy constant to device memory
		cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

		// set kernel grid size and block size
		dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
		dim3 grid_ThreadPerBlock(16, 16);

		// red-black gauss seidel relaxation (nPre times)
		for (int i = 0; i < nPre; i++)
		{
			relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
			//cudaDeviceSynchronize();
			relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
			//cudaDeviceSynchronize();
		}

		// residue calculation
		residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );
		//cudaDeviceSynchronize();

		// V-Cycle => from finer to coarsest grid
		for (int step = gridFrom + 1; step <= gridTo; step++)
		{
			iOne = 1 << (step - 1); 
			jOne = 1 << (step - 1);

			grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos += grid_RRow;

			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			// pre-compute constant memory
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;

			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;

			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);

			// restriction
			restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// zeroing V
			zeroingVPotential<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPre times)
			for (int i = 0; i < nPre; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
			}

			// residue calculation
			if (step < gridTo)
			{
				residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );
				//cudaDeviceSynchronize();

			}
		}
		/////////// innner w cycle
		/// up one down one

		// up one


		{
			int step = (gridTo - 1);
			iOne = iOne / 2;
			jOne = jOne / 2;
		
			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			grid_StartPos -= grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos -= grid_RRow;
		
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;
		
			tempRatioPhi = ratioPhi * iOne * iOne;
	
			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);
	
		// prolongation
			prolongation2DHalf<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
//			cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPost times)
			for (int i = 0; i < nPost; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
			}
		}

		// down one
		{
			residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );
				
			iOne = iOne * 2; 
			jOne = jOne * 2;

			grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos += grid_RRow;

			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			// pre-compute constant memory
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;

			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;


			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);

			// restriction
			restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// zeroing V
			zeroingVPotential<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPre times)
			for (int i = 0; i < nPre; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
			}
			
		}
		/// end up one down on
		
		/// up two down two
		// up two from gridTo - 1, to gridTo -3
		for (int step = (gridTo - 1); step >= gridTo - 3; step--)
		{
			iOne = iOne / 2;
			jOne = jOne / 2;
		
			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			grid_StartPos -= grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos -= grid_RRow;
		
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;
		
			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;

			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);
	
			// prolongation
			prolongation2DHalf<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
//			cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPost times)
			for (int i = 0; i < nPost; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
			}
		}
		
		// down to from gridTo - 1, to gridTo -3
		for (int step = gridTo - 3; step <= gridTo - 1; step++)
		{
			iOne = iOne * 2; 
			jOne = jOne * 2;

			grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos += grid_RRow;

			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			// pre-compute constant memory
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;

			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;

			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);

			// restriction
			restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// zeroing V
			zeroingVPotential<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPre times)
			for (int i = 0; i < nPre; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
			}

			// residue calculation
			if (step < gridTo)
			{
				residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );
				//cudaDeviceSynchronize();

			}
		}

		

		/// up one down one
		{
			int step = (gridTo - 1);
			iOne = iOne / 2;
			jOne = jOne / 2;
		
			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			grid_StartPos -= grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos -= grid_RRow;
		
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;
		
			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;
	
			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);
	
		// prolongation
			prolongation2DHalf<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
//			cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPost times)
			for (int i = 0; i < nPost; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
			}
		}

		// down one
		{
			residueCalculation<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_icoef4 );
				
			iOne = iOne * 2; 
			jOne = jOne * 2;

			grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos += grid_RRow;

			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			// pre-compute constant memory
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;

			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;


			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);

			// restriction
			restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_RhoChargeDensity, d_DeltaResidue, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// zeroing V
			zeroingVPotential<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
			//cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPre times)
			for (int i = 0; i < nPre; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
				//cudaDeviceSynchronize();
			}
			
		}
		/// end up one down one

		/////////// end inner w cyle

		// V-Cycle => from coarser to finer grid
		for (int step = (gridTo - 1); step >= gridFrom; step--)
		{
			iOne = iOne / 2;
			jOne = jOne / 2;
		
			grid_RRow		= ((RRow - 1) / iOne) + 1;
			grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

			grid_StartPos -= grid_RRow * grid_ZColumn * PhiSlice;
			coef_StartPos -= grid_RRow;
		
			h	= gridSizeR * iOne;
			h2	= h * h;
			ih2	= 1.0 / h2;
		
			tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
			tempRatioPhi = ratioPhi * iOne * iOne;

			// copy constant to device memory
			cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

			// set kernel grid size and block size
			dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
			dim3 grid_ThreadPerBlock(16, 16);
	
			// prolongation
			prolongation2DHalf<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );
//			cudaDeviceSynchronize();

			// red-black gauss seidel relaxation (nPost times)
			for (int i = 0; i < nPost; i++)
			{
				relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
				relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
//				cudaDeviceSynchronize();
			}
		}

	/*V-Cycle ends*/

		errorCalculation<<< error_BlockPerGrid, error_ThreadPerBlock >>> ( d_VPotentialPrev, d_VPotential, d_EpsilonError, RRow, ZColumn, PhiSlice);

		cudaMemcpy( EpsilonError, d_EpsilonError, 1 * sizeof(float), cudaMemcpyDeviceToHost );		
		

		errorConv[cycle] = *EpsilonError  / (RRow * ZColumn * PhiSlice);

		if (errorConv[cycle] < convErr)
		{
			//errorConv
			nCycle = cycle;
			iparam[3] = nCycle;
			break;
		}

		cudaMemcpy( d_VPotentialPrev, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToDevice );
		cudaMemset( d_EpsilonError, 0, 1 * sizeof(float) );
		
		
		
	}

	cudaDeviceSynchronize();
	// copy result from device to host
	cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );

	memcpy(VPotential, temp_VPotential, RRow * ZColumn * PhiSlice * sizeof(float));

	// free device memory
	cudaFree( d_VPotential );
	cudaFree( d_VPotentialPrev );
	cudaFree( d_EpsilonError );


	cudaFree( d_DeltaResidue );
	cudaFree( d_RhoChargeDensity );
	cudaFree( d_coef1 );
	cudaFree( d_coef2 );
	cudaFree( d_coef3 );
	cudaFree( d_coef4 );
	cudaFree( d_icoef4 );

	// free host memory
	free( coef1 );
	free( coef2 );
	free( coef3 );
	free( coef4 );
	free( icoef4 );
	free( temp_VPotential );
	//free( temp_VPotentialPrev );
}


/*extern function */
extern "C" void PoissonMultigrid3DSemiCoarseningGPUErrorFCycle
(
	float *VPotential, 
	float *RhoChargeDensity,
	const int RRow, 
	const int ZColumn,  
	const int PhiSlice,   
	const int Symmetry,
	float *fparam,
	int *iparam,
	bool isExactPresent,	
	float *errorConv,
	float *errorExact,
	float *VPotentialExact //allocation in the client
)
{
	// variables for CPU memory
	float *temp_VPotential;
	float *VPotentialPrev;
	float *EpsilonError;		

	// variables for GPU memory	
	float *d_VPotential;
	float *d_RhoChargeDensity;
	float *d_DeltaResidue;
	float *d_coef1;
	float *d_coef2;
	float *d_coef3;
	float *d_coef4;
	float *d_icoef4;
	float *d_VPotentialPrev;
	float *d_EpsilonError;
	

	// variables for coefficent calculations
	float *coef1;
	float *coef2;
	float *coef3;
	float *coef4;
	float *icoef4;
	float tempRatioZ;
	float tempRatioPhi;
	float radius;

	int gridFrom;
	int gridTo; 
	int loops;

	// variables passed from ALIROOT
	float gridSizeR		= fparam[0];
	//float gridSizePhi	= fparam[1];
	//float gridSizeZ		= fparam[2];
	float ratioPhi		= fparam[3];
	float ratioZ		= fparam[4];
	float convErr		= fparam[5];
	float IFCRadius		= fparam[6];
	int nPre	= iparam[0];
	int nPost	= iparam[1];
	int maxLoop	= iparam[2];
	int nCycle	= iparam[3];

	// variables for calculating GPU memory allocation
	int grid_RRow;
	int grid_ZColumn;
	int grid_PhiSlice = PhiSlice;
	int grid_Size = 0;
	int grid_StartPos;
	int coef_Size = 0;
	int coef_StartPos;
	int iOne, jOne;
	float h, h2, ih2;

	// variables for calculating multigrid maximum depth
	int depth_RRow = 0;
	int depth_ZColumn = 0;
	int temp_RRow = RRow;
	int temp_ZColumn = ZColumn;

	// calculate depth for multigrid
	while (temp_RRow >>= 1) depth_RRow++;  
	while (temp_ZColumn >>= 1) depth_ZColumn++;
  
	loops = (depth_RRow > depth_ZColumn) ? depth_ZColumn : depth_RRow;
	loops = (loops > maxLoop) ? maxLoop : loops;

	gridFrom = 1;
	gridTo = loops;

	// calculate GPU memory allocation for multigrid
	for (int step = gridFrom; step <= gridTo; step++)
	{
		grid_RRow = ((RRow - 1) / (1 << (step - 1))) + 1;
		grid_ZColumn = ((ZColumn - 1) / (1 << (step - 1))) + 1;
		
		grid_Size += grid_RRow * grid_ZColumn * grid_PhiSlice;
		coef_Size += grid_RRow;
	}

	// allocate memory for temporary output
	temp_VPotential 		= (float *) malloc(grid_Size * sizeof(float));
	VPotentialPrev = (float *) malloc(grid_Size * sizeof(float));
	EpsilonError = (float *) malloc(1 * sizeof(float));

	

	for (int i=0;i<grid_Size;i++) temp_VPotential[i] = 0.0;


	// allocate memory for relaxation coefficient
	coef1 = (float *) malloc(coef_Size * sizeof(float));
	coef2 = (float *) malloc(coef_Size * sizeof(float));
	coef3 = (float *) malloc(coef_Size * sizeof(float));
	coef4 = (float *) malloc(coef_Size * sizeof(float));
	icoef4 = (float *) malloc(coef_Size * sizeof(float));

	// pre-compute relaxation coefficient
	// restrict boundary
	coef_StartPos = 0;
	grid_StartPos = 0;

	iOne = 1 << (gridFrom - 1); 
	jOne = 1 << (gridFrom - 1);
	
	for (int step = gridFrom; step <= gridTo; step++)
	{
		grid_RRow = ((RRow - 1) / iOne) + 1;
		grid_ZColumn = ((ZColumn - 1) / iOne) + 1;

		h = gridSizeR * iOne;
		h2 = h * h;
		ih2 = 1.0 / h2;

		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		for (int i = 1; i < grid_RRow - 1; i++)
		{
			radius = IFCRadius + i * h;
			coef1[coef_StartPos + i] = 1.0 + h / (2 * radius);
			coef2[coef_StartPos + i] = 1.0 - h / (2 * radius);
			coef3[coef_StartPos + i] = tempRatioPhi / (radius * radius);
			coef4[coef_StartPos + i] = 0.5 / (1.0 + tempRatioZ + coef3[coef_StartPos + i]);
			icoef4[coef_StartPos + i] = 1.0 / coef4[coef_StartPos + i];
		}

		// call restrict boundary
		if (step == gridFrom) {
			// Copy original VPotential to tempPotential
			memcpy(temp_VPotential,     VPotential, RRow * ZColumn * PhiSlice * sizeof(float));
					
		} 
		// else 
		//{
		//	Restrict_Boundary(temp_VPotential, grid_RRow, grid_ZColumn, PhiSlice, grid_StartPos);
		//}

		
		coef_StartPos += grid_RRow;
		grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;


		iOne = 2 * iOne;
		jOne = 2 * jOne;
	}

	// device memory allocation
	cudaMalloc( &d_VPotential, grid_Size * sizeof(float) );
	cudaMalloc( &d_DeltaResidue, grid_Size * sizeof(float) );
	cudaMalloc( &d_RhoChargeDensity, grid_Size * sizeof(float) );
	cudaMalloc( &d_coef1, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef2, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef3, coef_Size * sizeof(float) );
	cudaMalloc( &d_coef4, coef_Size * sizeof(float) );
	cudaMalloc( &d_icoef4, coef_Size * sizeof(float) );
	cudaMalloc( &d_VPotentialPrev, grid_Size * sizeof(float) );
	cudaMalloc( &d_EpsilonError, 1 * sizeof(float) );	
		

	// set memory to zero
	cudaMemset( d_VPotential, 0, grid_Size * sizeof(float) );
	cudaMemset( d_DeltaResidue, 0, grid_Size * sizeof(float) );
	cudaMemset( d_RhoChargeDensity, 0, grid_Size * sizeof(float) );
	cudaMemset( d_VPotentialPrev, 0, grid_Size * sizeof(float) );
	cudaMemset( d_EpsilonError, 0, 1 * sizeof(float) );

	// set memory to zero
	cudaMemset( d_VPotential, 0, grid_Size * sizeof(float) );
	cudaMemset( d_DeltaResidue, 0, grid_Size * sizeof(float) );
	cudaMemset( d_RhoChargeDensity, 0, grid_Size * sizeof(float) );

	// copy data from host to devicei
	// case of FCycle you need to copy all boundary for all
	cudaMemcpy( d_VPotential, temp_VPotential, grid_Size * sizeof(float), cudaMemcpyHostToDevice ); //check
//	cudaMemcpy( d_VPotential, VPotential, grid_Size * isizeof(float), cudaMemcpyHostToDevice ); //check

	cudaMemcpy( d_RhoChargeDensity, RhoChargeDensity, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice ); //check
//	cudaMemcpy( d_RhoChargeDensity, temp_VPotentialPrev, grid_Size * sizeof(float), cudaMemcpyHostToDevice ); //check
	cudaMemcpy( d_coef1, coef1, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef2, coef2, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef3, coef3, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_coef4, coef4, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_icoef4, icoef4, coef_Size * sizeof(float), cudaMemcpyHostToDevice );
//	cudaMemcpy( d_VPotentialPrev, temp_VPotential, grid_Size * sizeof(float), cudaMemcpyHostToDevice );

//	cudaMemcpy( d_VPotentialPrev, VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyHostToDevice );
	
	// max exact
	
	float maxAbsExact = 1.0;

	if (isExactPresent == true)
		maxAbsExact = GetAbsMax(VPotentialExact, RRow * PhiSlice * ZColumn);
	
	

	// init iOne,grid_RRow, grid_ZColumn, grid_StartPos, coef_StartPos
	iOne = 1 << (gridFrom - 1); 
	jOne = 1 << (gridFrom - 1);

	grid_RRow		= ((RRow - 1) / iOne) + 1;
	grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

	grid_StartPos = 0;
	coef_StartPos = 0;


	//// Restrict Boundary and Rho	
	for (int step = gridFrom + 1; step <= gridTo; step++)
	{

		iOne = 1 << (step - 1); 
		jOne = 1 << (step - 1);

		grid_StartPos += grid_RRow * grid_ZColumn * PhiSlice;
		coef_StartPos += grid_RRow;

		grid_RRow		= ((RRow - 1) / iOne) + 1;
		grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

		// pre-compute constant memory
		h	= gridSizeR * iOne;
		h2	= h * h;
		ih2	= 1.0 / h2;

		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		// copy constant to device memory
		cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

		// set kernel grid size and block size
		dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
		dim3 grid_ThreadPerBlock(16, 16);

		// restriction
		restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_RhoChargeDensity, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice );
		
		// restrict boundary (already done in cpu)
///		cudaMemcpy( temp_VPotential, d_RhoChargeDensity + grid_StartPos , grid_RRow * grid_ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
//		PrintMatrix(temp_VPotential,grid_RRow * PhiSlice,grid_ZColumn);
		restriction2DFull<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );

		
	}

	dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
	dim3 grid_ThreadPerBlock(16, 16);


	// relax on the coarsest 
	// red-black gauss seidel relaxation (nPre times)
//	printf("rho\n");
//	cudaMemcpy( temp_VPotential, d_RhoChargeDensity + grid_StartPos , grid_RRow * grid_ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
//	PrintMatrix(temp_VPotential,grid_RRow,grid_ZColumn);
	
//	printf("v\n");
//	cudaMemcpy( temp_VPotential, d_VPotential + grid_StartPos , grid_RRow * grid_ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
//	PrintMatrix(temp_VPotential,grid_RRow,grid_ZColumn);
	for (int i = 0; i < nPre; i++)
	{
		relaxationGaussSeidelRed<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
		cudaDeviceSynchronize();
		relaxationGaussSeidelBlack<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, d_RhoChargeDensity, grid_RRow, grid_ZColumn, grid_PhiSlice, d_coef1, d_coef2, d_coef3, d_coef4 );
		cudaDeviceSynchronize();
	}

//	printf("v after relax\n");
//	cudaMemcpy( temp_VPotential, d_VPotential + grid_StartPos , grid_RRow * grid_ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
//	PrintMatrix(temp_VPotential,grid_RRow,grid_ZColumn);
	
	// V-Cycle => from coarser to finer grid
	for (int step = gridTo -1 ; step >= gridFrom; step--)
	{
		iOne = iOne / 2;
		jOne = jOne / 2;
	
		grid_RRow		= ((RRow - 1) / iOne) + 1;
		grid_ZColumn	= ((ZColumn - 1) / jOne) + 1;

		grid_StartPos -= grid_RRow * grid_ZColumn * PhiSlice;
		coef_StartPos -= grid_RRow;
	
		h	= gridSizeR * iOne;
		h2	= h * h;
		ih2	= 1.0 / h2;
	
		tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
		tempRatioPhi = ratioPhi * iOne * iOne;

		// copy constant to device memory
		cudaMemcpyToSymbol( d_grid_StartPos, &grid_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_coef_StartPos, &coef_StartPos, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_h2, &h2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_ih2, &ih2, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );
		cudaMemcpyToSymbol( d_tempRatioZ, &tempRatioZ, 1 * sizeof(float), 0, cudaMemcpyHostToDevice );

		

		// set kernel grid size and block size
		dim3 grid_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
		dim3 grid_ThreadPerBlock(16, 16);


		prolongation2DHalfNoAdd<<< grid_BlockPerGrid, grid_ThreadPerBlock >>>( d_VPotential, grid_RRow, grid_ZColumn, grid_PhiSlice );


		

		// just 
		
		// max exact
		cudaMemcpy( d_VPotentialPrev + grid_StartPos, d_VPotential + grid_StartPos, grid_RRow * grid_ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToDevice );
				
		float maxAbsExact = 1.0;

		if (isExactPresent == true)
			maxAbsExact = GetAbsMax(VPotentialExact, RRow * PhiSlice * ZColumn);
		dim3 error_BlockPerGrid((grid_RRow < 16) ? 1 : (grid_RRow / 16), (grid_ZColumn < 16) ? 1 : (grid_ZColumn / 16), PhiSlice);
		dim3 error_ThreadPerBlock(16, 16);		

		

		for (int cycle = 0; cycle < nCycle; cycle++)
		{

				
			if (step == gridFrom) {
				cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
				if (isExactPresent == true )errorExact[cycle] = GetErrorNorm2(temp_VPotential, VPotentialExact, RRow * PhiSlice,ZColumn, maxAbsExact); 
			}



			//cudaDeviceSynchronize();
			VCycleSemiCoarseningGPU(d_VPotential, d_RhoChargeDensity, d_DeltaResidue, d_coef1, d_coef2, d_coef3, d_coef4, d_icoef4, gridSizeR, ratioZ, ratioPhi, RRow, ZColumn, PhiSlice, step, gridTo, nPre, nPost);
			


				//if (step == gridFrom) {
				//cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );
	
				//errorConv[cycle] = GetErrorNorm2(temp_VPotential, VPotentialPrev, RRow * PhiSlice,ZColumn, 1.0); 

				errorCalculation<<< error_BlockPerGrid, error_ThreadPerBlock >>> ( d_VPotentialPrev + grid_StartPos, d_VPotential + grid_StartPos, d_EpsilonError, grid_RRow, grid_ZColumn, PhiSlice);

				cudaMemcpy( EpsilonError, d_EpsilonError, 1 * sizeof(float), cudaMemcpyDeviceToHost );		
				
				errorConv[cycle] = *EpsilonError  / (grid_RRow * grid_ZColumn * PhiSlice);

				if (errorConv[cycle]< convErr)
				{
					nCycle = cycle;			
					break;
				}

				cudaMemcpy( d_VPotentialPrev + grid_StartPos, d_VPotential + grid_StartPos, grid_RRow * grid_ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToDevice );
				cudaMemset( d_EpsilonError, 0, 1 * sizeof(float) );
				
		}
		
		
	}

	iparam[3] = nCycle;	

	// copy result from device to host
	cudaMemcpy( temp_VPotential, d_VPotential, RRow * ZColumn * PhiSlice * sizeof(float), cudaMemcpyDeviceToHost );

	memcpy(VPotential, temp_VPotential, RRow * ZColumn * PhiSlice * sizeof(float));

	// free device memory
	cudaFree( d_VPotential );
	cudaFree( d_DeltaResidue );
	cudaFree( d_RhoChargeDensity );
	cudaFree( d_coef1 );
	cudaFree( d_coef2 );
	cudaFree( d_coef3 );
	cudaFree( d_coef4 );
	cudaFree( d_icoef4 );

	// free host memory
	free( coef1 );
	free( coef2 );
	free( coef3 );
	free( coef4 );
	free( icoef4 );
	free( temp_VPotential );
	free( VPotentialPrev );
}

