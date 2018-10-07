#include "IntegrateEzGPU.h"

#include <cuda.h>
#include <math.h>

__device__ __constant__ float d_gridSizeZ;
__device__ __constant__ float d_ezField;
__device__ __constant__ int d_scanSize;
__device__ __constant__ int d_nRRow;
__device__ __constant__ int d_nZColumn;
__device__ __constant__ int d_phiSlice;
__device__ __constant__ int d_interpolationOrder;
__device__ int d_currentZIndex;


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

/// Solving cubic splines for system of splines
///
/// \param xList Double_t[] known position x
/// \param yList Double_t[] known y = f(x)
/// \param n Int_t length of splines
/// \param y2List Double_t[] calculated $d^2Y$ spline (output)
/// \param skip memory offset for xList
///
__device__ void initCubicSplineGPU(float *xList, float *yList, const int n, float *y2List, const int skip) {
  float u[6];
  float sig, p, qn, un;

  y2List[0] = 0.0;
  u[0] = 0.0; //natural condition


  for (int i = 1; i <= n - 2; i++) {
    sig = (xList[i] - xList[i - 1]) / (xList[i + 1] - xList[i - 1]);
    p = sig * y2List[(i - 1) * skip] + 2.0;
    y2List[i * skip] = (sig - 1.0) / p;
    u[i] = (yList[(i + 1) * skip] - yList[i * skip]) / (xList[i + 1] - xList[i]) -
           (yList[i * skip] - yList[(i - 1) * skip]) / (xList[i] - xList[i - 1]);
    u[i] = (6.0 * u[i] / (xList[i + 1] - xList[i - 1]) - sig * u[i - 1]) / p;
  }

  qn = un = 0.0;

  y2List[(n - 1) * skip] = (un - qn * u[n - 2]) / (qn * y2List[(n - 2) * skip] + 1.0);
  for (int k = n - 2; k >= 0; k--)
    y2List[k * skip] = y2List[k * skip] * y2List[(k + 1) * skip] + u[k];
}

/// Interpolate initialized cubic spline
///
/// \param xList
/// \param yList
/// \param y2List
/// \param nxList
/// \param nyList
/// \param ny2List
/// \param x
/// \param skip
/// \return
__device__  void interpolateCubicSplineGPU(float *xList, float *yList,  float *y2List, const int nxList, const int nyList, const int ny2List, float x, int skip, float *y) {
  int klo, khi, k;
  float h, b, a;
  klo = 0;
  khi = nxList - 1;

  while (khi - klo > 1) {
    k = (khi + klo) >> 1;
    if (xList[k] > x) khi = k;
    else klo = k;
  }

  h = xList[khi] - xList[klo];
  
  if (fabsf(h) < 1e-10) {
    *y =  0.0;
    return;
  }

  a = (xList[khi] - x) / h;
  b = (x - xList[klo]) / h;

  *y = a * yList[klo] + b * yList[khi] +
               ((a * a * a - a) * y2List[klo * skip] + (b * b * b - b) * y2List[khi * skip]) * (h * h) / 6.0;

  

}




// interpolation for phi direction
__device__ void interpolatePhiGPU
(
	float * xList,
	int iLow,
	int lenX,
	float * yList,
	float x,
	float *y
)
{
	
  	int i0 = iLow;
  	float xi0 = xList[iLow];
  	int  i1 = (iLow + 1) % lenX;
  	float xi1 = xList[i1];
  	int  i2 = (iLow + 2) % lenX;
  	float  xi2 = xList[i2];
    	float y2List[6];
   	float xListTemp[6];
	float dPhi;

  	if (xi1 < xi0) xi1 = 2*M_PI + xi1;
  	if (xi2 < xi1) xi2 = 2*M_PI + xi2;
  	if (x < xi0) x = 2*M_PI + x;

	if (d_interpolationOrder == 1)
    		*y = yList[0] + (yList[1] - yList[0]) * (x - xi0) / (xi1 - xi0);
	else if (d_interpolationOrder == 2)
	{
    		*y = (x - xi1) * (x - xi2) * yList[0] / ((xi0 - xi1) * (xi0 - xi2));
    		*y += (x - xi2) * (x - xi0) * yList[1] / ((xi1 - xi2) * (xi1 - xi0));
    		*y += (x - xi0) * (x - xi1) * yList[2] / ((xi2 - xi0) * (xi2 - xi1));
	} else {
		
    		dPhi = xList[1] - xList[0];
		for (int i = 0; i < d_interpolationOrder + 1; i++) {
		      xListTemp[i] = xList[iLow] + (dPhi * i);
	    	}
	 	if ((xListTemp[0] - x) > 1e-10) x = x + 2 * M_PI;
	        initCubicSplineGPU(xListTemp, yList, d_interpolationOrder + 1, y2List, 1);
    		interpolateCubicSplineGPU(xListTemp, yList, y2List,d_interpolationOrder + 1, d_interpolationOrder + 1, d_interpolationOrder + 1, x, 1,y);
	}
}

// interpolate in 1D
__device__ void interpolateGPU
(
	float * xList,
	float * yList,
	float x,
	float * y
)
{
	float y2List[6];
	int   nPoints;
	nPoints = d_interpolationOrder + 1;
	if(d_interpolationOrder == 1) // linear
    		*y = yList[0] + (yList[1] - yList[0]) * (x - xList[0]) / (xList[1] - xList[0]);
	else if (d_interpolationOrder == 2)  {
    		*y = (x - xList[1]) * (x - xList[2]) * yList[0] / ((xList[0] - xList[1]) * (xList[0] - xList[2]));
    		*y += (x - xList[2]) * (x - xList[0]) * yList[1] / ((xList[1] - xList[2]) * (xList[1] - xList[0]));
    		*y += (x - xList[0]) * (x - xList[1]) * yList[2] / ((xList[2] - xList[0]) * (xList[2] - xList[1]));
	} else {

    		initCubicSplineGPU(xList, yList, nPoints, y2List, 1);
		interpolateCubicSplineGPU(xList, yList, y2List, nPoints, nPoints, nPoints, x, 1,y);
	}

}

__device__ void interpolate3DCylindricalGPU
(	
	int iLow,
	int jLow,
	int kLow,
	float r,
	float z,
	float phi,
	float *rList,
	float *zList,
	float *phiList,
	float *valList,
	float *secondDerZ,
	float *interpolationValue
)
{

  	// do for each
	int m,index;
	float saveArray[6];
	float savedArray[6];

  	for (int k = 0; k < d_interpolationOrder + 1; k++) {
    		m = (kLow + k) % d_phiSlice;
    		// interpolate
    		for (int i = iLow; i < iLow + d_interpolationOrder + 1; i++) {
      			if (d_interpolationOrder <= 2) {
          			index = m * (d_nZColumn  * d_nRRow) + i * (d_nZColumn) + jLow;
          			interpolateGPU(&zList[jLow], &valList[index], z,&saveArray[i - iLow]);

      			} else {
				// cubic spline
        			index = m * (d_nZColumn * d_nRRow) + i * (d_nZColumn);
        			interpolateCubicSplineGPU(zList, &valList[index], &secondDerZ[index], d_nZColumn, d_nZColumn, d_nZColumn,z, 1,&saveArray[i-iLow]);
			
			}
    		}
    		interpolateGPU(&rList[iLow], saveArray, r,&savedArray[k]);
  	}
  	interpolatePhiGPU(phiList, kLow, d_phiSlice, savedArray, phi,interpolationValue);
}
		
			



__global__ void integrateDistEzDriftLineGPUKernel
(
	float *distDrDz,
	float *distDPhiRDz,
	float *distDz, 
	float *corrDrDz,
	float *corrDPhiRDz,
	float *corrDz, 
	float *GDistDrDz,
	float *GDistDPhiRDz,
	float *GDistDz, 
	float *GCorrDrDz,
	float *GCorrDPhiRDz,
	float *GCorrDz, 
	float *rList,
	float *zList,
	float *phiList,
	float *secondDerZDistDr,
	float *secondDerZDistDPhiR,
	float *secondDerZDistDz,
	float *secondDerZCorrDr,
	float *secondDerZCorrDPhiR,
	float *secondDerZCorrDz,
	float *GPrevDistDrDz,
	int operationType,
	int currentZIndex
)
{
	int index, index_x, index_y, index_z;
	
	// float gDistDrDz, gDistDPhiRDz, gDistDz;
	float lDistDrDz, lDistDPhiRDz, lDistDz;
	float lCorrDrDz, lCorrDPhiRDz, lCorrDz;
	float currentPhi,currentRadius, currentZ;
	
	// for interpolation
	int iLow,jLow,kLow;
	int corrCurrentZIndex;
	

	corrCurrentZIndex = d_nZColumn - currentZIndex - 2;

	index = (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	index_x = index / (d_nZColumn * d_nRRow);
	
	if (index_x == 0)
	{
		index_y = index / d_nRRow;
	}
	else	
	{
		index_y = (index % (index_x * d_nRRow * d_nZColumn)) / d_nRRow;
	}

	index_z = index % d_nZColumn;
	


	index = index_x * d_nRRow * d_nZColumn + index_y * d_nZColumn + index_z;	

	if (operationType == 0) {
	   if ((index_x >= 0) && (index_x < d_phiSlice) && (index_y >= 0) && (index_y < d_nRRow ) && (index_z >= 0) && (index_z < d_nZColumn)) 			{
			
		GDistDrDz[index] = 0.0;
		GPrevDistDrDz[index] = 0.0;
		GDistDPhiRDz[index] = 0.0;
		GDistDz[index] = 0.0;
		GCorrDrDz[index] = 0.0;
		GCorrDPhiRDz[index] = 0.0;
		GCorrDz[index] = 0.0;
		}
	} else {
	   if ((index_x >= 0) && (index_x < d_phiSlice) && (index_y >= 0) && (index_y < d_nRRow ) && (index_z >= 0) && (index_z < d_nZColumn) && (index_z <=  currentZIndex)) {
		lDistDrDz = 0.0;
		lDistDPhiRDz = 0.0;
		lDistDz = 0.0;
	
	
		currentRadius = rList[index_y] + GPrevDistDrDz[index];
		currentPhi = phiList[index_x] + (GDistDPhiRDz[index]/currentRadius);
		currentRadius = rList[index_y] + GDistDrDz[index];
		while (currentPhi < 0.0) currentPhi = 2 * M_PI + currentPhi;
		while (currentPhi > 2*M_PI) currentPhi = currentPhi - (2 * M_PI);
		currentZ =  zList[currentZIndex] + GDistDz[index];

		// get Local Distortion through interpolation
		// get nearest index
		for (iLow=0;iLow<d_nRRow;iLow++)  if (rList[iLow] > currentRadius) break;
		for (jLow=0;jLow<d_nZColumn;jLow++)  if (zList[jLow] > currentZ) break;
		for (kLow=0;kLow<d_phiSlice;kLow++)  if (phiList[kLow] > currentPhi) break;

		iLow--;jLow--;kLow--;
  		// order >= 3
  		kLow -= (d_interpolationOrder / 2);
  		iLow -= (d_interpolationOrder / 2);
  		jLow -= (d_interpolationOrder / 2);

  		// check if out of range
  		if (iLow < 0) iLow = 0;
  		if (jLow < 0) jLow = 0;
  		if (kLow < 0) kLow = d_phiSlice + kLow;
  		// check if out of range
 		if (iLow + d_interpolationOrder >= d_nRRow  - 1) iLow = d_nRRow- 1 - d_interpolationOrder;
  		if (jLow + d_interpolationOrder >= d_nZColumn - 1) jLow = d_nZColumn - 1 - d_interpolationOrder;

		interpolate3DCylindricalGPU(iLow,jLow,kLow,currentRadius,currentZ,currentPhi,rList,zList,phiList,distDrDz,secondDerZDistDr,&lDistDrDz);	
		interpolate3DCylindricalGPU(iLow,jLow,kLow,currentRadius,currentZ,currentPhi,rList,zList,phiList,distDPhiRDz,secondDerZDistDPhiR,&lDistDPhiRDz);	
		interpolate3DCylindricalGPU(iLow,jLow,kLow,currentRadius,currentZ,currentPhi,rList,zList,phiList,distDz,secondDerZDistDz,&lDistDz);	
			
		// update global distortion
		GPrevDistDrDz[index ] = GDistDrDz[index];
		GDistDrDz[index] += lDistDrDz;
		GDistDPhiRDz[index] += lDistDPhiRDz;
		GDistDz[index] += lDistDz;
		
		// Global distortion
		

				
	    }
	   if ((index_x >= 0) && (index_x < d_phiSlice) && (index_y >= 0) && (index_y < d_nRRow ) && (index_z >= 0) && (index_z < d_nZColumn - 1) && (index_z ==  corrCurrentZIndex)) {
		
		lCorrDrDz = 0.0;
		lCorrDPhiRDz = 0.0;
		lCorrDz = 0.0;

		
		GCorrDrDz[index] = GCorrDrDz[index + 1];
		GCorrDPhiRDz[index] = GCorrDPhiRDz[index + 1];
		GCorrDz[index] = GCorrDz[index + 1];
	 
	
		currentRadius = rList[index_y] + GCorrDrDz[index];
		currentPhi = phiList[index_x] + (GCorrDPhiRDz[index]/currentRadius);
		while (currentPhi < 0.0) currentPhi = 2 * M_PI + currentPhi;
		while (currentPhi > 2*M_PI) currentPhi = currentPhi - (2 * M_PI);
		currentZ =  zList[corrCurrentZIndex + 1] + GDistDz[index];
		
		// get Local Distortion through interpolation
		// get nearest index
		for (iLow=0;iLow<d_nRRow;iLow++)  if (rList[iLow] > currentRadius) break;
		for (jLow=0;jLow<d_nZColumn;jLow++)  if (zList[jLow] > currentZ) break;
		for (kLow=0;kLow<d_phiSlice;kLow++)  if (phiList[kLow] > currentPhi) break;

		iLow--;jLow--;kLow--;
  		// order >= 3
  		kLow -= (d_interpolationOrder / 2);
  		iLow -= (d_interpolationOrder / 2);
  		jLow -= (d_interpolationOrder / 2);

  		// check if out of range
  		if (iLow < 0) iLow = 0;
  		if (jLow < 0) jLow = 0;
  		if (kLow < 0) kLow = d_phiSlice + kLow;
  		// check if out of range
 		if (iLow + d_interpolationOrder >= d_nRRow  - 1) iLow = d_nRRow- 1 - d_interpolationOrder;
  		if (jLow + d_interpolationOrder >= d_nZColumn - 1) jLow = d_nZColumn - 1 - d_interpolationOrder;

		interpolate3DCylindricalGPU(iLow,jLow,kLow,currentRadius,currentZ,currentPhi,rList,zList,phiList,corrDrDz,secondDerZCorrDr,&lCorrDrDz);	
		interpolate3DCylindricalGPU(iLow,jLow,kLow,currentRadius,currentZ,currentPhi,rList,zList,phiList,corrDPhiRDz,secondDerZCorrDPhiR,&lCorrDPhiRDz);	
		interpolate3DCylindricalGPU(iLow,jLow,kLow,currentRadius,currentZ,currentPhi,rList,zList,phiList,corrDz,secondDerZCorrDz,&lCorrDz);	
			
		// update global distortion
		GCorrDrDz[index] += lCorrDrDz;
		GCorrDPhiRDz[index] += lCorrDPhiRDz;
		GCorrDz[index] += lCorrDz;
   	}
    }
}



extern "C" void IntegrateEzDriftLineGPU(
	float * distDrDz, float * distDPhiRDz, float * distDz, float *corrDrDz, float * corrDPhiRDz, float * corrDz,  
	float * GDistDrDz, float * GDistDPhiRDz, float * GDistDz, float * GCorrDrDz, float * GCorrDPhiRDz, float * GCorrDz,  
	float * rList, float * zList, float * phiList,   
	const int rows, const int columns, const int phislices, const int interpolationOrder,
	float * secondDerZDistDr, float *secondDerZDistDPhiR, float *secondDerZDistDz,
	float * secondDerZCorrDr, float *secondDerZCorrDPhiR, float *secondDerZCorrDz) {



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
	float *d_GPrevDistDrDz;
	float *d_rList;
	float *d_zList;
	float *d_phiList;
	float *d_secondDerZDistDr;
	float *d_secondDerZDistDPhiR;
	float *d_secondDerZDistDz;

	float *d_secondDerZCorrDr;
	float *d_secondDerZCorrDPhiR;
	float *d_secondDerZCorrDz;

	int currentZIndex;

	cudaError error;

	cudaMalloc( &d_distDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_distDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_distDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_corrDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_corrDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_corrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GDistDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GPrevDistDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GDistDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GDistDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GCorrDrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GCorrDPhiRDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_GCorrDz, rows * columns * phislices * sizeof(float) );
	cudaMalloc( &d_rList, rows *  sizeof(float) );
	cudaMalloc( &d_zList, columns *  sizeof(float) );
	cudaMalloc( &d_phiList,  phislices * sizeof(float) );

	cudaMalloc( &d_secondDerZDistDr, rows *  columns * phislices *  sizeof(float) );
	cudaMalloc( &d_secondDerZDistDPhiR, rows *  columns * phislices *  sizeof(float) );
	cudaMalloc( &d_secondDerZDistDz, rows *  columns * phislices *  sizeof(float) );
	
	cudaMalloc( &d_secondDerZCorrDr, rows *  columns * phislices *  sizeof(float) );
	cudaMalloc( &d_secondDerZCorrDPhiR, rows *  columns * phislices *  sizeof(float) );
	cudaMalloc( &d_secondDerZCorrDz, rows *  columns * phislices *  sizeof(float) );



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

	cudaMemcpy( d_rList, rList, rows  * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_zList, zList, columns *  sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( d_phiList, phiList,  phislices * sizeof(float), cudaMemcpyHostToDevice );

	if (d_interpolationOrder > 2) {	
		cudaMemcpy( d_secondDerZDistDr, secondDerZDistDr, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( d_secondDerZDistDPhiR, secondDerZDistDPhiR, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( d_secondDerZDistDz, secondDerZDistDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );

		cudaMemcpy( d_secondDerZCorrDr, secondDerZCorrDr, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( d_secondDerZCorrDPhiR, secondDerZCorrDPhiR, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( d_secondDerZCorrDz, secondDerZCorrDz, rows * columns * phislices * sizeof(float), cudaMemcpyHostToDevice );
	
	}
	cudaMemcpyToSymbol( d_interpolationOrder, &interpolationOrder, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_nRRow, &rows, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_nZColumn, &columns, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( d_phiSlice, &phislices, 1 * sizeof(int), 0, cudaMemcpyHostToDevice );
	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA memory copy host to device error: " << cudaGetErrorString(error) << '\n';
	}

	// call kernel
	// set grid size and block size
	dim3 gridSize((rows / 32) + 1, (columns / 32) + 1, phislices);
	dim3 blockSize(32, 32);

	integrateDistEzDriftLineGPUKernel<<< gridSize,blockSize >>>(
					  d_distDrDz,d_distDPhiRDz,d_distDz,
					  d_corrDrDz,d_corrDPhiRDz,d_corrDz,
					  d_GDistDrDz,d_GDistDPhiRDz,d_GDistDz,
					  d_GCorrDrDz,d_GCorrDPhiRDz,d_GCorrDz, 
					  d_rList,d_zList, d_phiList,
					  d_secondDerZDistDr, d_secondDerZDistDPhiR, d_secondDerZDistDz, 
					  d_secondDerZCorrDr, d_secondDerZCorrDPhiR, d_secondDerZCorrDz, 
					  d_GPrevDistDrDz, 0,currentZIndex);
	for (currentZIndex = 0; currentZIndex < columns;currentZIndex++) {	
		integrateDistEzDriftLineGPUKernel<<< gridSize,blockSize >>>(d_distDrDz,d_distDPhiRDz,d_distDz,
					  d_corrDrDz,d_corrDPhiRDz,d_corrDz,
					  d_GDistDrDz,d_GDistDPhiRDz,d_GDistDz,
					  d_GCorrDrDz,d_GCorrDPhiRDz,d_GCorrDz,
					  d_rList,d_zList, d_phiList,
					  d_secondDerZDistDr, d_secondDerZDistDPhiR, d_secondDerZDistDz, 
					  d_secondDerZCorrDr, d_secondDerZCorrDPhiR, d_secondDerZCorrDz, 
					  d_GPrevDistDrDz, 1,currentZIndex);

	}
	
	cudaMemcpy( GDistDrDz, d_GDistDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GDistDPhiRDz, d_GDistDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GDistDz, d_GDistDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GCorrDrDz, d_GCorrDrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GCorrDPhiRDz, d_GCorrDPhiRDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( GCorrDz, d_GCorrDz, rows * columns * phislices * sizeof(float), cudaMemcpyDeviceToHost );
	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA Global dist  memory copy device to host error: " << cudaGetErrorString(error) << '\n';
	}


	cudaFree( d_distDrDz );
	cudaFree( d_distDPhiRDz );
	cudaFree( d_distDz );
	cudaFree( d_corrDrDz );
	cudaFree( d_corrDPhiRDz );
	cudaFree( d_corrDz );
	cudaFree( d_GDistDrDz );
	cudaFree( d_GPrevDistDrDz );
	cudaFree( d_GDistDPhiRDz );
	cudaFree( d_GDistDz );
	cudaFree( d_GCorrDrDz );
	cudaFree( d_GCorrDPhiRDz );
	cudaFree( d_GCorrDz );
	cudaFree( d_rList );
	cudaFree( d_zList );
	cudaFree( d_phiList );
	cudaFree( d_secondDerZDistDr);
	cudaFree( d_secondDerZDistDPhiR);
	cudaFree( d_secondDerZDistDz);
	cudaFree( d_secondDerZCorrDr);
	cudaFree( d_secondDerZCorrDPhiR);
	cudaFree( d_secondDerZCorrDz);

	error = cudaGetLastError();	
	if ( error != cudaSuccess )
	{
		std::cout << "CUDA free allocated memory error: " << cudaGetErrorString(error) << '\n';
	}
}






