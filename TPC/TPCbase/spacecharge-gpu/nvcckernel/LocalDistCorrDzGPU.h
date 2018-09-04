#ifndef LOCAL_DISTCORR_DZ_GPU_H
#define LOCAL_DISTCORR_DZ_GPU_H

#include <ctime>
#include <iomanip>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

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
);


#endif
