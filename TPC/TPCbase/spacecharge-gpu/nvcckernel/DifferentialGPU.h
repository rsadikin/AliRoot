#ifndef DIFFERENTIAL_GPU_H
#define DIFFERENTIAL_GPU_H

#include <ctime>
#include <iomanip>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

extern "C" void DifferentialCalculationGPU (
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
);

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
);

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
);

#endif
