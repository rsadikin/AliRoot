#ifndef INTEGRATEEZGPU_H
#define INTEGRATEEZGPU_H

#include <ctime>
#include <iomanip>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>



extern "C" void IntegrateEzGPU(float *, float *, int, int, int, float, float);

// LDistDrDz lcoal distortion r
// LDistDphiDz, LDistDz local distortion phi and z
// Dist: distortion, Corr: correction
// interpolationType
// 1 	: linear
// 2	: quadratic
// 3	: qubic (TODO)
extern "C" void IntegrateEzDriftLineGPU(float * distDrDz, float * distDPhiRDz, float * distDz, float *corrDrDz, float * corrDPhiRDz, float * corrDz,  
	float * GDistDrDz, float * GDistDphiDz, float * GDistDz, float * GCorrDrDz, float * GCorrDphiDz, float * GCorrDz,  
	float * rList, float * zList, float * phiList,   
	const int rows, const int columns, const int phislices, const int integrationType);


#endif
