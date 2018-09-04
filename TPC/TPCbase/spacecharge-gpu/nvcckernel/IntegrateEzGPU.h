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
extern "C" void IntegrateEzDriftLineGPU(float * LDistDrDz, float * LDistDphiDz, float * LDistDz, const int rows, const int columns, const int phislices, const int symmetry, const float fgkIFCRadius, const float fgkOFCRadius, const float fgkTPCZ0, float * GDistDrDz, float * GDistDphiDz, float * GDistDz, float * GCorrDrDz, float * GCorrDphiDz, float * GCorrDz, int interpolationType);


#endif
