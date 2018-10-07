#ifndef POISSONSOLVER3DGPU_H
#define POISSONSOLVER3DGPU_H

#include <ctime>
#include <iomanip>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

/// \file PoissonSolver3DGPU.h
/// \brief Berkas ini berisi definisi fungsi extern yang dimiliki implementasi CUDA yang  didapat dipanggil CPU
///
/// \author Rifki Sadikin <rifki.sadikin@lipi.go.id>, Pusat Penelitian Informatika, Lembaga Ilmu Pengetahuan Indonesia
/// \author I Wayan Aditya Swardiana  <i.wayan.aditya.swardiana@lipi.go.id>, Pusat Penelitian Informatika, Lembaga Ilmu Pengetahuan Indonesia
/// \date Mar 4, 2015




/// Fungsi ini menghitung solusi terhadap Persamaan Poisson
/// 
///  \f[
///   \nabla^{2}(r,\phi,z) = \rho(r,\phi,z)
///  \f]
///
/// dengan diketahui nilai tepi (Boundary Value) pada \f$V\f$ (potensial) dan distribusi buatan \f$\rho\f$
///
/// \param[in,out] VPotential float[nrows*ncols*nphi] distribusi potensial. Input: hanya nilai tepi. Output: hasil perhitungan penyelesaian persamaan Poisson
/// \param[in]  RhoChangeDensity float[nrows*ncols*nphi] distributsi muatan listrik. 
///
/// \return A fixed number that has nothing to do with what the function does
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
);


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
);

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
);



#endif
