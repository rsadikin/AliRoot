/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/


/// \class AliTPCPoissonSolverCuda
/// \brief This class provides implementation of Poisson Eq
/// solver by MultiGrid Method using CUDA kernel
///
///
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017

#include "AliTPCPoissonSolverCuda.h"
#include <TMath.h>

/// \cond CLASSIMP
ClassImp(AliTPCPoissonSolverCuda)
/// \endcond

/// constructor
///
AliTPCPoissonSolverCuda::AliTPCPoissonSolverCuda()
  :  AliTPCPoissonSolver() {

	fErrorConvF = new TVectorF(fMgParameters.nMGCycle);
	fErrorExactF = new TVectorF(fMgParameters.nMGCycle);

}

/// Constructor
/// \param name name of the object
/// \param title title of the object
AliTPCPoissonSolverCuda::AliTPCPoissonSolverCuda(const char *name, const char *title)
  : AliTPCPoissonSolver(name, title) {
  /// constructor

	fErrorConvF = new TVectorF(fMgParameters.nMGCycle);
	fErrorExactF = new TVectorF(fMgParameters.nMGCycle);
}

/// destructor
AliTPCPoissonSolverCuda::~AliTPCPoissonSolverCuda() {
	delete fErrorConvF;
	delete fErrorExactF;
}

/// function overriding
void AliTPCPoissonSolverCuda::PoissonSolver3D(TMatrixD **matricesV, TMatrixD **matricesCharge,
                                          Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                          Int_t symmetry) {

  fNRRow = nRRow;
  fNZColumn = nZColumn;
  fPhiSlice = phiSlice;

  fVPotential = new TMatrixF(phiSlice * nRRow,  nZColumn);
  fromArrayOfMatrixToMatrixObj(matricesV,fVPotential,nRRow,nZColumn,phiSlice);	
  fRhoCharge = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesCharge,fRhoCharge,nRRow,nZColumn,phiSlice);	


  PoissonMultiGrid3D2D(fVPotential, fRhoCharge, nRRow, nZColumn, phiSlice, symmetry);
  

  fromMatrixObjToArrayOfMatrix(fVPotential,matricesV,nRRow,nZColumn,phiSlice);

  delete fVPotential;
  delete fRhoCharge;
}


// method to do multigrid3d2d
void AliTPCPoissonSolverCuda::PoissonMultiGrid3D2D(TMatrixF *VPotential, TMatrixF * RhoChargeDensities, Int_t nRRow,
                                               Int_t nZColumn, Int_t phiSlice, Int_t symmetry) {

	const Float_t  gridSizeR   =  (AliTPCPoissonSolver::fgkOFCRadius-AliTPCPoissonSolver::fgkIFCRadius) / (nRRow-1); // h_{r}
  	const Float_t  gridSizePhi =  TMath::TwoPi()/phiSlice;  // h_{phi}
  	const Float_t  gridSizeZ   =  AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn-1) ; // h_{z}
 	const Float_t  ratioPhi    =  gridSizeR*gridSizeR / (gridSizePhi*gridSizePhi) ;  // ratio_{phi} = gridsize_{r} / gridsize_{phi}
  	const Float_t  ratioZ      =  gridSizeR*gridSizeR / (gridSizeZ*gridSizeZ) ; // ratio_{Z} = gridsize_{r} / gridsize_{z}
	const Float_t  convErr 	   =  AliTPCPoissonSolver::fgConvergenceError;
	const Float_t  IFCRadius   =  AliTPCPoissonSolver::fgkIFCRadius;

	Int_t fparamsize = 8;
	Float_t * fparam = new Float_t[fparamsize];

	fparam[0] = gridSizeR;
  	fparam[1] = gridSizePhi;
	fparam[2] = gridSizeZ;
	fparam[3] = ratioPhi;
	fparam[4] = ratioZ;
	fparam[5] = convErr;
	fparam[6] = IFCRadius;

	Int_t iparamsize = 4;
	Int_t * iparam = new Int_t[iparamsize];

	iparam[0] = fMgParameters.nPre;	
	iparam[1] = fMgParameters.nPost;			
	iparam[2] = fMgParameters.maxLoop;
	iparam[3] = fMgParameters.nMGCycle;


	if (fMgParameters.cycleType == kFCycle)
	{
		PoissonMultigrid3DSemiCoarseningGPUErrorFCycle(VPotential->GetMatrixArray(), RhoChargeDensities->GetMatrixArray(),nRRow, nZColumn,phiSlice,symmetry, fparam, iparam, fErrorConvF->GetMatrixArray(), fErrorExactF->GetMatrixArray(), fExactSolutionF->GetMatrixArray());
	} else if (fMgParameters.cycleType == kWCycle) 
	{
		PoissonMultigrid3DSemiCoarseningGPUErrorWCycle(VPotential->GetMatrixArray(), RhoChargeDensities->GetMatrixArray(),nRRow, nZColumn,phiSlice,symmetry, fparam, iparam, fErrorConvF->GetMatrixArray(), fErrorExactF->GetMatrixArray(), fExactSolutionF->GetMatrixArray());
	} else 
	{
		PoissonMultigrid3DSemiCoarseningGPUError(VPotential->GetMatrixArray(), RhoChargeDensities->GetMatrixArray(),nRRow, nZColumn,phiSlice,symmetry, fparam, iparam, fErrorConvF->GetMatrixArray(), fErrorExactF->GetMatrixArray(), fExactSolutionF->GetMatrixArray());

	}	
	fIterations = iparam[3];
	delete[] fparam;
	delete[] iparam;
}



// helper function
// copy array of matrix to an obj of matrix
void AliTPCPoissonSolverCuda::fromArrayOfMatrixToMatrixObj(TMatrixD **matrices, TMatrixF *obj, Int_t nRRow, Int_t nZColumn, Int_t phiSlice) {
	TMatrixD *matrix;

	for (Int_t k=0; k< phiSlice;k++) {
		matrix = matrices[k];
		for (Int_t i=0;i<nRRow;i++) {
			for (Int_t j=0;j<nZColumn;j++) (*obj)(k*nRRow + i,j) = (Float_t)(*matrix)(i,j);
		}
	}

}


// helper function
// copy array of matrix to an obj of matrix
void AliTPCPoissonSolverCuda::fromMatrixObjToArrayOfMatrix(TMatrixF*obj,TMatrixD **matrices,  Int_t nRRow, Int_t nZColumn, Int_t phiSlice) {
	TMatrixD *matrix;

	for (Int_t k=0; k< phiSlice;k++) {
		matrix = matrices[k];
		for (Int_t i=0;i<nRRow;i++) {
			for (Int_t j=0;j<nZColumn;j++) (*matrix)(i,j) = (*obj)(k*nRRow + i,j);
		}
	}

}

void AliTPCPoissonSolverCuda::SetExactSolution(TMatrixD **exactSolution,Int_t nRRow, Int_t nZColumn, Int_t phiSlice) {
  	fNRRow = nRRow;
  	fNZColumn = nZColumn;
  	fPhiSlice = phiSlice;
	fExactSolutionF = new TMatrixF(fNRRow * fPhiSlice,fNZColumn);
  	fromArrayOfMatrixToMatrixObj(exactSolution,fExactSolutionF,fNRRow,fNZColumn,phiSlice);	
	fExactPresent = kTRUE;
	fMaxExact = TMath::Max(TMath::Abs((*fExactSolutionF).Max()), TMath::Abs((*fExactSolutionF).Min()));
}
