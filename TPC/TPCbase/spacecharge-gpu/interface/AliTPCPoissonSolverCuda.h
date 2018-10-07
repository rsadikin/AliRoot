#ifndef ALITPCPOISSONSOLVERCUDA_H
#define ALITPCPOISSONSOLVERCUDA_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


/// \class AliTPCPoissonSolver
/// \brief This class provides implementation of Poisson Eq
/// solver by MultiGrid Method
///
///
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017
#include <TNamed.h>
#include "TMatrixD.h"
#include "TMatrixF.h"
#include "TVectorD.h"
#include "AliTPCPoissonSolver.h"
#include "PoissonSolver3DGPU.h"



class AliTPCPoissonSolverCuda : public AliTPCPoissonSolver {
public:
  AliTPCPoissonSolverCuda();
  AliTPCPoissonSolverCuda(const char *name, const char *title, Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice);
  AliTPCPoissonSolverCuda(const char *name, const char *title);
  virtual ~AliTPCPoissonSolverCuda();
  void PoissonSolver3D(TMatrixD **matricesV, TMatrixD **matricesChargeDensities, Int_t nRRow, Int_t nZColumn,
                       Int_t phiSlice, Int_t maxIterations, Int_t symmetry);


  // setter and getter
  void SetStrategy(StrategyType strategy) {fStrategy = strategy;}
  void SetExactSolution(TMatrixD **exactSolution,Int_t nRRow, Int_t nZColumn, Int_t phiSlice);
  TMatrixF *fExactSolutionF;	
private:
  AliTPCPoissonSolverCuda(const AliTPCPoissonSolverCuda &);               // not implemented
  AliTPCPoissonSolverCuda &operator=(const AliTPCPoissonSolverCuda &);    // not implemented
  void PoissonMultiGrid3D2D(TMatrixF *VPotential, TMatrixF *RhoChargeDensities, Int_t nRRow,
                            Int_t nZColumn, Int_t phiSlice, Int_t symmetry);



  // store potential and charge
  TMatrixF * fVPotential; //-> hold potential in an object of tmatrixd 
  TMatrixF * fRhoCharge; //-> pointer to an object of tmatrixd for storing charge 
  Int_t fNRRow;
  Int_t fNZColumn;
  Int_t fPhiSlice;
  // error single precision for cuda-based
  TVectorF *fErrorConvF;
  TVectorF *fErrorExactF;
  void fromArrayOfMatrixToMatrixObj(TMatrixD **matrices, TMatrixF *obj, Int_t nRRow, Int_t nZColumn, Int_t phiSlice);
  void fromMatrixObjToArrayOfMatrix(TMatrixF*obj,TMatrixD **matrices,  Int_t nRRow, Int_t nZColumn, Int_t phiSlice);
/// \cond CLASSIMP
  ClassDef(AliTPCPoissonSolverCuda,5);
/// \endcond
};

#endif
