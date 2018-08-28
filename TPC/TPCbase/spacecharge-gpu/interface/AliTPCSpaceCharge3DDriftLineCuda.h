#ifndef ALI_TPC_SPACECHARGE3D_DRIFTLINE_CUDA_H
#define ALI_TPC_SPACECHARGE3D_DRIFTLINE_CUDA_H


/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

/// \class AliTPCSpaceCharge3DDriftLineCudaCuda
/// \brief This class provides distortion and correction map with integration following electron drift in
///        cuda implementation
/// TODO: validate distortion z by comparing with exisiting classes
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017


#include <TNamed.h>
#include "TMatrixD.h"
#include "TMatrixF.h"
#include "TVectorD.h"
#include "AliTPCSpaceCharge3DDriftLine.h"
#include "AliTPCPoissonSolverCuda.h"




class AliTPCSpaceCharge3DDriftLineCuda : public AliTPCSpaceCharge3DDriftLine {
public:
  AliTPCSpaceCharge3DDriftLineCuda();
  AliTPCSpaceCharge3DDriftLineCuda(const char *name, const char *title);
  AliTPCSpaceCharge3DDriftLineCuda(const char *name, const char *title, Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice);
  AliTPCSpaceCharge3DDriftLineCuda(const char *name, const char *title, Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice, Int_t interpolationOrder, Int_t irregularGridSize, Int_t rbfKernelType);
  virtual ~AliTPCSpaceCharge3DDriftLineCuda();
  void InitSpaceCharge3DPoissonIntegralDz(Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                          Double_t stopConvergence);

  
  void SetPoissonSolver(AliTPCPoissonSolverCuda *poissonSolver) {
    fPoissonSolverCuda = poissonSolver;
  }

  AliTPCPoissonSolver *GetPoissonSolver() { return fPoissonSolver; }


private:

  AliTPCPoissonSolverCuda *fPoissonSolverCuda;
  /**void ElectricField(TMatrixD **matricesV, TMatrixD **matricesEr, TMatrixD **matricesEPhi,
                                                 TMatrixD **matricesEz, const Int_t nRRow, const Int_t nZColumn,
                                                 const Int_t phiSlice,
                                                 const Float_t gridSizeR, const Float_t gridSizePhi,
                                                 const Float_t gridSizeZ,
                                                 const Int_t symmetry, const Float_t innerRadius);
  **/
  
  void fromArrayOfMatrixToMatrixObj(TMatrixD **matrices, TMatrixF *obj, Int_t nRRow, Int_t nZColumn, Int_t phiSlice);
  void fromMatrixObjToArrayOfMatrix(TMatrixF*obj,TMatrixD **matrices,  Int_t nRRow, Int_t nZColumn, Int_t phiSlice);
/// \cond CLASSIMP
  ClassDef(AliTPCSpaceCharge3DDriftLineCuda,
  1);
/// \endcond
};

#endif
