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
private:
  AliTPCPoissonSolverCuda *fPoissonSolverCuda;
/// \cond CLASSIMP
  ClassDef(AliTPCSpaceCharge3DDriftLineCuda,
  1);
/// \endcond
};

#endif
