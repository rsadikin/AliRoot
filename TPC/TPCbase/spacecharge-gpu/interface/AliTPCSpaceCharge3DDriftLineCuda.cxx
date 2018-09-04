/*************************************************************************
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


/* $Id$ */

/// \class AliTPCSpaceCharge3DDriftLineCuda
/// \brief This class provides distortion and correction map with integration following electron drift/// cuda implementation
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017

#include "TMath.h"
#include "TStopwatch.h"
#include "AliSysInfo.h"
#include "AliLog.h"
#include "AliTPCSpaceCharge3DDriftLineCuda.h"
#include "DifferentialGPU.h"
#include "LocalDistCorrDzGPU.h"
#include "IntegrateEzGPU.h"
/// \cond CLASSIMP
ClassImp(AliTPCSpaceCharge3DDriftLineCuda)
/// \endcond

/// Construction for AliTPCSpaceCharge3DDriftLineCuda class
/// Default values
/// ~~~
/// fInterpolationOrder = 5; // interpolation cubic spline with 5 points
/// fNRRows = 129;
/// fNPhiSlices = 180; // the maximum of phi-slices so far = (8 per sector)
/// fNZColumns = 129; // the maximum on column-slices so  ~ 2cm slicing
/// ~~~
AliTPCSpaceCharge3DDriftLineCuda::AliTPCSpaceCharge3DDriftLineCuda() : AliTPCSpaceCharge3DDriftLine() {
}


/// Construction for AliTPCSpaceCharge3DDriftLineCuda class
/// Default values
/// ~~~
/// fInterpolationOrder = 5; interpolation cubic spline with 5 points
/// fNRRows = 129;
/// fNPhiSlices = 180; // the maximum of phi-slices so far = (8 per sector)
/// fNZColumns = 129; // the maximum on column-slices so  ~ 2cm slicing
/// ~~~
///
AliTPCSpaceCharge3DDriftLineCuda::AliTPCSpaceCharge3DDriftLineCuda(const char *name, const char *title) : AliTPCSpaceCharge3DDriftLine(name,title) {
}




/// Construction for AliTPCSpaceCharge3DDriftLineCuda class
/// Member values from params
///
/// \param nRRow Int_t number of grid in r direction
/// \param nZColumn Int_t number of grid in z direction
/// \param nPhiSlice Int_t number of grid in \f$ \phi \f$ direction
///
AliTPCSpaceCharge3DDriftLineCuda::AliTPCSpaceCharge3DDriftLineCuda(const char *name, const char *title, Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice) :
  AliTPCSpaceCharge3DDriftLine(name,title,nRRow,nZColumn, nPhiSlice) {
}
/// Construction for AliTPCSpaceCharge3DDriftLineCuda class
/// Member values from params
///
/// \param nRRow Int_t number of grid in r direction
/// \param nZColumn Int_t number of grid in z direction
/// \param nPhiSlice Int_t number of grid in \f$ \phi \f$ direction
/// \param interpolationOrder Int_t order of interpolation
/// \param strategy Int_t strategy for global distortion
/// \param rbfKernelType Int_t strategy for global distortion
///
AliTPCSpaceCharge3DDriftLineCuda::AliTPCSpaceCharge3DDriftLineCuda(
  const char *name, const char *title, Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice, Int_t interpolationOrder, Int_t irregularGridSize, Int_t rbfKernelType)
  : AliTPCSpaceCharge3DDriftLine (name, title, nRRow, nZColumn, nPhiSlice, interpolationOrder, irregularGridSize, rbfKernelType) {
}

/// Destruction for AliTPCSpaceCharge3DDriftLineCuda
/// Deallocate memory for lookup table and charge distribution
///
AliTPCSpaceCharge3DDriftLineCuda::~AliTPCSpaceCharge3DDriftLineCuda() {

}





/// 
void AliTPCSpaceCharge3DDriftLineCuda::InitSpaceCharge3DPoissonIntegralDz(
  Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stoppingConvergence) {

  Int_t phiSlicesPerSector = phiSlice / kNumSector;
  const Float_t gridSizeR = (fgkOFCRadius - fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Double_t ezField = (fgkCathodeV - fgkGG) / fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;

  // local variables
  Float_t radius0, phi0, z0;

  // memory allocation for temporary matrices:
  // potential (boundary values), charge distribution
  TMatrixD *matricesV[phiSlice], *matricesCharge[phiSlice];
  TMatrixD *matricesEr[phiSlice], *matricesEPhi[phiSlice], *matricesEz[phiSlice];
  TMatrixD *matricesDistDrDz[phiSlice], *matricesDistDPhiRDz[phiSlice], *matricesDistDz[phiSlice];
  TMatrixD *matricesCorrDrDz[phiSlice], *matricesCorrDPhiRDz[phiSlice], *matricesCorrDz[phiSlice];
  TMatrixD *matricesGDistDrDz[phiSlice], *matricesGDistDPhiRDz[phiSlice], *matricesGDistDz[phiSlice];
  TMatrixD *matricesGCorrDrDz[phiSlice], *matricesGCorrDPhiRDz[phiSlice], *matricesGCorrDz[phiSlice];

  for (Int_t k = 0; k < phiSlice; k++) {
    matricesV[k] = new TMatrixD(nRRow, nZColumn);
    matricesCharge[k] = new TMatrixD(nRRow, nZColumn);
    matricesEr[k] = new TMatrixD(nRRow, nZColumn);
    matricesEPhi[k] = new TMatrixD(nRRow, nZColumn);
    matricesEz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDz[k] = new TMatrixD(nRRow, nZColumn);

  }

  // list of point as used in the poisson relaxation and the interpolation (for interpolation)
  Double_t rList[nRRow], zList[nZColumn], phiList[phiSlice];

  // pointer to current TF1 for potential boundary values
  TF1 *f1BoundaryIFC = NULL;
  TF1 *f1BoundaryOFC = NULL;
  TF1 *f1BoundaryROC = NULL;
  TStopwatch w;

  for (Int_t k = 0; k < phiSlice; k++) phiList[k] = gridSizePhi * k;
  for (Int_t i = 0; i < nRRow; i++) rList[i] = fgkIFCRadius + i * gridSizeR;

  for (Int_t j = 0; j < nZColumn; j++) zList[j] = j * gridSizeZ;


  // allocate look up local distortion
  AliTPCLookUpTable3DInterpolatorD *lookupLocalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesDistDrDz, rList, phiSlice, matricesDistDPhiRDz, phiList, nZColumn, matricesDistDz,
      zList, fInterpolationOrder);

  // allocate look up local correction
  AliTPCLookUpTable3DInterpolatorD *lookupLocalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesCorrDrDz, rList, phiSlice, matricesCorrDPhiRDz, phiList, nZColumn, matricesCorrDz,
      zList, fInterpolationOrder);

  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD *lookupGlobalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGDistDrDz, rList, phiSlice, matricesGDistDPhiRDz, phiList, nZColumn, matricesGDistDz,
      zList, fInterpolationOrder);
  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD *lookupGlobalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGCorrDrDz, rList, phiSlice, matricesGCorrDPhiRDz, phiList, nZColumn, matricesGCorrDz,
      zList, fInterpolationOrder);

  // should be set, in another place
  const Int_t symmetry = 0; // fSymmetry

  // for irregular
  TMatrixD **matricesIrregularDrDz = NULL;
  TMatrixD **matricesIrregularDPhiRDz = NULL;
  TMatrixD **matricesIrregularDz = NULL;
  TMatrixD **matricesPhiIrregular = NULL;
  TMatrixD **matricesRIrregular = NULL;
  TMatrixD **matricesZIrregular = NULL;

  // for charge
  TMatrixD **matricesLookUpCharge = NULL;
  AliTPC3DCylindricalInterpolator *chargeInterpolator = NULL;
  AliTPC3DCylindricalInterpolator *potentialInterpolator = NULL;
  Double_t *potentialBoundary = NULL;
  TMatrixD *matrixV;
  TMatrixD *matrixCharge;
  Int_t pIndex = 0;

  // do if look up table haven't be initialized
  if (!fInitLookUp) {
    // initialize for working memory
    for (Int_t side = 0; side < 2; side++) {

      // zeroing global distortion/correction
      for (Int_t k = 0; k < phiSlice; k++) {
        matricesDistDrDz[k]->Zero();
        matricesDistDPhiRDz[k]->Zero();
        matricesDistDz[k]->Zero();
        matricesCorrDrDz[k]->Zero();
        matricesCorrDPhiRDz[k]->Zero();
        matricesCorrDz[k]->Zero();

        matricesGDistDrDz[k]->Zero();
        matricesGDistDPhiRDz[k]->Zero();
        matricesGDistDz[k]->Zero();
        matricesGCorrDrDz[k]->Zero();
        matricesGCorrDPhiRDz[k]->Zero();
        matricesGCorrDz[k]->Zero();
      }
      if (side == 0) {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularA;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularA;
        matricesIrregularDz = fMatrixIntCorrDzIrregularA;

        matricesPhiIrregular = fMatrixPhiListIrregularA;
        matricesRIrregular = fMatrixRListIrregularA;
        matricesZIrregular = fMatrixZListIrregularA;
        matricesLookUpCharge = fMatrixChargeA;
        chargeInterpolator = fInterpolatorChargeA;
        potentialInterpolator = fInterpolatorPotentialA;
        fLookupDistA->SetLookUpR(matricesDistDrDz);
        fLookupDistA->SetLookUpPhi(matricesDistDPhiRDz);
        fLookupDistA->SetLookUpZ(matricesDistDz);


        fLookupElectricFieldA->SetLookUpR(matricesEr);
        fLookupElectricFieldA->SetLookUpPhi(matricesEPhi);
        fLookupElectricFieldA->SetLookUpZ(matricesEz);

        potentialBoundary = fListPotentialBoundaryA;
        f1BoundaryIFC = fFormulaBoundaryIFCA;
        f1BoundaryOFC = fFormulaBoundaryOFCA;
        f1BoundaryROC = fFormulaBoundaryROCA;
      } else {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularC;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularC;
        matricesIrregularDz = fMatrixIntCorrDzIrregularC;
        matricesPhiIrregular = fMatrixPhiListIrregularC;
        matricesRIrregular = fMatrixRListIrregularC;
        matricesZIrregular = fMatrixZListIrregularC;
        matricesLookUpCharge = fMatrixChargeC;
        chargeInterpolator = fInterpolatorChargeC;
        potentialInterpolator = fInterpolatorPotentialC;
        fLookupDistC->SetLookUpR(matricesDistDrDz);
        fLookupDistC->SetLookUpPhi(matricesDistDPhiRDz);
        fLookupDistC->SetLookUpZ(matricesDistDz);
        fLookupElectricFieldC->SetLookUpR(matricesEr);
        fLookupElectricFieldC->SetLookUpPhi(matricesEPhi);
        fLookupElectricFieldC->SetLookUpZ(matricesEz);

        potentialBoundary = fListPotentialBoundaryC;
        f1BoundaryIFC = fFormulaBoundaryIFCC;
        f1BoundaryOFC = fFormulaBoundaryOFCC;
        f1BoundaryROC = fFormulaBoundaryROCC;
      }

      // fill the potential boundary
      // guess the initial potential
      // fill also charge
      //pIndex = 0;

      //AliInfo(Form("Step = 0: Fill Boundary and Charge Densities"));
      for (Int_t k = 0; k < phiSlice; k++) {
        phi0 = k * gridSizePhi;
        matrixV = matricesV[k];
        matrixCharge = matricesCharge[k];
        for (Int_t i = 0; i < nRRow; i++) {
          radius0 = fgkIFCRadius + i * gridSizeR;
          for (Int_t j = 0; j < nZColumn; j++) {
            z0 = j * gridSizeZ;
            (*matrixCharge)(i, j) = chargeInterpolator->GetValue(rList[i], phiList[k], zList[j]);
            (*matrixV)(i, j) = 0.0; // fill zeros
            if (fFormulaPotentialV == NULL) {
              // boundary IFC
              if (i == 0) {
                if (f1BoundaryIFC != NULL) {
                  (*matrixV)(i, j) = f1BoundaryIFC->Eval(z0);
                }
              }
              if (i == (nRRow - 1)) {
                if (f1BoundaryOFC != NULL)
                  (*matrixV)(i, j) = f1BoundaryOFC->Eval(z0);
              }
              if (j == 0) {
                if (fFormulaBoundaryCE) {
                  (*matrixV)(i, j) = fFormulaBoundaryCE->Eval(radius0);
                }
              }
              if (j == (nZColumn - 1)) {
                if (f1BoundaryROC != NULL)
                  (*matrixV)(i, j) = f1BoundaryROC->Eval(radius0);
              }
            } else {
              if ((i == 0) || (i == (nRRow - 1)) || (j == 0) || (j == (nZColumn - 1))) {
                (*matrixV)(i, j) = fFormulaPotentialV->Eval(radius0, phi0, z0);
              }
            }
          }
        }
      }
      AliInfo(Form("Step 0: Preparing Charge interpolator: %f\n", w.CpuTime()));
      AliTPCPoissonSolver::fgConvergenceError = stoppingConvergence;

      fPoissonSolverCuda->SetStrategy(AliTPCPoissonSolver::kMultiGrid);
      (fPoissonSolverCuda->fMgParameters).cycleType = AliTPCPoissonSolver::kFCycle;
      (fPoissonSolverCuda->fMgParameters).isFull3D = kFALSE;
      (fPoissonSolverCuda->fMgParameters).nMGCycle = maxIteration;
      (fPoissonSolverCuda->fMgParameters).maxLoop = 6;


      w.Start();
      fPoissonSolverCuda->PoissonSolver3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration,
                                      symmetry);
      w.Stop();

      potentialInterpolator->SetValue(matricesV);
      potentialInterpolator->InitCubicSpline();


      AliInfo(Form("Step 1: Poisson solver: %f\n", w.CpuTime()));
      myProfile.poissonSolverTime = w.CpuTime();
      myProfile.iteration = fPoissonSolverCuda->fIterations;


      w.Start();
      ElectricField(matricesV,
                    matricesEr, matricesEPhi, matricesEz, nRRow, nZColumn, phiSlice,
                    gridSizeR, gridSizePhi, gridSizeZ, symmetry, fgkIFCRadius);
      w.Stop();

      myProfile.electricFieldTime = w.CpuTime();
      AliInfo(Form("Step 2: Electric Field Calculation: %f\n", w.CpuTime()));
      w.Start();
      LocalDistCorrDz(matricesEr, matricesEPhi, matricesEz,
                      matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz,
                      matricesCorrDrDz, matricesCorrDPhiRDz, matricesCorrDz,
                      nRRow, nZColumn, phiSlice, gridSizeZ, ezField);
      w.Stop();
      myProfile.localDistortionTime = w.CpuTime();

      // copy to interpolator
      if (side == 0) {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistA->CopyFromMatricesToInterpolator();
        fLookupElectricFieldA->CopyFromMatricesToInterpolator();
      } else {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistC->CopyFromMatricesToInterpolator();
        fLookupElectricFieldC->CopyFromMatricesToInterpolator();
      }

      AliInfo(Form("Step 3: Local distortion and correction: %f\n", w.CpuTime()));
      w.Start();
      if (fIntegrationStrategy == kNaive)
	IntegrateDistCorrDriftLineDz(
          lookupLocalDist,
          matricesGDistDrDz, matricesGDistDPhiRDz, matricesGDistDz,
          lookupLocalCorr,
          matricesGCorrDrDz, matricesGCorrDPhiRDz, matricesGCorrDz,
          matricesIrregularDrDz, matricesIrregularDPhiRDz, matricesIrregularDz,
          matricesRIrregular, matricesPhiIrregular, matricesZIrregular,
          nRRow, nZColumn, phiSlice, rList, phiList, zList
        );
      else
	IntegrateDistCorrDriftLineDzWithLookUp ( 
          lookupLocalDist,
          matricesGDistDrDz, matricesGDistDPhiRDz, matricesGDistDz,
          lookupLocalCorr,
          matricesGCorrDrDz, matricesGCorrDPhiRDz, matricesGCorrDz,
          nRRow, nZColumn, phiSlice, rList, phiList, zList
	);

      w.Stop();
      AliInfo(Form("Step 4: Global correction/distortion: %f\n", w.CpuTime()));
      myProfile.globalDistortionTime = w.CpuTime();
      w.Start();

      //// copy to 1D interpolator /////
      lookupGlobalDist->CopyFromMatricesToInterpolator();
      lookupGlobalCorr->CopyFromMatricesToInterpolator();
      ////


       w.Stop();
       AliInfo(Form("Step 5: Filling up the look up: %f\n", w.CpuTime()));

      if (side == 0) {
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzA, fMatrixIntDistDPhiREzA, fMatrixIntDistDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        FillLookUpTable(lookupGlobalCorr,
                        fMatrixIntCorrDrEzA, fMatrixIntCorrDPhiREzA, fMatrixIntCorrDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        fLookupIntDistA->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0)
          fLookupIntCorrA->CopyFromMatricesToInterpolator();
        else
          fLookupIntCorrIrregularA->CopyFromMatricesToInterpolator();

        AliInfo(" A side done");
      }
      if (side == 1) {
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzC, fMatrixIntDistDPhiREzC, fMatrixIntDistDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        FillLookUpTable(lookupGlobalCorr,
                        fMatrixIntCorrDrEzC, fMatrixIntCorrDPhiREzC, fMatrixIntCorrDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        fLookupIntDistC->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0)
          fLookupIntCorrC->CopyFromMatricesToInterpolator();
        else
          fLookupIntCorrIrregularC->CopyFromMatricesToInterpolator();
        AliInfo(" C side done");
      }

    }

    fInitLookUp = kTRUE;
  }




  // memory de-allocation for temporary matrices
  for (Int_t k = 0; k < phiSlice; k++) {
    delete matricesV[k];
    delete matricesCharge[k];
    delete matricesEr[k];
    delete matricesEPhi[k];
    delete matricesEz[k];
    delete matricesDistDrDz[k];
    delete matricesDistDPhiRDz[k];
    delete matricesDistDz[k];

    delete matricesCorrDrDz[k];
    delete matricesCorrDPhiRDz[k];
    delete matricesCorrDz[k];
    delete matricesGDistDrDz[k];
    delete matricesGDistDPhiRDz[k];
    delete matricesGDistDz[k];

    delete matricesGCorrDrDz[k];
    delete matricesGCorrDPhiRDz[k];
    delete matricesGCorrDz[k];

  }
  delete lookupLocalDist;
  delete lookupLocalCorr;
  delete lookupGlobalDist;
  delete lookupGlobalCorr;
}



void AliTPCSpaceCharge3DDriftLineCuda::ElectricField(TMatrixD **matricesV, TMatrixD **matricesEr, TMatrixD **matricesEPhi,
                                                 TMatrixD **matricesEz, const Int_t nRRow, const Int_t nZColumn,
                                                 const Int_t phiSlice,
                                                 const Float_t gridSizeR, const Float_t gridSizePhi,
                                                 const Float_t gridSizeZ,
                                                 const Int_t symmetry, const Float_t innerRadius) {


  TMatrixF * VPotential = new TMatrixF(phiSlice * nRRow,  nZColumn);
  fromArrayOfMatrixToMatrixObj(matricesV,VPotential,nRRow,nZColumn,phiSlice);	
  TMatrixF * Er = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesEr,Er,nRRow,nZColumn,phiSlice);	
  TMatrixF * EPhi = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesEPhi,EPhi,nRRow,nZColumn,phiSlice);	
  TMatrixF * Ez = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesEz,Ez,nRRow,nZColumn,phiSlice);	

  DifferentialCalculationGPU (
	VPotential->GetMatrixArray(),
	Er->GetMatrixArray(),
	Ez->GetMatrixArray(),
	EPhi->GetMatrixArray(), 
	nRRow,
	nZColumn,
	phiSlice,
	symmetry,
	fgkIFCRadius,
 	fgkOFCRadius,
	fgkTPCZ0
  );


  fromMatrixObjToArrayOfMatrix(Er,matricesEr,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(EPhi,matricesEPhi,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(Ez,matricesEz,nRRow,nZColumn,phiSlice);

  delete VPotential;
  delete Er;
  delete EPhi;
  delete Ez;
}


void AliTPCSpaceCharge3DDriftLineCuda::LocalDistCorrDz(TMatrixD **matricesEr, TMatrixD **matricesEPhi, TMatrixD **matricesEz, TMatrixD **matricesDistDrDz, TMatrixD **matricesDistDPhiRDz, TMatrixD **matricesDistDz, TMatrixD **matricesCorrDrDz,TMatrixD **matricesCorrDPhiRDz, TMatrixD **matricesCorrDz, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Float_t gridSizeZ, const Double_t ezField) {

  TMatrixF * Er = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesEr,Er,nRRow,nZColumn,phiSlice);	
  TMatrixF * EPhi = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesEPhi,EPhi,nRRow,nZColumn,phiSlice);	
  TMatrixF * Ez = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesEz,Ez,nRRow,nZColumn,phiSlice);	
  TMatrixF * distDrDz = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesDistDrDz,distDrDz,nRRow,nZColumn,phiSlice);	
  TMatrixF * distDPhiRDz = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesDistDPhiRDz,distDPhiRDz,nRRow,nZColumn,phiSlice);	
  TMatrixF * distDz = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesDistDz,distDz,nRRow,nZColumn,phiSlice);	
  TMatrixF * corrDrDz = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesCorrDrDz,corrDrDz,nRRow,nZColumn,phiSlice);	
  TMatrixF * corrDPhiRDz = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesCorrDPhiRDz,corrDPhiRDz,nRRow,nZColumn,phiSlice);	
  TMatrixF * corrDz = new TMatrixF(phiSlice * nRRow,  nZColumn);	  
  fromArrayOfMatrixToMatrixObj(matricesCorrDz,corrDz,nRRow,nZColumn,phiSlice);	

  
  LocalDistCorrDzGPU (Er->GetMatrixArray(),Ez->GetMatrixArray(),EPhi->GetMatrixArray(), distDrDz->GetMatrixArray(),distDPhiRDz->GetMatrixArray(),distDz->GetMatrixArray(),corrDrDz->GetMatrixArray(),corrDPhiRDz->GetMatrixArray(),corrDz->GetMatrixArray(), nRRow, nZColumn, phiSlice, gridSizeZ,ezField,fC0,fC1,fgkdvdE); 

  fromMatrixObjToArrayOfMatrix(distDrDz,matricesDistDrDz,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(distDPhiRDz,matricesDistDPhiRDz,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(distDz,matricesCorrDz,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(corrDrDz,matricesCorrDrDz,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(corrDPhiRDz,matricesCorrDPhiRDz,nRRow,nZColumn,phiSlice);
  fromMatrixObjToArrayOfMatrix(corrDz,matricesCorrDz,nRRow,nZColumn,phiSlice);
  delete Er;
  delete EPhi;
  delete Ez;
  delete distDrDz;
  delete distDPhiRDz;
  delete distDz;
  delete corrDrDz;
  delete corrDPhiRDz;
  delete corrDz;
}

// helper function
// copy array of matrix to an obj of matrix
void AliTPCSpaceCharge3DDriftLineCuda::fromArrayOfMatrixToMatrixObj(TMatrixD **matrices, TMatrixF *obj, Int_t nRRow, Int_t nZColumn, Int_t phiSlice) {
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
void AliTPCSpaceCharge3DDriftLineCuda::fromMatrixObjToArrayOfMatrix(TMatrixF*obj,TMatrixD **matrices,  Int_t nRRow, Int_t nZColumn, Int_t phiSlice) {
	TMatrixD *matrix;

	for (Int_t k=0; k< phiSlice;k++) {
		matrix = matrices[k];
		for (Int_t i=0;i<nRRow;i++) {
			for (Int_t j=0;j<nZColumn;j++) (*matrix)(i,j) = (*obj)(k*nRRow + i,j);
		}
	}

}
