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

#include "AliTPCSpaceCharge3DDriftLineCuda.h"

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
