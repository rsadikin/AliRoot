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

/*
$Log$
Revision 1.10  2002/10/14 14:55:35  hristov
Merging the VirtualMC branch to the main development branch (HEAD)

Revision 1.8.6.1  2002/07/30 15:00:39  alibrary
Correction in initialization of dx (Enrique Gamez Flores).

Revision 1.9  2002/07/30 13:48:03  morsch

Correction in initialization of dx (Enrique Gamez Flores).

Revision 1.8  2002/02/08 16:50:50  morsch
Add name and title in constructor.

Revision 1.7  2001/07/27 17:09:35  morsch
Use local SetTrack, KeepTrack and SetHighWaterMark methods
to delegate either to local stack or to stack owned by AliRun.
(Piotr Skowronski, A.M.)

Revision 1.6  2000/12/21 16:24:06  morsch
Coding convention clean-up

Revision 1.5  2000/12/06 15:11:38  morsch
Correct double declared data members.

Revision 1.4  2000/11/30 07:12:50  alibrary
Introducing new Rndm and QA classes

Revision 1.3  2000/10/02 21:28:06  fca
Removal of useless dependecies via forward declarations

Revision 1.2  2000/06/09 20:37:51  morsch
All coding rule violations except RS3 corrected

Revision 1.1  2000/02/23 16:25:14  morsch
First commit of this file

*/

// As AliGenScan,  generation of particles on a 3-dim grid
// but here double hits with a predefined distance are generated.
// The second particle is generated at a constant distance but with random phi.
// Generator can be used to evaluate double hit resolutions.
// Author: andreas.morsch@cern.ch

#include "AliGenDoubleScan.h"
#include "AliRun.h"

 ClassImp(AliGenDoubleScan)
    
 AliGenDoubleScan::AliGenDoubleScan()
	 :AliGenScan(-1)
{
}

AliGenDoubleScan::AliGenDoubleScan(Int_t npart)
    :AliGenScan(npart)
{
// Constructor
    fName = "Double Scan";
    fTitle= "Particle Generator for two correlated particles on a grid";
}

//____________________________________________________________
AliGenDoubleScan::~AliGenDoubleScan()
{
// Destructor
}

//____________________________________________________________
void AliGenDoubleScan::Generate()
{
    //
    // Generate one trigger
    //
  
    Float_t polar[3]= {0,0,0};
    //
    Float_t origin[3];
    Float_t p[3];
    Int_t nt;
    Float_t pmom, theta, phi;
    //
    Float_t random[6];
    Float_t dx,dy,dz;
    
    //
    if (fNx > 0) {
	dx=(fXCmax-fXCmin)/fNx;
    } else {
	dx=1e10;
    }

    if (fNy > 0) {
	dy=(fYCmax-fYCmin)/fNy;
    } else {
	dy=1e10;
    }
    
    if (fNz > 0) {
      dz=(fZmax-fZmin)/fNz;
    } else {
	dz=1e10;
    }
    for (Int_t ix=0; ix<fNx; ix++) {
      for (Int_t iy=0; iy<fNy; iy++) {
	  for (Int_t iz=0; iz<fNz; iz++){
	      Rndm(random,6);
	      origin[0]=fXCmin+ix*dx+2*(random[0]-0.5)*fOsigma[0];
	      origin[1]=fYCmin+iy*dy+2*(random[1]-0.5)*fOsigma[1];
	      origin[2]=fZmin+iz*dz+2*(random[2]-0.5)*fOsigma[2];	     
	      pmom=fPMin+random[3]*(fPMax-fPMin);
	      theta=fThetaMin+random[4]*(fThetaMax-fThetaMin);
	      phi=fPhiMin+random[5]*(fPhiMax-fPhiMin);
	      p[0] = pmom*TMath::Cos(phi)*TMath::Sin(theta);
	      p[1] = pmom*TMath::Sin(phi)*TMath::Sin(theta);
	      p[2] = pmom*TMath::Cos(theta);
	      SetTrack(fTrackIt,-1,fIpart,p,origin,polar,0,kPPrimary,nt);
//
// Generate 2nd particle at distance fDistance from  the first
//
	      Rndm(random,6);
	      Float_t phi2=2.*TMath::Pi()*random[0];
	      Float_t dx  =fDistance*TMath::Sin(phi2);
	      Float_t dy  =fDistance*TMath::Cos(phi2);	      
	      origin[0]=origin[0]+dx;
	      origin[1]=origin[1]+dy;	      
	      pmom=fPMin+random[1]*(fPMax-fPMin);
	      theta=fThetaMin+random[2]*(fThetaMax-fThetaMin);
	      phi=fPhiMin+random[3]*(fPhiMax-fPhiMin);
	      p[0] = pmom*TMath::Cos(phi)*TMath::Sin(theta);
	      p[1] = pmom*TMath::Sin(phi)*TMath::Sin(theta);
	      p[2] = pmom*TMath::Cos(theta);
	      SetTrack(fTrackIt,-1,fIpart,p,origin,polar,0,kPPrimary,nt);
	  }
      }
  }
}













