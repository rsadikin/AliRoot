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

// $Id$

#include "AliMUONPedestal.h"
#include "AliMUONErrorCounter.h"
#include "AliMUONVStore.h"
#include "AliMUON2DMap.h"
#include "AliMUONCalibParamND.h"
#include "AliMpConstants.h"
#include <TString.h>
#include <TTimeStamp.h>
#include <TMath.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <Riostream.h>

#include <sstream>

//-----------------------------------------------------------------------------
/// \class AliMUONPedestal
///
/// Implementation of the pedestal computing
///
/// add
/// 
///
/// \author Alberto Baldisseri, JL Charvet (05/05/2009)
//-----------------------------------------------------------------------------

/// \cond CLASSIMP
ClassImp(AliMUONPedestal)
/// \endcond

//______________________________________________________________________________
AliMUONPedestal::AliMUONPedestal()
: TObject(),
//fN(0),
fNEvents(0),
fRunNumber(0),
fNChannel(0),
fNManu(0),
fNManuConfig(0),
fConfig(1),
fErrorBuspatchTable(new AliMUON2DMap(kFALSE)),
fManuBuspatchTable(new AliMUON2DMap(kFALSE)),
fManuBPoutofconfigTable(new AliMUON2DMap(kFALSE)),
fDate(new TTimeStamp()),
fFilcout(0),
fHistoFileName(),
fPedestalStore(new AliMUON2DMap(kTRUE)),
fIndex(-1),
fPrefixDA()
{
/// Default constructor
}

//______________________________________________________________________________
AliMUONPedestal::~AliMUONPedestal()
{
/// Destructor
  delete fErrorBuspatchTable;
  delete fManuBuspatchTable;
  delete fPedestalStore;
  delete fManuBPoutofconfigTable;
}

//______________________________________________________________________________
const char* 
AliMUONPedestal::GetHistoFileName() const
{
  /// Return the name of file we use to store histograms
  return fHistoFileName.Data();
}

//______________________________________________________________________________
void AliMUONPedestal::LoadConfig(const char* dbfile)
{
  /// Load MUONTRK configuration from ascii file "dbfile" (in DetDB)

  Int_t manuId;
  Int_t busPatchId;

  ifstream filein(dbfile,ios::in);

  // check if the 1st caracter of the 1st line is # (Config read from the OCDB => OffLine)
  // NO NEED ANYMORE : change configuration tested by the Shuttle (from 16/02/10)
//   string line; 
//   getline(filein, line, '\n');
//   cout << " line 1: " << line ;
//   if ( int(line[0]) == 35 )  // ascii code of # character
//     {
//       cout << " ==>  1st caracter = " << line[0] << " (ascii code =" << int(line[0]) << ")" << endl;    
//     }
//   else  
//     { filein.clear();  filein.seekg(0);  // rewind
//       cout << " ==> rewind configuration file: "<< dbfile << endl;           
//     } 
  
  while (!filein.eof())
    { 
      filein >> busPatchId >> manuId;

      AliMUONErrorCounter* manuCounter;
      AliMUONVCalibParam* ped = 
	static_cast<AliMUONVCalibParam*>(fPedestalStore ->FindObject(busPatchId, manuId));

      if (!ped) {
	fNManuConfig++;
	fNChannel+=64;
  ped = new AliMUONCalibParamND(2, AliMpConstants::ManuNofChannels(),busPatchId, manuId, -1.); // put default wise -1, not connected channel
	fPedestalStore ->Add(ped);  

	if (!(manuCounter = static_cast<AliMUONErrorCounter*>(fManuBuspatchTable->FindObject(busPatchId,manuId))))
	  {
	    // New (buspatch,manu)
	    manuCounter = new AliMUONErrorCounter(busPatchId,manuId);
	    fManuBuspatchTable->Add(manuCounter);
	  }
      }
    }
} 
//______________________________________________________________________________
void AliMUONPedestal::MakePed(Int_t busPatchId, Int_t manuId, Int_t channelId, Int_t charge)
{
  static Int_t warn=0;
  /// Compute pedestals values
  AliMUONVCalibParam* ped = 
    static_cast<AliMUONVCalibParam*>(fPedestalStore ->FindObject(busPatchId, manuId));

  if (!ped)   
    {
      if(fConfig) 
	{  // Fill out_of_config (buspatch,manu) table
	  if (!(static_cast<AliMUONErrorCounter*>(fManuBPoutofconfigTable->FindObject(busPatchId,manuId))))
	    fManuBPoutofconfigTable->Add(new AliMUONErrorCounter(busPatchId,manuId));
	  if(warn<10) cout << " !!! WARNING  : busPatchId = " << busPatchId << " manuId = " << manuId << " not in the Detector configuration " << endl;
	  else if(warn==10) cout << " !!! see .log file for an exhaustive list of (busPatchId, manuId) out of Detector configuration \n" << endl; 
	   warn++;
	   (*fFilcout) << " !!! WARNING  : busPatchId = " << busPatchId << " manuId = " << manuId << " not in the Detector configuration " << endl; 
	}
      else {fNManu++;}
      fNChannel+=64;
      // put default wise -1, not connected channel
      ped = new AliMUONCalibParamND(2, AliMpConstants::ManuNofChannels(),busPatchId, manuId, -1.); 
      fPedestalStore ->Add(ped);  
    }

  // Initialization for the first value
  if (ped->ValueAsDouble(channelId, 0) == -1)  
    { 
      if(fConfig && channelId == 0){fNManu++;}
      ped->SetValueAsDouble(channelId, 0, 0.);
    }
  if (ped->ValueAsDouble(channelId, 1) == -1) ped->SetValueAsDouble(channelId, 1, 0.);

  Double_t pedMean  = ped->ValueAsDouble(channelId, 0) + (Double_t) charge;
  Double_t pedSigma = ped->ValueAsDouble(channelId, 1) + (Double_t) charge*charge;

  ped->SetValueAsDouble(channelId, 0, pedMean);
  ped->SetValueAsDouble(channelId, 1, pedSigma);

  AliMUONErrorCounter* manuCounter;
  if (!(manuCounter = static_cast<AliMUONErrorCounter*>(fManuBuspatchTable->FindObject(busPatchId,manuId))))
    {
      // New (buspatch,manu)
      manuCounter = new AliMUONErrorCounter(busPatchId,manuId);
      fManuBuspatchTable->Add(manuCounter);
    }
  else
    {
      // Existing buspatch
      manuCounter->Increment();
    }	
}
//______________________________________________________________________________
void AliMUONPedestal::Finalize()
{
  /// final polishing of the store
  
  Double_t pedMean;
  Double_t pedSigma;
  Int_t busPatchId;
  Int_t manuId;
  Int_t channelId;

  // print in logfile
  if (fErrorBuspatchTable->GetSize())
    {
      cout<<"\n* Buspatches with less statistics (due to parity errors)"<<endl;
      (*fFilcout)<<"\n* Buspatches with less statistics (due to parity errors)"<<endl;
      TIter nextParityError(fErrorBuspatchTable->CreateIterator());
      AliMUONErrorCounter* parityerror;
      while((parityerror = static_cast<AliMUONErrorCounter*>(nextParityError())))
	{
	  cout<<"  bp "<<parityerror->BusPatch()<<": events used = "<<fNEvents-parityerror->Events()<<endl;
	  (*fFilcout)<<"  bp "<<parityerror->BusPatch()<<": events used = "<<fNEvents-parityerror->Events()<<endl;
	}
    }

  // iterator over pedestal
  TIter next(fPedestalStore ->CreateIterator());
  AliMUONVCalibParam* ped;

  while ( ( ped = dynamic_cast<AliMUONVCalibParam*>(next() ) ) )
    {
      busPatchId              = ped->ID0();
      manuId                  = ped->ID1();
      if(manuId==0)
	{
	  cout << " !!! BIG WARNING: ManuId = " << manuId << " !!! in  BP = " << busPatchId << endl;
	  (*fFilcout) << " !!! BIG WARNING: ManuId = " << manuId << " !!! in  BP = " << busPatchId << endl;
	}
      Int_t eventCounter;
      // Correct the number of events for buspatch with errors
      AliMUONErrorCounter* errorCounter;
      if ((errorCounter = (AliMUONErrorCounter*)fErrorBuspatchTable->FindObject(busPatchId)))
	{
	  eventCounter = fNEvents - errorCounter->Events();
	}
      else
	{
	  eventCounter = fNEvents;
	}

      Int_t occupancy=0; // channel missing in raw data or read but rejected (case of parity error)
      // value of (buspatch, manu) occupancy
      AliMUONErrorCounter* manuCounter;
      manuCounter = static_cast<AliMUONErrorCounter*>(fManuBuspatchTable->FindObject(busPatchId,manuId));
      if(eventCounter>0)occupancy = manuCounter->Events()/64/eventCounter;
      if(occupancy>1)
	{
	  cout << " !!! BIG WARNING: ManuId = " << manuId << " !!! in  BP = " << busPatchId << " occupancy (>1) = " << occupancy << endl;
	  (*fFilcout) << " !!! BIG WARNING: ManuId = " << manuId << " !!! in  BP = " << busPatchId << " occupancy (>1) = " << occupancy <<endl;
	}

      for (channelId = 0; channelId < ped->Size() ; ++channelId) 
	{
	  pedMean  = ped->ValueAsDouble(channelId, 0);

	  if (pedMean > 0) // connected channels
	    {
	      ped->SetValueAsDouble(channelId, 0, pedMean/(Double_t)eventCounter);
	      pedMean  = ped->ValueAsDouble(channelId, 0);
	      pedSigma = ped->ValueAsDouble(channelId, 1);
	      ped->SetValueAsDouble(channelId, 1, TMath::Sqrt(TMath::Abs(pedSigma/(Double_t)eventCounter - pedMean*pedMean)));
	      if(manuId == 0)
		{
		  ped->SetValueAsDouble(channelId, 0, ADCMax());
		  ped->SetValueAsDouble(channelId, 1, ADCMax());
		}
	      if(occupancy>1)
		{
		  ped->SetValueAsDouble(channelId, 0, ADCMax());
		  ped->SetValueAsDouble(channelId, 1, ADCMax());
		  if(channelId==0)ped->SetValueAsDouble(channelId, 0, ADCMax()+occupancy);
		}
	    }
	  else
	    {
	      ped->SetValueAsDouble(channelId, 0, ADCMax());
	      ped->SetValueAsDouble(channelId, 1, ADCMax());
	    }
	}
    }
}
//______________________________________________________________________________
void AliMUONPedestal::MakeASCIIoutput(ostream& out) const
{
  /// put pedestal store in the output stream

  out<<"//===========================================================================" << endl;
  out<<"//                 Pedestal file calculated by "<< fPrefixDA.Data() << endl;
  out<<"//===========================================================================" << endl;
  out<<"//       * Run           : " << fRunNumber << endl; 
  out<<"//       * Date          : " << fDate->AsString("l") <<endl;
  out<<"//       * Statictics    : " << fNEvents << endl;
  if(fConfig)
    out<<"//       * # of MANUS    : " << fNManuConfig << " read in the Det. config. " << endl;
  out<<"//       * # of MANUS    : " << fNManu << " read in raw data " << endl;
  out<<"//       * # of MANUS    : " << fNChannel/64 << " written in pedestal file " << endl;
  out<<"//       * # of channels : " << fNChannel << endl;
  if (fErrorBuspatchTable->GetSize())
    {
      out<<"//"<<endl;
      out<<"//    * Buspatches with less statistics (due to parity errors)"<<endl;
      TIter next(fErrorBuspatchTable->CreateIterator());
      AliMUONErrorCounter* parityerror;
      while((parityerror = static_cast<AliMUONErrorCounter*>(next())))
	{
	  out<<"//      BusPatch = "<<parityerror->BusPatch()<<"\t Nevents used = "<<fNEvents-parityerror->Events()<<endl;
	}
    }  

//   out<<"//"<<endl;
//   out<<"//    * Puzzling (Buspatch,Manu) read in raw data ?"<<endl;
  Int_t writitle=0;
  Int_t occupancy=1;
  if(occupancy)
    {
      TIter next(fPedestalStore ->CreateIterator());
      AliMUONVCalibParam* ped;
      while ( ( ped = dynamic_cast<AliMUONVCalibParam*>(next() ) ) )
	{
	  Int_t busPatchId = ped->ID0();
	  Int_t manuId = ped->ID1();
	  Double_t pedMean  = ped->ValueAsDouble(0, 0); // check pedestal value for channelId=0

	  if(pedMean>ADCMax()) 
	    {
	      writitle++;
	      if(writitle==1){ 
		out<<"//"<<endl;
		out<<"//    * Puzzling (Buspatch,Manu) read in raw data ?"<<endl;}
	      occupancy=TMath::Nint(pedMean-ADCMax());
	      ped->SetValueAsDouble(0, 0, ADCMax());
	      out<<"//      BusPatch = "<< busPatchId <<"\t ManuId =  "<< manuId << "\t occupancy = " << occupancy  <<endl;
	    }

	  if (manuId==0 || (fConfig && static_cast<AliMUONErrorCounter*>(fManuBPoutofconfigTable->FindObject(busPatchId,manuId))))
	    {
	      writitle++;
	      if(writitle==1){ 
		out<<"//"<<endl;
		out<<"//    * Puzzling (Buspatch,Manu) read in raw data ?"<<endl;}
	      out<<"//      BusPatch = "<< busPatchId <<"\t ManuId =  "<< manuId << "\t missing in the mapping" << endl;
	    }
	}
    }


  out<<"//"<<endl;
  out<<"//---------------------------------------------------------------------------" << endl;
  out<<"//---------------------------------------------------------------------------" << endl;
  out<<"//      BP     MANU     CH.      MEAN    SIGMA"<<endl;
  out<<"//---------------------------------------------------------------------------" << endl;

  // iterator over pedestal
  TIter next(fPedestalStore ->CreateIterator());
  AliMUONVCalibParam* ped;
  
  while ( ( ped = dynamic_cast<AliMUONVCalibParam*>(next() ) ) )
    {
      Int_t busPatchId = ped->ID0();
      Int_t manuId = ped->ID1();

      for ( Int_t channelId = 0; channelId < ped->Size(); ++channelId ) 
	{
	  Double_t pedMean  = ped->ValueAsDouble(channelId, 0);
	  Double_t pedSigma = ped->ValueAsDouble(channelId, 1);

	  out << "\t" << busPatchId << "\t" << manuId <<"\t"<< channelId << "\t" << pedMean <<"\t"<< pedSigma << endl;
	}
    }
}

//______________________________________________________________________________
void AliMUONPedestal::MakeControlHistos()
{
  /// Create control histograms
  if (fIndex>=0) return; // Pedestal run (fIndex=-1)

  Double_t pedMean;
  Double_t pedSigma;
  Int_t busPatchId;
  Int_t manuId;
  Int_t channelId;

// histo
  TFile*  histoFile = 0;
  TTree* tree = 0;
  TH1F* pedMeanHisto = 0;
  TH1F* pedSigmaHisto = 0;
    
  fHistoFileName=Form("%s.root",fPrefixDA.Data());
  histoFile = new TFile(fHistoFileName,"RECREATE","MUON Tracking pedestals");

  Char_t name[255];
  Char_t title[255];
  sprintf(name,"pedmean_allch");
  sprintf(title,"Pedestal mean all channels");
  Int_t nx = ADCMax()+1;
  Int_t xmin = 0;
  Int_t xmax = ADCMax(); 
  pedMeanHisto = new TH1F(name,title,nx,xmin,xmax);
  pedMeanHisto->SetDirectory(histoFile);

  sprintf(name,"pedsigma_allch");
  sprintf(title,"Pedestal sigma all channels");
  nx = 201;
  xmin = 0;
  xmax = 200; 
  pedSigmaHisto = new TH1F(name,title,nx,xmin,xmax);
  pedSigmaHisto->SetDirectory(histoFile);

  tree = new TTree("t","Pedestal tree");
  tree->Branch("bp",&busPatchId,"bp/I");
  tree->Branch("manu",&manuId,",manu/I");
  tree->Branch("channel",&channelId,",channel/I");
  tree->Branch("pedMean",&pedMean,",pedMean/D");
  tree->Branch("pedSigma",&pedSigma,",pedSigma/D");

  // iterator over pedestal
  TIter next(fPedestalStore ->CreateIterator());
  AliMUONVCalibParam* ped;
  
  while ( ( ped = dynamic_cast<AliMUONVCalibParam*>(next() ) ) )
  {
    busPatchId = ped->ID0();
    manuId = ped->ID1();
    
    for ( channelId = 0; channelId < ped->Size(); ++channelId ) 
    {
      pedMean  = ped->ValueAsDouble(channelId, 0);
      pedSigma = ped->ValueAsDouble(channelId, 1);
          
      pedMeanHisto->Fill(pedMean);
      pedSigmaHisto->Fill(pedSigma);
      tree->Fill();  
    }
  }
    
  histoFile->Write();  
  histoFile->Close(); 

}
