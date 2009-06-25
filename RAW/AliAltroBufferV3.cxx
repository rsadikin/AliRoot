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

// Interface to the Altro format
// to read and write digits
// To be used in Alice Data Challenges 
// and in the compression of the RAW data

#include "AliAltroBufferV3.h"
#include "AliRawDataHeaderSim.h"
#include "AliLog.h"
#include "AliFstream.h"

ClassImp(AliAltroBufferV3)

//_____________________________________________________________________________
AliAltroBufferV3::AliAltroBufferV3(const char* fileName, AliAltroMapping *mapping):
AliAltroBuffer(fileName,mapping),
  fN(0)
{
  // Constructor
}

//_____________________________________________________________________________
AliAltroBufferV3::~AliAltroBufferV3()
{
// destructor

  if (fVerbose) Info("~AliAltroBufferV3", "File Created");

  delete fFile;

}

//_____________________________________________________________________________
AliAltroBufferV3::AliAltroBufferV3(const AliAltroBufferV3& source):
  AliAltroBuffer(source),
  fN(source.fN)
{
// Copy Constructor

  Fatal("AliAltroBufferV3", "copy constructor not implemented");
}

//_____________________________________________________________________________
AliAltroBufferV3& AliAltroBufferV3::operator = (const AliAltroBufferV3& /*source*/)
{
//Assigment operator

  Fatal("operator =", "assignment operator not implemented");
  return *this;
}

//_____________________________________________________________________________
void AliAltroBufferV3::FillBuffer(Int_t val)
{
//Fills the Buffer with 16 ten bits words and write into a file 

  if ((val > 0x3FF) || (val < 0)) {
    Error("FillBuffer", "Value out of range (10 bits): %d", val);
    val = 0x3FF;
  }

  if (fN >= (kMaxWords-1)) {
    Error("FillBuffer","Altro channel can't have more than 1024 10-bit words!");
    return;
  }

  fArray[fN++] = val;
}

//_____________________________________________________________________________
void AliAltroBufferV3::WriteTrailer(Int_t wordsNumber, Short_t hwAddress)
{
  //Writes a trailer (header) of 32 bits using
  //a given hardware adress
  UInt_t temp = hwAddress & 0xFFF;
  temp = (wordsNumber << 16) & 0x3FF;
  temp |= (0x1 << 30);

  fFile->WriteBuffer((char *)(&temp),sizeof(UInt_t));

  ReverseAndWrite();
}

//_____________________________________________________________________________
void AliAltroBufferV3::ReverseAndWrite()
{
  // Reverse the altro data order and
  // write the buffer to the file
  UInt_t temp = 0;
  Int_t shift = 20;
  for(Int_t i = fN; i >= 0; i--) {
    temp |= (fArray[i] << shift);
    shift -= 10;
    if (shift < 0) {
      fFile->WriteBuffer((char *)(&temp),sizeof(UInt_t));
      temp = 0;
      shift = 20;
    }
  }

  if (shift != 20) {
    fFile->WriteBuffer((char *)(&temp),sizeof(UInt_t));
  }

  fN = 0;
}

//_____________________________________________________________________________
void AliAltroBufferV3::WriteRCUTrailer(Int_t rcuId)
{
  // Writes the RCU trailer
  // rcuId the is serial number of the corresponding
  // RCU. The basic format of the trailer can be
  // found in the RCU manual.
  // This method should be called at the end of
  // raw data writing.

  UInt_t currentFilePos = fFile->Tellp();
  UInt_t size = currentFilePos-fDataHeaderPos;
  size -= sizeof(AliRawDataHeader);
  
  if ((size % 5) != 0) {
    AliFatal(Form("The current raw data payload is not a mutiple of 5 (%d) ! Can not write the RCU trailer !",size));
    return;
  }

  // Now put the size in unit of number of 40bit words
  size /= 5;
  fFile->WriteBuffer((char *)(&size),sizeof(UInt_t));

  // Now several not yet full defined fields
  // In principle they are supposed to contain
  // information about the sampling frequency,
  // L1 phase, list of 'dead' FECs, etc.
  //  UInt_t buffer[n];
  //  fFile->WriteBuffer((char *)(buffer),sizeof(UInt_t)*n);
  
  //  Now the RCU identifier and size of the trailer
  //  FOr the moment the triler size is 2 32-bit words
  UInt_t buffer = (2 & 0x7F);
  buffer |= ((rcuId & 0x1FF) << 7);
  buffer |= 0xAAAA << 16;
  fFile->WriteBuffer((char *)(&buffer),sizeof(UInt_t));

}
