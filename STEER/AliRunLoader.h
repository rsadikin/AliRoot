#ifndef ALIRUNLoader_H
#define ALIRUNLoader_H

//This class aims to be the only one interface for manging data
//It stores Loaders for all modules which knows the filenames 
//of the data files to be stored.
//It aims to substitude AliRun in automatic maging of data positioning
//thus there won't be necessity of loading gAlice from file in order to 
//get fast ccess to the data
//
//logical place for putting the Loader specific to the given detector is detector itself
// but, to load detector one need to load gAlice, and by the way all other detectors
// with their geometrieces and so on. 
// So, if one need to open TPC clusters there is no principal nedd to read everything

#include <TNamed.h>
#include <TFile.h>
#include "AliConfig.h"

class TString;
class TFolder;
class TObjArray;
class TTree;
class TTask;
class TParticle;

class AliRun;
class AliLoader;
class AliDetector;
class AliHeader;
class AliStack;
class AliRunDigitizer;


class AliRunLoader: public TNamed
{
  public:
    
    AliRunLoader();
    AliRunLoader(const char* topfoldername);
    AliRunLoader(TFolder* topfolder);
    
    virtual ~AliRunLoader();
    
    static AliRunLoader* Open(const char* filename = "galice.root",
                              const char* eventfoldername = AliConfig::fgkDefaultEventFolderName,
	          Option_t* option = "READ");

    
    Int_t       GetEventNumber() const {return fCurrentEvent;}

    Int_t       GetEvent(Int_t evno);//sets the event number and reloads data in folders properly
    Int_t       GetNextEvent(){return GetEvent(fCurrentEvent+1);}//gets next event 
    Int_t       SetEventNumber(Int_t evno); //cleans folders and sets the root dirs in files (do not reload data)
    Int_t       SetNextEvent(){return SetEventNumber(fCurrentEvent+1);}
    
    Int_t       GetNumberOfEvents();
    
    void        MakeTree(Option_t *option);
    void        MakeHeader();
    void        MakeStack();
    
    Int_t       LoadgAlice();
    Int_t       LoadHeader();
    Int_t       LoadKinematics(Option_t* option = "READ");
    Int_t       LoadTrackRefs(Option_t* option = "READ");
    
    void        UnloadHeader();
    void        UnloadKinematics();
    void        UnloadgAlice();
    void        UnloadTrackRefs();
    
    void        SetKineFileName(const TString& fname){fKineFileName = fname;}
    void        SetTrackRefsFileName(const TString& fname){fTrackRefsFileName = fname;}
    
    TTree*      TreeE() const; //returns the tree from folder; shortcut method
    AliHeader*  GetHeader() const;
    
    AliStack*   Stack() const {return fStack;}
    
    TTree*      TreeK() const; //returns the tree from folder; shortcut method
    TTree*      TreeTR() const; //returns the tree from folder; shortcut method    
    
    AliRun*     GetAliRun()const;
        
    Int_t       WriteGeometry(Option_t* opt="");
    Int_t       WriteHeader(Option_t* opt="");
    Int_t       WriteAliRun(Option_t* opt="");
    Int_t       WriteKinematics(Option_t* opt="");
    Int_t       WriteTrackRefs(Option_t* opt="");
    Int_t       WriteRunLoader(Option_t* opt="");
    
    Int_t       WriteHits(Option_t* opt=""); 
    Int_t       WriteSDigits(Option_t* opt="");
    Int_t       WriteDigits(Option_t* opt="");
    Int_t       WriteRecPoints(Option_t* opt="");
    Int_t       WriteTracks(Option_t* opt="");
    
    Int_t       LoadHits(Option_t* detectors = "all",Option_t* opt = "RAED");
    Int_t       LoadSDigits(Option_t* detectors = "all",Option_t* opt = "RAED");
    Int_t       LoadDigits(Option_t* detectors = "all",Option_t* opt = "RAED");
    Int_t       LoadRecPoints(Option_t* detectors = "all",Option_t* opt = "RAED");
    Int_t       LoadTracks(Option_t* detectors = "all",Option_t* opt = "RAED");
    
    
    void        AddLoader(AliLoader* loader);
    void        AddLoader(AliDetector* det);
    AliLoader*  GetLoader(const char* detname) const;
    AliLoader*  GetLoader(AliDetector* det) const;
    Int_t       SetEventFolderName(const TString& name = AliConfig::fgkDefaultEventFolderName);//sets top folder name for this run; of alread
    void        CleanFolders();//removes all abjects from folder structure
    void        CleanDetectors();
    void        CleanKinematics(){Clean(fgkKineContainerName);}
    void        CleanTrackRefs(){Clean(fgkTrackRefsContainerName);}
    
    void        RemoveEventFolder(); //remove folder structure from top folder 
    void        SetCompressionLevel(Int_t cl);

    TFolder*    GetEventFolder() const {return fEventFolder;}
    void        CdGAFile();

    void        MakeTrackRefsContainer();
    void        SetDirName(TString& dirname);
    Int_t       GetFileOffset() const;
    void        SetNumberOfEventsPerFile(Int_t nevpf){fNEventsPerFile = nevpf;}
    
    void        SetDigitsFileNameSuffix(const TString& suffix);//adds the suffix before ".root", 
                                                               //e.g. TPC.Digits.root -> TPC.DigitsMerged.root
                                                               //made on Jiri Chudoba demand

    const TObjArray* GetArrayOfLoaders() const {return fLoaders;}
    
    
  protected:
    TObjArray     *fLoaders;          //  List of Detectors
    TFolder       *fEventFolder;      //!top folder for this run
    
    Int_t          fCurrentEvent;//!Number of current event
    
    TFile         *fGAFile;//!
    AliHeader     *fHeader;//!
    AliStack      *fStack;//!
    
//    AliLoaderDataInfo *fKine;//data connected to kinematics data
//    AliLoaderDataInfo *fRefsFile;//data connected to track reference data
    
    TFile*         fKineFile;    //!pointer to file with kinematics
    TDirectory*    fKineDir;     //!pointer to Dir with current event in Kine File
    TString        fKineFileName;//name of file with hits

    TFile*         fTrackRefsFile;    //!pointer to file with kinematics
    TDirectory*    fTrackRefsDir;     //!pointer to Dir with current event in Kine File
    TString        fTrackRefsFileName;//name of file with hits
    
    Int_t          fNEventsPerFile;
    
    static const TString   fgkDefaultKineFileName;//default file name with kinamatics
    static const TString   fgkDefaultTrackRefsFileName;//default file name with kinamatics
    
    void           SetGAliceFile(TFile* gafile);//sets the pointer to gAlice file
    Int_t          PostKinematics();
    Int_t          PostTrackRefs();
    Int_t          OpenKineFile(Option_t* opt);
    Int_t          OpenTrackRefsFile(Option_t* opt);

    Int_t          OpenDataFile(const TString& filename,TFile*& file,TDirectory*& dir,Option_t* opt);
    void           SetUnixDir(const TString& udirname);
    const TString  SetFileOffset(const TString& fname);//adds the proper number before .root
  private:
    void  GetListOfDetectors(const char * namelist,TObjArray& pointerarray) const;

    void  CleanHeader(){Clean(fgkHeaderContainerName);}
    void  Clean(const TString& name);
    
    Int_t SetEvent();
  public:
  /******************************************/
  /*****   Public S T A T I C Stuff   *******/
  /******************************************/
    static AliRunLoader* GetRunLoader(const char* eventfoldername);

//    static AliRunDigitizer* GetRunDigitizer();
    static TTask*           GetRunDigitizer();
    static TTask*           GetRunSDigitizer();
    static TTask*           GetRunReconstructioner();
    static TTask*           GetRunTracker();
    static TTask*           GetRunQATask();
    
    static const TString   fgkRunLoaderName;
    static const TString   fgkHeaderContainerName;//default name of the kinematics container (TREE) name - TreeE
    static const TString   fgkKineContainerName;//default name of the kinematics container (TREE) name - TreeK
    static const TString   fgkTrackRefsContainerName;//default name of the track references container (TREE) name - TreeTR
    static const TString   fgkHeaderBranchName;
    static const TString   fgkKineBranchName;
    static const TString   fgkGAliceName;
    
    ClassDef(AliRunLoader,1)
};

#endif
