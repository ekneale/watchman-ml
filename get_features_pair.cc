// Extracts the MC and pmt hit information from events.
// Outputs relevant data to root file.
// Also outputs MC and pmt hit data for pairs of events to csvfile for use
// with vertex_reconstruction*.py
// Author: Elisabeth Kneale, November 2020
// Adapted in part from bonsai.cc (M.Smy) for rat-pac
// To compile (requires Makefile):
// make get_features_pair
// To run:
// ./get_features_pair infile.root outfile.root outfile.csv
// (where infile.root is raw rat output)

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#include <RAT/DS/Run.hh>
#include <RAT/DS/PMTInfo.hh>
#include <RAT/DS/Root.hh>
#include <RAT/DS/MC.hh>
#include <RAT/DS/MCParticle.hh>
#include <RAT/DS/EV.hh>
#include <RAT/DS/PMT.hh>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>

//Need to separate the Inner-Detector tubes from the Outer-Detector tubes
static const int innerPMTcode = 1;
static const int vetoPMTcode  = 2;


int main(int argc, char** argv)
{

  // check if minimum arguments exist
  if (argc<5)
    {
      printf("Less than the required number of arguments\n");
      return -1;
    }
 
  int detector_threshold = 10;
  int charge_threshold = 0.25;
  int id;
    
  Int_t    gtid=0, mcid=0, subid=0, tot_nhit=0, vetoHit=0;
  Int_t    innerHit=0,innerHitPrev=0,vetoHitPrev=0,triggers=0;

  Double_t totPE=0., totPEprev = 0., innerPE=0., vetoPE=0.;
  Double_t trueX=0., trueY=0., trueZ=0., trueT=0., trueU=0., trueV=0., trueW=0.;
  Double_t trueXprev=0., trueYprev=0., trueZprev=0., trueTprev=0., trueUprev=0., trueVprev=0., trueWprev=0.;
  Double_t trueEnergy=0., trueEnergyPrev=0.;
  Double_t timestamp=0., timestampPrev=0., dt_sub=0., dtPrev_us=0.;
  Int_t sub_event_tally[20] = {};
  Double_t pmtBoundR=0.,pmtBoundZ=0.;

  // root stuff
  TFile *f;
  TTree *rat_tree,*run_tree,*data;
  Int_t n_events;
  TTree *run_summary;

  // rat stuff
  RAT::DS::Root *ds=new RAT::DS::Root();
  RAT::DS::Run  *run=new RAT::DS::Run();
  RAT::DS::EV *ev;
  RAT::DS::PMTInfo *pmtinfo;
  RAT::DS::MC *mc;
  RAT::DS::MCParticle *prim;
  RAT::DS::PMT *pmt;


  // BONSAI stuff
  vector< vector<float> >       hitpmt; // assumes that there will be no more than 500 hit pmts in an event
  vector< vector<float> >       hitpmtPrev;
  int         event,sub_event,n,count;
  int         inpmt,vetopmt;
  int         pmtindex,hit,nhit;

  // open input file
  f= new TFile(argv[1]);

  rat_tree=(TTree*) f->Get("T");
  rat_tree->SetBranchAddress("ds", &ds);
  run_tree=(TTree*) f->Get("runT");
  if (rat_tree==0x0 || run_tree==0x0)
    {
      printf("can't find trees T and runT in this file\n");
      return -1;
    }
  run_tree->SetBranchAddress("run", &run);
  if (run_tree->GetEntries() != 1)
    {
      printf("More than one run! Ignoring all but the geometry for the first run\n");
      //return -1;
    }

  // open output files
  TFile *out=new TFile(argv[2],"RECREATE");
  data=new TTree("data","low-energy detector-triggered events");
  ofstream mc_csvfile;
  mc_csvfile.open (argv[3],ofstream::trunc);
  mc_csvfile << "# trueXprompt,  trueYprompt,  trueZprompt, trueTprompt,  trueUprompt,  trueVprompt,  trueWprompt \n";
  ofstream hit_csvfile;
  hit_csvfile.open (argv[4],ofstream::trunc);
  hit_csvfile << "# [x, y, z, t, q]^T for each hit for each sub-event: \n# 5 rows (features) for each sub-event record \n# one column for each hit per 5 rows\n";
  

  //Define the Integer Tree Leaves
  data->Branch("gtid",&gtid,"gtid/I");
  data->Branch("mcid",&mcid,"mcid/I");
  data->Branch("subid",&subid,"subid/I");
  data->Branch("innerHit",&innerHit,"innerHit/I");//inner detector    
  data->Branch("innerHitPrev",&innerHitPrev,"innerHitPrev/I");//inner detector
  data->Branch("vetoHit",&vetoHit,"vetoHit/I");//veto detector
  data->Branch("vetoHitPrev",&vetoHitPrev,"vetoHitPrev/I");//veto detector
  //Define the double Tree Leaves
  data->Branch("pe",&totPE,"pe/D");
  data->Branch("innerPE",&innerPE,"innerPE/D");
  data->Branch("vetoPE",&vetoPE,"vetoPE/D");
  data->Branch("trueEnergy",&trueEnergy,"trueEnergy/D");
  data->Branch("trueEnergyPrev",&trueEnergyPrev,"trueEnergyPrev/D");
  data->Branch("trueX",&trueX,"trueX/D"); data->Branch("trueY",&trueY,"trueY/D");
  data->Branch("trueZ",&trueZ,"trueZ/D"); data->Branch("trueT",&trueT,"trueT/D");
  data->Branch("trueXprev",&trueXprev,"trueXprev/D"); data->Branch("trueYprev",&trueYprev,"trueYprev/D");
  data->Branch("trueZprev",&trueZprev,"trueZprev/D"); data->Branch("trueTprev",&trueTprev,"trueTprev/D");
  data->Branch("trueU",&trueU,"trueU/D"); data->Branch("trueV",&trueV,"trueV/D");
  data->Branch("trueW",&trueW,"trueW/D"); 
  data->Branch("trueUprev",&trueUprev,"trueUprev/D"); data->Branch("trueVprev",&trueVprev,"trueVprev/D");
  data->Branch("trueWprev",&trueWprev,"trueWprev/D"); 
  
  data->Branch("dt_sub", &dt_sub, "dt_sub/D"); //time of the sub-event trigger from start of the event mc
  data->Branch("dtPrev_us",&dtPrev_us,"dtPrev_us/D"); //global time between consecutive events in us
  data->Branch("timestamp",&timestamp,"timestamp/D"); //trigger time of sub event from start of run
  data->Branch("timestampPrev",&timestampPrev,"timestampPrev/D"); //trigger time of sub event from start of run


  run_summary=new TTree("runSummary","mc run summary");
  run_summary->Branch("nEvents",&n_events,"nEvents/I");
  run_summary->Branch("subEventTally",sub_event_tally,"subEventTally[20]/I");

  run_tree->GetEntry(0);


  // loop over PMTs and find positions and location of PMT support
  pmtinfo=run->GetPMTInfo();
  n=pmtinfo->GetPMTCount();
  inpmt = 0; vetopmt =0;

  //Determines the number of inner and veto pmts
  for(pmtindex=0; pmtindex<n; pmtindex++)
    {
      if (pmtinfo->GetType(pmtindex)==innerPMTcode)     ++inpmt;
      else if (pmtinfo->GetType(pmtindex)==vetoPMTcode) ++vetopmt;
      else
	printf("PMT does not have valid identifier: %d \n",
	       pmtinfo->GetType(pmtindex));
    }
  if (n != (inpmt+vetopmt))
    printf("Mis-match in total PMT numbers: %d, %d \n",n, inpmt+vetopmt);
    

  // get pmt information
  {
    float xyz[3*inpmt+1];

    printf("In total there are  %d PMTs in WATCHMAN\n",n);
    
    for(pmtindex=count=0; pmtindex<n; pmtindex++)
      {
	if(pmtinfo->GetType(pmtindex)==innerPMTcode)
	  {
	    TVector3 pos=pmtinfo->GetPosition(pmtindex);
	    xyz[3*count]=pos[0]*0.1;
	    xyz[3*count+1]=pos[1]*0.1;
	    xyz[3*count+2]=pos[2]*0.1;
	    if (pos[0]>pmtBoundR) pmtBoundR = pos[0];
	    if (pos[2]>pmtBoundZ) pmtBoundZ = pos[2];
	    ++count;
	  }
      }
    
    printf("There are %d inner pmts and %d veto pmts \n ",inpmt,vetopmt);
    printf("Inner PMT boundary (r,z):(%4.1f mm %4.1f, mm)\n",pmtBoundR,pmtBoundZ);

    if (count!= inpmt)
      printf("There is a descrepancy in inner PMTS %d vs %d",count,inpmt);

  }

  n_events = rat_tree->GetEntries();
  // loop over all events
  for (event = 0; event < n_events; event++)
    {
      if (event%1000==0)
        printf("Evaluating event %d of %d (%d sub events)\n",event,n_events,
	      ds->GetEVCount());
      rat_tree->GetEntry(event);


      sub_event_tally[ds->GetEVCount()]++;
      // loop over all subevents
      for(sub_event=0;sub_event<ds->GetEVCount();sub_event++)
        {
        gtid += 1;
        mcid = event;
        subid = sub_event;
     
        ev = ds->GetEV(sub_event);
        totPE = ev->GetTotalCharge();

        trueEnergyPrev = trueEnergy;
        trueXprev = trueX;trueYprev=trueY;trueZprev=trueZ;

        TVector3 temp;
      
        mc = ds->GetMC();
        prim = mc->GetMCParticle(sub_event); 
        trueEnergy = prim->GetKE();
        temp = prim->GetPosition();
        trueX = temp.X()*0.1;
        trueY = temp.Y()*0.1;
        trueZ = temp.Z()*0.1;
        trueT    = prim->GetTime(); // local emission time
        if (subid>0)
          {
          temp = prim->GetEndPosition();
          trueX = temp.X()*0.1; 
          trueY = temp.Y()*0.1; 
          trueZ = temp.Z()*0.1; 
          trueT    = prim->GetEndTime(); // should be the time of the neutron capture, may cause an issue with re-triggers
          }
        // get true event timings
        // times are in ns unless specified
        timestamp = 1e6*mc->GetUTC().GetSec() + 1e-3*mc->GetUTC().GetNanoSec() + 1e-3*ev->GetCalibratedTriggerTime() - 1e6*run->GetStartTime().GetSec()-1e-3*run->GetStartTime().GetNanoSec(); //global time of subevent trigger (us)
        dtPrev_us = timestamp-timestampPrev; //time since the previous trigger (us)
        dt_sub = ev->GetCalibratedTriggerTime(); //trigger time (first pmt hit time) from start of event mc

        nhit=ev->GetPMTCount();

        // loop over all PMT hits for each subevent
        innerPE=0;vetoPE=0;    
        for(hit=innerHit=vetoHit=0; hit<nhit; hit++)
          {
          pmt=ev->GetPMT(hit);
          id = pmt->GetID();
          //only use information from the inner pmts
          if(pmtinfo->GetType(id) == innerPMTcode)
            {
            vector <float> hittemp;
            TVector3 pos=pmtinfo->GetPosition(id);
            hittemp.push_back(pos[0]*0.1);       //x
            hittemp.push_back(pos[1]*0.1);       //y
            hittemp.push_back(pos[2]*0.1);       //z
            hittemp.push_back(pmt->GetTime());   //t
            hittemp.push_back(pmt->GetCharge()); //q
            innerPE += pmt->GetCharge();
            innerHit++;
	    hitpmt.push_back(hittemp);
            }
          else if(pmtinfo->GetType(id)== vetoPMTcode)
            {
            vetoPE += pmt->GetCharge();    
            vetoHit++;
            }
          else
            printf("Unidentified PMT type: (%d,%d) \n",count,pmtinfo->GetType(id));
          } // end of loop over all PMT hits
        if (innerHit<detector_threshold)
          {
          break; // do not look at the neutron if the positron didn't pass the trigger
          }
        triggers++;

        // get momentum vector and normalize it (find direction)
        temp = prim->GetMomentum();
        temp = temp.Unit();
        trueU = temp.X();trueV = temp.Y();trueW = temp.Z();

	// write the pair data to the csvfiles (only if true pair)
	if (subid==1 && dtPrev_us < 500)
          {
          data->Fill();
	  // write the hit data to the csvfile
	  for(int feature=0;feature<5;feature++)
            {
            for(hit=0;hit<innerHitPrev;hit++)
              {
              hit_csvfile << hitpmtPrev[hit][feature] << ",";
              }
            }

	  for(int feature=0;feature<5;feature++)
            {
            for(hit=0; hit<innerHit; hit++)
              {
              if (hit==innerHit-1) hit_csvfile << hitpmt[hit][feature];
	      else hit_csvfile << hitpmt[hit][feature] << ",";
              }
            hit_csvfile << "\n";
  	    }

          // write the mc data to the other csvfile
          mc_csvfile << trueXprev << "," << trueYprev << "," << trueZprev << "," << trueTprev << "," << trueUprev << "," << trueVprev << "," << trueWprev << "\n";
          }

        //save reference values for the next subevent
        trueEnergyPrev = trueEnergy;
        trueXprev = trueX; trueYprev = trueY; trueZprev = trueZ;
	trueUprev = trueU; trueVprev = trueV; trueWprev = trueW;
	trueTprev = trueT;
	totPEprev = totPE;
        timestampPrev = timestamp;
        innerHitPrev = innerHit;
        vetoHitPrev = vetoHit;
	hitpmtPrev = hitpmt;
	hitpmt.clear();
      } 
    }
  cout << triggers << " triggered events" << endl;
  out->cd();
  data->Write();
  run_summary->Fill();
  run_summary->Write();
  out->Close();
  return 0;
}

