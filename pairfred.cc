#include <iostream>
#include <iomanip>

using namespace std;

#include <RAT/DS/Run.hh>
#include <RAT/DS/PMTInfo.hh>
#include <RAT/DS/Root.hh>
#include <RAT/DS/MC.hh>
#include <RAT/DS/MCParticle.hh>
#include <RAT/DS/EV.hh>
#include <RAT/DS/PMT.hh>
#include <RAT/DS/PathFit.hh>
#include <RAT/DS/BonsaiFit.hh>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TApplication.h>

#include <TRandom.h>
#include <TRandom3.h>

#include "ariadne.h"
#include "azimuth_ks.h"
#include "distpmt.h"
#include <pmt_geometry.h>
#include <goodness.h>
#include <searchgrid.h>
#include <fourhitgrid.h>
#include <combinedgrid.h>

//Need to separate the Inner-Detector tubes from the Outer-Detector tubes
static const int innerPMTcode = 1;
static const int vetoPMTcode  = 2;

extern "C"{
void lfariadn2_(float *av,int *anhit,float *apmt,float *adir, float *amsg, float *aratio,int *anscat,float *acosscat);
}

int nwin(RAT::DS::PMTInfo *pmtinfo,
	 float twin,float *v,int nfit, int *cfit, float *tfit, int *cwin);

TVector3 Fitting_Likelihood_Ascent(RAT::DS::Root* ds, RAT::DS::PMTInfo* pmtinfo, RAT::DS::EV *ev ,TH2F *PDF){

  //---- Reconstructed position vector
  TVector3 Position = {-1e9,-1e9,-1e9};
  TRandom3* gRandom = new TRandom3();

  if (ds->GetEVCount()){

    //---- Define some variables. 
    float_t Likelihood_Best = -1e10; float Jump = 1000.0; int iWalk_Max = 70; TVector3 Test_Vertex = {0,0,0}; TVector3 Best_Vertex; float Start = 3000.0; TVector3 End_Vector = {0,0,0}; vector<TVector3> Vector_List = {{0,0,0},{Start,0,0},{-Start,0,0},{0,Start,0},{0,-Start,0},{0,0,Start},{0,0,-Start},{Start,Start,0},{Start,-Start,0},{-Start,Start,0},{-Start,-Start,0},{Start,Start,Start},{Start,-Start,Start},{-Start,Start,Start},{-Start,-Start,Start},{Start,Start,-Start},{Start,-Start,-Start},{-Start,Start,-Start},{-Start,-Start,-Start}};

    //---- Go for a walk...
    for(int iWalk = 0; iWalk < iWalk_Max; iWalk++){
      if (iWalk < 19){Test_Vertex = Vector_List[iWalk];}
      else if (iWalk == 19){gRandom->Sphere(End_Vector[0],End_Vector[1],End_Vector[2],Jump); Test_Vertex = Position + End_Vector;}

      float_t Likelihood = 0;

      for(long iPMT = 0; iPMT < ev->GetPMTCount(); iPMT++ ){
        int PMT_ID             = ev->GetPMT(iPMT)->GetID();
        int pmt_type           = pmtinfo->GetType(PMT_ID);
        if( pmt_type != 1 ) continue;
        TVector3 PMT_Position  = pmtinfo->GetPosition(PMT_ID);
        TVector3 PMT_Direction = pmtinfo->GetDirection(PMT_ID);
        TVector3 R_Test_Vector = Test_Vertex - PMT_Position;
        float_t Angle          = cos(R_Test_Vector.Angle(PMT_Direction));
        Likelihood += ev->GetPMT(iPMT)->GetCharge()*log(PDF->GetBinContent(PDF->FindBin(R_Test_Vector.Mag(),Angle)));
      }

      for (long ipmt = 0; ipmt < pmtinfo->GetPMTCount(); ipmt++){
        int pmt_type           = pmtinfo->GetType(ipmt);
        if( pmt_type != 1 ) continue;
        TVector3 PMT_Position  = pmtinfo->GetPosition(ipmt);
        TVector3 PMT_Direction = pmtinfo->GetDirection(ipmt);
        TVector3 R_Test_Vector = Test_Vertex - PMT_Position;
        float_t Angle          = cos(R_Test_Vector.Angle(PMT_Direction));
        Likelihood -= PDF->GetBinContent(PDF->FindBin(R_Test_Vector.Mag(),Angle));
      }

      //---- If we find a test vertex with a larger likelihood, that is the new reconstructed position 
      if (Likelihood > Likelihood_Best){Likelihood_Best = Likelihood; iWalk--; Jump=Jump/1.05; Position = Test_Vertex;
        if (End_Vector[0] != 0 && End_Vector[1] !=0 && End_Vector[2] !=0){Test_Vertex = Position + End_Vector;}
        else {gRandom->Sphere(End_Vector[0],End_Vector[1],End_Vector[2],Jump); Test_Vertex = Position + End_Vector;}
      }
      else{gRandom->Sphere(End_Vector[0],End_Vector[1],End_Vector[2],Jump); Test_Vertex = Position + End_Vector;}
    }
  }

  //---- After scanning for the number of iterations defined, the position with the largest likelihood gives the reconstructed position
  return Position;
}

int main(int argc, char **argv)
{
  float darkNoise,offsetT,minT, maxT;
  char do_clusfit,do_QFit,useAngle;
  int nsct,detector_threshold;
  int crash_count=0;
  int tot_inner,tot_veto,id;

  printf("\n\nWelcome to FRED (Functions to Reconstruct Events in the Detector). The function can take no less than two input and up\n");
  printf("to 11 inputs: infile,outfile,darkNoise,detector_threshold,time_nX,useAngle,timeOffset,minTime,maxTime,do_clusfit,do_QFIT\n\n");
  printf("%d input arguments in function bonsai.\n",argc);
  
  // check if minimum arguments exist
  if (argc<3)
    {
      printf("Less than the required number of arguments\n");
      return -1;
    }
  // set default values
  darkNoise = 3000.; //As agreed for Path A/ Path B comparison
  offsetT   = 800.;
  minT      = -500.;
  maxT      = 1000.;
  useAngle  = 1;
  float time_nX = 9;
  do_clusfit = 0;
  do_QFit   = 0;
  detector_threshold = 9; 
  switch(argc)
    {
    case 3:
      printf("Only input file and output file are provided. All other values set to default.\n");
      break;
    case 4:
      printf("Only input file and output file and dark noise rate are provided. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));  
      break;
    case 5:
      // printf("Only input file and output file are provide. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      break;
    case 6:
      // printf("Only input file and output file are provided. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      break;
    case 7:
      // printf("Only input file and output file are provide. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =         atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      break;
    case 8:
      // printf("Only input file and output file are provide. All other values set to default.\n");
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =         atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT  = float(strtol(argv[7],NULL,10));
      break;
    case 9:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =         atoi(argv[6]);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      break;
    case 10:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold  = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =          atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      maxT      = float(strtol(argv[9],NULL,10));
      break;
    case 11:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX  = float(strtol(argv[5],NULL,10));
      useAngle =          atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      maxT      = float(strtol(argv[9],NULL,10));
      do_clusfit =       atoi(argv[10]);//sscanf(argv[10],"%d",&do_clusfit);
      break;
    case 12:
      darkNoise = float(strtol(argv[3],NULL,10));
      detector_threshold = atoi(argv[4]);
      time_nX   = float(strtol(argv[5],NULL,10));
      useAngle  =         atoi(argv[6]);//sscanf(argv[6],"%d",&useAngle);
      offsetT   = float(strtol(argv[7],NULL,10));
      minT      = float(strtol(argv[8],NULL,10));
      maxT      = float(strtol(argv[9],NULL,10));
      do_clusfit =       atoi(argv[10]);//sscanf(argv[10],"%d",&do_clusfit);
      do_QFit =          atoi(argv[11]);//sscanf(argv[11],"%d",&do_QFit);
      break;
  }
  
  printf("\n\nUsing:\n");
  printf("(1) Infile \t%20s\n",argv[1]); 
  printf("(2) Outfile \t%20s\n",argv[2]);
  printf("(3) darkNoise \t%20.1f\n", darkNoise); 
  printf("(4) detector_threshold \t%12d\n",detector_threshold);
  printf("(5) time_nX \t%20.1f\n",time_nX);
  printf("(6) useAngle \t%20d\n",useAngle);
  printf("(7) offsetT \t%20.1f\n",offsetT);
  printf("(8) minT \t%20.1f\n",minT); 
  printf("(9) maxT  \t%20.1f\n",maxT);
  printf("(10) do clusft \t %19d\n",do_clusfit);
  printf("(11) do qfit \t %19d\n\n",do_QFit);

 
  TVector3 Best_Fit;
  TFile *fPDF      = new TFile("PDF.root");
  if(!fPDF->IsOpen()){
    if(do_QFit){
      printf("PDF.root file does not exist. Exiting program. Please\ncopy over a PDF.root or turn QFit option off.\n");
      return -1;
    }
  }
  TH2F *PDF    = (TH2F*)fPDF->Get("h_R_Cos_Theta");
 
    
  Int_t    gtid=0, mcid=0, subid=0, tot_nhit=0, nhits=0, veto_hit=0;
  Int_t    totVHIT=0,inner_hit=0,inner_hit_prev=0,veto_hit_prev=0;

  Int_t    nsel=0;
  Double_t n9=0., nX = 0., nOff = 0., n100 = 0., n400 = 0., bonsai_goodness=0., dir_goodness=0., azi_ks;
  Double_t n9_prev = 0., nX_prev = 0., n100_prev = 0., n400_prev = 0.;
  Double_t bonsai_goodness_prev = 0., dir_goodness_prev = 0., azi_ks_prev = 0.;
  Double_t clusfit_goodness=0.;

  Double_t totPE=0., innerPE=0., vetoPE=0.;
  Double_t dist_pmt;
  double   wall[3];
  Double_t x=0., y=0., z=0., t=0., t_prev=0., u=0., v=0., w=0.;
  Double_t cx=0., cy=0., cz=0., ct=0.;
  Double_t mc_x=0., mc_y=0., mc_z=0., mct=0., mc_u=0., mc_v=0., mc_w=0.;
  Double_t mc_x_prev=0., mc_y_prev=0., mc_z_prev=0.;
  Double_t closestPMT=0.,mc_energy=0.,mc_energy_prev;
  Double_t dxx=0.,dyy=0.,dzz=0.,drr=0.,dxmcx=0.,dymcy=0.,dzmcz=0.,drmcr=0.;
  Double_t prev_x = -1e9,prev_y= -1e9,prev_z= -1e9, timestamp_prev,p2W,p2ToB;
  Double_t timestamp=0., timestamp0=0., dt_sub=0., dt_prev_us=0.;
  Int_t sub_event_tally[20] = {};
  Double_t pmtBoundR=0.,pmtBoundZ=0.;

  // dark noise stuff
  TRandom rnd;
  int npmt;
  float darkrate,tmin,tmax;
  int ndark,darkhit;
  int vhit;

  // Ariadne stuff
  float adir[3],agoodn,aqual,cosscat;

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
  fit_param       bspar;
  bonsaifit       *bsfit,*bspairfit,*cffit;
  pmt_geometry    *bsgeom;
  likelihood      *bslike;
  pairlikelihood  *bspairlike;
  goodness        *bsgdn[2];
  fourhitgrid     *bsgrid[2];
  combinedgrid    *bspairgrid;
  int         cables[5000],veto_cables[5000];
  int	      cables_win[500],veto_cables_win[5000];
  float       times[5000],veto_times[5000];
  float       charges[5000],veto_charges[5000];
  int         event,sub_event,n,count;
  int         inpmt;
  int         hit,nhit,veto_count;
  float       bonsai_vtxfit[4];
  float       bonsaipair_vtxfit[3];
  double      vertex[3],dir[3];
  float       goodn[2];
  int         iscorrelated,pair;

  // likelihood information
  Int_t num_tested;
  Double_t best_like, worst_like, average_like, average_like_05m;
  Double_t best_like_prev, worst_like_prev, average_like_prev, average_like_05m_prev;

  Double_t xQFit =-999999.99,yQFit=-999999.99, zQFit=-999999.99,closestPMTQFit=-999999.99;
  Double_t xQFit_prev =-999999.99,yQFit_prev=-999999.99, zQFit_prev=-999999.99;
  Double_t closestPMTQFit_prev=-999999.99, drrQFit=-999999.99;
  Int_t  QFit = 0;

  rnd.SetSeed();  
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

  // open output file
  TFile *out=new TFile(argv[2],"RECREATE");

  data=new TTree("data","low-energy detector triggered events");

  int time_nX_int = static_cast<int>(time_nX);

  //Define the Integer Tree Leaves
  data->Branch("gtid",&gtid,"gtid/I");
  data->Branch("mcid",&mcid,"mcid/I");
  data->Branch("subid",&subid,"subid/I");
  //data->Branch("nhit",&nhits,"nhit/I");
  data->Branch("inner_hit",&inner_hit,"inner_hit/I");//inner detector    
  data->Branch("inner_hit_prev",&inner_hit_prev,"inner_hit_prev/I");//inner detector
  if (do_clusfit)
    data->Branch("ncherenkovhit",&nsel,"ncherenkovhit/I");// # of selected hits
  data->Branch("id_plus_dr_hit",&tot_nhit,"id_plus_dr_hit/I");//Inner detector plus dark rate hits
  data->Branch("veto_hit",&veto_hit,"veto_hit/I");//veto detector
  data->Branch("veto_plus_dr_hit",&totVHIT,"veto_plus_dr_hit/I");//veto detector plus dark rate hits  
  data->Branch("veto_hit_prev",&veto_hit_prev,"veto_hit_prev/I");//veto detector
  //Define the double Tree Leaves
  data->Branch("pe",&totPE,"pe/D");
  data->Branch("innerPE",&innerPE,"innerPE/D");
  data->Branch("vetoPE",&vetoPE,"vetoPE/D");
  if(do_QFit){
   data->Branch("xQFit",&xQFit,"xQFit/D"); data->Branch("yQFit",&yQFit,"yQFit/D");
   data->Branch("zQFit",&zQFit,"zQFit/D");data->Branch("QFit",&QFit,"QFit/I");
   data->Branch("closestPMTQFit",&closestPMTQFit,"closestPMTQFit/D");
   data->Branch("closestPMTQFit_prev",&closestPMTQFit_prev,"closestPMTQFit_prev/D");
  }		
  data->Branch("n9",&n9,"n9/D");
  data->Branch("n9_prev",&n9_prev,"n9_prev/D");
  data->Branch("nOff",&nOff,"nOff/D");
  data->Branch("n100",&n100,"n100/D");
  data->Branch("n100_prev",&n100_prev,"n100_prev/D");
  data->Branch("n400",&n400,"n400/D");
  data->Branch("n400_prev",&n400_prev,"n400_prev/D");
  data->Branch("nX",&nX,"nX/D");
  data->Branch("nX_prev",&nX_prev,"nX_prev/D");
  data->Branch("good_pos",&bonsai_goodness,"good_pos/D");
  data->Branch("good_pos_prev",&bonsai_goodness_prev,"good_pos_prev/D");
  if (do_clusfit)
    data->Branch("good_cpos",&clusfit_goodness,"good_cpos/D");
  data->Branch("good_dir",&dir_goodness,"good_dir/D");
  data->Branch("good_dir_prev",&dir_goodness_prev,"good_dir_prev/D");
  data->Branch("x",&x,"x/D"); data->Branch("y",&y,"y/D");
  data->Branch("z",&z,"z/D"); data->Branch("t",&t,"t/D");
  if (do_clusfit)
    {
      data->Branch("cx",&cx,"cx/D"); data->Branch("cy",&cy,"cy/D");
      data->Branch("cz",&cz,"cz/D"); data->Branch("ct",&ct,"ct/D");
    }
  data->Branch("u",&u,"u/D"); data->Branch("v",&v,"v/D");
  data->Branch("w",&w,"w/D");
  data->Branch("azimuth_ks",&azi_ks,"azimuth_ks/D");
  data->Branch("azimuth_ks_prev",&azi_ks_prev,"azimuth_ks_prev/D");
  data->Branch("distpmt",&dist_pmt,"distpmt/D");
  data->Branch("mc_energy",&mc_energy,"mc_energy/D");
  data->Branch("mc_energy_prev",&mc_energy_prev,"mc_energy_prev/D");
  data->Branch("mcx",&mc_x,"mcx/D"); data->Branch("mcy",&mc_y,"mcy/D");
  data->Branch("mcz",&mc_z,"mcz/D"); data->Branch("mct",&mct,"mct/D");
  data->Branch("mcx_prev",&mc_x_prev,"mcx_prev/D"); data->Branch("mcy_prev",&mc_y_prev,"mcy_prev/D");
  data->Branch("mcz_prev",&mc_z_prev,"mcz_prev/D");
  data->Branch("mcu",&mc_u,"mcu/D"); data->Branch("mcv",&mc_v,"mcv/D");
  data->Branch("mcw",&mc_w,"mcw/D");
  // data->Branch("code",&code,"code/I");
  data->Branch("closestPMT",&closestPMT,"closestPMT/D");//Proximity to PMT wall
  data->Branch("dxPrevx",&dxx,"dxPrevx/D");
  data->Branch("dyPrevy",&dyy,"dyPrevy/D");
  data->Branch("dzPrevz",&dzz,"dzPrevz/D");
  data->Branch("drPrevr",&drr,"drPrevr/D");
  data->Branch("drPrevrQFit",&drrQFit,"drPrevrQFit/D");
  data->Branch("dxmcx",&dxmcx,"dxmcx/D");
  data->Branch("dymcy",&dymcy,"dymcy/D");
  data->Branch("dzmcz",&dzmcz,"dzmcz/D");
  data->Branch("drmcr",&drmcr,"drmcr/D");

  data->Branch("dt_sub", &dt_sub, "dt_sub/D"); //time of the sub-event trigger from start of the event mc
  data->Branch("dt_prev_us",&dt_prev_us,"dt_prev_us/D"); //global time between consecutive events in us
  data->Branch("timestamp",&timestamp,"timestamp/D"); //trigger time of sub event from start of run

  // likelihood information from bonsai
  data->Branch("num_tested",&num_tested,"num_tested/I"); // number of tested points
  data->Branch("best_like",&best_like,"best_like/D"); // the best log likelihood
  data->Branch("best_like_prev",&best_like_prev,"best_like_prev/D");
  data->Branch("worst_like",&worst_like,"worst_like/D"); // the worst log likelihood
  data->Branch("worst_like_prev",&worst_like_prev,"worst_like_prev/D"); // the worst log likelihood
  data->Branch("average_like",&average_like,"average_like/D"); // the total average log likelihood
  data->Branch("average_like_prev",&average_like_prev,"average_like_prev/D"); // the total average log likelihood
  data->Branch("average_like_05m",&average_like_05m,"average_like_05m/D"); // the average log likelihood excluding a 0.5m sphere around the best fit
  data->Branch("average_like_05m_prev",&average_like_05m_prev,"average_like_05m_prev/D"); // the average log likelihood excluding a 0.5m sphere around the best fit


  run_summary=new TTree("runSummary","mc run summary");
  run_summary->Branch("nEvents",&n_events,"nEvents/I");
  run_summary->Branch("subEventTally",sub_event_tally,"subEventTally[20]/I");
  run_summary->Branch("X",&time_nX_int,"X/I"); //Produce plot with labels showing time window
  run_summary->Branch("darkNoise",&darkNoise,"darkNoise/F"); 
  run_summary->Branch("detector_threshold",&detector_threshold,"detector_threshold/I");
  run_summary->Branch("time_nX",&time_nX,"time_nX/F");
  run_summary->Branch("useAngle",&useAngle,"useAngle/B");
  run_summary->Branch("offsetT",&offsetT,"offsetT/F");
  run_summary->Branch("minT",&minT,"minT/F"); 
  run_summary->Branch("maxT",&maxT,"maxT/F");
  run_summary->Branch("do_clusfit",&do_clusfit,"do_clusfit/B");
  run_summary->Branch("do_QFit",&do_QFit,"do_QFit/B");

  run_tree->GetEntry(0);


  // loop over PMTs and find positions and location of PMT support
  pmtinfo=run->GetPMTInfo();
  n=pmtinfo->GetPMTCount();
  tot_inner = 0; tot_veto =0;

  //Determines the number of inner and veto pmts
  for(hit=count=0; hit<n; hit++)
    {
      if (pmtinfo->GetType(hit)==innerPMTcode)     ++tot_inner;
      else if (pmtinfo->GetType(hit)==vetoPMTcode) ++tot_veto;
      else
	printf("PMT does not have valid identifier: %d \n",
	       pmtinfo->GetType(hit));
    }
  if (n != (tot_inner+tot_veto))
    printf("Mis-match in total PMT numbers: %d, %d \n",n, tot_inner+tot_veto);
    
  inpmt= tot_inner;

  // generate BONSAI geometry object
  {
    float xyz[3*inpmt+1];

    printf("In total there are  %d PMTs in WATCHMAN\n",n);
    
    for(hit=0; hit<n; hit++)
      {
	if(pmtinfo->GetType(hit)==innerPMTcode)
	  {
	    TVector3 pos=pmtinfo->GetPosition(hit);
	    xyz[3*count]=pos[0]*0.1;
	    xyz[3*count+1]=pos[1]*0.1;
	    xyz[3*count+2]=pos[2]*0.1;
	    if (pos[0]>pmtBoundR) pmtBoundR = pos[0];
	    if (pos[2]>pmtBoundZ) pmtBoundZ = pos[2];
	    ++count;
	  }
      }
    
    printf("There are %d inner pmts and %d veto pmts \n ",tot_inner,tot_veto);
    printf("Inner PMT boundary (r,z):(%4.1f mm %4.1f, mm)\n",pmtBoundR,pmtBoundZ);

    if (count!= tot_inner)
      printf("There is a descreptancy in inner PMTS %d vs %d",count,tot_inner);

    // create BONSAI objects from the PMT position array
    bsgeom=new pmt_geometry(inpmt,xyz);
    bslike = new likelihood(bsgeom->cylinder_radius(),bsgeom->cylinder_height());
    bsfit = new bonsaifit(bslike);
    bspairlike=new pairlikelihood(bsgeom->cylinder_radius(),bsgeom->cylinder_height());
    bspairfit=new bonsaifit(bspairlike);
  }

  n_events = rat_tree->GetEntries();
  pair=0;
  // loop over all events
  for (event = 0; event < n_events; event++)
    {
//      if (event%1000==0)
        printf("Evaluating event %d of %d (%d sub events)\n",event,n_events,
	       ds->GetEVCount());
      rat_tree->GetEntry(event);


      sub_event_tally[ds->GetEVCount()]++;
      // loop over all subevents
      for(sub_event=0;sub_event<ds->GetEVCount();sub_event++)
	{
          iscorrelated = sub_event;
          // reset output variables
          nOff = -999999;
          n9 =  -999999;
          n100 = -999999;
          n400 = -999999;
          nX =  -999999;
          nsel = -999999;
          bonsai_goodness = dir_goodness = x = y = z = t  = u = v = w = azi_ks = dist_pmt = closestPMT = -999999.99;
          clusfit_goodness=cx=cy=cz=ct=-999999.99;
          dxx = dyy = dzz = drr = dxmcx = dymcy = dzmcz = drmcr = drrQFit = closestPMTQFit = -999999.99;

	  gtid += 1;
	  mcid = event;
	  subid = sub_event;
     
	  ev = ds->GetEV(sub_event);
	  totPE = ev->GetTotalCharge();

          mc_energy_prev = mc_energy;
          mc_x_prev = mc_x;mc_y_prev=mc_y;mc_z_prev=mc_z;

	  TVector3 temp;
      
	  mc = ds->GetMC();
	  prim = mc->GetMCParticle(sub_event);
	  mc_energy = prim->GetKE();
	  temp = prim->GetPosition();
	  mc_x = temp.X();
	  mc_y = temp.Y();
	  mc_z = temp.Z();
          mct    = prim->GetTime(); //local emission time
          if (subid>0){
             temp = prim->GetEndPosition();
             mc_x = temp.X(); 
             mc_y = temp.Y(); 
             mc_z = temp.Z(); 
             mct    = prim->GetEndTime();
          }
          // get true event timings
          // times are in ns unless specified
          timestamp = 1e6*mc->GetUTC().GetSec() + 1e-3*mc->GetUTC().GetNanoSec() + 1e-3*ev->GetCalibratedTriggerTime() - 1e6*run->GetStartTime().GetSec()-1e-3*run->GetStartTime().GetNanoSec(); //global time of subevent trigger (us)
          dt_prev_us = timestamp-timestamp_prev; //time since the previous trigger (us)
          dt_sub = ev->GetCalibratedTriggerTime(); //trigger time (first pmt hit time) from start of event mc

	  nhit=ev->GetPMTCount();

          // loop over all PMT hits for each subevent
	  innerPE=0;vetoPE=0;    
	  for(hit=count=veto_count=0; hit<nhit; hit++)
	    {
	      pmt=ev->GetPMT(hit);
	      id = pmt->GetID();
	      //only use information from the inner pmts
	      if(pmtinfo->GetType(id) == innerPMTcode)
		{
		  cables[count]=pmt->GetID()+1;
		  times[count]=pmt->GetTime()+offsetT;
		  charges[count]=pmt->GetCharge();
		  innerPE += pmt->GetCharge();
		  count++;
		}
	      else if(pmtinfo->GetType(id)== vetoPMTcode)
		{
		  veto_cables[veto_count]=pmt->GetID()+1;
		  veto_times[veto_count]=pmt->GetTime()+offsetT;
		  veto_charges[veto_count]=pmt->GetCharge();
		  vetoPE += pmt->GetCharge();    
		  veto_count++;
		}
	      else
		printf("Unidentified PMT type: (%d,%d) \n",count,pmtinfo->GetType(id));
	    } // end of loop over all PMT hits
	  veto_hit = veto_count;
	  inner_hit = count;
	  nhit = count;
          printf("nhit: %d\n",nhit);
          if (inner_hit<detector_threshold){
            //  printf("Event did not pass trigger threshold (%d:%d)\n",inner_hit,triggerThreshold);
              //data->Fill();
              continue;
          }
	  //Inner PMT Dark Rate
	  npmt=tot_inner;
	  darkrate=darkNoise*npmt;
	  tmin=minT+offsetT;
	  tmax= maxT+offsetT;
	  ndark=rnd.Poisson((tmax-tmin)*1e-9*darkrate);

	  //int inhit= count;
	  //loop over (randomly generated) dark hits and
	  //assign random dark rate where event rate is 
	  //below dark rate for the inner detector
	  for(darkhit=0; darkhit<ndark; darkhit++)
	    {
	      int darkcable= (int)(npmt*rnd.Rndm()+1);
	      float darkt=tmin+(tmax-tmin)*rnd.Rndm();
	      // loop over all inner PMT hits
	      for(hit=0; hit<nhit; hit++)
		if (cables[hit]==darkcable) break;
	      if (hit==nhit)
		{
		  cables[hit]=darkcable;
		  times[hit]=darkt;
		  charges[hit]=1;
		  nhit++;
		} 
	      else
		{
		  if (darkt<times[hit]) times[hit]=darkt;
		  charges[hit]++;
		} //end of loop over all inner  PMT hits
	    } // end of loop over inner dark hits
	  //Veto PMT
	  //Inner PMT Dark Rate
      
	  npmt=tot_veto;
	  darkrate=darkNoise*npmt;
	  ndark=rnd.Poisson((tmax-tmin)*1e-9*darkrate);
	  vhit= veto_count;
          //loop over (randomly generated) dark hits and
          //assign random dark rate where event rate is
          //below dark rate for the veto detector
	  for(darkhit=0; darkhit<ndark; darkhit++)
	    {
	      int darkcable=(int) (npmt*rnd.Rndm()+1);
	      float darkt=tmin+(tmax-tmin)*rnd.Rndm();
              // loop over all inner PMT hits
	      for(hit=0; hit<vhit; hit++)
		if (veto_cables[hit]==darkcable) break;
	      if (hit==vhit)
		{
		  veto_cables[hit]=darkcable;
		  veto_times[hit]=darkt;
		  veto_charges[hit]=1;
		  vhit++;
		}
	      else
		{
		  if (darkt<veto_times[hit]) veto_times[hit]=darkt;
		  veto_charges[hit]++;
		} //end of loop over all veto  PMT hits
	    } // end of loop over vetp dark hits 
	  totVHIT= vhit;
	  tot_nhit= nhit;
	  //Determines how many events before crash
	  crash_count++;

	  // generate BONSAI objects
	  bsgdn[pair]=new goodness(bspairlike->sets(),bspairlike->chargebins(),
			     bsgeom,nhit,cables,times,charges);
	  nsel=bsgdn[pair]->nselected();

	  if(nsel<4) {
	    // four selected hits required to form a fourhitgrid.
	    // we will crash if we continue.
	              //Perform Qfit
            if(do_QFit ==1){
              Best_Fit = Fitting_Likelihood_Ascent(ds, pmtinfo, ev, PDF);
              //printf("Qfit: %f",Best_Fit[0]);
              xQFit = Best_Fit[0];
              yQFit = Best_Fit[1];
              zQFit = Best_Fit[2];
              QFit  = 1;

              // calculate smallest distance to any pmt
              p2W = pmtBoundR-sqrt(xQFit*xQFit+yQFit*yQFit);
              p2ToB = pmtBoundZ-sqrt(zQFit*zQFit);
              closestPMTQFit = TMath::Min(p2W,p2ToB);

              // calculate distance from previous subevent
              drrQFit = sqrt(pow(xQFit-xQFit_prev,2)+pow(yQFit-yQFit_prev,2)+pow(zQFit-zQFit_prev,2));
            }
            data->Fill();
            QFit  = 0 ;
            xQFit = yQFit = zQFit = -999999.0;
	    continue;
	  }

          bsgrid[pair] = new fourhitgrid(bsgeom->cylinder_radius(),
                                 bsgeom->cylinder_height(),bsgdn[pair]);
          bslike->set_hits(bsgdn[pair]); 

	  if (do_clusfit)
	    {
	      // Clusfit
	      cffit=new bonsaifit(bsgdn[1]);
	      bsgdn[pair]->maximize(cffit,bsgrid[1]);
	      cx=10.*cffit->xfit();
	      cy=10.*cffit->yfit();
	      cz=10.*cffit->zfit();
	      ct=10.*bsgdn[pair]->get_zero()-offsetT;
	      clusfit_goodness=cffit->maxq();
	      delete cffit;
	    }
          // only attempt the pair fit from the second triggered event
          // don't try to reconstruct if the time between exceeds
          // loose time cut
          
	  // fit
          bool use_cherenkov_angle = true;
          if(useAngle == 0) use_cherenkov_angle = false;

          if (pair==0) {
            pair=1;
            mc_x_prev= mc_x; mc_y_prev = mc_y; mc_z_prev = mc_z;
            mc_energy_prev = mc_energy;
            timestamp_prev = timestamp;
            inner_hit_prev = inner_hit;
            veto_hit_prev = veto_hit;
            data->Fill();
            continue;
          }
          bspairgrid = new combinedgrid(bsgeom->cylinder_radius(),
                           bsgeom->cylinder_height(),
                           bsgrid[0],bsgrid[1]);
          bspairlike->set_hits(bsgdn[0],bsgdn[1]);
          bspairlike->maximize(bspairfit, bspairgrid);
	  x= 10.*bspairfit->xfit();y= 10.*bspairfit->yfit();z= 10.*bspairfit->zfit();
     
	  // calculate n9 and goodness
	  *bonsaipair_vtxfit=bspairfit->xfit();
	  bonsaipair_vtxfit[1]=bspairfit->yfit();
	  bonsaipair_vtxfit[2]=bspairfit->zfit();
          n9 =  bspairlike->nwind(1,bonsaipair_vtxfit,-3,6);
          n9_prev =  bspairlike->nwind(0,bonsaipair_vtxfit,-3,6);
          nOff = bspairlike->nwind(1,bonsaipair_vtxfit,-150,-50);
          n100 =  bspairlike->nwind(1,bonsaipair_vtxfit,-10,90);
          n100_prev =  bspairlike->nwind(0,bonsaipair_vtxfit,-10,90);
          n400 =  bspairlike->nwind(1,bonsaipair_vtxfit,-10,390);
          n400_prev =  bspairlike->nwind(0,bonsaipair_vtxfit,-10,390);
          if (time_nX == 9.) {
            nX = bspairlike->nwind(1,bonsaipair_vtxfit,-3,6);
          }
          else {
            nX =  bspairlike->nwind(1,bonsaipair_vtxfit,-10,time_nX-10); //Stephen Wilson - how to decide the time interval
          }
          bspairlike->ntgood(0,bonsaipair_vtxfit,0,goodn[0]);
          bspairlike->ntgood(1,bonsaipair_vtxfit,0,goodn[1]);
          bonsai_goodness = goodn[1];
          bonsai_goodness_prev = goodn[0];          
          // get the reconstructed emission time
          t=bspairlike->get_zero(2)-offsetT;
          t_prev=bspairlike->get_zero(1)-offsetT;

            float ave;
            int nfit;

          best_like = (Double_t)bspairlike->get_ll(2);
          best_like_prev = (Double_t)bspairlike->get_ll(1);
//            nfit=bspairlike->nfit();
//            num_tested = nfit;
//            worst_like = bspairlike->worstquality();

//            nfit=bspairlike->average_quality(ave,bonsaipair_vtxfit,-1);
//            average_like= ave;
//            nfit=bspairlike->average_quality(ave,bonsaipair_vtxfit,50);
//            average_like_05m = ave;


	  // get momentum vector and normalize it
	  temp = prim->GetMomentum();
	  temp = temp.Unit();
	  mc_u = temp.X();mc_v = temp.Y();mc_w = temp.Z();
	  tot_nhit = nhit;

	  // get distance and time difference to previous fit event
	  dxx = x-prev_x;dyy = y-prev_y;dzz = z-prev_z;
	  drr = sqrt(dxx*dxx+dyy*dyy+dzz*dzz);
          if(dxx>1e6)
	    {
	      dxx = 0.;dyy = 0.;dzz = 0.; drr = 0.;
	    }
	  dxmcx = x-mc_x;dymcy = y-mc_y;dzmcz = z-mc_z;
          drmcr = sqrt(dxmcx*dxmcx+dymcy*dymcy+dzmcz*dzmcz);

	  // calculate smallest distance to any pmt
	  p2W = pmtBoundR-sqrt(x*x+y*y);
	  p2ToB = pmtBoundZ-sqrt(z*z);
	  closestPMT = TMath::Min(p2W,p2ToB);

	  // do direction fit
	  // find all PMTs within 9 nsec window
	  int n9win=nwin(pmtinfo,9,bonsaipair_vtxfit,nhit,cables,times,cables_win);
	  float apmt[3*n9win];
	  // fill PMT positions into an array
	  for(hit=0; hit<n9win; hit++)
	    {
	      TVector3 n9pos=pmtinfo->GetPosition(cables_win[hit]-1);
	      apmt[3*hit]=n9pos.X()*0.1;
	      apmt[3*hit+1]=n9pos.Y()*0.1;
	      apmt[3*hit+2]=n9pos.Z()*0.1;
	    }
	  // call direction fit and save results
	  adir[0]=adir[1]=adir[2]=-2;
	  {
	    ariadne ari(bonsaipair_vtxfit,n9win,apmt,0.719);
	    ari.fit();
	    agoodn=ari.dir_goodness();
	    if (agoodn>=0) ari.dir(adir);
	    dir_goodness = agoodn;
	    u=adir[0];
	    v=adir[1];
	    w=adir[2];
	  }
	  azi_ks=azimuth_ks(n9win,apmt,bonsaipair_vtxfit,adir);
	  vertex[0]=x;
	  vertex[1]=y;
	  vertex[2]=z;
	  dir[0]=adir[0];
	  dir[1]=adir[1];
	  dir[2]=adir[2];
	  if ((dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]>1.00001) ||
	      (dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]<0.99999))
	    dist_pmt=-1;
	  else
	    dist_pmt=distpmt(vertex,dir,pmtBoundR,pmtBoundZ,wall);
	   
          //Perform Qfit
          if(do_QFit ==1){
            Best_Fit = Fitting_Likelihood_Ascent(ds, pmtinfo, ev, PDF);
            //printf("Qfit: %f",Best_Fit[0]);
            xQFit = Best_Fit[0];
            yQFit = Best_Fit[1];
            zQFit = Best_Fit[2];
            QFit  = 1;
          
            // calculate smallest distance to any pmt
            p2W = pmtBoundR-sqrt(xQFit*xQFit+yQFit*yQFit);
            p2ToB = pmtBoundZ-sqrt(zQFit*zQFit);
            closestPMTQFit = TMath::Min(p2W,p2ToB);

            // calculate distance from previous subevent
            drrQFit = sqrt(pow((xQFit-xQFit_prev),2)+pow((yQFit-yQFit_prev),2)+pow((zQFit-zQFit_prev),2));
          }
	  data->Fill();
          //save reference values for the next subevent
          mc_energy_prev = mc_energy;
          mc_x_prev = mc_x;mc_y_prev=mc_y;mc_z_prev=mc_z;
	  prev_x=x;prev_y=y;prev_z=z;
	  timestamp_prev = timestamp;
          inner_hit_prev = inner_hit;
          veto_hit_prev = veto_hit;
          nX_prev = nX;
          n9_prev = n9;
          n100_prev = n100;
          n400_prev = n400;
          azi_ks_prev = azi_ks;
          xQFit_prev = xQFit;
          yQFit_prev = yQFit;
          zQFit_prev = zQFit;
          closestPMTQFit_prev = closestPMTQFit;
          QFit  = 0 ;
          xQFit = yQFit = zQFit = -999999.0;
          bsgdn[0]=bsgdn[1];
          bsgrid[0]=bsgrid[1];
	  // delete BONSAI objects and reset likelihoods
	  delete bspairgrid;
	} 
    }
  out->cd();
  data->Write();
  run_summary->Fill();
  run_summary->Write();
  out->Close();
  return 0;
}

int nwin(RAT::DS::PMTInfo *pmtinfo,
         float twin,float *v,int nfit,int *cfit,float *tfit,int *cwin)
{
    if (nfit<=0) return(0);

    float ttof[nfit],tsort[nfit],dx,dy,dz;
    int   hit,nwin=0,nwindow,hstart_test,hstart,hstop;

    // calculate t-tof for each hit
    for(hit=0; hit<nfit; hit++)
    {
        TVector3 pos=pmtinfo->GetPosition(cfit[hit]-1);
        dx=pos.X()*0.1-v[0];
        dy=pos.Y()*0.1-v[1];
        dz=pos.Z()*0.1-v[2];
        tsort[hit]=ttof[hit]=tfit[hit]-sqrt(dx*dx+dy*dy+dz*dz)*CM_TO_NS;
    }
    sort(tsort,tsort+nfit);

    // find the largest number of hits in a time window <= twin
    nwindow=1;
    hstart_test=hstart=0;
    while(hstart_test<nfit-nwindow)
    {
        hstop=hstart_test+nwindow;
        while((hstop<nfit) && (tsort[hstop]-tsort[hstart_test]<=twin))
        {
            hstart=hstart_test;
            nwindow++;
            hstop++;
        }
        hstart_test++;
    }
    hstop=hstart+nwindow-1;
    for(hit=0; hit<nfit; hit++)
    {
        if (ttof[hit]<tsort[hstart]) continue;
        if (ttof[hit]>tsort[hstop]) continue;
        cwin[nwin++]=cfit[hit];
    }
    if (nwin!=nwindow) printf("nwin error %d!=%d\n",nwin,nwindow);
    return(nwindow);
}

