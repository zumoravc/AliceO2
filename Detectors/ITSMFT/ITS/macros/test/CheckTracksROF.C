/// \file CheckTracks.C
/// \brief Simple macro to check ITSU tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>

#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TString.h>
#include <TGraph.h>

#include "ITSBase/GeometryTGeo.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"

#endif

using namespace std;

struct DataFrames {
  void update(int frame, long index)
  {
    if (frame < firstFrame) {
      firstFrame = frame;
      firstIndex = index;
    }
    if (frame > lastFrame) {
      lastFrame = frame;
      lastIndex = index;
    }
    if (frame == firstFrame && index < firstIndex) {
      firstIndex = index;
    }
    if (frame == lastFrame && index > lastIndex) {
      lastIndex = index;
    }
  }

  long firstFrame = 10000;
  int firstIndex = 1;
  long lastFrame = -10000;
  int lastIndex = -1;
};

void CheckTracksROF(std::string tracfile = "o2trac_its.root", std::string clusfile = "o2clus_its.root", std::string kinefile = "o2sim_Kine.root")
{
  bool filterMultiROFTracks = 1;

  const int arrDimension = 50;
  const int lowerCut = 10;
  const int minNOfPartInROF = 25;

  using namespace o2::itsmft;
  using namespace o2::its;

  TFile* f = TFile::Open("CheckTracksROF.root", "recreate");
  TNtuple* nt = new TNtuple("ntt", "track ntuple",
                            //"mcYOut:recYOut:"
                            "mcZOut:recZOut:"
                            "mcPhiOut:recPhiOut:"
                            "mcThetaOut:recThetaOut:"
                            "mcPhi:recPhi:"
                            "mcLam:recLam:"
                            "mcPt:recPt:"
                            "ipD:ipZ:label");

  // Geometry
  o2::base::GeometryManager::loadGeometry();
  auto gman = o2::its::GeometryTGeo::Instance();

  // MC tracks
  TFile* file0 = TFile::Open(kinefile.data());
  TTree* mcTree = (TTree*)gFile->Get("o2sim");
  mcTree->SetBranchStatus("*", 0); //disable all branches
  //mcTree->SetBranchStatus("MCEventHeader.*",1);
  mcTree->SetBranchStatus("MCTrack*", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);

  /*
  std::vector<o2::TrackReference>* mcTrackRefs = nullptr;
  mcTree->SetBranchAddress("TrackRefs", &mcTrackRefs);
  */

  // Clusters
  TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  // Reconstructed tracks
  TFile* file1 = TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);
  // Track MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);

  Int_t lastEventIDcl = -1, cf = 0;
  Int_t nev = mcTree->GetEntriesFast();

  Int_t nb = 100;
  Double_t xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
  Double_t a = TMath::Log(ptcuth / ptcutl) / nb;
  for (Int_t i = 0; i <= nb; i++)
    xbins[i] = ptcutl * TMath::Exp(i * a);
  TH1D* num = new TH1D("num", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", nb, xbins);
  num->Sumw2();
  TH1D* fak = new TH1D("fak", ";#it{p}_{T} (GeV/#it{c});Fak", nb, xbins);
  fak->Sumw2();
  TH1D* clone = new TH1D("clone", ";#it{p}_{T} (GeV/#it{c});Clone", nb, xbins);
  clone->Sumw2();

  TH1D* den = new TH1D("den", ";#it{p}_{T} (GeV/#it{c});Den", nb, xbins);
  den->Sumw2();

  TH2D* corr = new TH2D("corr", "Correlation; # of reconstructed tracks; # of simulated tracks", 100, 0, 100, 100, 0, 100);
  corr->Sumw2();

  TH1D* hMultiplicity = new TH1D("hMultiplicity", "Multiplicity; # of charged; Events", 20, 0, 100);
  TH1D* hMultFrame = new TH1D("hMultFrame", "Multiplicity per ROF; # of charged; Events", 200, 0, 200);
  TH2D* hPurity = new TH2D("hPurity", "Purity", 2, 0, 2, arrDimension, 0, arrDimension);

  int statGenAr[arrDimension] = {0};
  int statGooAr[arrDimension] = {0};

  // search for MC events in frames

  cout << "Find mc events in cluster frames.. " << endl;

  int loadedEventClust = -1;

  vector<DataFrames> clusterFrames(nev);

  for (int frame = 0; frame < clusTree->GetEntriesFast(); frame++) { // Cluster frames
    if (!clusTree->GetEvent(frame))
      continue;
    loadedEventClust = frame;
    for (unsigned int i = 0; i < clusArr->size(); i++) { // Find the last MC event within this reconstructed entry
      auto lab = (clusLabArr->getLabels(i))[0];
      if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() < 0 || lab.getEventID() >= nev)
        continue;
      clusterFrames[lab.getEventID()].update(frame, i);
    }
  }

  cout << "Find mc events in track frames.. " << endl;

  int loadedEventTracks = -1;

  printf("entries: %d \n", recTree->GetEntriesFast());

  vector<DataFrames> trackFrames(nev);
  for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // Track frames
    if (!recTree->GetEvent(frame))
      continue;
    int loadedEventTracks = frame;

    for (unsigned int i = 0; i < recArr->size(); i++) { // Find the last MC event within this reconstructed entry
      auto lab = (trkLabArr->getLabels(i))[0];
      if (!lab.isValid()) {
        const TrackITS& recTrack = (*recArr)[i];
        fak->Fill(recTrack.getPt());
      }
      if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() < 0 || lab.getEventID() >= nev)
        continue;
      trackFrames[lab.getEventID()].update(frame, i);
    }

    int nOfPartInFrame = 0;
    for (unsigned int i = 0; i < recArr->size(); i++) { // Find the last MC event within this reconstructed entry
      const TrackITS& recTrack = (*recArr)[i];
      auto lab = (trkLabArr->getLabels(i))[0];
      if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() < 0 || lab.getEventID() >= nev)
        continue;
      if(TMath::Abs(recTrack.getEta()) < 1.1 ) nOfPartInFrame++;
    }
    printf("Frame %d with %d particles. \n", frame, nOfPartInFrame);
  }

  cout << "Process mc events.. " << endl;

  Int_t statGen = 0, statGoo = 0, statFak = 0, statClone = 0;
  int nDataEvents = 0;

  int firstFramePrevious = -1;
  int nOfParticlesInFrame = 0;
  int nHMevents = 0;
  bool highMultiEvent = false;
  int purity[2][arrDimension] = {0};

  for (Int_t n = 0; n < nev; n++) { // loop over MC events
  // for (Int_t n = 0; n < 10; n++) { // loop over MC events

    std::cout << "\nMC event " << n << '/' << nev << std::endl;
    DataFrames f = clusterFrames[n];
    if ((f.firstFrame > f.lastFrame) ||
        ((f.firstFrame == f.lastFrame) && (f.firstIndex > f.lastIndex)))
      continue;

    if (!mcTree->GetEvent(n))
      continue;
    //cout<<"mc event loaded"<<endl;

    Int_t nmc = mcArr->size();
    //Int_t nmcrefs = mcTrackRefs->size();
    Int_t nOfRecTracks = 0; //within certain range

    std::cout << "N of MC tracks (total): " << nmc << std::endl;

    std::vector<int> clusMap(nmc, 0);
    std::vector<int> clusRofMap(nmc, -1);
    std::vector<int> trackMap(nmc, -1);
    std::vector<int> mapNFakes(nmc, 0);
    std::vector<int> mapNClones(nmc, 0);

    std::vector<TrackITS> trackStore;
    trackStore.reserve(10000);

    f = trackFrames[n];
    cout << "track frames: " << f.firstFrame << ", " << f.firstIndex << " <-> " << f.lastFrame << ", " << f.lastIndex << endl;

    if(firstFramePrevious != f.firstFrame){
      if(n != 0 || f.lastIndex < 0) {
        hMultFrame->Fill(nOfParticlesInFrame);
        for(int cut(0); cut < arrDimension; cut++){
          if(nOfParticlesInFrame > minNOfPartInROF + cut){
            if(highMultiEvent) {
              hPurity->Fill(1.0, cut);
              purity[1][cut]++;
            }
            else {
              purity[0][cut]++;
              hPurity->Fill(0.0, cut);
            }
          }
        }
      }
      nOfParticlesInFrame = 0;
      firstFramePrevious = f.firstFrame;
      highMultiEvent = false;
    }

    for (int frame = f.firstFrame; frame <= f.lastFrame; frame++) {
      if (frame != loadedEventTracks) {
        loadedEventTracks = -1;
        if (!recTree->GetEvent(frame))
          continue;
        loadedEventTracks = frame;
      }
      long nentr = recArr->size();

      long firstIndex = (frame == f.firstFrame) ? f.firstIndex : 0;
      long lastIndex = (frame == f.lastFrame) ? f.lastIndex : nentr - 1;

      for (long i = firstIndex; i <= lastIndex; i++) {
        auto lab = (trkLabArr->getLabels(i))[0];
        if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() != n)
          continue;
        int mcid = lab.getTrackID();
        if (mcid < 0 || mcid >= nmc) {
          cout << "track mc label is too big!!!" << endl;
          continue;
        }
        if (lab.isFake()) {
          mapNFakes[mcid]++;
        } else if (trackMap[mcid] >= 0) {
          mapNClones[mcid]++;
        } else {
          trackMap[mcid] = trackStore.size();
          const TrackITS& recTrack = (*recArr)[i];
          trackStore.emplace_back(recTrack);
          if(TMath::Abs(recTrack.getEta()) < 1.1 ) nOfRecTracks++;
        }
      }
    }

    f = clusterFrames[n];
    cout << "cluster frames: " << f.firstFrame << ", " << f.firstIndex << " <-> " << f.lastFrame << ", " << f.lastIndex << endl;
    int nClusters = 0;
    for (int frame = f.firstFrame; frame <= f.lastFrame; frame++) {
      if (frame != loadedEventClust) {
        loadedEventClust = -1;
        if (!clusTree->GetEvent(frame))
          continue;
        loadedEventClust = frame;
      }
      long nentr = clusArr->size();

      long firstIndex = (frame == f.firstFrame) ? f.firstIndex : 0;
      long lastIndex = (frame == f.lastFrame) ? f.lastIndex : nentr - 1;

      for (int i = firstIndex; i <= lastIndex; i++) {
        auto lab = (clusLabArr->getLabels(i))[0];
        if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() != n)
          continue;

        int mcid = lab.getTrackID();
        if (mcid < 0 || mcid >= nmc) {
          cout << "cluster mc label is too big!!!" << endl;
          continue;
        }

        if (!lab.isCorrect())
          continue;

        const CompClusterExt& c = (*clusArr)[i];

        /* FIXME
        if (clusRofMap[mcid] < 0) {
          clusRofMap[mcid] = c.getROFrame();
        }
        if (filterMultiROFTracks && (clusRofMap[mcid] != (int)c.getROFrame()))
          continue;
*/
        nClusters++;

        int& ok = clusMap[mcid];
        auto layer = gman->getLayer(c.getSensorID());
        float r = 0.f;
        if (layer == 0)
          ok |= 0b1;
        if (layer == 1)
          ok |= 0b10;
        if (layer == 2)
          ok |= 0b100;
        if (layer == 3)
          ok |= 0b1000;
        if (layer == 4)
          ok |= 0b10000;
        if (layer == 5)
          ok |= 0b100000;
        if (layer == 6)
          ok |= 0b1000000;
      }
    } // cluster frames

    if (nClusters > 0)
      nDataEvents++;

    Int_t nGen = 0, nGoo = 0, nFak = 0, nClone = 0;

    Int_t nOfCharged = 0;

    for (int mc = 0; mc < nmc; mc++) {

      const auto& mcTrack = (*mcArr)[mc];

      Int_t mID = mcTrack.getMotherTrackId();
      if (mID >= 0) {
        const auto& mom = (*mcArr)[mID];
        int pdg = std::abs(mom.GetPdgCode());
        if (pdg > 100 || (pdg < 20 && pdg > 10))
          continue; // Select primary particles
      }
      Int_t pdg = mcTrack.GetPdgCode();
      if (TMath::Abs(pdg) != 211 && TMath::Abs(pdg) != 321 && TMath::Abs(pdg) != 2212 && TMath::Abs(pdg) != 1 && TMath::Abs(pdg) != 13)
        continue; // Select pions

      if(TMath::Abs( mcTrack.GetEta() ) > 1.1 ) continue;

      // nOfCharged++;

      if (clusMap[mc] != 0b1111111)
        continue;

      nOfCharged++;
      nOfParticlesInFrame++;
    } // loop over mc particles

    if(nOfCharged > 25) {
      nHMevents++;
      highMultiEvent = true;
    }

    if(true){
    // if(nOfCharged > 30){
      for (int mc = 0; mc < nmc; mc++) {

        const auto& mcTrack = (*mcArr)[mc];

        if (mapNFakes[mc] > 0) { // Fake-track rate calculation
          nFak += mapNFakes[mc];
          fak->Fill(mcTrack.GetPt(), mapNFakes[mc]);
        }

        Int_t mID = mcTrack.getMotherTrackId();
        if (mID >= 0) {
          const auto& mom = (*mcArr)[mID];
          int pdg = std::abs(mom.GetPdgCode());
          if (pdg > 100 || (pdg < 20 && pdg > 10))
            continue; // Select primary particles
        }
        Int_t pdg = mcTrack.GetPdgCode();
        if (TMath::Abs(pdg) != 211 && TMath::Abs(pdg) != 321 && TMath::Abs(pdg) != 2212 && TMath::Abs(pdg) != 1 && TMath::Abs(pdg) != 13)
          continue; // Select pions

        if(TMath::Abs( mcTrack.GetEta() ) > 1.1 ) continue;

        // nOfCharged++;

        if (clusMap[mc] != 0b1111111)
          continue;

        nGen++; // Generated tracks for the efficiency calculation

        // Float_t mcYOut=-1., recYOut=-1.;
        Float_t mcZOut = -1., recZOut = -1.;
        Float_t mcPhiOut = -1., recPhiOut = -1.;
        Float_t mcThetaOut = -1., recThetaOut = -1.;
        Float_t mcPx = mcTrack.GetStartVertexMomentumX();
        Float_t mcPy = mcTrack.GetStartVertexMomentumY();
        Float_t mcPz = mcTrack.GetStartVertexMomentumZ();
        Float_t mcPhi = TMath::ATan2(mcPy, mcPx), recPhi = -1.;
        Float_t mcPt = mcTrack.GetPt(), recPt = -1.;
        Float_t mcLam = TMath::ATan2(mcPz, mcPt), recLam = -1.;
        Float_t ip[2]{0., 0.};
        Float_t label = -123456789.;

        den->Fill(mcPt);

        if (trackMap[mc] >= 0) {
          nGoo++; // Good found tracks for the efficiency calculation
          num->Fill(mcPt);

          const TrackITS& recTrack = trackStore[trackMap[mc]];
          auto out = recTrack.getParamOut();
          // recYOut = out.getY();
          recZOut = out.getZ();
          recPhiOut = out.getPhi();
          recThetaOut = out.getTheta();
          std::array<float, 3> p;
          recTrack.getPxPyPzGlo(p);
          recPt = recTrack.getPt();
          recPhi = TMath::ATan2(p[1], p[0]);
          recLam = TMath::ATan2(p[2], recPt);
          Float_t vx = 0., vy = 0., vz = 0.; // Assumed primary vertex
          Float_t bz = 5.;                   // Assumed magnetic field
          recTrack.getImpactParams(vx, vy, vz, bz, ip);

          nt->Fill( // mcYOut,recYOut,
            mcZOut, recZOut, mcPhiOut, recPhiOut, mcThetaOut, recThetaOut, mcPhi, recPhi, mcLam, recLam, mcPt, recPt, ip[0],
            ip[1], mc);
        }

        if (mapNClones[mc] > 0) { // Clone-track rate calculation
          nClone += mapNClones[mc];
          clone->Fill(mcPt, mapNClones[mc]);
        }
      } // end particle loop

    } // end nOfCharged > 30

    hMultiplicity->Fill(nOfCharged);

    statGen += nGen;
    statGoo += nGoo;
    statFak += nFak;
    statClone += nClone;

    for(int cut(0); cut < arrDimension; cut++){
      if(nOfCharged > lowerCut + cut){
        statGenAr[cut] += nGen;
        statGooAr[cut] += nGoo;
      }
    }

    if (nGen > 0) {
      Float_t eff = nGoo / Float_t(nGen);
      Float_t rat = nFak / Float_t(nGen);
      Float_t clonerat = nClone / Float_t(nGen);
      std::cout << "Good found tracks: " << nGoo << ",  efficiency: " << eff << ",  fake-track rate: " << rat << " clone rate " << clonerat << std::endl;

      corr->Fill(nOfRecTracks, nGen);
    }


  } // mc events

  cout << "\nOverall efficiency: " << endl;
  if (statGen > 0) {
    Float_t eff = statGoo / Float_t(statGen);
    Float_t rat = statFak / Float_t(statGen);
    Float_t clonerat = statClone / Float_t(statGen);
    std::cout << "Good found tracks/event: " << statGoo / nDataEvents << ",  efficiency: " << eff << ",  fake-track rate: " << rat << " clone rate " << clonerat << std::endl;
  }

  Float_t x[arrDimension], y[arrDimension];
  for(int cut(0); cut < arrDimension; cut++){
    x[cut] = lowerCut+cut;
    if(statGenAr[cut] > 0) y[cut] = statGooAr[cut] / Float_t(statGenAr[cut]);
    else y[cut] = 0.0;
  }

  TCanvas* c3 = new TCanvas();
  hMultFrame->Draw();

  //purity plot
  TCanvas* c5 = new TCanvas();
  Float_t xPur[arrDimension], yPur[arrDimension], yLoss[arrDimension];
  for(int cut(0); cut < arrDimension; cut++){
    xPur[cut] = minNOfPartInROF+cut;
    Float_t total = Float_t(purity[0][cut]) + Float_t(purity[1][cut]);
    if(total > 0) yPur[cut] = Float_t(purity[1][cut])/total;
    else yPur[cut] = 0.0;
    yLoss[cut] = Float_t(purity[1][0] - purity[1][cut]) / Float_t(purity[1][0]);
  }
  TGraph *pur = new TGraph(arrDimension, xPur, yPur);
  pur->SetTitle("; Min # of charged tracks per ROF; ROF with HM events / all ROF");
  pur->Draw();
  c5->SaveAs("purity.pdf");

  //loss count
  TCanvas* c7 = new TCanvas();
  TGraph* loss = new TGraph(arrDimension, xPur, yLoss);
  loss->SetTitle("; Min # of charged tracks per ROF; Rel. loss of HM events");
  loss->Draw();
  c7->SaveAs("relLoss.pdf");


  //2d distribution
  TCanvas* c6 = new TCanvas();
  hPurity->Draw();

  // hMultiplicity->Draw();
  // c3->SetLogy();
  // c3->SaveAs("multi_withClusterMap.pdf");
}
