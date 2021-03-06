/// \file CheckLines.C
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

#include "ITSBase/GeometryTGeo.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/ClusterLines.h"


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

void CheckLines(std::string tracfile = "vertexer_serial_data.root", std::string clusfile = "o2clus_its.root", std::string kinefile = "o2sim_Kine.root")
{
  bool filterMultiROFTracks = 1;

  using namespace o2::itsmft;
  using namespace o2::its;

  TFile* f = TFile::Open("CheckLines.root", "recreate");
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
  std::vector<o2::its::Line>* linesArr = nullptr;
  recTree->SetBranchAddress("LinesITS", &linesArr);
  // Track MC labels
  // o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkLabArr = nullptr;
  // recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);

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

  cout << "Find mc events in lines frames.. " << endl;

  int loadedEventTracks = -1;

  vector<DataFrames> trackFrames(nev);
  for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // Track frames
    if (!recTree->GetEvent(frame))
      continue;
    int loadedEventTracks = frame;
    for (unsigned int i = 0; i < linesArr->size(); i++) { // Find the last MC event within this reconstructed entry
      const Line& recLine = (*linesArr)[i];
      auto lab = recLine.clusterLabels[0];
      // if (!lab.isValid()) {
      //   // const TrackITS& recTrack = (*recArr)[i];
      //   fak->Fill(recTrack.getPt());
      // }
      if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() < 0 || lab.getEventID() >= nev)
        continue;
      trackFrames[lab.getEventID()].update(frame, i);
    }
  }

  cout << "Process mc events.. " << endl;

  Int_t statGen = 0, statGoo = 0, statFak = 0, statClone = 0;
  int nDataEvents = 0;

  for (Int_t n = 0; n < nev; n++) { // loop over MC events

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

    std::vector<int> clusRofMap(nmc, -1);
    std::vector<int> trackMap(nmc, -1);
    std::vector<int> mapNFakes(nmc, 0);
    std::vector<int> mapNClones(nmc, 0);

    std::vector<o2::its::Line> lineStore;
    lineStore.reserve(10000);

    f = trackFrames[n];
    cout << "track frames: " << f.firstFrame << ", " << f.firstIndex << " <-> " << f.lastFrame << ", " << f.lastIndex << endl;

    for (int frame = f.firstFrame; frame <= f.lastFrame; frame++) {
      if (frame != loadedEventTracks) {
        loadedEventTracks = -1;
        if (!recTree->GetEvent(frame))
          continue;
        loadedEventTracks = frame;
      }
      long nentr = linesArr->size();

      long firstIndex = (frame == f.firstFrame) ? f.firstIndex : 0;
      long lastIndex = (frame == f.lastFrame) ? f.lastIndex : nentr - 1;

      for (long i = firstIndex; i <= lastIndex; i++) {
        const Line& recLine = (*linesArr)[i];

        if(recLine.clusterLabels[0].getEventID() != n) continue;
        if(recLine.clusterLabels[0].getEventID() != recLine.clusterLabels[1].getEventID() || recLine.clusterLabels[1].getEventID() != recLine.clusterLabels[2].getEventID()) continue;
        if(!recLine.clusterLabels[0].isValid() || !recLine.clusterLabels[1].isValid() || !recLine.clusterLabels[2].isValid()) continue;
        if(recLine.clusterLabels[0].getSourceID() != 0 || recLine.clusterLabels[1].getSourceID() != 0 || recLine.clusterLabels[2].getSourceID() != 0) continue;


        if(recLine.clusterLabels[0].getTrackID() != recLine.clusterLabels[1].getTrackID() || recLine.clusterLabels[1].getTrackID() != recLine.clusterLabels[2].getTrackID()) continue;

        std::cout << "track ID " << recLine.clusterLabels[0].getTrackID() << " " << recLine.clusterLabels[1].getTrackID() << " " << recLine.clusterLabels[2].getTrackID() << std::endl;

        auto lab = recLine.clusterLabels[0];
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
          trackMap[mcid] = lineStore.size();
          // const TrackITS& recTrack = (*recArr)[i];
          lineStore.emplace_back(recLine);
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

      }
    } // cluster frames

    if (nClusters > 0)
      nDataEvents++;

    Int_t nGen = 0, nGoo = 0, nFak = 0, nClone = 0;

    for (int mc = 0; mc < nmc; mc++) {

      const auto& mcTrack = (*mcArr)[mc];

      if (mapNFakes[mc] > 0) { // Fake-track rate calculation
        nFak += mapNFakes[mc];
        fak->Fill(mcTrack.GetPt(), mapNFakes[mc]);
      }

      Int_t mID = mcTrack.getMotherTrackId();
      if (mID >= 0)
        continue; // Select primary particles
      Int_t pdg = mcTrack.GetPdgCode();
      if (TMath::Abs(pdg) != 211 && TMath::Abs(pdg) != 321 && TMath::Abs(pdg) != 2212)
        continue; // Select pions

      // if (clusMap[mc] != 0b1111111)
      //   continue;

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

        // const TrackITS& recTrack = trackStore[trackMap[mc]];
        // auto out = recTrack.getParamOut();
        // // recYOut = out.getY();
        // recZOut = out.getZ();
        // recPhiOut = out.getPhi();
        // recThetaOut = out.getTheta();
        // std::array<float, 3> p;
        // recTrack.getPxPyPzGlo(p);
        // recPt = recTrack.getPt();
        // recPhi = TMath::ATan2(p[1], p[0]);
        // recLam = TMath::ATan2(p[2], recPt);
        // Float_t vx = 0., vy = 0., vz = 0.; // Assumed primary vertex
        // Float_t bz = 5.;                   // Assumed magnetic field
        // recTrack.getImpactParams(vx, vy, vz, bz, ip);

        nt->Fill( // mcYOut,recYOut,
          mcZOut, recZOut, mcPhiOut, recPhiOut, mcThetaOut, recThetaOut, mcPhi, recPhi, mcLam, recLam, mcPt, recPt, ip[0],
          ip[1], mc);
      }

      if (mapNClones[mc] > 0) { // Clone-track rate calculation
        nClone += mapNClones[mc];
        clone->Fill(mcPt, mapNClones[mc]);
      }
    }

    statGen += nGen;
    statGoo += nGoo;
    statFak += nFak;
    statClone += nClone;

    if (nGen > 0) {
      Float_t eff = nGoo / Float_t(nGen);
      Float_t rat = nFak / Float_t(nGen);
      Float_t clonerat = nClone / Float_t(nGen);
      std::cout << "Good found tracks: " << nGoo << ",  efficiency: " << eff << ",  fake-track rate: " << rat << " clone rate " << clonerat << std::endl;
    }

  } // mc events

  cout << "\nOverall efficiency: " << endl;
  if (statGen > 0) {
    Float_t eff = statGoo / Float_t(statGen);
    Float_t rat = statFak / Float_t(statGen);
    Float_t clonerat = statClone / Float_t(statGen);
    std::cout << "Good found tracks/event: " << statGoo / nDataEvents << ",  efficiency: " << eff << ",  fake-track rate: " << rat << " clone rate " << clonerat << std::endl;
  }

  // "recPt>0" means "found tracks only"
  // "label>0" means "found good tracks only"

  /*
  new TCanvas;
  nt->Draw("ipD", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcLam-recLam", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcPt-recPt", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcZOut-recZOut", "recPt>0 && label>0 && abs(mcZOut-recZOut)<0.025");
  new TCanvas;
  nt->Draw("mcPhiOut-recPhiOut", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcThetaOut-recThetaOut", "recPt>0 && label>0");
  */
  TCanvas* c1 = new TCanvas;
  c1->SetLogx();
  c1->SetGridx();
  c1->SetGridy();
  num->Divide(num, den, 1, 1, "b");
  num->Draw("histe");
  fak->Divide(fak, den, 1, 1, "b");
  fak->SetLineColor(2);
  fak->Draw("histesame");
  clone->Divide(clone, den, 1, 1, "b");
  clone->SetLineColor(3);
  clone->Draw("histesame");
  f->Write();
  f->Close();
}
