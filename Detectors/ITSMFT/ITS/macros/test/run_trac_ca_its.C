#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>
#include <chrono>
#include <iostream>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>
#include "DetectorsCommonDataFormats/NameConf.h"

#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ClusterLines.h"
#include "ReconstructionDataFormats/Vertex.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

#include <TGraph.h>
#include <TH2D.h>
#include <TCanvas.h>

#include "ITStracking/Configuration.h"

using namespace o2::gpu;
using o2::its::MemoryParameters;
using o2::its::TrackingParameters;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

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

void run_trac_ca_its(std::string path = "./",
                     std::string outputfile = "o2trac_its.root",
                     std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                     std::string dictfile = "",
                     std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root",
                     std::string paramfilename = "o2sim_par.root",
                     std::string kinefile = "o2sim_Kine.root")
{

  gSystem->Load("libO2ITStracking.so");

  std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance());
  // std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance("CUDA", true)); // for GPU with CUDA
  auto* chainITS = rec->AddChain<GPUChainITS>();
  rec->Init();

  o2::its::Tracker tracker(chainITS->GetITSTrackerTraits());
  //o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  o2::its::ROframe event(0);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object";
  }
  o2::base::GeometryManager::loadGeometry(path);
  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma";
  }
  double origD[3] = {0., 0., 0.};
  tracker.setBz(field->getBz(origD));

  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms


  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  if (!itsClusters.GetBranch("ITSClusterComp")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterComp in the input tree";
  }
  std::vector<o2::itsmft::CompClusterExt>* cclusters = nullptr;
  itsClusters.SetBranchAddress("ITSClusterComp", &cclusters);

  if (!itsClusters.GetBranch("ITSClusterPatt")) {
    LOG(FATAL) << "Did not find ITS cluster patterns branch ITSClusterPatt in the input tree";
  }
  std::vector<unsigned char>* patterns = nullptr;
  itsClusters.SetBranchAddress("ITSClusterPatt", &patterns);

  MCLabCont* labels = nullptr;
  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(WARNING) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  } else {
    itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);
  }

  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }

  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersMC2ROF")) {
    LOG(WARNING) << "Did not find ITS clusters branch ITSClustersMC2ROF in the input tree";
  } else
    itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);

  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);

  itsClusters.GetEntry(0);

  //
  // Clusters
  TFile::Open(inputClustersITS.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
  //-------------------------------------------------

  o2::itsmft::TopologyDictionary dict;
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
    dict.readBinaryFile(dictfile);
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

  // MC tracks
  TFile* file0 = TFile::Open(kinefile.data());
  if(!file0) printf("Error with %s file! \n", kinefile.data());
  TTree* mcTree = (TTree*)gFile->Get("o2sim");
  mcTree->SetBranchStatus("*", 0); //disable all branches
  //mcTree->SetBranchStatus("MCEventHeader.*",1);
  mcTree->SetBranchStatus("MCTrack*", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);

  //-------------------------------------------------
  std::vector<o2::its::TrackITSExt> tracks;
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  MCLabCont trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);

  o2::its::VertexerTraits* traits = o2::its::createVertexerTraits();
  o2::its::Vertexer vertexer(traits);

  std::vector<double> ncls;
  std::vector<double> time;

  const float kmaxDCAxy1[5] = /*{1.f,0.5,0.5,1.7,3.};/*/{1.f*2.0,0.4f*2.0,0.4f*2.0,2.0f*2.0,3.f*2.0};
  const float kmaxDCAz1[5] = /*{2.f,0.8,0.8,3.,5.};/*/{1.f*2.0,0.4f*2.0,0.4f*2.0,2.0f*2.0,3.f*2.0};
  const float kmaxDN1[4] = /*{0.006f,0.0045f,0.01f,0.04f};/*/{0.005f*2.0,0.0035f*2.0,0.009f*2.0,0.03f*2.0};
  const float kmaxDP1[4] = /*{0.04f,0.01f,0.012f,0.014f};/*/{0.02f*2.0,0.005f*2.0,0.006f*2.0,0.007f*2.0};
  const float kmaxDZ1[6] = /*{2.0f,2.0f,2.f,2.f,2.f,2.f};/*/{1.f*2.0,1.f*2.0,2.0f*2.0,2.0f*2.0,2.0f*2.0,2.0f*2.0};
  const float kDoublTanL1 = /*0.12f;/*/0.05f*5.;
  const float kDoublPhi1 = /*0.4f;/*/0.2f*5.;

  std::vector<TrackingParameters> trackParams(2);
  // trackParams[0].TrackletMaxDeltaPhi = 0.05f;
  // trackParams[1].TrackletMaxDeltaPhi = 0.1f;
  trackParams[1].MinTrackLength = 7;
  trackParams[1].TrackletMaxDeltaPhi = 0.3;
  trackParams[1].CellMaxDeltaPhi = 0.2*2;
  trackParams[1].CellMaxDeltaTanLambda = 0.05*2;
  std::copy(kmaxDZ1, kmaxDZ1 + 6, trackParams[1].TrackletMaxDeltaZ);
  std::copy(kmaxDCAxy1,kmaxDCAxy1+5,trackParams[1].CellMaxDCA);
  std::copy(kmaxDCAz1, kmaxDCAz1+5, trackParams[1].CellMaxDeltaZ);
  std::copy(kmaxDP1, kmaxDP1+4, trackParams[1].NeighbourMaxDeltaCurvature);
  std::copy(kmaxDN1, kmaxDN1+4, trackParams[1].NeighbourMaxDeltaN);

  std::vector<MemoryParameters> memParams(2);
  for (auto& coef : memParams[1].CellsMemoryCoefficients)
    coef *= 40;
  for (auto& coef : memParams[1].TrackletsMemoryCoefficients)
    coef *= 40;

  tracker.setParameters(memParams, trackParams);

  int currentEvent = -1;
  gsl::span<const unsigned char> patt(patterns->data(), patterns->size());
  auto pattIt = patt.begin();
  auto clSpan = gsl::span(cclusters->data(), cclusters->size());

  //plotting purity + efficiency
  const int arrDimension = 50;
  const int minNOfContributorsPerVertex = 25;

  TH2D* hPurity = new TH2D("hPurity", "Purity", 2, 0, 2, arrDimension, minNOfContributorsPerVertex, minNOfContributorsPerVertex+arrDimension);

  TH1D* nOfDifEvents = new TH1D("nOfDifEvents", "nOfDifEvents", 10, 0, 10); //# of different events contributing to the created vertex (reconstruced with vertexer)
  TH1D* nOfVerPerROF = new TH1D("nOfVerPerROF", "nOfVerPerROF", 15, -1, 14); //# of light vertices from vertexer

  // search for MC events in frames
  //imporant for checking cluster map

  std::cout << "Find mc events in cluster frames.. " << std::endl;

  int loadedEventClust = -1;
  Int_t nev = mcTree->GetEntriesFast();

  std::vector<DataFrames> clusterFrames(nev);

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

  //

  int purity[2][arrDimension] = {0};

  for (auto& rof : *rofs) {
    auto start = std::chrono::high_resolution_clock::now();
    auto it = pattIt;
    o2::its::ioutils::loadROFrameData(rof, event, clSpan, pattIt, dict, labels);

    vertexer.initialiseVertexer(&event);
    vertexer.findTracklets();
    // vertexer.filterMCTracklets(); // to use MC check
    vertexer.validateTracklets();
    vertexer.findVertices();
    o2::its::VertexingParameters verPar = vertexer.getVertParameters(); //needed for DCA check

    //light vertices needed for this study as "normal" Vertex object does not have MC info or info about lines
    std::vector<o2::its::lightVertex> vertITS = vertexer.exportLightVertices();
    nOfVerPerROF->Fill(vertITS.size());
    // std::vector<Vertex> vertITS = vertexer.exportVertices();
    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
    vtxROF.setNEntries(vertITS.size());

    for (const auto& vtx : vertITS) {
      vertices.emplace_back(Point3D<float>(vtx.mX, vtx.mY, vtx.mZ), vtx.mRMS2, vtx.mContributors, vtx.mAvgDistance2);
      vertices.back().setTimeStamp(vtx.mTimeStamp);
    }

    for(const auto& vtx : vertITS){
      int eventId = vtx.mEventId;
      if (!mcTree->GetEvent(eventId)) continue;

      Int_t nmc = mcArr->size();
      std::vector<int> clusMap(nmc, 0);

      //this part only needed for cluster map cut of MC tracks
      DataFrames f = clusterFrames[eventId];
      if ((f.firstFrame > f.lastFrame) || ((f.firstFrame == f.lastFrame) && (f.firstIndex > f.lastIndex))) continue;

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
          if (!lab.isValid() || lab.getSourceID() != 0 || lab.getEventID() != eventId) continue;
          int mcid = lab.getTrackID();
          if (mcid < 0 || mcid >= nmc) {
            std::cout << "cluster mc label is too big!!!" << std::endl;
            continue;
          }
          if (!lab.isCorrect()) continue;

          const o2::itsmft::CompClusterExt& c = (*clusArr)[i];
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

      //# of MC particles that pass all cuts (charged, primary, PID, pseudorapidy, cluster map)
      //imporant for finding out whether the triggered event was real high multiplicity event

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
        if (TMath::Abs(pdg) != 211 && TMath::Abs(pdg) != 321 && TMath::Abs(pdg) != 2212 && TMath::Abs(pdg) != 1 && TMath::Abs(pdg) != 13) continue; // Select PID
        if(TMath::Abs( mcTrack.GetEta() ) > 1.1 ) continue;
        if (clusMap[mc] != 0b1111111) continue;
        nOfCharged++;
      } // end mc tracks

      int contributors = vtx.mContributors;
      std::vector<o2::its::Line> lines = vtx.mLines;
      std::array<float, 3> tmpVertex{vtx.mX, vtx.mY, vtx.mZ};

      std::vector<int> evIdDif; //vector with all MC event IDs of contributing lines
      for(int iLine(0); iLine < lines.size(); iLine++){
        //(additional) cut on DCA between the vertex and its contributors
        //already (to some extent) checked in vertexer when reconstructing the vertex
        if(o2::its::Line::getDistanceFromPoint(lines[iLine], tmpVertex) >= verPar.pairCut) continue;
        evIdDif.push_back(lines[iLine].evtId);
      }
      sort(evIdDif.begin(),evIdDif.end());
      evIdDif.erase(unique(evIdDif.begin(),evIdDif.end()),evIdDif.end());
      nOfDifEvents->Fill(evIdDif.size()); //# of unique events contributing to the vertex

      bool isRealHMevent = kFALSE;
      if(nOfCharged > 25) isRealHMevent = kTRUE;
      for(int cut(0); cut < arrDimension; cut++){
        if(contributors > minNOfContributorsPerVertex + cut){
          if(isRealHMevent) {
            purity[1][cut]++;
            hPurity->Fill(1.0, minNOfContributorsPerVertex+cut);
          }
          else {
            purity[0][cut]++;
            hPurity->Fill(0.0, minNOfContributorsPerVertex+cut);
          }
        }
      }
    } // end vertices

    if (vertITS.empty()) {
        std::cout << " - Vertex not reconstructed, tracking skipped " << std::endl;
    }
    trackClIdx.clear();
    tracksITS.clear();
    tracker.clustersToTracks(event);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_t{end - start};

    ncls.push_back(event.getTotalClusters());
    time.push_back(diff_t.count());

    tracks.swap(tracker.getTracks());
    for (auto& trc : tracks) {
      trc.setFirstClusterEntry(trackClIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters();
      for (int ic = 0; ic < ncl; ic++) {
        trackClIdx.push_back(trc.getClusterIndex(ic));
      }
      tracksITS.emplace_back(trc);
    }

    trackLabels = tracker.getTrackLabels(); /// FIXME: assignment ctor is not optimal.
    outTree.Fill();
  }

  outFile.cd();
  outTree.Write();


  TGraph* graph = new TGraph(ncls.size(), ncls.data(), time.data());
  graph->SetMarkerStyle(20);
  graph->Draw("AP");

  //purity and efficiency calculation and plotting
  TCanvas* c2 = new TCanvas();
  Float_t xPur[arrDimension], yPur[arrDimension], yLoss[arrDimension], yEff[arrDimension];
  for(int cut(0); cut < arrDimension; cut++){
    xPur[cut] = minNOfContributorsPerVertex+cut;
    Float_t total = Float_t(purity[0][cut]) + Float_t(purity[1][cut]);
    if(total > 0) yPur[cut] = Float_t(purity[1][cut])/total;
    else yPur[cut] = 0.0;
    yLoss[cut] = Float_t(purity[1][0] - purity[1][cut]) / Float_t(purity[1][0]);
    yEff[cut] = 1.0 - yLoss[cut];
  }
  TGraph *pur = new TGraph(arrDimension, xPur, yPur);
  pur->SetTitle("; Min # of contributors per vertex; HM events / all events");
  pur->Draw();
  pur->SetName("pur");
  pur->Write();
  c2->SaveAs("purity.pdf");

  //loss count
  TCanvas* c3 = new TCanvas();
  TGraph* loss = new TGraph(arrDimension, xPur, yLoss);
  loss->SetTitle("; Min # of charged tracks per ROF; Rel. loss of HM events");
  loss->Draw();
  loss->Write();
  c3->SaveAs("relLoss.pdf");

  //efficiency
  TCanvas* c4 = new TCanvas();
  TGraph* effi = new TGraph(arrDimension, xPur, yEff);
  effi->SetTitle("; Min # of contributors per vertex; Efficiency");
  effi->Draw();
  effi->SetName("effi");
  effi->Write();
  c4->SaveAs("efficiency.pdf");

  nOfDifEvents->Write();
  nOfVerPerROF->Write();
  hPurity->Write();

  outFile.Close();
}

#endif
