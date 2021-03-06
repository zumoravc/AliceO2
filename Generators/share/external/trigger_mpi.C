// MPI trigger
//
//   usage: o2sim --trigger external --extTrgFile trigger_mpi.C
// options:                          --extTrgFunc "trigger_mpi()"
//

/// \author R+Preghenella - February 2020

#include "Generators/Trigger.h"
#include "Pythia8/Pythia.h"
#include "TPythia6.h"
#include "FairLogger.h"

  o2::eventgen::DeepTrigger
trigger_mpi(int mpiMin = 30)
{
  return [mpiMin](void* interface, std::string name) -> bool {
    int nMPI = 0;
    if (!name.compare("pythia8")) {
      auto py8 = reinterpret_cast<Pythia8::Pythia*>(interface);
      nMPI = py8->info.nMPI();
    } else if (!name.compare("pythia6")) {
      auto py6 = reinterpret_cast<TPythia6*>(interface);
      nMPI = py6->GetMSTI(31);
    } else
      LOG(FATAL) << "Cannot define MPI for generator interface \'" << name << "\'";
    LOG(INFO) << "nMPIs: " << nMPI << "\'";

    // only 10% of events are HM
    if(gRandom->Integer(10) == 0) return nMPI >= mpiMin;
    return nMPI >= 1;
  };
}

