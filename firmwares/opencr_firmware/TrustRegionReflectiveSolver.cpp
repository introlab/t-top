#include "TrustRegionReflectiveSolver.h"

#include <utility>

using namespace std;

TrustRegionReflectiveParameters::TrustRegionReflectiveParameters() : ftol(1e-8), xtol(1e-8), gtol(1e-8), maxNfev(20) {}

TrustRegionReflectiveSolver::TrustRegionReflectiveSolver(TrustRegionReflectiveParameters parameters)
    : m_parameters(move(parameters))
{
}
