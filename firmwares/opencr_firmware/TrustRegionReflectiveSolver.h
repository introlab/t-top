#ifndef TRUST_REGION_REFLECTIVE_SOLVER_H
#define TRUST_REGION_REFLECTIVE_SOLVER_H

#include <Eigen.h>

#include <cstddef>
#include <cmath>
#include <limits>

class TrustRegionReflectiveSolver;

// See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
class TrustRegionReflectiveParameters {
  float ftol;
  float xtol;
  float gtol;
  size_t maxNfev;

public:
  TrustRegionReflectiveParameters();

  bool setFtol(float ftol);
  bool setXtol(float xtol);
  bool setGtol(float gtol);
  bool setMaxNfev(size_t maxNfev);

  friend TrustRegionReflectiveSolver;
};

inline bool TrustRegionReflectiveParameters::setFtol(float newFtol) {
  if (newFtol <= 0.0) {
    return false;
  }
  ftol = newFtol;
  return true;
}

inline bool TrustRegionReflectiveParameters::setXtol(float newXtol) {
  if (newXtol <= 0.0) {
    return false;
  }
  xtol = newXtol;
  return true;
}

inline bool TrustRegionReflectiveParameters::setGtol(float newGtol) {
  if (newGtol <= 0.0) {
    return false;
  }
  gtol = newGtol;
  return true;
}

inline bool TrustRegionReflectiveParameters::setMaxNfev(size_t newMaxNfev) {
  maxNfev = newMaxNfev;
  return true;
}

template<int S>
struct TrustRegionReflectiveSolverResult {
  Eigen::Matrix<float, S, 1> x;
  float cost;
};

class TrustRegionReflectiveSolver {
  TrustRegionReflectiveParameters m_parameters;

public:
  TrustRegionReflectiveSolver(TrustRegionReflectiveParameters parameters);

  template<class F, int S>
  TrustRegionReflectiveSolverResult<S> solve(F f,
      const Eigen::Matrix<float, S, 1>& x0,
      const Eigen::Matrix<float, S, 1>& lowerBound,
      const Eigen::Matrix<float, S, 1>& upperBound);

private:
  template<int S>
  void makeStrictlyFeasible(Eigen::Matrix<float, S, 1>& x,
      const Eigen::Matrix<float, S, 1>& lowerBound,
      const Eigen::Matrix<float, S, 1>& upperBound);

  template<int S>
  float loss(const Eigen::Matrix<float, S, 1>& f);

  template<int S>
  void computeGrad(Eigen::Matrix<float, S, 1>& grad,
      const Eigen::Matrix<float, S, 1>& f,
      const Eigen::Matrix<float, S, S>& J);

  template<int S>
  void clScalingVector(Eigen::Matrix<float, S, 1>& v,
      Eigen::Matrix<float, S, 1>& dv,
      const Eigen::Matrix<float, S, 1>& x,
      const Eigen::Matrix<float, S, 1>& grad,
      const Eigen::Matrix<float, S, 1>& lowerBound,
      const Eigen::Matrix<float, S, 1>& upperBound);

  template<int S>
  void solveFAugmented(Eigen::Matrix<float, S, S>& Jh,
      Eigen::Matrix<float, S, 1>& s,
      Eigen::Matrix<float, S, S>& V,
      Eigen::Matrix<float, S, 1>& uf,
      const Eigen::Matrix<float, S, 1>& f,
      const Eigen::Matrix<float, S, S>& J,
      const Eigen::Matrix<float, S, 1>& d,
      const Eigen::Matrix<float, S, 1>& diagH);

  template<int S>
  void solveLsqTrustRegion(Eigen::Matrix<float, S, 1>& p,
      float& alpha,
      const Eigen::Matrix<float, S, 1>& s,
      const Eigen::Matrix<float, S, S>& V,
      const Eigen::Matrix<float, S, 1>& uf,
      float delta);

  template<int S>
  void phiAndDerivative(float& phi, float& phiPrime,
      float alpha, const Eigen::Matrix<float, S, 1>& suf, const Eigen::Matrix<float, S, 1>& s, float delta);

  template<int S>
  void selectStep(Eigen::Matrix<float, S, 1>& step,
      Eigen::Matrix<float, S, 1>& stepH,
      float& predictedReduction,
      const Eigen::Matrix<float, S, 1>& x,
      const Eigen::Matrix<float, S, S>& Jh,
      const Eigen::Matrix<float, S, 1>& diagH,
      const Eigen::Matrix<float, S, 1>& gH,
      const Eigen::Matrix<float, S, 1>& p,
      const Eigen::Matrix<float, S, 1>& pH,
      const Eigen::Matrix<float, S, 1>& d,
      float delta,
      const Eigen::Matrix<float, S, 1>& lowerBound,
      const Eigen::Matrix<float, S, 1>& upperBound,
      float theta);

  template<int S>
  bool inBounds(const Eigen::Matrix<float, S, 1>& x,
      const Eigen::Matrix<float, S, 1>& lowerBound,
      const Eigen::Matrix<float, S, 1>& upperBound);

  template<int S>
  float evaluateQuadratic(const Eigen::Matrix<float, S, S>& J,
      const Eigen::Matrix<float, S, 1>& g,
      const Eigen::Matrix<float, S, 1>& s,
      const Eigen::Matrix<float, S, 1>& diag);

  template<int S>
  void stepSizeToBound(float& minStep,
      Eigen::Matrix<int, S, 1>& hits,
      const Eigen::Matrix<float, S, 1>& x,
      const Eigen::Matrix<float, S, 1>& s,
      const Eigen::Matrix<float, S, 1>& lowerBound,
      const Eigen::Matrix<float, S, 1>& upperBound);

  template<int S>
  void intersectTrustRegion(float& t1, float& t2,
      const Eigen::Matrix<float, S, 1>& x, const Eigen::Matrix<float, S, 1>& s, float delta);

  template<int S>
  void buildQuadratic1d(float& a, float& b, float& c,
      const Eigen::Matrix<float, S, S>& J,
      const Eigen::Matrix<float, S, 1>& g,
      const Eigen::Matrix<float, S, 1>& s,
      const Eigen::Matrix<float, S, 1>& diag);

  template<int S>
  void buildQuadratic1d(float& a, float& b, float& c,
      const Eigen::Matrix<float, S, S>& J,
      const Eigen::Matrix<float, S, 1>& g,
      const Eigen::Matrix<float, S, 1>& s,
      const Eigen::Matrix<float, S, 1>& diag,
      const Eigen::Matrix<float, S, 1>& s0);

  void minimizeQuadratic1d(float& x, float& y, float a, float b, float c, float lowerBound, float upperBound);

  void updateTrRadius(float& deltaNew, float& ratio,
      float delta, float actualReduction, float predictedReduction, float stepNorm, bool boundHit);
};

// Heavily inspired by https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
template<class F, int S>
TrustRegionReflectiveSolverResult<S> TrustRegionReflectiveSolver::solve(F fun,
    const Eigen::Matrix<float, S, 1>& x0,
    const Eigen::Matrix<float, S, 1>& lowerBound,
    const Eigen::Matrix<float, S, 1>& upperBound)
{
  Eigen::Matrix<float, S, 1> x(x0);
  makeStrictlyFeasible(x, lowerBound, upperBound);

  Eigen::Matrix<float, S, 1> f;
  Eigen::Matrix<float, S, S> J;
  fun(f, J, x);

  float cost = loss(f);
  Eigen::Matrix<float, S, 1> g;
  computeGrad(g, f, J);

  Eigen::Matrix<float, S, 1> v, dv;
  clScalingVector(v, dv, x, g, lowerBound, upperBound);

  float delta = (x.cwiseQuotient(v.cwiseSqrt())).norm();
  if (delta == 0) {
    delta = 1.f;
  }

  float alpha = 0;
  size_t nfev = 0;

  while (true) {
    clScalingVector(v, dv, x, g, lowerBound, upperBound);
    float gNorm = g.cwiseProduct(v).template lpNorm<Eigen::Infinity>();

    if (gNorm < m_parameters.gtol || nfev >= m_parameters.maxNfev) {
      break;
    }

    Eigen::Matrix<float, S, 1> d = v.cwiseSqrt();
    Eigen::Matrix<float, S, 1> diagH = g.cwiseProduct(dv);
    Eigen::Matrix<float, S, 1> gH = d.cwiseProduct(g);

    Eigen::Matrix<float, S, S> Jh;
    Eigen::Matrix<float, S, 1> s;
    Eigen::Matrix<float, S, S> V; //2nd col sign error
    Eigen::Matrix<float, S, 1> uf; //2nd row sign error
    solveFAugmented(Jh, s, V, uf, f, J, d, diagH);

    float theta = std::max(0.995f, 1 - gNorm);
    float actualReduction = -1.f;
    while (actualReduction <= 0 && nfev < m_parameters.maxNfev) {
      Eigen::Matrix<float, S, 1> pH;
      solveLsqTrustRegion(pH, alpha, s, V, uf, delta);

      Eigen::Matrix<float, S, 1> p = d.cwiseProduct(pH);
      Eigen::Matrix<float, S, 1> step;
      Eigen::Matrix<float, S, 1> stepH;
      float predictedReduction;
      selectStep(step, stepH, predictedReduction, x, Jh, diagH, gH, p, pH, d, delta, lowerBound, upperBound, theta);

      Eigen::Matrix<float, S, 1> xNew = x + step;
      makeStrictlyFeasible(xNew, lowerBound, upperBound);

      Eigen::Matrix<float, S, 1> fNew;
      Eigen::Matrix<float, S, S> JNew;
      fun(fNew, JNew, xNew);
      nfev++;

      float stepHNorm = stepH.norm();
      if (!fNew.allFinite()) {
        delta = 0.25f * stepHNorm;
        continue;
      }

      float costNew = loss(fNew);
      actualReduction = cost - costNew;

      float deltaNew, ratio;
      updateTrRadius(deltaNew, ratio, delta, actualReduction, predictedReduction, stepHNorm, stepHNorm > 0.95 * delta);

      float stepNorm = step.norm();
      if ((actualReduction < m_parameters.ftol * cost && ratio > 0.25f) ||
          (stepNorm < m_parameters.xtol * (m_parameters.xtol + x.norm()))) {
        break;
      }

      alpha *= delta / deltaNew;
      delta = deltaNew;

      if (actualReduction > 0) {
        x = xNew;
        f = fNew;
        cost = costNew;
        J = JNew;
        computeGrad(g, f, J);
      }
    }
  }

  return TrustRegionReflectiveSolverResult<S>{x, cost};
}

template<int S>
void TrustRegionReflectiveSolver::makeStrictlyFeasible(Eigen::Matrix<float, S, 1>& x,
    const Eigen::Matrix<float, S, 1>& lowerBound,
    const Eigen::Matrix<float, S, 1>& upperBound) {
  constexpr float STEP = 1e-8;
  for (int i = 0; i < S; i++) {
    if (x[i] < lowerBound[i]) {
      x[i] = lowerBound[i] + STEP;
    }
    else if (x[i] > upperBound[i]) {
      x[i] = upperBound[i] - STEP;
    }
  }
}

template<int S>
float TrustRegionReflectiveSolver::loss(const Eigen::Matrix<float, S, 1>& f)
{
  return 0.5 * f.dot(f);
}

template<int S>
void TrustRegionReflectiveSolver::computeGrad(Eigen::Matrix<float, S, 1>& grad,
    const Eigen::Matrix<float, S, 1>& f,
    const Eigen::Matrix<float, S, S>& J) {
  grad = J.transpose() * f;
}

template<int S>
void TrustRegionReflectiveSolver::clScalingVector(Eigen::Matrix<float, S, 1>& v,
    Eigen::Matrix<float, S, 1>& dv,
    const Eigen::Matrix<float, S, 1>& x,
    const Eigen::Matrix<float, S, 1>& grad,
    const Eigen::Matrix<float, S, 1>& lowerBound,
    const Eigen::Matrix<float, S, 1>& upperBound) {
  for (int i = 0; i < S; i++) {
    if (grad[i] < 0 && std::isfinite(upperBound[i])) {
      v[i] = upperBound[i] - x[i];
      dv[i] = -1;
    }
    else if (grad[i] > 0 && std::isfinite(lowerBound[i])) {
      v[i] = x[i] - lowerBound[i];
      dv[i] = 1;
    }
    else {
      v[i] = 1;
      dv[i] = 0;
    }
  }
}

template<int S>
void TrustRegionReflectiveSolver::solveFAugmented(Eigen::Matrix<float, S, S>& Jh,
    Eigen::Matrix<float, S, 1>& s,
    Eigen::Matrix<float, S, S>& V,
    Eigen::Matrix<float, S, 1>& uf,
    const Eigen::Matrix<float, S, 1>& f,
    const Eigen::Matrix<float, S, S>& J,
    const Eigen::Matrix<float, S, 1>& d,
    const Eigen::Matrix<float, S, 1>& diagH) {
  Eigen::Matrix<float, 2 * S, 1> fAugmented;
  fAugmented.setZero();
  fAugmented.topRows(S) = f;

  Eigen::Matrix<float, 2 * S, S> JAugmented;
  Jh = J.array().rowwise() * d.transpose().array();
  JAugmented.topRows(S) = Jh;
  JAugmented.bottomRows(S) = diagH.cwiseSqrt().asDiagonal();

  auto svd = JAugmented.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  s = svd.singularValues();
  V = svd.matrixV();
  uf = svd.matrixU().leftCols(S).transpose() * fAugmented;
}

template<int S>
void TrustRegionReflectiveSolver::solveLsqTrustRegion(Eigen::Matrix<float, S, 1>& p,
    float& alpha,
    const Eigen::Matrix<float, S, 1>& s,
    const Eigen::Matrix<float, S, S>& V,
    const Eigen::Matrix<float, S, 1>& uf,
    float delta) {
  constexpr float EPS = 1e-7;
  constexpr int MAX_ITER = 10;
  constexpr float RTOL = 0.01f;

  Eigen::Matrix<float, S, 1> suf = s.cwiseProduct(uf);

  float threshold = EPS * S * s[0];
  bool fullRank = s[S - 1] > threshold;

  if (fullRank) {
    p = -V * (uf.cwiseQuotient(s));
    if (p.norm() <= delta) {
      alpha = 0.f;
      return;
    }
  }

  float alphaUpper = suf.norm() / delta;
  float alphaLower = 0.f;
  if (fullRank) {
    float phi, phiPrime;
    phiAndDerivative(phi, phiPrime, 0.f, suf, s, delta);
    alphaLower = -phi / phiPrime;
  }

  if (!fullRank && alpha == 0.f) {
    alpha = std::max(0.001f * alphaUpper, std::sqrt(alphaLower * alphaUpper));
  }

  for (int i = 0; i < MAX_ITER; i++) {
    if (alpha < alphaLower || alpha > alphaUpper) {
      alpha = std::max(0.001f * alphaUpper, std::sqrt(alphaLower * alphaUpper));
    }

    float phi, phiPrime;
    phiAndDerivative(phi, phiPrime, alpha, suf, s, delta);

    if (phi < 0.f) {
      alphaUpper = alpha;
    }

    float ratio = phi / phiPrime;
    alphaLower = std::max(alphaLower, alpha - ratio);
    alpha -= (phi + delta) * ratio / delta;

    if (std::abs(phi) < RTOL * delta) {
      break;
    }
  }

  p =-V * suf.cwiseQuotient((s.array().pow(2) + alpha).matrix());
  p *= delta / p.norm();
}

template<int S>
void TrustRegionReflectiveSolver::phiAndDerivative(float& phi, float& phiPrime,
    float alpha, const Eigen::Matrix<float, S, 1>& suf, const Eigen::Matrix<float, S, 1>& s, float delta) {
  Eigen::Matrix<float, S, 1> denum = s.array().pow(2) + alpha;
  float pNorm = suf.cwiseQuotient(denum).norm();
  phi = pNorm - delta;
  phiPrime = -suf.array().pow(2).cwiseQuotient(denum.array().pow(3)).sum() / pNorm;
}

template<int S>
void TrustRegionReflectiveSolver::selectStep(Eigen::Matrix<float, S, 1>& step,
    Eigen::Matrix<float, S, 1>& stepH,
    float& predictedReduction,
    const Eigen::Matrix<float, S, 1>& x,
    const Eigen::Matrix<float, S, S>& Jh,
    const Eigen::Matrix<float, S, 1>& diagH,
    const Eigen::Matrix<float, S, 1>& gH,
    const Eigen::Matrix<float, S, 1>& p,
    const Eigen::Matrix<float, S, 1>& pH,
    const Eigen::Matrix<float, S, 1>& d,
    float delta,
    const Eigen::Matrix<float, S, 1>& lowerBound,
    const Eigen::Matrix<float, S, 1>& upperBound,
    float theta) {
  Eigen::Matrix<float, S, 1> xAddP = x + p;
  if (inBounds(xAddP, lowerBound, upperBound)) {
    step = p;
    stepH = pH;
    predictedReduction = -evaluateQuadratic(Jh, gH, pH, diagH);
    return;
  }

  float pStride;
  Eigen::Matrix<int, S, 1> hits;
  stepSizeToBound(pStride, hits, x, p, lowerBound, upperBound);

  Eigen::Matrix<float, S, 1> rH = pH;
  for (int i = 0; i < S; i++) {
    if (hits[i] != 0) {
      rH[i] *= 1;
    }
  }

  Eigen::Matrix<float, S, 1> r = d.cwiseProduct(rH);

  Eigen::Matrix<float, S, 1> pScaled = p * pStride;
  Eigen::Matrix<float, S, 1> pHScaled = pH * pStride;
  Eigen::Matrix<float, S, 1> xOnBound = x + pScaled;

  float _s, toTr, toBound;
  Eigen::Matrix<int, S, 1> _v;
  intersectTrustRegion(_s, toTr, pHScaled, rH, delta);
  stepSizeToBound(toBound, _v, xOnBound, r, lowerBound, upperBound);

  float rStride = std::min(toBound, toTr);
  float rStrideL = 0.f;
  float rStrideU = -1.f;
  if (rStride > 0.f) {
    rStrideL = (1 - theta) * pStride / rStride;
    if (rStride == toBound) {
      rStrideU = theta * toBound;
    }
    else {
      rStrideU = toTr;
    }
  }

  float rValue;
  if (rStrideL <= rStrideU) {
    float a, b, c;
    buildQuadratic1d(a, b, c, Jh, gH, rH, diagH, pHScaled);

    minimizeQuadratic1d(rStride, rValue, a, b, c, rStrideL, rStrideU);

    rH *= rStride;
    rH += pHScaled;
    r = d.cwiseProduct(rH);
  }
  else {
    rValue = std::numeric_limits<float>::infinity();
  }

  pScaled *= theta;
  pHScaled *= theta;
  float pValue = evaluateQuadratic(Jh, gH, pHScaled, diagH);

  Eigen::Matrix<float, S, 1> agH = -gH;
  Eigen::Matrix<float, S, 1> ag = d.cwiseProduct(agH);

  toTr = delta / agH.norm();
  stepSizeToBound(toBound, _v, x, ag, lowerBound, upperBound);

  float agStride;
  if (toBound < toTr) {
    agStride = theta * toBound;
  }
  else {
    agStride = toTr;
  }

  float a, b, c;
  buildQuadratic1d(a, b, c, Jh, gH, agH, diagH);
  float agValue;
  minimizeQuadratic1d(agStride, agValue, a, b, c, 0.f, agStride);
  agH *= agStride;
  ag *= agStride;

  if (pValue < rValue && pValue < agValue) {
    step = pScaled;
    stepH = pHScaled;
    predictedReduction = -pValue;
  }
  else if (rValue < pValue && rValue < agValue) {
    step = r;
    stepH = rH;
    predictedReduction = -rValue;
  }
  else {
    step = ag;
    stepH = agH;
    predictedReduction = -agValue;
  }
}

template<int S>
bool TrustRegionReflectiveSolver::inBounds(const Eigen::Matrix<float, S, 1>& x,
    const Eigen::Matrix<float, S, 1>& lowerBound,
    const Eigen::Matrix<float, S, 1>& upperBound) {
  return (x.array() >= lowerBound.array() && x.array() <= upperBound.array()).all();
}

template<int S>
float TrustRegionReflectiveSolver::evaluateQuadratic(const Eigen::Matrix<float, S, S>&J,
    const Eigen::Matrix<float, S, 1>& g,
    const Eigen::Matrix<float, S, 1>& s,
    const Eigen::Matrix<float, S, 1>& diag) {
  Eigen::Matrix<float, S, 1> Js = J * s;
  float q = Js.dot(Js);
  q += s.cwiseProduct(diag).dot(s);

  float l = s.dot(g);
  return 0.5f * q + l;
}

template<int S>
void TrustRegionReflectiveSolver::stepSizeToBound(float& minStep,
    Eigen::Matrix<int, S, 1>& hits,
    const Eigen::Matrix<float, S, 1>& x,
    const Eigen::Matrix<float, S, 1>& s,
    const Eigen::Matrix<float, S, 1>& lowerBound,
    const Eigen::Matrix<float, S, 1>& upperBound) {
  Eigen::Matrix<float, S, 1> steps;
  steps.setConstant(std::numeric_limits<float>::infinity());

  for (int i = 0; i < S; i++) {
    if (s[i] == 0.f) {
      continue;
    }
    steps[i] = std::max((lowerBound[i] - x[i]) / s[i], (upperBound[i] - x[i]) / s[i]);
  }

  int minStepIndex;
  minStep = steps.minCoeff(&minStepIndex);

  hits.setZero();
  for (int i = 0; i < S; i++) {
    if (s[i] != 0.f && (i == minStepIndex || steps[i] == minStep)) {
      if (s[i] < 0.f) {
        hits[i] = -1;
      }
      else if (s[i] > 0.f) {
        hits[i] = 1;
      }
    }
  }
}

template<int S>
void TrustRegionReflectiveSolver::intersectTrustRegion(float& t1, float& t2,
    const Eigen::Matrix<float, S, 1>& x, const Eigen::Matrix<float, S, 1>& s, float delta) {
  float a = s.dot(s);
  float b = x.dot(s);
  float c = x.dot(x) - delta * delta;
  float d = std::sqrt(b * b - a * c);

  float q = -(b + std::copysign(d, b));
  t1 = q / a;
  t2 = c / q;

  if (t2 > t1) {
    std::swap(t1, t2);
  }
}

template<int S>
void TrustRegionReflectiveSolver::buildQuadratic1d(float& a, float& b, float& c,
    const Eigen::Matrix<float, S, S>& J,
    const Eigen::Matrix<float, S, 1>& g,
    const Eigen::Matrix<float, S, 1>& s,
    const Eigen::Matrix<float, S, 1>& diag) {

  Eigen::Matrix<float, S, 1> v = J * s;
  a = v.dot(v);
  a += s.cwiseProduct(diag).dot(s);
  a *= 0.5;

  b = g.dot(s);
  c = 0.f;
}

template<int S>
void TrustRegionReflectiveSolver::buildQuadratic1d(float& a, float& b, float& c,
    const Eigen::Matrix<float, S, S>& J,
    const Eigen::Matrix<float, S, 1>& g,
    const Eigen::Matrix<float, S, 1>& s,
    const Eigen::Matrix<float, S, 1>& diag,
    const Eigen::Matrix<float, S, 1>& s0) {

  Eigen::Matrix<float, S, 1> v = J * s;
  a = v.dot(v);
  a += s.cwiseProduct(diag).dot(s);
  a *= 0.5;

  b = g.dot(s);
  c = 0.f;

  Eigen::Matrix<float, S, 1> u = J * s0;
  b += u.dot(v);
  c = 0.5 * u.dot(u) + g.dot(s0);

  b += s0.cwiseProduct(diag).dot(s);
  c += 0.5 * s0.cwiseProduct(diag).dot(s0);
}

inline void TrustRegionReflectiveSolver::minimizeQuadratic1d(float& x, float& y,
    float a, float b, float c, float lowerBound, float upperBound) {
  Eigen::Array<float, 3, 1> t;
  t << lowerBound, upperBound, upperBound;

  if (a != 0.f) {
    float extremum = -0.5f * b / a;
    if (lowerBound < extremum && extremum < upperBound) {
      t[2] = extremum;
    }
  }

  Eigen::Array<float, 3, 1> allY = t * (a * t + b) + c;
  int minIndex;
  y = allY.matrix().minCoeff(&minIndex);
  x = t[minIndex];
}

inline void TrustRegionReflectiveSolver::updateTrRadius(float& deltaNew, float& ratio,
    float delta, float actualReduction, float predictedReduction, float stepNorm, bool boundHit)
{
  if (predictedReduction > 0) {
    ratio = actualReduction / predictedReduction;
  }
  else if (predictedReduction == 0.f && actualReduction == 0.f) {
    ratio = 1;
  }
  else {
    ratio = 0;
  }

  if (ratio < 0.25f) {
    deltaNew = 0.25f * stepNorm;
  }
  else if (ratio > 0.75f && boundHit) {
    deltaNew = 2.f * delta;
  }
  else {
    deltaNew = delta;
  }
}

#endif
