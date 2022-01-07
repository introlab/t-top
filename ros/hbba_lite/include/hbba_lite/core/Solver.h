#ifndef HBBA_LITE_CORE_SOLVER_H
#define HBBA_LITE_CORE_SOLVER_H

#include <hbba_lite/core/Desire.h>
#include <hbba_lite/core/Strategy.h>
#include <hbba_lite/utils/ClassMacros.h>

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <typeindex>

struct SolverResult
{
    size_t desireIndex;
    size_t strategyIndex;

    SolverResult(size_t desireIndex, size_t strategyIndex);
};

inline bool operator==(const SolverResult& a, const SolverResult& b)
{
    return a.desireIndex == b.desireIndex && a.strategyIndex == b.strategyIndex;
}

template<>
struct std::hash<SolverResult>
{
    std::size_t operator()(SolverResult const& x) const noexcept
    {
        std::size_t h1 = std::hash<size_t>()(x.desireIndex);
        std::size_t h2 = std::hash<size_t>()(x.strategyIndex);
        return h1 ^ (h2 << 1);
    }
};

class Solver
{
public:
    Solver();
    virtual ~Solver() = default;

    DECLARE_NOT_COPYABLE(Solver);
    DECLARE_NOT_MOVABLE(Solver);

    // Return the strategy to activate (Desire Type, strategy index)
    virtual std::unordered_set<SolverResult> solve(const std::vector<std::unique_ptr<Desire>>& desires,
        const std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>>& strategiesByDesireType,
        const std::unordered_map<std::string, uint16_t>& systemResourcesByName) = 0;
};

void checkDesireStrategies(const std::vector<std::unique_ptr<Desire>>& desires,
    const std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>>& strategiesByDesireType);
void checkStrategyResources(const std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>>& strategiesByDesireType,
    const std::unordered_map<std::string, uint16_t>& systemResourcesByName);

std::vector<size_t> selectMostIntenseEnabledDesireIndexes(const std::vector<std::unique_ptr<Desire>>& desires);

#endif
