#ifndef HBBA_LITE_CORE_SOLVER_H
#define HBBA_LITE_CORE_SOLVER_H

#include <hbba_lite/core/Desire.h>
#include <hbba_lite/core/Strategy.h>

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <typeindex>

template<>
struct std::hash<std::pair<std::type_index, size_t>>
{
    std::size_t operator()(std::pair<std::type_index, size_t> const& x) const noexcept
    {
        std::size_t h1 = std::hash<std::type_index>()(x.first);
        std::size_t h2 = std::hash<size_t>()(x.second);
        return h1 ^ (h2 << 1);
    }
};

class Solver
{
public:
    Solver();
    virtual ~Solver() = default;

    // Return the strategy to activate (Desire Type, strategy index)
    virtual std::unordered_set<std::pair<std::type_index, size_t>> solve(const std::vector<std::unique_ptr<Desire>>& desires,
        const std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>>& strategiesByDesireType,
        const std::unordered_map<std::string, uint16_t>& systemResourcesByName) = 0;

protected:
    void checkDesires(const std::vector<std::unique_ptr<Desire>>& desires,
        const std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>>& strategiesByDesireType);
};

#endif
