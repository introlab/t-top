#ifndef HBBA_LITE_CORE_GECODE_SOLVER_H
#define HBBA_LITE_CORE_GECODE_SOLVER_H

#include <hbba_lite/core/Solver.h>

class GecodeSolver : public Solver
{
public:
    GecodeSolver();
    ~GecodeSolver() override = default;

    DECLARE_NOT_COPYABLE(GecodeSolver);
    DECLARE_NOT_MOVABLE(GecodeSolver);

    std::unordered_set<SolverResult> solve(
        const std::vector<std::unique_ptr<Desire>>& desires,
        const std::unordered_map<std::type_index, std::vector<std::unique_ptr<BaseStrategy>>>& strategiesByDesireType,
        const std::unordered_map<std::string, uint16_t>& systemResourcesByName) override;
};

#endif
