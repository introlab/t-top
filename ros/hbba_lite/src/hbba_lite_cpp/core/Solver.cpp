#include <hbba_lite/core/Solver.h>

#include <hbba_lite/utils/HbbaLiteException.h>

using namespace std;

Solver::Solver()
{
}

void Solver::checkDesires(const vector<unique_ptr<Desire>>& desires,
    const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType)
{
    for (auto& desire : desires)
    {
        auto it = strategiesByDesireType.find(desire->type());
        if (it == strategiesByDesireType.end())
        {
            throw HbbaLiteException("No strategy found for \"" + string(desire->type().name()));
        }
    }
}
