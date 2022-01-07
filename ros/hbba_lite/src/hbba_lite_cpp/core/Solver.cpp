#include <hbba_lite/core/Solver.h>

#include <hbba_lite/utils/HbbaLiteException.h>

using namespace std;

SolverResult::SolverResult(size_t desireIndex, size_t strategyIndex) :
        desireIndex(desireIndex),
        strategyIndex(strategyIndex)
{
}

Solver::Solver()
{
}

void checkDesireStrategies(const vector<unique_ptr<Desire>>& desires,
    const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType)
{
    for (auto& desire : desires)
    {
        auto it = strategiesByDesireType.find(desire->type());
        if (it == strategiesByDesireType.end() || it->second.size() == 0)
        {
            throw HbbaLiteException("No strategy found for \"" + string(desire->type().name()));
        }
    }
}

void checkStrategyResources(const unordered_map<type_index, vector<unique_ptr<BaseStrategy>>>& strategiesByDesireType,
    const unordered_map<string, uint16_t>& systemResourcesByName)
{
    for (auto& strategiesPair : strategiesByDesireType)
    {
        for (auto& strategy : strategiesPair.second)
        {
            auto strategyResourcesByName = strategy->resourcesByName();
            for (auto& resourcePair : strategyResourcesByName)
            {
                auto it = systemResourcesByName.find(resourcePair.first);
                if (it == systemResourcesByName.end())
                {
                    throw HbbaLiteException("Invalid resource (name=" + resourcePair.first + ")");
                }
            }
        }
    }
}

vector<size_t> selectMostIntenseEnabledDesireIndexes(const vector<unique_ptr<Desire>>& desires)
{
    unordered_map<type_index, pair<uint16_t, size_t>> mostIntenseDesiresByType;

    for (size_t i = 0; i < desires.size(); i++)
    {
        if (!desires[i]->enabled())
        {
            continue;
        }

        if (mostIntenseDesiresByType.find(desires[i]->type()) == mostIntenseDesiresByType.end() ||
            desires[i]->intensity() > mostIntenseDesiresByType[desires[i]->type()].first)
        {
            mostIntenseDesiresByType[desires[i]->type()] = pair<uint16_t, size_t>(desires[i]->intensity(), i);
        }
    }

    vector<size_t> mostIntenseDesireIndexes;
    for (auto& pair : mostIntenseDesiresByType)
    {
        mostIntenseDesireIndexes.emplace_back(pair.second.second);
    }

    return mostIntenseDesireIndexes;
}
