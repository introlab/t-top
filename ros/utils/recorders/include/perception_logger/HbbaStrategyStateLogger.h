#ifndef RECORDERS_PERCEPTION_LOGGER_HBBA_STRATEGY_STATE_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_HBBA_STRATEGY_STATE_LOGGER_H

#include <perception_logger/PerceptionLogger.h>

#include <string>

struct HbbaStrategyState
{
    Timestamp timestamp;
    std::string desireTypeName;
    std::string strategyTypeName;
    bool enabled;

    HbbaStrategyState(Timestamp timestamp, std::string desireTypeName, std::string strategyTypeName, bool enabled);
};

class HbbaStrategyStateLogger
{
public:
    HbbaStrategyStateLogger();
    virtual ~HbbaStrategyStateLogger();

    virtual int64_t log(const HbbaStrategyState& state) = 0;
};

#endif
