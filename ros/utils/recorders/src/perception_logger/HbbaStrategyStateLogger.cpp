#include <perception_logger/HbbaStrategyStateLogger.h>

using namespace std;

HbbaStrategyState::HbbaStrategyState(Timestamp timestamp, string desireTypeName, string strategyTypeName, bool enabled)
    : timestamp(timestamp),
      desireTypeName(move(desireTypeName)),
      strategyTypeName(move(strategyTypeName)),
      enabled(enabled)
{
}

HbbaStrategyStateLogger::HbbaStrategyStateLogger() {}

HbbaStrategyStateLogger::~HbbaStrategyStateLogger() {}
