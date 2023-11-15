#ifndef RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_HBBA_STRATEGY_STATE_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_HBBA_STRATEGY_STATE_LOGGER_H

#include <perception_logger/HbbaStrategyStateLogger.h>

#include <SQLiteCpp/SQLiteCpp.h>

class SQLiteHbbaStrategyStateLogger : public HbbaStrategyStateLogger
{
    SQLite::Database& m_database;

public:
    SQLiteHbbaStrategyStateLogger(SQLite::Database& database);
    ~SQLiteHbbaStrategyStateLogger() override;

    int64_t log(const HbbaStrategyState& state) override;
};


#endif
