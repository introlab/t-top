#ifndef RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_SPEECH_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_SPEECH_LOGGER_H

#include <perception_logger/SpeechLogger.h>
#include <perception_logger/sqlite/SQLitePerceptionLogger.h>

class SQLiteSpeechLogger : public SpeechLogger, private SQLitePerceptionLogger
{
public:
    SQLiteSpeechLogger(SQLite::Database& database);
    ~SQLiteSpeechLogger() override;

    int64_t log(const Speech& speech) override;
};


#endif
