#ifndef RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_AUDIO_ANALYSIS_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_AUDIO_ANALYSIS_LOGGER_H

#include <perception_logger/AudioAnalysisLogger.h>
#include <perception_logger/sqlite/SQLitePerceptionLogger.h>

class SQLiteAudioAnalysisLogger : public AudioAnalysisLogger, private SQLitePerceptionLogger
{
public:
    SQLiteAudioAnalysisLogger(SQLite::Database& database);
    ~SQLiteAudioAnalysisLogger() override;

    int64_t log(const AudioAnalysis& analysis) override;
};


#endif
