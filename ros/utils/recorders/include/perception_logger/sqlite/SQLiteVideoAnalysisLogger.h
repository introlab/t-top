#ifndef RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_VIDEO_ANALYSIS_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_SQLITE_SQLITE_VIDEO_ANALYSIS_LOGGER_H

#include <perception_logger/VideoAnalysisLogger.h>
#include <perception_logger/sqlite/SQLitePerceptionLogger.h>

class SQLiteVideoAnalysisLogger : public VideoAnalysisLogger, private SQLitePerceptionLogger
{
public:
    SQLiteVideoAnalysisLogger(SQLite::Database& database);
    ~SQLiteVideoAnalysisLogger() override;

    int64_t log(const VideoAnalysis& analysis) override;
};


#endif
