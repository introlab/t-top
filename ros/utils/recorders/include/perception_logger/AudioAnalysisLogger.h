#ifndef RECORDERS_PERCEPTION_LOGGER_AUDIO_ANALYSIS_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_AUDIO_ANALYSIS_LOGGER_H

#include <perception_logger/PerceptionLogger.h>

#include <optional>
#include <string>
#include <vector>

struct AudioAnalysis
{
    Timestamp timestamp;
    Direction direction;
    int64_t trackingId;
    std::string classes;
    std::optional<std::vector<float>> voiceDescriptor;

    AudioAnalysis(Timestamp timestamp, Direction direction, int64_t trackingId, std::string classes);
    AudioAnalysis(
        Timestamp timestamp,
        Direction direction,
        int64_t trackingId,
        std::string classes,
        std::vector<float> voiceDescriptor);
};

class AudioAnalysisLogger
{
public:
    AudioAnalysisLogger();
    virtual ~AudioAnalysisLogger();

    virtual int64_t log(const AudioAnalysis& analysis) = 0;
};

#endif
