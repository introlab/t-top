#include <perception_logger/AudioAnalysisLogger.h>

using namespace std;

AudioAnalysis::AudioAnalysis(Timestamp timestamp, Direction direction, int64_t trackingId, string classes)
    : timestamp(timestamp),
      direction(direction),
      trackingId(trackingId),
      classes(move(classes))
{
}

AudioAnalysis::AudioAnalysis(
    Timestamp timestamp,
    Direction direction,
    int64_t trackingId,
    string classes,
    vector<float> voiceDescriptor)
    : timestamp(timestamp),
      direction(direction),
      trackingId(trackingId),
      classes(move(classes)),
      voiceDescriptor(move(voiceDescriptor))
{
}

AudioAnalysisLogger::AudioAnalysisLogger() {}

AudioAnalysisLogger::~AudioAnalysisLogger() {}
