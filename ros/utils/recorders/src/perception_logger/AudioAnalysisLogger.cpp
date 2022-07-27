#include <perception_logger/AudioAnalysisLogger.h>

using namespace std;

AudioAnalysis::AudioAnalysis(Timestamp timestamp, Direction direction, string classes) : timestamp(timestamp), direction(direction), classes(move(classes))
{
}

AudioAnalysis::AudioAnalysis(Timestamp timestamp, Direction direction, string classes, vector<float> voiceDescriptor) : timestamp(timestamp), direction(direction), classes(move(classes)), voiceDescriptor(move(voiceDescriptor))
{
}

AudioAnalysisLogger::AudioAnalysisLogger()
{
}

AudioAnalysisLogger::~AudioAnalysisLogger()
{
}
