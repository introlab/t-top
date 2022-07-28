#include <perception_logger/VideoAnalysisLogger.h>

using namespace std;

VideoAnalysis::VideoAnalysis(Timestamp timestamp, Position position, Direction direction, string objectClass)
    : timestamp(timestamp),
      position(position),
      direction(direction),
      objectClass(move(objectClass))
{
}

VideoAnalysis::VideoAnalysis(
    Timestamp timestamp,
    Position position,
    Direction direction,
    string objectClass,
    vector<Position> personPose,
    vector<float> personPoseConfidence)
    : timestamp(timestamp),
      position(position),
      direction(direction),
      objectClass(move(objectClass)),
      personPose(move(personPose)),
      personPoseConfidence(move(personPoseConfidence))
{
}

VideoAnalysis::VideoAnalysis(
    Timestamp timestamp,
    Position position,
    Direction direction,
    string objectClass,
    vector<Position> personPose,
    vector<float> personPoseConfidence,
    vector<float> faceDescriptor)
    : timestamp(timestamp),
      position(position),
      direction(direction),
      objectClass(move(objectClass)),
      personPose(move(personPose)),
      personPoseConfidence(move(personPoseConfidence)),
      faceDescriptor(move(faceDescriptor))
{
}

VideoAnalysisLogger::VideoAnalysisLogger() {}

VideoAnalysisLogger::~VideoAnalysisLogger() {}
