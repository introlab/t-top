#include <perception_logger/VideoAnalysisLogger.h>

using namespace std;

VideoAnalysis::VideoAnalysis(
    Timestamp timestamp,
    Position position,
    Direction direction,
    string objectClass,
    BoundingBox boundingBox)
    : timestamp{timestamp},
      position{position},
      direction{direction},
      objectClass{move(objectClass)},
      boundingBox{boundingBox},
      personPoseImage{std::nullopt},
      personPose{std::nullopt},
      personPoseConfidence{std::nullopt},
      faceDescriptor{std::nullopt}
{
}
VideoAnalysis::VideoAnalysis(
    Timestamp timestamp,
    Position position,
    Direction direction,
    string objectClass,
    BoundingBox boundingBox,
    vector<ImagePosition> personPoseImage,
    vector<Position> personPose,
    vector<float> personPoseConfidence)
    : timestamp{timestamp},
      position{position},
      direction{direction},
      objectClass{move(objectClass)},
      boundingBox{boundingBox},
      personPoseImage{move(personPoseImage)},
      personPose{move(personPose)},
      personPoseConfidence{move(personPoseConfidence)},
      faceDescriptor{std::nullopt}
{
}
VideoAnalysis::VideoAnalysis(
    Timestamp timestamp,
    Position position,
    Direction direction,
    string objectClass,
    BoundingBox boundingBox,
    vector<ImagePosition> personPoseImage,
    vector<Position> personPose,
    vector<float> personPoseConfidence,
    vector<float> faceDescriptor)
    : timestamp{timestamp},
      position{position},
      direction{direction},
      objectClass{move(objectClass)},
      boundingBox{boundingBox},
      personPoseImage{move(personPoseImage)},
      personPose{move(personPose)},
      personPoseConfidence{move(personPoseConfidence)},
      faceDescriptor{move(faceDescriptor)}
{
}

VideoAnalysisLogger::VideoAnalysisLogger() = default;

VideoAnalysisLogger::~VideoAnalysisLogger() = default;
