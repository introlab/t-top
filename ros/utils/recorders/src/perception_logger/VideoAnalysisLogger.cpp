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
      personPoseImage{tl::nullopt},
      personPose{tl::nullopt},
      personPoseConfidence{tl::nullopt},
      faceDescriptor{tl::nullopt}
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
      faceDescriptor{tl::nullopt}
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
