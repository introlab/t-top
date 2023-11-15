#include <perception_logger/VideoAnalysisLogger.h>

using namespace std;

VideoAnalysis::VideoAnalysis(
    Timestamp timestamp,
    Position position,
    Direction direction,
    string objectClass,
    float objectConfidence,
    float objectClassProbability,
    BoundingBox boundingBox)
    : timestamp{timestamp},
      position{position},
      direction{direction},
      objectClass{move(objectClass)},
      objectConfidence{objectConfidence},
      objectClassProbability{objectClassProbability},
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
    float objectConfidence,
    float objectClassProbability,
    BoundingBox boundingBox,
    vector<ImagePosition> personPoseImage,
    vector<Position> personPose,
    vector<float> personPoseConfidence)
    : timestamp{timestamp},
      position{position},
      direction{direction},
      objectClass{move(objectClass)},
      objectConfidence(objectConfidence),
      objectClassProbability(objectClassProbability),
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
    float objectConfidence,
    float objectClassProbability,
    BoundingBox boundingBox,
    vector<ImagePosition> personPoseImage,
    vector<Position> personPose,
    vector<float> personPoseConfidence,
    vector<float> faceDescriptor,
    int32_t faceAlignmentKeypointCount,
    float faceSharpnessScore)
    : timestamp{timestamp},
      position{position},
      direction{direction},
      objectClass{move(objectClass)},
      objectConfidence(objectConfidence),
      objectClassProbability(objectClassProbability),
      boundingBox{boundingBox},
      personPoseImage{move(personPoseImage)},
      personPose{move(personPose)},
      personPoseConfidence{move(personPoseConfidence)},
      faceDescriptor{move(faceDescriptor)},
      faceAlignmentKeypointCount{faceAlignmentKeypointCount},
      faceSharpnessScore{faceSharpnessScore}
{
}

VideoAnalysisLogger::VideoAnalysisLogger() = default;

VideoAnalysisLogger::~VideoAnalysisLogger() = default;
