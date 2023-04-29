#ifndef RECORDERS_PERCEPTION_LOGGER_VIDEO_ANALYSIS_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_VIDEO_ANALYSIS_LOGGER_H

#include <perception_logger/PerceptionLogger.h>

#include <optional>
#include <string>
#include <vector>

struct VideoAnalysis
{
    Timestamp timestamp;
    Position position;
    Direction direction;

    std::string objectClass;
    float objectConfidence;
    float objectClassProbability;
    BoundingBox boundingBox;

    std::optional<std::vector<ImagePosition>> personPoseImage;
    std::optional<std::vector<Position>> personPose;
    std::optional<std::vector<float>> personPoseConfidence;

    std::optional<std::vector<float>> faceDescriptor;
    std::optional<int32_t> faceAlignmentKeypointCount;
    std::optional<float> faceSharpnessScore;

    VideoAnalysis(
        Timestamp timestamp,
        Position position,
        Direction direction,
        std::string objectClass,
        float objectConfidence,
        float objectClassProbability,
        BoundingBox boundingBox);
    VideoAnalysis(
        Timestamp timestamp,
        Position position,
        Direction direction,
        std::string objectClass,
        float objectConfidence,
        float objectClassProbability,
        BoundingBox boundingBox,
        std::vector<ImagePosition> personPoseImage,
        std::vector<Position> personPose,
        std::vector<float> personPoseConfidence);
    VideoAnalysis(
        Timestamp timestamp,
        Position position,
        Direction direction,
        std::string objectClass,
        float objectConfidence,
        float objectClassProbability,
        BoundingBox boundingBox,
        std::vector<ImagePosition> personPoseImage,
        std::vector<Position> personPose,
        std::vector<float> personPoseConfidence,
        std::vector<float> faceDescriptor,
        int32_t faceAlignmentKeypointCount,
        float faceSharpnessScore);
};

class VideoAnalysisLogger
{
public:
    VideoAnalysisLogger();
    virtual ~VideoAnalysisLogger();

    virtual int64_t log(const VideoAnalysis& analysis) = 0;
};

#endif
