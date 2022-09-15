#ifndef RECORDERS_PERCEPTION_LOGGER_VIDEO_ANALYSIS_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_VIDEO_ANALYSIS_LOGGER_H

#include <perception_logger/PerceptionLogger.h>

#include <tl/optional.hpp>

#include <string>
#include <vector>

struct VideoAnalysis
{
    Timestamp timestamp;
    Position position;
    Direction direction;

    std::string objectClass;

    tl::optional<std::vector<Position>> personPose;
    tl::optional<std::vector<float>> personPoseConfidence;

    tl::optional<std::vector<float>> faceDescriptor;

    VideoAnalysis(Timestamp timestamp, Position position, Direction direction, std::string objectClass);
    VideoAnalysis(
        Timestamp timestamp,
        Position position,
        Direction direction,
        std::string objectClass,
        std::vector<Position> personPose,
        std::vector<float> personPoseConfidence);
    VideoAnalysis(
        Timestamp timestamp,
        Position position,
        Direction direction,
        std::string objectClass,
        std::vector<Position> personPose,
        std::vector<float> personPoseConfidence,
        std::vector<float> faceDescriptor);
};

inline bool operator==(const Position& l, const Position& r)
{
    return l.x == r.x && l.y == r.y && l.z == r.z;
}

class VideoAnalysisLogger
{
public:
    VideoAnalysisLogger();
    virtual ~VideoAnalysisLogger();

    virtual int64_t log(const VideoAnalysis& analysis) = 0;
};

#endif
