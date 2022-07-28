#ifndef SMART_SPEAKER_STATES_RSS_RSS_VALID_TASK_STATE_H
#define SMART_SPEAKER_STATES_RSS_RSS_VALID_TASK_STATE_H

#include "../common/ValidTaskState.h"

constexpr const char* CURRENT_WEATHER_TASK = "CURRENT_WEATHER";
constexpr const char* WEATHER_FORECAST_TASK = "WEATHER_FORECAST";
constexpr const char* STORY_TASK = "STORY";
constexpr const char* DANCE_TASK = "DANCE";
constexpr const char* DANCE_PLAYED_SONG_TASK = "DANCE_PLAYED_SONG";

class RssValidTaskState : public ValidTaskState
{
public:
    RssValidTaskState(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    ~RssValidTaskState() override = default;

    DECLARE_NOT_COPYABLE(RssValidTaskState);
    DECLARE_NOT_MOVABLE(RssValidTaskState);

protected:
    std::type_index type() const override;

    void switchState(const std::string& task) override;
};

inline std::type_index RssValidTaskState::type() const
{
    return std::type_index(typeid(RssValidTaskState));
}

#endif
