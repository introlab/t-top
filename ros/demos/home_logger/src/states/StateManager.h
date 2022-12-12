#ifndef HOME_LOGGER_STATES_STATE_MANAGER_H
#define HOME_LOGGER_STATES_STATE_MANAGER_H

#include "State.h"

#include <unordered_map>

class StateManager : private DesireSetObserver
{
    std::shared_ptr<DesireSet> m_desireSet;
    ros::NodeHandle& m_nodeHandle;

    std::unordered_map<StateType, std::unique_ptr<State>> m_states;
    State* m_currentState;

    ros::Subscriber m_speechToTextSubscriber;
    ros::Subscriber m_robotNameDetectedSubscriber;
    ros::Subscriber m_videoAnalysisSubscriber;
    ros::Subscriber m_audioAnalysisSubscriber;
    ros::Subscriber m_personNamesSubscriber;
    ros::Subscriber m_baseStatusSubscriber;

    ros::Timer m_stateTimeoutTimer;
    ros::Timer m_everyMinuteTimer;
    ros::Timer m_everyTenMinuteTimer;

public:
    StateManager(std::shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle);
    virtual ~StateManager();

    DECLARE_NOT_COPYABLE(StateManager);
    DECLARE_NOT_MOVABLE(StateManager);

    void addState(std::unique_ptr<State> state);

    template<class T>
    void switchTo(const StateParameter& parameter = StateParameter());
    void switchTo(StateType type, const StateParameter& parameter = StateParameter());

private:
    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& desires) override;

    void onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg);
    void onRobotNameDetected(const std_msgs::Empty::ConstPtr& msg);
    void onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg);
    void onAudioAnalysisReceived(const audio_analyzer::AudioAnalysis::ConstPtr& msg);
    void onPersonNamesDetected(const person_identification::PersonNames::ConstPtr& msg);
    void onBaseStatusChanged(const daemon_ros_client::BaseStatus::ConstPtr& msg);

    void onStateTimeout(const ros::TimerEvent& event);
    void onEveryMinuteTimeout(const ros::TimerEvent& event);
    void onEveryTenMinutesTimeout(const ros::TimerEvent& event);
};

template<class T>
inline void StateManager::switchTo(const StateParameter& parameter)
{
    switchTo(StateType::get<T>(), parameter);
}

#endif
