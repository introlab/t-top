#ifndef HOME_LOGGER_STATES_STATE_MANAGER_H
#define HOME_LOGGER_STATES_STATE_MANAGER_H

#include "State.h"

#include <unordered_map>

class StateManager : private DesireSetObserver
{
    std::shared_ptr<DesireSet> m_desireSet;
    rclcpp::Node::SharedPtr m_node;

    std::unordered_map<StateType, std::unique_ptr<State>> m_states;
    State* m_currentState;

    rclcpp::Subscription<speech_to_text::msg::Transcript>::SharedPtr m_speechToTextSubscriber;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_robotNameDetectedSubscriber;
    rclcpp::Subscription<video_analyzer::msg::VideoAnalysis>::SharedPtr m_videoAnalysisSubscriber;
    rclcpp::Subscription<audio_analyzer::msg::AudioAnalysis>::SharedPtr m_audioAnalysisSubscriber;
    rclcpp::Subscription<person_identification::msg::PersonNames>::SharedPtr m_personNamesSubscriber;
    rclcpp::Subscription<daemon_ros_client::msg::BaseStatus>::SharedPtr m_baseStatusSubscriber;

    rclcpp::TimerBase::SharedPtr m_stateTimeoutTimer;
    rclcpp::TimerBase::SharedPtr m_everyMinuteTimer;
    rclcpp::TimerBase::SharedPtr m_everyTenMinuteTimer;

public:
    StateManager(std::shared_ptr<DesireSet> desireSet, rclcpp::Node::SharedPtr node);
    virtual ~StateManager();

    DECLARE_NOT_COPYABLE(StateManager);
    DECLARE_NOT_MOVABLE(StateManager);

    void addState(std::unique_ptr<State> state);

    template<class T>
    void switchTo(const StateParameter& parameter = StateParameter());
    void switchTo(StateType type, const StateParameter& parameter = StateParameter());

private:
    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& desires) override;

    void onSpeechToTextTranscriptReceived(const speech_to_text::msg::Transcript::SharedPtr& msg);
    void onRobotNameDetected(const std_msgs::msg::Empty::SharedPtr& msg);
    void onVideoAnalysisReceived(const video_analyzer::msg::VideoAnalysis::SharedPtr& msg);
    void onAudioAnalysisReceived(const audio_analyzer::msg::AudioAnalysis::SharedPtr& msg);
    void onPersonNamesDetected(const person_identification::msg::PersonNames::SharedPtr& msg);
    void onBaseStatusChanged(const daemon_ros_client::msg::BaseStatus::SharedPtr& msg);

    void onStateTimeout();
    void onEveryMinuteTimeout();
    void onEveryTenMinutesTimeout();
};

template<class T>
inline void StateManager::switchTo(const StateParameter& parameter)
{
    switchTo(StateType::get<T>(), parameter);
}

#endif
