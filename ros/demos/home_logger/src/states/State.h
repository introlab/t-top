#ifndef HOME_LOGGER_STATES_STATE_H
#define HOME_LOGGER_STATES_STATE_H

#include <ros/ros.h>

#include <std_msgs/Empty.h>
#include <daemon_ros_client/BaseStatus.h>

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/utils/ClassMacros.h>

#include <speech_to_text/Transcript.h>
#include <video_analyzer/VideoAnalysis.h>
#include <audio_analyzer/AudioAnalysis.h>
#include <person_identification/PersonNames.h>

#include <memory>
#include <numeric>
#include <vector>
#include <typeindex>
#include <numeric>
#include <type_traits>

class State;
class StateManager;

class StateParameter
{
public:
    StateParameter();
    virtual ~StateParameter();

    virtual std::string toString() const;
};

class StateType
{
    std::type_index m_type;

    explicit StateType(std::type_index type);

public:
    template<class T>
    static StateType get();
    static StateType null();

    bool operator==(const StateType& other) const;
    bool operator!=(const StateType& other) const;
    bool operator<(const StateType& other) const;
    bool operator<=(const StateType& other) const;
    bool operator>(const StateType& other) const;
    bool operator>=(const StateType& other) const;

    const char* name() const;
    std::size_t hashCode() const;
};

template<class T>
inline StateType StateType::get()
{
    static_assert(std::is_base_of<State, T>::value, "T must be a subclass of State");
    return StateType(std::type_index(typeid(T)));
}

inline StateType StateType::null()
{
    return StateType(std::type_index(typeid(std::nullptr_t)));
}

inline bool StateType::operator==(const StateType& other) const
{
    return m_type == other.m_type;
}

inline bool StateType::operator!=(const StateType& other) const
{
    return m_type != other.m_type;
}

inline bool StateType::operator<(const StateType& other) const
{
    return m_type < other.m_type;
}

inline bool StateType::operator<=(const StateType& other) const
{
    return m_type <= other.m_type;
}

inline bool StateType::operator>(const StateType& other) const
{
    return m_type > other.m_type;
}

inline bool StateType::operator>=(const StateType& other) const
{
    return m_type >= other.m_type;
}

inline const char* StateType::name() const
{
    return m_type.name();
}

inline std::size_t StateType::hashCode() const
{
    return m_type.hash_code();
}

namespace std
{
    template<>
    struct hash<StateType>
    {
        inline std::size_t operator()(const StateType& type) const { return type.hashCode(); }
    };
}

#define DECLARE_STATE_PROTECTED_METHODS(className)                                                                     \
    StateType type() const override { return StateType::get<className>(); }

class State
{
    bool m_enabled;

protected:
    StateManager& m_stateManager;
    std::shared_ptr<DesireSet> m_desireSet;
    ros::NodeHandle& m_nodeHandle;

public:
    State(StateManager& stateManager, std::shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle);
    virtual ~State();

    DECLARE_NOT_COPYABLE(State);
    DECLARE_NOT_MOVABLE(State);

protected:
    bool enabled() const;
    virtual StateType type() const = 0;

    void enable(const StateParameter& parameter, const StateType& previousStateType);
    void disable();

    virtual void onEnabling(const StateParameter& parameter, const StateType& previousStateType) = 0;
    virtual void onDisabling() = 0;

    virtual void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& desires);

    virtual void onSpeechToTextTranscriptReceived(const speech_to_text::Transcript::ConstPtr& msg);
    virtual void onRobotNameDetected();
    virtual void onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg);
    virtual void onAudioAnalysisReceived(const audio_analyzer::AudioAnalysis::ConstPtr& msg);
    virtual void onPersonNamesDetected(const person_identification::PersonNames::ConstPtr& msg);
    virtual void onBaseStatusChanged(const daemon_ros_client::BaseStatus::ConstPtr& msg);

    virtual void onStateTimeout();
    virtual void onEveryMinuteTimeout();
    virtual void onEveryTenMinutesTimeout();

    friend StateManager;
};

inline bool State::enabled() const
{
    return m_enabled;
}

bool containsAtLeastOnePerson(const video_analyzer::VideoAnalysis::ConstPtr& msg);

#endif
