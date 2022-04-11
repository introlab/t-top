#ifndef SMART_SPEAKER_STATES_STATE_MANAGER_H
#define SMART_SPEAKER_STATES_STATE_MANAGER_H

#include "State.h"

#include <ros/ros.h>

#include <hbba_lite/utils/ClassMacros.h>

#include <memory>
#include <unordered_map>

class StateManager
{
    std::unordered_map<std::type_index, std::unique_ptr<State>> m_states;
    State* m_currentState;

public:
    StateManager();
    virtual ~StateManager();

    DECLARE_NOT_COPYABLE(StateManager);
    DECLARE_NOT_MOVABLE(StateManager);

    void addState(std::unique_ptr<State> state);

    template<class T>
    void switchTo(const std::string& parameter = "");
    void switchTo(std::type_index type, const std::string& parameter = "");
};

template<class T>
void StateManager::switchTo(const std::string& parameter)
{
    switchTo(std::type_index(typeid(T)), parameter);
}

inline void StateManager::switchTo(std::type_index stateType, const std::string& parameter)
{
    std::type_index previousStageType(typeid(State));
    if (m_currentState != nullptr)
    {
        ROS_INFO("Disabling %s", m_currentState->type().name());
        m_currentState->disable();
        previousStageType = m_currentState->type();
    }

    ROS_INFO("Enabling %s (%s)", stateType.name(), parameter.c_str());
    m_currentState = m_states.at(stateType).get();
    m_currentState->enable(parameter, previousStageType);
}

#endif
