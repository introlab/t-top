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

    template <class T>
    void switchTo(const std::string& parameter = "");
};

template <class T>
void StateManager::switchTo(const std::string& parameter)
{
    if (m_currentState != nullptr)
    {
        ROS_INFO("Disabling %s", m_currentState->type().name());
        m_currentState->disable();
    }

    m_currentState = m_states.at(std::type_index(typeid(T))).get();
    ROS_INFO("Enabling %s (%s)", m_currentState->type().name(), parameter.c_str());
    m_currentState->enable(parameter);
}

#endif
