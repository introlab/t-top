#include "StateManager.h"

using namespace std;

StateManager::StateManager(rclcpp::Node::SharedPtr node) : m_currentState(nullptr), m_node(move(node)) {}

StateManager::~StateManager()
{
    if (m_currentState != nullptr)
    {
        m_currentState->disable();
        m_currentState = nullptr;
    }
}

void StateManager::addState(unique_ptr<State> state)
{
    m_states.emplace(state->type(), move(state));
}
