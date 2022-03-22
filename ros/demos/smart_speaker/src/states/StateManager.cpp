#include "StateManager.h"

using namespace std;

StateManager::StateManager() : m_currentState(nullptr) {}

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
