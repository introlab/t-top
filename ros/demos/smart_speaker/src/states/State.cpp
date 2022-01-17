#include "State.h"

using namespace std;

State::State(StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        m_stateManager(stateManager),
        m_desireSet(move(desireSet)),
        m_nodeHandle(nodeHandle)
{
}

void State::enable(const std::string& parameter)
{
    m_enabled = true;
}

void State::disable()
{
    m_enabled = false;

    auto transaction = m_desireSet->beginTransaction();
    for (auto id : m_desireIds)
    {
        m_desireSet->removeDesire(id);
    }
}
