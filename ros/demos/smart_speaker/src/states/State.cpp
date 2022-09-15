#include "State.h"

using namespace std;

State::State(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : m_enabled(false),
      m_language(language),
      m_stateManager(stateManager),
      m_desireSet(move(desireSet)),
      m_nodeHandle(nodeHandle),
      m_previousStageType(typeid(State))
{
}

void State::enable(const string& parameter, const type_index& previousStageType)
{
    m_enabled = true;
    m_previousStageType = previousStageType;
}

void State::disable()
{
    m_enabled = false;

    auto transaction = m_desireSet->beginTransaction();
    for (auto id : m_desireIds)
    {
        m_desireSet->removeDesire(id);
    }
    m_desireIds.clear();
}
