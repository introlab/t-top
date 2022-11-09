#include "AlarmState.h"
#include "IdleState.h"
#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

AlarmStateParameter::AlarmStateParameter() {}

AlarmStateParameter::AlarmStateParameter(vector<int> alarmIds) : alarmIds(move(alarmIds)) {}

AlarmStateParameter::~AlarmStateParameter() {}

string AlarmStateParameter::toString() const
{
    stringstream ss;
    ss << "size=" << alarmIds.size();
    return ss.str();
}

AlarmState::AlarmState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    AlarmManager& alarmManager,
    string alarmPath)
    : SoundFaceFollowingState(stateManager, move(desireSet), nodeHandle),
      m_alarmManager(alarmManager),
      m_alarmPath(move(alarmPath))
{
}

AlarmState::~AlarmState() {}

void AlarmState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    m_parameter = dynamic_cast<const AlarmStateParameter&>(parameter);
    m_playSoundDesireId = m_desireSet->addDesire<PlaySoundDesire>(m_alarmPath);
}

void AlarmState::onDisabling()
{
    if (m_playSoundDesireId.has_value())
    {
        m_desireSet->removeDesire(m_playSoundDesireId.value());
        m_playSoundDesireId = nullopt;
    }

    m_alarmManager.informPerformedAlarms(m_parameter.alarmIds);
}

void AlarmState::onDesireSetChanged(const vector<unique_ptr<Desire>>& _)
{
    if (!(m_playSoundDesireId.has_value() && m_desireSet->contains(m_playSoundDesireId.value())))
    {
        m_stateManager.switchTo<IdleState>();
    }
}
