#include "SleepState.h"
#include "IdleState.h"
#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

SleepState::SleepState(StateManager& stateManager, std::shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle, Time sleepTime, Time wakeUpTime)
: State(stateManager, desireSet, nodeHandle),
      m_sleepTime(sleepTime),
      m_wakeUpTime(wakeUpTime)
{
}

SleepState::~SleepState()
{
}

void SleepState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    m_hadCamera3dRecordingDesire = m_desireSet->containsAnyDesiresOfType<Camera3dRecordingDesire>();
    m_hadCamera2dWideRecordingDesire = m_desireSet->containsAnyDesiresOfType<Camera2dWideRecordingDesire>();
    m_hadAudioAnalyzerDesire = m_desireSet->containsAnyDesiresOfType<AudioAnalyzerDesire>();
    m_hadFastVideoAnalyzer3dDesire = m_desireSet->containsAnyDesiresOfType<FastVideoAnalyzer3dDesire>();

    m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>("sleep");
}

void SleepState::onDisabling()
{
    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = tl::nullopt;
    }

    if (m_hadCamera3dRecordingDesire)
    {
        m_desireSet->addDesire<Camera3dRecordingDesire>();
    }
    if (m_hadCamera2dWideRecordingDesire)
    {
        m_desireSet->addDesire<Camera2dWideRecordingDesire>();
    }
    if (m_hadAudioAnalyzerDesire)
    {
        m_desireSet->addDesire<AudioAnalyzerDesire>();
    }
    if (m_hadFastVideoAnalyzer3dDesire)
    {
        m_desireSet->addDesire<FastVideoAnalyzer3dDesire>();
    }
}

void SleepState::onEveryMinuteTimeout()
{
    Time now = Time::now();
    if (now.between(m_wakeUpTime, m_sleepTime))
    {
        m_stateManager.switchTo<IdleState>();
        return;
    }
}
