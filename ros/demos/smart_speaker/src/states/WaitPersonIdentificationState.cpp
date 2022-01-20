#include "WaitPersonIdentificationState.h"
#include "StateManager.h"
#include "AskTaskState.h"
#include "IdleState.h"

#include "../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

using namespace std;

WaitPersonIdentificationState::WaitPersonIdentificationState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(language, stateManager, desireSet, nodeHandle)
{
    m_personNamesSubscriber = nodeHandle.subscribe("person_names", 1,
        &WaitPersonIdentificationState::personNamesSubscriberCallback, this);
}

void WaitPersonIdentificationState::enable(const string& parameter)
{
    State::enable(parameter);

    auto audioAnalyzerDesire = make_unique<AudioAnalyzerDesire>();
    auto videoAnalyzerDesire = make_unique<FastVideoAnalyzerDesire>();
    auto soundFollowingDesire = make_unique<SoundFollowingDesire>();
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");

    m_desireIds.emplace_back(audioAnalyzerDesire->id());
    m_desireIds.emplace_back(videoAnalyzerDesire->id());
    m_desireIds.emplace_back(soundFollowingDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(audioAnalyzerDesire));
    m_desireSet->addDesire(move(videoAnalyzerDesire));
    m_desireSet->addDesire(move(soundFollowingDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));

    constexpr bool oneshot = true;
    m_timeoutTimer = m_nodeHandle.createTimer(ros::Duration(TIMEOUT_S),
        &WaitPersonIdentificationState::timeoutTimerCallback, this, oneshot);
}

void WaitPersonIdentificationState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void WaitPersonIdentificationState::personNamesSubscriberCallback(
    const person_identification::PersonNames::ConstPtr& msg)
{
    if (!enabled() || msg->names.size() == 0)
    {
        return;
    }

    auto names = mergeStrings(msg->names, ", ");
    m_stateManager.switchTo<AskTaskState>(names);
}

void WaitPersonIdentificationState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<IdleState>();
}
