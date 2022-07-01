#include "RssWaitPersonIdentificationState.h"
#include "RssAskTaskState.h"
#include "RssIdleState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

RssWaitPersonIdentificationState::RssWaitPersonIdentificationState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : State(language, stateManager, desireSet, nodeHandle)
{
    m_personNamesSubscriber =
        nodeHandle.subscribe("person_names", 1, &RssWaitPersonIdentificationState::personNamesSubscriberCallback, this);
}

void RssWaitPersonIdentificationState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    auto audioAnalyzerDesire = make_unique<AudioAnalyzerDesire>();
    auto videoAnalyzerDesire = make_unique<FastVideoAnalyzer3dDesire>();
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
    m_timeoutTimer = m_nodeHandle.createTimer(
        ros::Duration(TIMEOUT_S),
        &RssWaitPersonIdentificationState::timeoutTimerCallback,
        this,
        oneshot);
}

void RssWaitPersonIdentificationState::disable()
{
    State::disable();

    if (m_timeoutTimer.isValid())
    {
        m_timeoutTimer.stop();
    }
}

void RssWaitPersonIdentificationState::personNamesSubscriberCallback(
    const person_identification::PersonNames::ConstPtr& msg)
{
    if (!enabled() || msg->names.size() == 0)
    {
        return;
    }

    vector<string> names;
    transform(
        msg->names.begin(),
        msg->names.end(),
        back_inserter(names),
        [](const person_identification::PersonName& name) { return name.name; });

    auto mergedNames = mergeNames(names, getAndWord());
    m_stateManager.switchTo<RssAskTaskState>(mergedNames);
}

void RssWaitPersonIdentificationState::timeoutTimerCallback(const ros::TimerEvent& event)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<RssIdleState>();
}
