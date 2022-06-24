#include "RssIdleState.h"
#include "RssWaitPersonIdentificationState.h"
#include "RssAskTaskState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

#include <t_top_hbba_lite/Desires.h>

using namespace std;

RssIdleState::RssIdleState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : State(language, stateManager, desireSet, nodeHandle)
{
    m_robotNameDetectedSubscriber =
        nodeHandle.subscribe("robot_name_detected", 1, &RssIdleState::robotNameDetectedSubscriberCallback, this);
    m_personNamesSubscriber =
        nodeHandle.subscribe("person_names", 1, &RssIdleState::personNamesSubscriberCallback, this);
}

void RssIdleState::enable(const string& parameter, const type_index& previousStageType)
{
    State::enable(parameter, previousStageType);

    auto robotNameDetectorDesire = make_unique<RobotNameDetectorDesire>();
    auto videoAnalyzerDesire = make_unique<SlowVideoAnalyzer3dDesire>();
    auto exploreDesire = make_unique<ExploreDesire>();
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");

    m_desireIds.emplace_back(robotNameDetectorDesire->id());
    m_desireIds.emplace_back(videoAnalyzerDesire->id());
    m_desireIds.emplace_back(exploreDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(robotNameDetectorDesire));
    m_desireSet->addDesire(move(videoAnalyzerDesire));
    m_desireSet->addDesire(move(exploreDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));
}

void RssIdleState::robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<RssWaitPersonIdentificationState>();
}

void RssIdleState::personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg)
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
