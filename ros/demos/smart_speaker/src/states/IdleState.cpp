#include "IdleState.h"
#include "StateManager.h"
#include "WaitPersonIdentificationState.h"
#include "AskTaskState.h"

#include "../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

using namespace std;

IdleState::IdleState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle) :
        State(language, stateManager, desireSet, nodeHandle)
{
    m_robotNameDetectedSubscriber = nodeHandle.subscribe("robot_name_detected", 1,
        &IdleState::robotNameDetectedSubscriberCallback, this);
    m_personNamesSubscriber = nodeHandle.subscribe("person_names", 1,
        &IdleState::personNamesSubscriberCallback, this);
}

void IdleState::enable(const string& parameter)
{
    State::enable(parameter);

    auto robotNameDetectorDesire = make_unique<RobotNameDetectorDesire>();
    auto videoAnalyzerDesire = make_unique<SlowVideoAnalyzerDesire>();
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

void IdleState::robotNameDetectedSubscriberCallback(const std_msgs::Empty::ConstPtr& msg)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<WaitPersonIdentificationState>();
}

void IdleState::personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg)
{
    if (!enabled() || msg->names.size() == 0)
    {
        return;
    }

    auto names = mergeStrings(msg->names, ", ");
    m_stateManager.switchTo<AskTaskState>(names);
}
