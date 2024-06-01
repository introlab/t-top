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
    rclcpp::Node::SharedPtr node)
    : State(language, stateManager, desireSet, move(node))
{
    m_robotNameDetectedSubscriber =
        m_node->create_subscription<std_msgs::msg::Empty>("robot_name_detected", 1, [this](const std_msgs::msg::Empty::SharedPtr msg) { robotNameDetectedSubscriberCallback(msg); });
    m_personNamesSubscriber =
        m_node->create_subscription<person_identification::msg::PersonNames>("person_names", 1, [this] (const person_identification::msg::PersonNames::SharedPtr msg) { personNamesSubscriberCallback(msg); });
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

void RssIdleState::robotNameDetectedSubscriberCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<RssWaitPersonIdentificationState>();
}

void RssIdleState::personNamesSubscriberCallback(const person_identification::msg::PersonNames::SharedPtr msg)
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
        [](const person_identification::msg::PersonName& name) { return name.name; });

    auto mergedNames = mergeNames(names, getAndWord());
    m_stateManager.switchTo<RssAskTaskState>(mergedNames);
}
