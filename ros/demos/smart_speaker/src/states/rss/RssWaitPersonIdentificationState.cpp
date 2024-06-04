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
    rclcpp::Node::SharedPtr node)
    : State(language, stateManager, desireSet, move(node))
{
    m_personNamesSubscriber = m_node->create_subscription<person_identification::msg::PersonNames>(
        "person_names",
        1,
        [this](const person_identification::msg::PersonNames::SharedPtr msg) { personNamesSubscriberCallback(msg); });
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

    m_timeoutTimer = m_node->create_wall_timer(chrono::seconds(TIMEOUT_S), [this]() { timeoutTimerCallback(); });
}

void RssWaitPersonIdentificationState::disable()
{
    State::disable();

    if (m_timeoutTimer)
    {
        m_timeoutTimer->cancel();
        m_timeoutTimer = nullptr;
    }
}

void RssWaitPersonIdentificationState::personNamesSubscriberCallback(
    const person_identification::msg::PersonNames::SharedPtr msg)
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

void RssWaitPersonIdentificationState::timeoutTimerCallback()
{
    if (!enabled())
    {
        return;
    }

    m_stateManager.switchTo<RssIdleState>();
}
