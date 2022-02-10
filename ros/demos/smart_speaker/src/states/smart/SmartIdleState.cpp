#include "SmartIdleState.h"
#include "SmartAskTaskState.h"

#include "../StateManager.h"

#include "../../StringUtils.h"

#include <t_top/hbba_lite/Desires.h>

#include <algorithm>

using namespace std;

SmartIdleState::SmartIdleState(Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    double personDistanceThreshold) :
        State(language, stateManager, desireSet, nodeHandle),
        m_personDistanceThreshold(personDistanceThreshold)
{
    m_personNamesSubscriber = nodeHandle.subscribe("person_names", 1,
        &SmartIdleState::personNamesSubscriberCallback, this);

    m_videoAnalysisSubscriber = nodeHandle.subscribe("audio_analysis", 1,
        &SmartIdleState::videoAnalysisSubscriberCallback, this);
}

void SmartIdleState::enable(const string& parameter)
{
    State::enable(parameter);

    auto videoAnalyzerDesire = make_unique<FastVideoAnalyzerDesire>();
    auto gestureDesire = make_unique<GestureDesire>("origin_all");
    auto faceAnimationDesire = make_unique<FaceAnimationDesire>("blink");

    m_desireIds.emplace_back(videoAnalyzerDesire->id());
    m_desireIds.emplace_back(gestureDesire->id());
    m_desireIds.emplace_back(faceAnimationDesire->id());

    auto transaction = m_desireSet->beginTransaction();
    m_desireSet->addDesire(move(videoAnalyzerDesire));
    m_desireSet->addDesire(move(gestureDesire));
    m_desireSet->addDesire(move(faceAnimationDesire));
}

void SmartIdleState::personNamesSubscriberCallback(const person_identification::PersonNames::ConstPtr& msg)
{
    if (!enabled() || msg->names.size() == 0)
    {
        return;
    }

    // TODO check distance
    //m_stateManager.switchTo<SmartAskTaskState>(mergedNames);
}

void SmartIdleState::videoAnalysisSubscriberCallback(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    // TODO check distance and wait for person identification
}
