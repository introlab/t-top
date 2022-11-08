#include "SoundFaceFollowingState.h"

#include <t_top_hbba_lite/Desires.h>

constexpr uint64_t VIDEO_ANALYSIS_WITHOUT_PERSON_COUNT_THRESHOLD = 5;

using namespace std;

SoundFaceFollowingState::SoundFaceFollowingState(
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : State(stateManager, move(desireSet), nodeHandle),
      m_followingDesireType(DesireType::null()),
      m_videoAnalysisWithoutPersonCount(0)
{
}

SoundFaceFollowingState::~SoundFaceFollowingState() {}

void SoundFaceFollowingState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    m_videoAnalysisWithoutPersonCount = 0;
    setFollowingDesire(make_unique<SoundFollowingDesire>());
}

void SoundFaceFollowingState::onDisabling()
{
    setFollowingDesire(nullptr);
}

void SoundFaceFollowingState::onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    bool containsAPerson = containsAtLeastOnePerson(msg);

    if (containsAPerson && m_followingDesireType != DesireType::get<NearestFaceFollowingDesire>())
    {
        m_videoAnalysisWithoutPersonCount = 0;
        setFollowingDesire(make_unique<NearestFaceFollowingDesire>());
    }
    else if (!containsAPerson && m_followingDesireType != DesireType::get<SoundFollowingDesire>())
    {
        m_videoAnalysisWithoutPersonCount++;
        if (m_videoAnalysisWithoutPersonCount > VIDEO_ANALYSIS_WITHOUT_PERSON_COUNT_THRESHOLD)
        {
            setFollowingDesire(make_unique<SoundFollowingDesire>());
        }
    }
}

void SoundFaceFollowingState::setFollowingDesire(unique_ptr<Desire> desire)
{
    if (m_followingDesireId.has_value())
    {
        m_desireSet->removeDesire(m_followingDesireId.value());
    }

    if (desire != nullptr)
    {
        m_followingDesireType = desire->type();
        m_followingDesireId = m_desireSet->addDesire(move(desire));
    }
    else
    {
        m_followingDesireType = DesireType::null();
        m_followingDesireId = nullopt;
    }
}
