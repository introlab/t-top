#include "TalkState.h"

#include <t_top_hbba_lite/Desires.h>

#include <sstream>

using namespace std;

TalkStateParameter::TalkStateParameter()
    : nextState(StateType::null())

          TalkStateParameter::TalkStateParameter(string text, StateType nextState)
    : text(move(text)),
      nextState(nextState),
      nextStateParameter(make_unique<StateParameter>())
{
}

TalkStateParameter(string text, StateType nextState, unique_ptr<StateParameter> nextStateParameter)
    : text(move(text)),
      nextState(nextState),
      nextStateParameter(move(nextStateParameter))
{
}

TalkStateParameter::TalkStateParameter(string text, string gestureName, string faceAnimationName, StateType nextState)
    : text(move(text)),
      gestureName(move(gestureName)),
      faceAnimationName(move(faceAnimationName)),
      nextState(nextState),
      nextStateParameter(make_unique<StateParameter>())
{
}

TalkStateParameter::TalkStateParameter(
    std::string text,
    std::string gestureName,
    std::string faceAnimationName,
    StateType nextState,
    std::unique_ptr<StateParameter> nextStateParameter)
    : text(move(text)),
      gestureName(move(gestureName)),
      faceAnimationName(move(faceAnimationName)),
      nextState(nextState),
      nextStateParameter(move(nextStateParameter))
{
}

TalkStateParameter::~TalkStateParameter() override {}

string TalkStateParameter::toString() const
{
    stringstream ss;
    ss << "text=" << text;
    ss << ", gesture_name=" << gestureName;
    ss << ", face_animation=" << faceAnimationName;
    ss << ", nextState=" << nextState.name();

    return ss.str();
}


TalkState::TalkState(StateManager& stateManager, shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle)
    : State(stateManager, desireSet, nodeHandle),
      m_talkDesireId(MAX_DESIRE_ID),
      m_gestureDesireId(MAX_DESIRE_ID),
      m_faceAnimationDesireId(MAX_DESIRE_ID)
{
}

TalkState::~TalkState() {}

void TalkState::onEnabling(const StateParameter& parameter, const StateType& previousStateType) override
{
    SoundFaceFollowingState::onEnabling(parameter, previousStateType);

    m_parameter = dynamic_cast<TalkStateParameter>(parameter);

    m_faceFollowingDesireId = m_desireSet->addDesire<NearestFaceFollowingDesire>();
    m_talkDesireId = m_desireSet->addDesire<TalkDesire>(m_parameter.text);

    if (m_parameter.gestureName != "")
    {
        m_gestureDesireId = m_desireSet->addDesire<GestureDesire>(m_parameter.gestureName, 2);
    }
    if (m_parameter.faceAnimationName != "")
    {
        m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>(m_parameter.faceAnimationName);
    }
}

void TalkState::onDisabling() override
{
    SoundFaceFollowingState::onDisabling();

    m_desireSet->removeDesire(m_faceFollowingDesireId);
    m_desireSet->removeDesire(m_talkDesireId);
    m_desireSet->removeDesire(m_gestureDesireId);
    m_desireSet->removeDesire(m_faceAnimationDesireId);

    m_faceFollowingDesireId = MAX_DESIRE_ID;
    m_talkDesireId = MAX_DESIRE_ID;
    m_gestureDesireId = MAX_DESIRE_ID;
    m_faceAnimationDesireId = MAX_DESIRE_ID;

    m_parameter = TalkStateParameter();
}

void TalkState::onDesireSetChanged(const vector<unique_ptr<Desire>>& _)
{
    if (!m_desireSet->contains(m_talkDesireId) && !m_disireSet->contains(m_gestureDesireId))
    {
        m_stateManager.switchTo(m_parameter.nextState, *m_parameter.nextStateParameter);
    }
}
