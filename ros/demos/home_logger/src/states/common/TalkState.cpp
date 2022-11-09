#include "TalkState.h"
#include "../StateManager.h"

#include <t_top_hbba_lite/Desires.h>

#include <sstream>

using namespace std;

TalkStateParameter::TalkStateParameter() : nextState(StateType::null()) {}

TalkStateParameter::TalkStateParameter(string text, StateType nextState)
    : text(move(text)),
      nextState(nextState),
      nextStateParameter(make_shared<StateParameter>())
{
}

TalkStateParameter::TalkStateParameter(string text, StateType nextState, shared_ptr<StateParameter> nextStateParameter)
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
      nextStateParameter(make_shared<StateParameter>())
{
}

TalkStateParameter::TalkStateParameter(
    string text,
    string gestureName,
    string faceAnimationName,
    StateType nextState,
    shared_ptr<StateParameter> nextStateParameter)
    : text(move(text)),
      gestureName(move(gestureName)),
      faceAnimationName(move(faceAnimationName)),
      nextState(nextState),
      nextStateParameter(move(nextStateParameter))
{
}

TalkStateParameter::~TalkStateParameter() {}

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
    : State(stateManager, move(desireSet), nodeHandle)
{
}

TalkState::~TalkState() {}

void TalkState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    m_parameter = dynamic_cast<const TalkStateParameter&>(parameter);

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

void TalkState::onDisabling()
{
    if (m_faceFollowingDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceFollowingDesireId.value());
        m_faceFollowingDesireId = nullopt;
    }
    if (m_talkDesireId.has_value())
    {
        m_desireSet->removeDesire(m_talkDesireId.value());
        m_talkDesireId = nullopt;
    }
    if (m_gestureDesireId.has_value())
    {
        m_desireSet->removeDesire(m_gestureDesireId.value());
        m_gestureDesireId = nullopt;
    }
    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = nullopt;
    }

    m_parameter = TalkStateParameter();
}

void TalkState::onDesireSetChanged(const vector<unique_ptr<Desire>>& _)
{
    if (!(m_talkDesireId.has_value() && m_desireSet->contains(m_talkDesireId.value())) &&
        !(m_gestureDesireId.has_value() && m_desireSet->contains(m_gestureDesireId.value())))
    {
        shared_ptr<StateParameter> nextStateParameter = m_parameter.nextStateParameter;
        m_stateManager.switchTo(m_parameter.nextState, *nextStateParameter);
    }
}
