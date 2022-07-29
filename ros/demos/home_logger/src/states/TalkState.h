#ifndef HOME_LOGGER_STATES_TALK_STATE_H
#define HOME_LOGGER_STATES_TALK_STATE_H

#include "State.h"

class TalkStateParameter : public StateParameter
{
public:
    std::string text;
    std::string gestureName;
    std::string faceAnimationName;
    StateType nextState;
    std::unique_ptr<StateParameter> nextStateParameter;

    TalkStateParameter();
    TalkStateParameter(std::string text, StateType nextState);
    TalkStateParameter(std::string text, StateType nextState, std::unique_ptr<StateParameter> nextStateParameter);
    TalkStateParameter(std::string text, std::string gestureName, std::string faceAnimationName, StateType nextState);
    TalkStateParameter(
        std::string text,
        std::string gestureName,
        std::string faceAnimationName,
        StateType nextState,
        std::unique_ptr<StateParameter> nextStateParameter);
    ~TalkStateParameter() override;

    std::string toString() const;
};

class TalkState : public State
{
    TalkStateParameter m_parameter;

    uint64_t m_faceFollowingDesireId;
    uint64_t m_talkDesireId;
    uint64_t m_gestureDesireId;
    uint64_t m_faceAnimationDesireId;

public:
    TalkState(StateManager& stateManager, std::shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle);
    ~TalkState() override;

protected:
    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;
};

#endif
