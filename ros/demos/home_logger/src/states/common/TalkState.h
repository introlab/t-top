#ifndef HOME_LOGGER_STATES_COMMON_TALK_STATE_H
#define HOME_LOGGER_STATES_COMMON_TALK_STATE_H

#include "../State.h"

#include <optional>

class TalkStateParameter : public StateParameter
{
public:
    std::string text;
    std::string gestureName;
    std::string faceAnimationName;
    StateType nextState;
    std::shared_ptr<StateParameter> nextStateParameter;

    TalkStateParameter();
    TalkStateParameter(std::string text, StateType nextState);
    TalkStateParameter(std::string text, StateType nextState, std::shared_ptr<StateParameter> nextStateParameter);
    TalkStateParameter(std::string text, std::string gestureName, std::string faceAnimationName, StateType nextState);
    TalkStateParameter(
        std::string text,
        std::string gestureName,
        std::string faceAnimationName,
        StateType nextState,
        std::shared_ptr<StateParameter> nextStateParameter);
    ~TalkStateParameter() override;

    std::string toString() const override;
};

class TalkState : public State
{
    TalkStateParameter m_parameter;

    std::optional<uint64_t> m_faceFollowingDesireId;
    std::optional<uint64_t> m_talkDesireId;
    std::optional<uint64_t> m_gestureDesireId;
    std::optional<uint64_t> m_faceAnimationDesireId;

public:
    TalkState(StateManager& stateManager, std::shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle);
    ~TalkState() override;

protected:
    DECLARE_STATE_PROTECTED_METHODS(TalkState)

    void onEnabling(const StateParameter& parameter, const StateType& previousStateType) override;
    void onDisabling() override;

    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;
};

#endif
