#include "IdleState.h"
#include "SleepState.h"
#include "../StateManager.h"
#include "../common/TalkState.h"

#include <home_logger_common/language/StringRessources.h>

#include <t_top_hbba_lite/Desires.h>

using namespace std;

constexpr float LOW_STATE_OF_CHARGE = 25;
constexpr chrono::minutes BATTERY_LOW_MESSAGE_INTERVAL(5);

IdleState::IdleState(StateManager& stateManager, shared_ptr<DesireSet> desireSet, ros::NodeHandle& nodeHandle, Time sleepTime, Time wakeUpTime)
    : SoundFaceFollowingState(stateManager, desireSet, nodeHandle),
      m_sleepTime(sleepTime),
      m_wakeUpTime(wakeUpTime),
      m_chargeNeeded(false)
{
}

IdleState::~IdleState() {}

void IdleState::onEnabling(const StateParameter& parameter, const StateType& previousStateType)
{
    SoundFaceFollowingState::onEnabling(parameter, previousStateType);

    m_faceAnimationDesireId = m_desireSet->addDesire<FaceAnimationDesire>("blink");
}

void IdleState::onDisabling()
{
    SoundFaceFollowingState::onDisabling();

    if (m_faceAnimationDesireId.has_value())
    {
        m_desireSet->removeDesire(m_faceAnimationDesireId.value());
        m_faceAnimationDesireId = tl::nullopt;
    }
}

void IdleState::onVideoAnalysisReceived(const video_analyzer::VideoAnalysis::ConstPtr& msg)
{
    SoundFaceFollowingState::onVideoAnalysisReceived(msg);

    if (!containsAtLeastOnePerson(msg))
    {
        return;
    }

    auto now = chrono::system_clock::now();

    if (m_chargeNeeded && (now - m_lastChargingMessageTime) >= BATTERY_LOW_MESSAGE_INTERVAL)
    {
        m_lastChargingMessageTime = now;
        m_stateManager.switchTo<TalkState>(TalkStateParameter(
            StringRessources::getValue("dialogs.low_battery"),
            "",  // No gesture
            "fear",
            StateType::get<IdleState>()));
    }

    // TODO greatings
    // TODO check reminder face descriptor
}

void IdleState::onRobotNameDetected()
{
    // TODO Wait command
}

void IdleState::onBaseStatusChanged(
    float stateOfCharge,
    float voltage,
    float current,
    bool isPsuConnected,
    bool isBatteryCharging)
{
    m_chargeNeeded = stateOfCharge <= LOW_STATE_OF_CHARGE && !isPsuConnected;
}

void IdleState::onEveryMinuteTimeout()
{
    Time now = Time::now();
    if (now.between(m_sleepTime, m_wakeUpTime))
    {
        m_stateManager.switchTo<SleepState>();
        return;
    }

    // TODO check alarms
    // TODO check reminders
}
