#include "SmartAskOtherTaskState.h"
#include "SmartWaitAnswerState.h"
#include "SmartThankYouState.h"

#include "../StateManager.h"

#include "../common/InvalidTaskState.h"

#include "../../StringUtils.h"

using namespace std;

SmartAskOtherTaskState::SmartAskOtherTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle,
    bool singleTaskPerPerson)
    : TalkState(language, stateManager, desireSet, nodeHandle, type_index(typeid(SmartWaitAnswerState))),
      m_singleTaskPerPerson(singleTaskPerPerson)
{
}

void SmartAskOtherTaskState::enable(const string& parameter, const type_index& previousStageType)
{
    if (m_singleTaskPerPerson && previousStageType != type_index(typeid(InvalidTaskState)))
    {
        m_stateManager.switchTo<SmartThankYouState>();
    }
    else
    {
        TalkState::enable(parameter, previousStageType);
    }
}

string SmartAskOtherTaskState::generateEnglishText(const string& personName)
{
    return "What else can I do for you?";
}

string SmartAskOtherTaskState::generateFrenchText(const string& personName)
{
    return "Qu'est-ce que je peux faire d'autre pour vous?";
}
