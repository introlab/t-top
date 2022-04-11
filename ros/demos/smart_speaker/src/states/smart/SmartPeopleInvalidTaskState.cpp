#include "SmartPeopleInvalidTaskState.h"

#include "../common/AfterTaskDelayState.h"

using namespace std;

SmartPeopleInvalidTaskState::SmartPeopleInvalidTaskState(
    Language language,
    StateManager& stateManager,
    shared_ptr<DesireSet> desireSet,
    ros::NodeHandle& nodeHandle)
    : InvalidTaskState(language, stateManager, desireSet, nodeHandle, type_index(typeid(AfterTaskDelayState)))
{
}

string SmartPeopleInvalidTaskState::generateText()
{
    switch (language())
    {
        case Language::ENGLISH:
            return "I cannot do that because people are waiting.";
        case Language::FRENCH:
            return "Je ne peux pas faire cela, car il y a des personnes qui attendent.";
    }

    return "";
}
