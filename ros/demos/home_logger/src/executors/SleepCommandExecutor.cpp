#include "SleepCommandExecutor.h"

#include "../states/specific/SleepState.h"

using namespace std;

SleepCommandExecutor::SleepCommandExecutor(StateManager& stateManager)
    : SpecificCommandExecutor<SleepCommand>(stateManager)
{
}

SleepCommandExecutor::~SleepCommandExecutor() {}

void SleepCommandExecutor::executeSpecific([[maybe_unused]] const shared_ptr<SleepCommand>& command)
{
    m_stateManager.switchTo<SleepState>();
}
