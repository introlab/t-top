#ifndef HOME_LOGGER_STATES_STATE_H
#define HOME_LOGGER_STATES_STATE_H

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/utils/ClassMacros.h>

class StateManager;

class State : protected DesireSetObserver
{

public:
    State()

protected:
    void onDesireSetChanged(const std::vector<std::unique_ptr<Desire>>& _) override;

    friend StateManager;
};

#endif
