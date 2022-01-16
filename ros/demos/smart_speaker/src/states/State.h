#ifndef SMART_SPEAKER_SMART_STATES_H
#define SMART_SPEAKER_SMART_STATES_H

#include <hbba_lite/utils/ClassMacros.h>

class State
{
    bool m_enabled;

public:
    State();
    virtual ~State() = default;

    DECLARE_NOT_COPYABLE(State);
    DECLARE_NOT_MOVABLE(State);
};

#endif
