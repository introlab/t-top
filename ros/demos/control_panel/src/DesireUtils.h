#ifndef CONTROL_PANEL_DESIRE_UTILS_H
#define CONTROL_PANEL_DESIRE_UTILS_H

#include <hbba_lite/core/DesireSet.h>

#include <t_top_hbba_lite/Desires.h>

inline void removeAllMovementDesires(DesireSet& desireSet)
{
    desireSet.removeAllDesiresOfType<GestureDesire>();
    desireSet.removeAllDesiresOfType<NearestFaceFollowingDesire>();
    desireSet.removeAllDesiresOfType<SpecificFaceFollowingDesire>();
    desireSet.removeAllDesiresOfType<SoundFollowingDesire>();
    desireSet.removeAllDesiresOfType<SoundObjectPersonFollowingDesire>();
    desireSet.removeAllDesiresOfType<DanceDesire>();
    desireSet.removeAllDesiresOfType<ExploreDesire>();
}

inline void removeAllLedDesires(DesireSet& desireSet)
{
    desireSet.removeAllDesiresOfType<DanceDesire>();
    desireSet.removeAllDesiresOfType<LedEmotionDesire>();
}

#endif
