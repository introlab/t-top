#ifndef CONTROL_PANEL_DESIRE_UTILS_H
#define CONTROL_PANEL_DESIRE_UTILS_H

#include <hbba_lite/core/DesireSet.h>

#include <t_top_hbba_lite/Desires.h>

inline void removeAllMovementDesires(DesireSet& desireSet)
{
    desireSet.removeDesires(std::type_index(typeid(GestureDesire)));
    desireSet.removeDesires(std::type_index(typeid(NearestFaceFollowingDesire)));
    desireSet.removeDesires(std::type_index(typeid(SpecificFaceFollowingDesire)));
    desireSet.removeDesires(std::type_index(typeid(SoundFollowingDesire)));
    desireSet.removeDesires(std::type_index(typeid(DanceDesire)));
    desireSet.removeDesires(std::type_index(typeid(ExploreDesire)));
}

#endif
