#include "ControlPanelDesires.h"

using namespace std;

FaceAnimationDesire::FaceAnimationDesire(string name) : Desire(1), m_name(name)
{
}

TalkDesire::TalkDesire(string text) : Desire(1), m_text(text)
{
}

ListenDesire::ListenDesire() : Desire(1)
{
}

GestureDesire::GestureDesire(string name) : Desire(1), m_name(name)
{
}

FaceFollowingDesire::FaceFollowingDesire() : Desire(1)
{
}

SoundFollowingDesire::SoundFollowingDesire() : Desire(1)
{
}

DanceDesire::DanceDesire() : Desire(1)
{
}

ExploreDesire::ExploreDesire() : Desire(1)
{
}

VideoAnalyzerDesire::VideoAnalyzerDesire() : Desire(1)
{
}

AudioAnalyzerDesire::AudioAnalyzerDesire() : Desire(1)
{
}

RobotNameDetectorDesire::RobotNameDetectorDesire() : Desire(1)
{
}

void removeAllMovementDesires(DesireSet& desireSet)
{
    desireSet.removeDesires(type_index(typeid(GestureDesire)));
    desireSet.removeDesires(type_index(typeid(FaceFollowingDesire)));
    desireSet.removeDesires(type_index(typeid(SoundFollowingDesire)));
    desireSet.removeDesires(type_index(typeid(DanceDesire)));
    desireSet.removeDesires(type_index(typeid(ExploreDesire)));
}
