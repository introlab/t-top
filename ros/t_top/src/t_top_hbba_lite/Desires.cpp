#include <t_top/hbba_lite/Desires.h>

using namespace std;

RobotNameDetectorDesire::RobotNameDetectorDesire(uint16_t intensity) : Desire(intensity)
{
}

SlowVideoAnalyzerDesire::SlowVideoAnalyzerDesire(uint16_t intensity) : Desire(intensity)
{
}

FastVideoAnalyzerDesire::FastVideoAnalyzerDesire(uint16_t intensity) : Desire(intensity)
{
}

FastVideoAnalyzerWithAnalyzedImageDesire::FastVideoAnalyzerWithAnalyzedImageDesire(uint16_t intensity) : Desire(intensity)
{
}

AudioAnalyzerDesire::AudioAnalyzerDesire(uint16_t intensity) : Desire(intensity)
{
}

SpeechToTextDesire::SpeechToTextDesire(uint16_t intensity) : Desire(intensity)
{
}


ExploreDesire::ExploreDesire(uint16_t intensity) : Desire(intensity)
{
}

FaceAnimationDesire::FaceAnimationDesire(string name, uint16_t intensity) : Desire(intensity), m_name(move(name))
{
}

SoundFollowingDesire::SoundFollowingDesire(uint16_t intensity) : Desire(intensity)
{
}

FaceFollowingDesire::FaceFollowingDesire(uint16_t intensity) : Desire(intensity)
{
}

TalkDesire::TalkDesire(string text, uint16_t intensity) : Desire(intensity), m_text(move(text))
{
}

GestureDesire::GestureDesire(string name, uint16_t intensity) : Desire(intensity), m_name(move(name))
{
}

DanceDesire::DanceDesire(uint16_t intensity) : Desire(intensity)
{
}

PlaySoundDesire::PlaySoundDesire(string path, uint16_t intensity) : Desire(intensity), m_path(move(path))
{
}
