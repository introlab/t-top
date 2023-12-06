#include <t_top_hbba_lite/Desires.h>

using namespace std;

Camera3dRecordingDesire::Camera3dRecordingDesire(uint16_t intensity) : Desire(intensity) {}

Camera2dWideRecordingDesire::Camera2dWideRecordingDesire(uint16_t intensity) : Desire(intensity) {}

RobotNameDetectorDesire::RobotNameDetectorDesire(uint16_t intensity) : Desire(intensity) {}

RobotNameDetectorWithLedStatusDesire::RobotNameDetectorWithLedStatusDesire(uint16_t intensity) : Desire(intensity) {}

SlowVideoAnalyzer3dDesire::SlowVideoAnalyzer3dDesire(uint16_t intensity) : Desire(intensity) {}

FastVideoAnalyzer3dDesire::FastVideoAnalyzer3dDesire(uint16_t intensity) : Desire(intensity) {}

FastVideoAnalyzer3dWithAnalyzedImageDesire::FastVideoAnalyzer3dWithAnalyzedImageDesire(uint16_t intensity)
    : Desire(intensity)
{
}

SlowVideoAnalyzer2dWideDesire::SlowVideoAnalyzer2dWideDesire(uint16_t intensity) : Desire(intensity) {}

FastVideoAnalyzer2dWideDesire::FastVideoAnalyzer2dWideDesire(uint16_t intensity) : Desire(intensity) {}

FastVideoAnalyzer2dWideWithAnalyzedImageDesire::FastVideoAnalyzer2dWideWithAnalyzedImageDesire(uint16_t intensity)
    : Desire(intensity)
{
}

AudioAnalyzerDesire::AudioAnalyzerDesire(uint16_t intensity) : Desire(intensity) {}

VadDesire::VadDesire(uint16_t intensity) : Desire(intensity) {}

SpeechToTextDesire::SpeechToTextDesire(uint16_t intensity) : Desire(intensity) {}


ExploreDesire::ExploreDesire(uint16_t intensity) : Desire(intensity) {}

FaceAnimationDesire::FaceAnimationDesire(string name, uint16_t intensity) : Desire(intensity), m_name(move(name)) {}

LedEmotionDesire::LedEmotionDesire(string name, uint16_t intensity) : Desire(intensity), m_name(move(name)) {}

LedAnimationDesire::LedAnimationDesire(
    string name,
    vector<daemon_ros_client::LedColor> colors,
    double speed,
    double durationS,
    uint16_t intensity)
    : Desire(intensity),
      m_name(move(name)),
      m_colors(move(colors)),
      m_speed(speed),
      m_durationS(durationS)
{
}

SoundFollowingDesire::SoundFollowingDesire(uint16_t intensity) : Desire(intensity) {}

NearestFaceFollowingDesire::NearestFaceFollowingDesire(uint16_t intensity) : Desire(intensity) {}

SpecificFaceFollowingDesire::SpecificFaceFollowingDesire(string targetName, uint16_t intensity)
    : Desire(intensity),
      m_targetName(move(targetName))
{
}

SoundObjectPersonFollowingDesire::SoundObjectPersonFollowingDesire(uint16_t intensity) : Desire(intensity) {}

TalkDesire::TalkDesire(string text, uint16_t intensity) : Desire(intensity), m_text(move(text)) {}

GestureDesire::GestureDesire(string name, uint16_t intensity) : Desire(intensity), m_name(move(name)) {}

DanceDesire::DanceDesire(uint16_t intensity) : Desire(intensity) {}

PlaySoundDesire::PlaySoundDesire(string path, uint16_t intensity) : Desire(intensity), m_path(move(path)) {}

TelepresenceDesire::TelepresenceDesire(uint16_t intensity) : Desire(intensity) {}

TeleoperationDesire::TeleoperationDesire(uint16_t intensity) : Desire(intensity) {}

TooCloseReactionDesire::TooCloseReactionDesire(uint16_t intensity) : Desire(intensity) {}
