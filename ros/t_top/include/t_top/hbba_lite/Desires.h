#ifndef T_TOP_HBBA_LITE_DESIRES_H
#define T_TOP_HBBA_LITE_DESIRES_H

#include <hbba_lite/core/Desire.h>
#include <hbba_lite/core/DesireSet.h>

#include <string>

class RobotNameDetectorDesire : public Desire
{
public:
    RobotNameDetectorDesire(uint16_t intensity = 1);
    ~RobotNameDetectorDesire() override = default;

    DECLARE_DESIRE_METHODS(RobotNameDetectorDesire)
};


class SlowVideoAnalyzerDesire : public Desire
{
public:
    SlowVideoAnalyzerDesire(uint16_t intensity = 1);
    ~SlowVideoAnalyzerDesire() override = default;

    DECLARE_DESIRE_METHODS(SlowVideoAnalyzerDesire)
};


class FastVideoAnalyzerDesire : public Desire
{
public:
    FastVideoAnalyzerDesire(uint16_t intensity = 1);
    ~FastVideoAnalyzerDesire() override = default;

    DECLARE_DESIRE_METHODS(FastVideoAnalyzerDesire)
};


class FastVideoAnalyzerWithAnalyzedImageDesire : public Desire
{
public:
    FastVideoAnalyzerWithAnalyzedImageDesire(uint16_t intensity = 1);
    ~FastVideoAnalyzerWithAnalyzedImageDesire() override = default;

    DECLARE_DESIRE_METHODS(FastVideoAnalyzerWithAnalyzedImageDesire)
};


class AudioAnalyzerDesire : public Desire
{
public:
    AudioAnalyzerDesire(uint16_t intensity = 1);
    ~AudioAnalyzerDesire() override = default;

    DECLARE_DESIRE_METHODS(AudioAnalyzerDesire)
};


class SpeechToTextDesire : public Desire
{
public:
    SpeechToTextDesire(uint16_t intensity = 1);
    ~SpeechToTextDesire() override = default;

    DECLARE_DESIRE_METHODS(SpeechToTextDesire)
};


class ExploreDesire : public Desire
{
public:
    ExploreDesire(uint16_t intensity = 1);
    ~ExploreDesire() override = default;

    DECLARE_DESIRE_METHODS(ExploreDesire)
};


class FaceAnimationDesire : public Desire
{
    std::string m_name;

public:
    explicit FaceAnimationDesire(std::string name, uint16_t intensity = 1);
    ~FaceAnimationDesire() override = default;

    DECLARE_DESIRE_METHODS(FaceAnimationDesire)

    const std::string& name() const;
};

inline const std::string& FaceAnimationDesire::name() const
{
    return m_name;
}


class SoundFollowingDesire : public Desire
{
public:
    SoundFollowingDesire(uint16_t intensity = 1);
    ~SoundFollowingDesire() override = default;

    DECLARE_DESIRE_METHODS(SoundFollowingDesire)
};


class FaceFollowingDesire : public Desire
{
public:
    FaceFollowingDesire(uint16_t intensity = 1);
    ~FaceFollowingDesire() override = default;

    DECLARE_DESIRE_METHODS(FaceFollowingDesire)
};


class TalkDesire : public Desire
{
    std::string m_text;

public:
    explicit TalkDesire(std::string text, uint16_t intensity = 1);
    ~TalkDesire() override = default;

    DECLARE_DESIRE_METHODS(TalkDesire)

    const std::string& text() const;
};

inline const std::string& TalkDesire::text() const
{
    return m_text;
}


class GestureDesire : public Desire
{
    std::string m_name;

public:
    explicit GestureDesire(std::string name, uint16_t intensity = 1);
    ~GestureDesire() override = default;

    DECLARE_DESIRE_METHODS(GestureDesire)

    const std::string& name() const;
};

inline const std::string& GestureDesire::name() const
{
    return m_name;
}


class DanceDesire : public Desire
{
public:
    DanceDesire(uint16_t intensity = 1);
    ~DanceDesire() override = default;

    DECLARE_DESIRE_METHODS(DanceDesire)
};


class PlaySoundDesire : public Desire
{
    std::string m_path;

public:
    explicit PlaySoundDesire(std::string path, uint16_t intensity = 1);
    ~PlaySoundDesire() override = default;

    DECLARE_DESIRE_METHODS(PlaySoundDesire)

    const std::string& path() const;
};

inline const std::string& PlaySoundDesire::path() const
{
    return m_path;
}

#endif
