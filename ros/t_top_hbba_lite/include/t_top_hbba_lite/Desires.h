#ifndef T_TOP_HBBA_LITE_DESIRES_H
#define T_TOP_HBBA_LITE_DESIRES_H

#include <hbba_lite/core/Desire.h>
#include <hbba_lite/core/DesireSet.h>
#include <string>

class RobotNameDetectorDesire : public Desire
{
public:
    explicit RobotNameDetectorDesire(uint16_t intensity = 1);
    ~RobotNameDetectorDesire() override = default;

    DECLARE_DESIRE_METHODS(RobotNameDetectorDesire)
};


class SlowVideoAnalyzer3dDesire : public Desire
{
public:
    explicit SlowVideoAnalyzer3dDesire(uint16_t intensity = 1);
    ~SlowVideoAnalyzer3dDesire() override = default;

    DECLARE_DESIRE_METHODS(SlowVideoAnalyzer3dDesire)
};


class FastVideoAnalyzer3dDesire : public Desire
{
public:
    explicit FastVideoAnalyzer3dDesire(uint16_t intensity = 1);
    ~FastVideoAnalyzer3dDesire() override = default;

    DECLARE_DESIRE_METHODS(FastVideoAnalyzer3dDesire)
};


class FastVideoAnalyzer3dWithAnalyzedImageDesire : public Desire
{
public:
    explicit FastVideoAnalyzer3dWithAnalyzedImageDesire(uint16_t intensity = 1);
    ~FastVideoAnalyzer3dWithAnalyzedImageDesire() override = default;

    DECLARE_DESIRE_METHODS(FastVideoAnalyzer3dWithAnalyzedImageDesire)
};

class SlowVideoAnalyzer2dWideDesire : public Desire
{
public:
    explicit SlowVideoAnalyzer2dWideDesire(uint16_t intensity = 1);
    ~SlowVideoAnalyzer2dWideDesire() override = default;

    DECLARE_DESIRE_METHODS(SlowVideoAnalyzer2dWideDesire)
};


class FastVideoAnalyzer2dWideDesire : public Desire
{
public:
    explicit FastVideoAnalyzer2dWideDesire(uint16_t intensity = 1);
    ~FastVideoAnalyzer2dWideDesire() override = default;

    DECLARE_DESIRE_METHODS(FastVideoAnalyzer2dWideDesire)
};


class FastVideoAnalyzer2dWideWithAnalyzedImageDesire : public Desire
{
public:
    explicit FastVideoAnalyzer2dWideWithAnalyzedImageDesire(uint16_t intensity = 1);
    ~FastVideoAnalyzer2dWideWithAnalyzedImageDesire() override = default;

    DECLARE_DESIRE_METHODS(FastVideoAnalyzer2dWideWithAnalyzedImageDesire)
};


class AudioAnalyzerDesire : public Desire
{
public:
    explicit AudioAnalyzerDesire(uint16_t intensity = 1);
    ~AudioAnalyzerDesire() override = default;

    DECLARE_DESIRE_METHODS(AudioAnalyzerDesire)
};


class SpeechToTextDesire : public Desire
{
public:
    explicit SpeechToTextDesire(uint16_t intensity = 1);
    ~SpeechToTextDesire() override = default;

    DECLARE_DESIRE_METHODS(SpeechToTextDesire)
};


class ExploreDesire : public Desire
{
public:
    explicit ExploreDesire(uint16_t intensity = 1);
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
    explicit SoundFollowingDesire(uint16_t intensity = 1);
    ~SoundFollowingDesire() override = default;

    DECLARE_DESIRE_METHODS(SoundFollowingDesire)
};


class NearestFaceFollowingDesire : public Desire
{
public:
    explicit NearestFaceFollowingDesire(uint16_t intensity = 1);
    ~NearestFaceFollowingDesire() override = default;

    DECLARE_DESIRE_METHODS(NearestFaceFollowingDesire)
};


class SpecificFaceFollowingDesire : public Desire
{
    std::string m_targetName;

public:
    explicit SpecificFaceFollowingDesire(std::string targetName, uint16_t intensity = 1);
    ~SpecificFaceFollowingDesire() override = default;

    DECLARE_DESIRE_METHODS(SpecificFaceFollowingDesire)

    const std::string& targetName() const;
};

inline const std::string& SpecificFaceFollowingDesire::targetName() const
{
    return m_targetName;
}


class SoundObjectPersonFollowingDesire : public Desire
{
public:
    explicit SoundObjectPersonFollowingDesire(uint16_t intensity = 1);
    ~SoundObjectPersonFollowingDesire() override = default;

    DECLARE_DESIRE_METHODS(SoundObjectPersonFollowingDesire)
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
    explicit DanceDesire(uint16_t intensity = 1);
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


class TelepresenceDesire : public Desire
{
public:
    explicit TelepresenceDesire(uint16_t intensity = 1);
    ~TelepresenceDesire() override = default;

    DECLARE_DESIRE_METHODS(TelepresenceDesire);
};

class TeleoperationDesire : public Desire
{
public:
    explicit TeleoperationDesire(uint16_t intensity = 1);
    ~TeleoperationDesire() override = default;

    DECLARE_DESIRE_METHODS(TeleoperationDesire);
};


#endif
