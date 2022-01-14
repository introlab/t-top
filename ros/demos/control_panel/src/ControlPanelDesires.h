#ifndef CONTROL_PANEL_CONTROL_PANEL_DESIRES_H
#define CONTROL_PANEL_CONTROL_PANEL_DESIRES_H

#include <hbba_lite/core/Desire.h>
#include <hbba_lite/core/DesireSet.h>

#include <string>

class FaceAnimationDesire : public Desire
{
    std::string m_name;

public:
    explicit FaceAnimationDesire(std::string name);
    ~FaceAnimationDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;

    const std::string& name() const;
};

inline std::unique_ptr<Desire> FaceAnimationDesire::clone()
{
    return std::make_unique<FaceAnimationDesire>(*this);
}

inline std::type_index FaceAnimationDesire::type()
{
    return std::type_index(typeid(*this));
}

inline const std::string& FaceAnimationDesire::name() const
{
    return m_name;
}


class TalkDesire : public Desire
{
    std::string m_text;

public:
    explicit TalkDesire(std::string text);
    ~TalkDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;

    const std::string& text() const;
};

inline std::unique_ptr<Desire> TalkDesire::clone()
{
    return std::make_unique<TalkDesire>(*this);
}

inline std::type_index TalkDesire::type()
{
    return std::type_index(typeid(*this));
}

inline const std::string& TalkDesire::text() const
{
    return m_text;
}


class ListenDesire : public Desire
{
public:
    ListenDesire();
    ~ListenDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> ListenDesire::clone()
{
    return std::make_unique<ListenDesire>(*this);
}

inline std::type_index ListenDesire::type()
{
    return std::type_index(typeid(*this));
}


class GestureDesire : public Desire
{
    std::string m_name;

public:
    explicit GestureDesire(std::string name);
    ~GestureDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;

    const std::string& name() const;
};

inline std::unique_ptr<Desire> GestureDesire::clone()
{
    return std::make_unique<GestureDesire>(*this);
}

inline std::type_index GestureDesire::type()
{
    return std::type_index(typeid(*this));
}

inline const std::string& GestureDesire::name() const
{
    return m_name;
}


class FaceFollowingDesire : public Desire
{
public:
    FaceFollowingDesire();
    ~FaceFollowingDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> FaceFollowingDesire::clone()
{
    return std::make_unique<FaceFollowingDesire>(*this);
}

inline std::type_index FaceFollowingDesire::type()
{
    return std::type_index(typeid(*this));
}


class SoundFollowingDesire : public Desire
{
public:
    SoundFollowingDesire();
    ~SoundFollowingDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> SoundFollowingDesire::clone()
{
    return std::make_unique<SoundFollowingDesire>(*this);
}

inline std::type_index SoundFollowingDesire::type()
{
    return std::type_index(typeid(*this));
}


class DanceDesire : public Desire
{
public:
    DanceDesire();
    ~DanceDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> DanceDesire::clone()
{
    return std::make_unique<DanceDesire>(*this);
}

inline std::type_index DanceDesire::type()
{
    return std::type_index(typeid(*this));
}


class ExploreDesire : public Desire
{
public:
    ExploreDesire();
    ~ExploreDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> ExploreDesire::clone()
{
    return std::make_unique<ExploreDesire>(*this);
}

inline std::type_index ExploreDesire::type()
{
    return std::type_index(typeid(*this));
}


class VideoAnalyzerDesire : public Desire
{
public:
    VideoAnalyzerDesire();
    ~VideoAnalyzerDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> VideoAnalyzerDesire::clone()
{
    return std::make_unique<VideoAnalyzerDesire>(*this);
}

inline std::type_index VideoAnalyzerDesire::type()
{
    return std::type_index(typeid(*this));
}


class AudioAnalyzerDesire : public Desire
{
public:
    AudioAnalyzerDesire();
    ~AudioAnalyzerDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> AudioAnalyzerDesire::clone()
{
    return std::make_unique<AudioAnalyzerDesire>(*this);
}

inline std::type_index AudioAnalyzerDesire::type()
{
    return std::type_index(typeid(*this));
}


class RobotNameDetectorDesire : public Desire
{
public:
    RobotNameDetectorDesire();
    ~RobotNameDetectorDesire() override = default;

    std::unique_ptr<Desire> clone() override;
    std::type_index type() override;
};

inline std::unique_ptr<Desire> RobotNameDetectorDesire::clone()
{
    return std::make_unique<RobotNameDetectorDesire>(*this);
}

inline std::type_index RobotNameDetectorDesire::type()
{
    return std::type_index(typeid(*this));
}

void removeAllMovementDesires(DesireSet& desireSet);

#endif
