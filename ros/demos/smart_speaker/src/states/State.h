#ifndef SMART_SPEAKER_STATES_STATE_H
#define SMART_SPEAKER_STATES_STATE_H

#include <hbba_lite/core/DesireSet.h>
#include <hbba_lite/utils/ClassMacros.h>

#include <ros/ros.h>

#include <memory>
#include <numeric>
#include <vector>
#include <typeindex>

constexpr double TIMEOUT_S = 10;
constexpr uint64_t MAX_DESIRE_ID = std::numeric_limits<uint64_t>::max();  // TODO change to optional with C++17
constexpr int FLOAT_NUMBER_PRECISION = 3;

enum class Language  // TODO Use a resource manager for the strings
{
    ENGLISH,
    FRENCH
};

inline bool languageFromString(const std::string& str, Language& language)
{
    if (str == "en")
    {
        language = Language::ENGLISH;
    }
    else if (str == "fr")
    {
        language = Language::FRENCH;
    }
    else
    {
        return false;
    }

    return true;
}

class StateManager;

class State
{
    bool m_enabled;
    Language m_language;
    std::type_index m_previousStageType;

protected:
    StateManager& m_stateManager;
    std::shared_ptr<DesireSet> m_desireSet;
    ros::NodeHandle& m_nodeHandle;

    std::vector<uint64_t> m_desireIds;

public:
    State(
        Language language,
        StateManager& stateManager,
        std::shared_ptr<DesireSet> desireSet,
        ros::NodeHandle& nodeHandle);
    virtual ~State() = default;

    DECLARE_NOT_COPYABLE(State);
    DECLARE_NOT_MOVABLE(State);

protected:
    bool enabled() const;
    Language language() const;
    std::type_index previousStageType() const;
    virtual std::type_index type() const = 0;

    virtual void enable(const std::string& parameter, const std::type_index& previousStageType);
    virtual void disable();

    std::string getAndWord();

    friend StateManager;
};

inline bool State::enabled() const
{
    return m_enabled;
}

inline Language State::language() const
{
    return m_language;
}

inline std::type_index State::previousStageType() const
{
    return m_previousStageType;
}

inline std::string State::getAndWord()
{
    switch (language())
    {
        case Language::ENGLISH:
            return "and";
        case Language::FRENCH:
            return "et";
    }

    return "";
}

#endif
