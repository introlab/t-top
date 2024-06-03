#ifndef RECORDERS_PERCEPTION_LOGGER_SPEECH_LOGGER_H
#define RECORDERS_PERCEPTION_LOGGER_SPEECH_LOGGER_H

#include <perception_logger/PerceptionLogger.h>

#include <string>

enum class SpeechSource
{
    ROBOT,
    HUMAN
};

std::string speechSourceToString(SpeechSource source);

struct Speech
{
    Timestamp timestamp;
    SpeechSource source;
    std::string text;

    Speech(Timestamp timestamp, SpeechSource source, std::string text);
};

class SpeechLogger
{
public:
    SpeechLogger();
    virtual ~SpeechLogger();

    virtual int64_t log(const Speech& speech) = 0;
};

#endif
