#include <perception_logger/SpeechLogger.h>

using namespace std;

Speech::Speech(Timestamp timestamp, SpeechSource source, std::string text)
    : timestamp(timestamp),
      source(source),
      text(move(text))
{
}

string speechSourceToString(SpeechSource source)
{
    switch (source)
    {
        case SpeechSource::ROBOT:
            return "robot";
        case SpeechSource::HUMAN:
            return "human";
    }

    return "";
}

SpeechLogger::SpeechLogger() {}

SpeechLogger::~SpeechLogger() {}
