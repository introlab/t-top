#ifndef HOME_LOGGER_COMMON_COMMANDS_COMMAND_PARSER_H
#define HOME_LOGGER_COMMON_COMMANDS_COMMAND_PARSER_H

#include <home_logger_common/StringUtils.h>
#include <home_logger_common/commands/Commands.h>

#include <memory>
#include <vector>
#include <string>

class CommandParser
{
public:
    CommandParser();
    virtual ~CommandParser();

    virtual std::unique_ptr<Command> parse(const std::string& transcript) = 0;
};


class SynonymKeywords
{
    std::vector<std::string> m_keywords;

public:
    SynonymKeywords(const std::vector<std::string>& keywords);
    SynonymKeywords(std::initializer_list<std::string> keywords);

    bool empty() const;

    std::vector<std::string>::const_iterator begin() const;
    std::vector<std::string>::const_iterator end() const;
};

inline bool SynonymKeywords::empty() const
{
    return m_keywords.empty();
}

inline std::vector<std::string>::const_iterator SynonymKeywords::begin() const
{
    return m_keywords.begin();
}

inline std::vector<std::string>::const_iterator SynonymKeywords::end() const
{
    return m_keywords.end();
}


template<class T>
class KeywordCommandParser : public CommandParser
{
    std::vector<SynonymKeywords> m_keywords;
    std::vector<std::string> m_notKeywords;

public:
    KeywordCommandParser(std::vector<SynonymKeywords> keywords, std::vector<std::string> notKeywords);
    ~KeywordCommandParser() override;

    std::unique_ptr<Command> parse(const std::string& transcript) override;

private:
    template<class Container>
    bool containsAnyKeywords(const Container& keywords, const std::string& transcript);
};

template<class T>
inline KeywordCommandParser<T>::KeywordCommandParser(
    std::vector<SynonymKeywords> keywords,
    std::vector<std::string> notKeywords)
    : m_keywords(keywords)
{
    for (auto& notKeyword : notKeywords)
    {
        m_notKeywords.push_back(toLowerString(notKeyword));
    }
}

template<class T>
inline KeywordCommandParser<T>::~KeywordCommandParser()
{
}

template<class T>
std::unique_ptr<Command> KeywordCommandParser<T>::parse(const std::string& transcript)
{
    std::string lowerTranscript = toLowerString(transcript);
    if (containsAnyKeywords(m_notKeywords, lowerTranscript))
    {
        return nullptr;
    }

    for (auto& keywords : m_keywords)
    {
        if (!keywords.empty() && !containsAnyKeywords(keywords, lowerTranscript))
        {
            return nullptr;
        }
    }

    return std::make_unique<T>(lowerTranscript);
}

template<class T>
template<class Container>
bool KeywordCommandParser<T>::containsAnyKeywords(const Container& keywords, const std::string& transcript)
{
    for (auto& keyword : keywords)
    {
        if (transcript.find(keyword) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

#endif
