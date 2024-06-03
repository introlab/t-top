#include <home_logger_common/commands/CommandParsers.h>

using namespace std;

CommandParser::CommandParser() {}

CommandParser::~CommandParser() {}

SynonymKeywords::SynonymKeywords(const vector<string>& keywords)
{
    for (auto& keyword : keywords)
    {
        m_keywords.push_back(toLowerString(keyword));
    }
}

SynonymKeywords::SynonymKeywords(initializer_list<string> keywords)
{
    for (auto& keyword : keywords)
    {
        m_keywords.push_back(toLowerString(keyword));
    }
}
