#ifndef HOME_LOGGER_COMMON_COMMANDS_ALL_COMMAND_PARSER_H
#define HOME_LOGGER_COMMON_COMMANDS_ALL_COMMAND_PARSER_H

#include <home_logger_common/commands/CommandParsers.h>

#include <memory>
#include <vector>

class AllCommandParser
{
    std::vector<std::unique_ptr<CommandParser>> m_parsers;

public:
    AllCommandParser();
    virtual ~AllCommandParser();

    std::vector<std::unique_ptr<Command>> parse(const std::string& transcript);
};

#endif
