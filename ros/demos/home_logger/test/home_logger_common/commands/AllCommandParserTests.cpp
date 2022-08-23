#include <home_logger_common/commands/AllCommandParser.h>
#include <home_logger_common/commands/Commands.h>

#include "../loadStringResources.h"

#include <gtest/gtest.h>

using namespace std;

#define EXPECT_ONE_COMMAND(commands, commandClassName)                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        auto commandsVar = commands;                                                                                   \
        ASSERT_EQ(commandsVar.size(), 1);                                                                              \
        EXPECT_EQ(commandsVar[0]->type(), CommandType::get<commandClassName>());                                       \
    } while (false)

class AllCommandParserFrenchTests : public ::testing::Test
{
protected:
    static unique_ptr<AllCommandParser> parser;

    static void SetUpTestSuite()
    {
        loadFrenchStringResources();
        parser = make_unique<AllCommandParser>();
    }

    static void TearDownTestSuite() { parser = nullptr; }
};

unique_ptr<AllCommandParser> AllCommandParserFrenchTests::parser;

class AllCommandParserEnglishTests : public ::testing::Test
{
protected:
    static unique_ptr<AllCommandParser> parser;

    static void SetUpTestSuite()
    {
        loadEnglishStringResources();
        parser = make_unique<AllCommandParser>();
    }

    static void TearDownTestSuite() { parser = nullptr; }
};

unique_ptr<AllCommandParser> AllCommandParserEnglishTests::parser;

TEST_F(AllCommandParserFrenchTests, parse_WeatherCommand_shouldReturnOneWeatherCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu me dire la météo"), WeatherCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu me dire la température"), WeatherCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu me dire le bulletin météorologique"), WeatherCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu me dire la météorologie"), WeatherCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_IncreaseVolumeCommand_shouldReturnOneIncreaseVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu augmenter le volume"), IncreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu monter le volume"), IncreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu hausser le volume"), IncreaseVolumeCommand);

    EXPECT_ONE_COMMAND(parser->parse("Peux-tu augmenter le son"), IncreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu monter le son"), IncreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu hausser le son"), IncreaseVolumeCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_DecreaseVolumeCommand_shouldReturnOneDecreaseVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu diminuer le volume"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu descendre le volume"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu abaisser le volume"), DecreaseVolumeCommand);

    EXPECT_ONE_COMMAND(parser->parse("Peux-tu diminuer le son"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu descendre le son"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu abaisser le son"), DecreaseVolumeCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_SetVolumeCommand_shouldReturnOneSetVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu mettre le volume à 50%"), SetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu mettre le son à 25%"), SetVolumeCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_GetVolumeCommand_shouldReturnOneGetVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu donner le volume"), GetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu m'indiquer le volume"), GetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Quel est le volume"), GetVolumeCommand);

    EXPECT_ONE_COMMAND(parser->parse("Peux-tu donner le son"), GetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu m'indiquer le son"), GetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Quel est le son"), GetVolumeCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_SleepCommand_shouldReturnOneSleepCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu me dormir"), SleepCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu me faire dodo"), SleepCommand);
    EXPECT_ONE_COMMAND(parser->parse("T-Top, dors"), SleepCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_CurrentDateCommand_shouldReturnOneCurrentDateCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Quelle est la date?"), CurrentDateCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_CurrentTimeCommand_shouldReturnOneCurrentTimeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Quelle est l'heure?"), CurrentTimeCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_CurrentDateTimeCommand_shouldReturnOneCurrentDateTimeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Quelle est l'heure et la date?"), CurrentDateTimeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Quelle est la date et l'heure?"), CurrentDateTimeCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_AddAlarmCommand_shouldReturnOneAddAlarmCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu ajouter une alarme?"), AddAlarmCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu mettre une alarme?"), AddAlarmCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_ListAlarmsCommand_shouldReturnOneListAlarmsCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu lister les alarmes?"), ListAlarmsCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_RemoveAlarmCommand_shouldReturnOneRemoveAlarmCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu enlever une alarme?"), RemoveAlarmCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_AddReminderCommand_shouldReturnOneAddReminderCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Ajoute un rappel?"), AddReminderCommand);
    EXPECT_ONE_COMMAND(parser->parse("Crée un rappel?"), AddReminderCommand);

    EXPECT_ONE_COMMAND(parser->parse("Ajoute un aide-mémoire?"), AddReminderCommand);
    EXPECT_ONE_COMMAND(parser->parse("Crée un aide-mémoire?"), AddReminderCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_ListRemindersCommand_shouldReturnOneListRemindersCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu lister les rappels?"), ListRemindersCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu lister les aide-mémoires?"), ListRemindersCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_RemoveReminderCommand_shouldReturnOneRemoveReminderCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu enlever un rappel?"), RemoveReminderCommand);
    EXPECT_ONE_COMMAND(parser->parse("Peux-tu enlever un aide-mémoire?"), RemoveReminderCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_ListCommandsCommand_shouldReturnOneListCommandsCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("liste tes commandes"), ListCommandsCommand);
    EXPECT_ONE_COMMAND(parser->parse("Que peux-tu faire?"), ListCommandsCommand);
}

TEST_F(AllCommandParserFrenchTests, parse_NothingCommand_shouldReturnOneNothingCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("rien"), NothingCommand);
}


TEST_F(AllCommandParserEnglishTests, parse_WeatherCommand_shouldReturnOneWeatherCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Give me the current weather"), WeatherCommand);
    EXPECT_ONE_COMMAND(parser->parse("Give me today climate forecast"), WeatherCommand);
    EXPECT_ONE_COMMAND(parser->parse("Give me this week temperature forecast"), WeatherCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_IncreaseVolumeCommand_shouldReturnOneIncreaseVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Increase the volume"), IncreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Turn up the volume"), IncreaseVolumeCommand);

    EXPECT_ONE_COMMAND(parser->parse("Increase the sound"), IncreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Turn up the sound"), IncreaseVolumeCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_DecreaseVolumeCommand_shouldReturnOneDecreaseVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Decrease the volume"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Turn down the volume"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Lower the volume"), DecreaseVolumeCommand);

    EXPECT_ONE_COMMAND(parser->parse("Decrease the sound"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Turn down the sound"), DecreaseVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Lower the sound"), DecreaseVolumeCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_SetVolumeCommand_shouldReturnOneSetVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("Set the volume to 50%"), SetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("Set the sound to 50%"), SetVolumeCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_GetVolumeCommand_shouldReturnOneGetVolumeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("get me the volume"), GetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("what is the volume"), GetVolumeCommand);

    EXPECT_ONE_COMMAND(parser->parse("get me the sound level"), GetVolumeCommand);
    EXPECT_ONE_COMMAND(parser->parse("what is the sound level"), GetVolumeCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_SleepCommand_shouldReturnOneSleepCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("sleep"), SleepCommand);
    EXPECT_ONE_COMMAND(parser->parse("Take a nap"), SleepCommand);
    EXPECT_ONE_COMMAND(parser->parse("Go to bed"), SleepCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_CurrentDateCommand_shouldReturnOneCurrentDateCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("what is the date?"), CurrentDateCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_CurrentTimeCommand_shouldReturnOneCurrentTimeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("What time is it?"), CurrentTimeCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_CurrentDateTimeCommand_shouldReturnOneCurrentDateTimeCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("What is the time and date?"), CurrentDateTimeCommand);
    EXPECT_ONE_COMMAND(parser->parse("What is the date and time?"), CurrentDateTimeCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_AddAlarmCommand_shouldReturnOneAddAlarmCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("add an alarm"), AddAlarmCommand);
    EXPECT_ONE_COMMAND(parser->parse("set a new alarm"), AddAlarmCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_ListAlarmsCommand_shouldReturnOneListAlarmsCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("list alarms?"), ListAlarmsCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_RemoveAlarmCommand_shouldReturnOneRemoveAlarmCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("remove an alarm"), RemoveAlarmCommand);
    EXPECT_ONE_COMMAND(parser->parse("unset an alarm"), RemoveAlarmCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_AddReminderCommand_shouldReturnOneAddReminderCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("add a reminder"), AddReminderCommand);
    EXPECT_ONE_COMMAND(parser->parse("create a reminder"), AddReminderCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_ListRemindersCommand_shouldReturnOneListRemindersCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("list reminders"), ListRemindersCommand);
    EXPECT_ONE_COMMAND(parser->parse("list reminders"), ListRemindersCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_RemoveReminderCommand_shouldReturnOneRemoveReminderCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("remove a reminder"), RemoveReminderCommand);
    EXPECT_ONE_COMMAND(parser->parse("delete a reminder"), RemoveReminderCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_ListCommandsCommand_shouldReturnOneListCommandsCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("list the commands"), ListCommandsCommand);
    EXPECT_ONE_COMMAND(parser->parse("What can you do?"), ListCommandsCommand);
}

TEST_F(AllCommandParserEnglishTests, parse_NothingCommand_shouldReturnOneNothingCommand)
{
    EXPECT_ONE_COMMAND(parser->parse("nothing"), NothingCommand);
}
