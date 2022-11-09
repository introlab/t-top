#ifndef HOME_LOGGER_COMMON_MANAGERS_REMINDER_MANAGER_H
#define HOME_LOGGER_COMMON_MANAGERS_REMINDER_MANAGER_H

#include <home_logger_common/DateTime.h>
#include <home_logger_common/commands/Commands.h>

#include <SQLiteCpp/SQLiteCpp.h>

#include <hbba_lite/utils/ClassMacros.h>

#include <memory>
#include <string>
#include <vector>

class Reminder
{
    std::optional<int> m_id;
    std::string m_text;
    DateTime m_datetime;
    FaceDescriptor m_faceDescriptor;

public:
    Reminder(std::string text, DateTime datetime, FaceDescriptor faceDescriptor);
    Reminder(int id, std::string text, DateTime datetime, FaceDescriptor faceDescriptor);
    virtual ~Reminder();

    std::optional<int> id() const;
    const std::string& text() const;
    DateTime datetime() const;
    const FaceDescriptor& faceDescriptor() const;
};

inline std::optional<int> Reminder::id() const
{
    return m_id;
}

inline const std::string& Reminder::text() const
{
    return m_text;
}

inline DateTime Reminder::datetime() const
{
    return m_datetime;
}

inline const FaceDescriptor& Reminder::faceDescriptor() const
{
    return m_faceDescriptor;
}


class ReminderManager
{
    SQLite::Database& m_database;

public:
    explicit ReminderManager(SQLite::Database& database);
    virtual ~ReminderManager();

    DECLARE_NOT_COPYABLE(ReminderManager);
    DECLARE_NOT_MOVABLE(ReminderManager);

    void insertReminder(const Reminder& reminder);
    void removeReminder(int id);
    void removeRemindersOlderThan(DateTime datetime);
    std::vector<Reminder> listReminders();
    std::vector<Reminder> listReminders(const Date& date);

private:
    int getNextId();
    Reminder reminderFromRow(SQLite::Statement& query);
};

#endif
