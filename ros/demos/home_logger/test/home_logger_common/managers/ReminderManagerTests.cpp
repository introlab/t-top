#include <home_logger_common/managers/ReminderManager.h>

#include <gtest/gtest.h>

#include <limits>

using namespace std;

TEST(ReminderManagerTests, reminderConstructors_shouldSetAttributes)
{
    Reminder testee0("bob", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f}));
    EXPECT_EQ(testee0.id(), nullopt);
    EXPECT_EQ(testee0.text(), "bob");
    EXPECT_EQ(testee0.datetime(), DateTime(Date(2022, 5, 10), Time(17, 30)));
    EXPECT_EQ(testee0.faceDescriptor().data(), vector<float>({1.f, 2.f}));

    Reminder testee1(9, "bobby", DateTime(Date(2022, 5, 11), Time(17, 32)), FaceDescriptor({2.f, 1.f}));
    EXPECT_EQ(testee1.id(), 9);
    EXPECT_EQ(testee1.text(), "bobby");
    EXPECT_EQ(testee1.datetime(), DateTime(Date(2022, 5, 11), Time(17, 32)));
    EXPECT_EQ(testee1.faceDescriptor().data(), vector<float>({2.f, 1.f}));
}

TEST(ReminderManagerTests, insertListRemove_shouldInsertListAndRemove)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);
    ReminderManager testee(database);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("b", DateTime(Date(2022, 4, 10), Time(17, 30)), FaceDescriptor({2.f, 1.f})));
    testee.insertReminder(Reminder("c", DateTime(Date(2022, 3, 10), Time(17, 30)), FaceDescriptor({2.f, 2.f})));

    auto reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 3);

    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[0].text(), "a");
    EXPECT_EQ(reminders[0].datetime(), DateTime(Date(2022, 5, 10), Time(17, 30)));
    EXPECT_EQ(reminders[0].faceDescriptor().data(), vector<float>({1.f, 2.f}));

    EXPECT_EQ(reminders[1].id(), 2);
    EXPECT_EQ(reminders[1].text(), "b");
    EXPECT_EQ(reminders[1].datetime(), DateTime(Date(2022, 4, 10), Time(17, 30)));
    EXPECT_EQ(reminders[1].faceDescriptor().data(), vector<float>({2.f, 1.f}));

    EXPECT_EQ(reminders[2].id(), 3);
    EXPECT_EQ(reminders[2].text(), "c");
    EXPECT_EQ(reminders[2].datetime(), DateTime(Date(2022, 3, 10), Time(17, 30)));
    EXPECT_EQ(reminders[2].faceDescriptor().data(), vector<float>({2.f, 2.f}));

    testee.removeReminder(2);
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 2);

    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[0].text(), "a");
    EXPECT_EQ(reminders[0].datetime(), DateTime(Date(2022, 5, 10), Time(17, 30)));
    EXPECT_EQ(reminders[0].faceDescriptor().data(), vector<float>({1.f, 2.f}));

    EXPECT_EQ(reminders[1].id(), 3);
    EXPECT_EQ(reminders[1].text(), "c");
    EXPECT_EQ(reminders[1].datetime(), DateTime(Date(2022, 3, 10), Time(17, 30)));
    EXPECT_EQ(reminders[1].faceDescriptor().data(), vector<float>({2.f, 2.f}));
}

TEST(ReminderManagerTests, insert_shouldReplaceIds)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);
    ReminderManager testee(database);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));

    auto reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 3);
    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[1].id(), 2);
    EXPECT_EQ(reminders[2].id(), 3);

    testee.removeReminder(1);
    testee.removeReminder(3);
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 1);
    EXPECT_EQ(reminders[0].id(), 2);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 2);
    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[1].id(), 2);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 4);
    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[1].id(), 2);
    EXPECT_EQ(reminders[2].id(), 3);
    EXPECT_EQ(reminders[3].id(), 4);

    testee.removeReminder(2);
    testee.removeReminder(3);
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 2);
    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[1].id(), 4);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 3);
    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[1].id(), 2);
    EXPECT_EQ(reminders[2].id(), 4);
}

TEST(ReminderManagerTests, listReminders_date_shouldReturnRemindersOfADate)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);
    ReminderManager testee(database);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 5, 10), Time(10, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("b", DateTime(Date(2022, 5, 10), Time(17, 30)), FaceDescriptor({2.f, 1.f})));
    testee.insertReminder(Reminder("c", DateTime(Date(2022, 5, 11), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("d", DateTime(Date(2022, 6, 10), Time(17, 30)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("e", DateTime(Date(2021, 5, 10), Time(10, 30)), FaceDescriptor({1.f, 2.f})));

    auto reminders = testee.listReminders(Date(2022, 5, 10));
    ASSERT_EQ(reminders.size(), 2);

    EXPECT_EQ(reminders[0].id(), 1);
    EXPECT_EQ(reminders[0].text(), "a");
    EXPECT_EQ(reminders[0].datetime(), DateTime(Date(2022, 5, 10), Time(10, 30)));
    EXPECT_EQ(reminders[0].faceDescriptor().data(), vector<float>({1.f, 2.f}));

    EXPECT_EQ(reminders[1].id(), 2);
    EXPECT_EQ(reminders[1].text(), "b");
    EXPECT_EQ(reminders[1].datetime(), DateTime(Date(2022, 5, 10), Time(17, 30)));
    EXPECT_EQ(reminders[1].faceDescriptor().data(), vector<float>({2.f, 1.f}));
}

TEST(ReminderManagerTests, removeRemindersOlderThan_shouldRemove)
{
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);
    ReminderManager testee(database);

    testee.insertReminder(Reminder("a", DateTime(Date(2022, 1, 2), Time(3, 4)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("b", DateTime(Date(2022, 1, 2), Time(3, 5)), FaceDescriptor({2.f, 1.f})));
    testee.insertReminder(Reminder("c", DateTime(Date(2022, 1, 2), Time(4, 0)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("d", DateTime(Date(2022, 1, 3), Time(0, 0)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("e", DateTime(Date(2022, 2, 1), Time(0, 0)), FaceDescriptor({1.f, 2.f})));
    testee.insertReminder(Reminder("e", DateTime(Date(2023, 1, 1), Time(0, 0)), FaceDescriptor({1.f, 2.f})));

    auto reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2022, 1, 2), Time(3, 4)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2022, 1, 2), Time(3, 5)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 5);
    EXPECT_EQ(reminders[0].id(), 2);
    EXPECT_EQ(reminders[1].id(), 3);
    EXPECT_EQ(reminders[2].id(), 4);
    EXPECT_EQ(reminders[3].id(), 5);
    EXPECT_EQ(reminders[4].id(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2022, 1, 2), Time(4, 0)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 4);
    EXPECT_EQ(reminders[0].id(), 3);
    EXPECT_EQ(reminders[1].id(), 4);
    EXPECT_EQ(reminders[2].id(), 5);
    EXPECT_EQ(reminders[3].id(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2022, 1, 3), Time(0, 0)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 3);
    EXPECT_EQ(reminders[0].id(), 4);
    EXPECT_EQ(reminders[1].id(), 5);
    EXPECT_EQ(reminders[2].id(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2022, 1, 4), Time(0, 0)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 2);
    EXPECT_EQ(reminders[0].id(), 5);
    EXPECT_EQ(reminders[1].id(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2022, 3, 1), Time(0, 0)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 1);
    EXPECT_EQ(reminders[0].id(), 6);

    testee.removeRemindersOlderThan(DateTime(Date(2024, 1, 1), Time(0, 0)));
    reminders = testee.listReminders();
    ASSERT_EQ(reminders.size(), 0);
}
