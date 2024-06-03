#include <perception_logger/sqlite/SQLiteMigration.h>

#include <gtest/gtest.h>

using namespace std;

TEST(SQLiteMigrationTests, applyMigrations_shouldApplyAllMigrationsOnlyOnce)
{
    const string loggerName = "audio";
    SQLite::Database database(":memory:", SQLite::OPEN_READWRITE);

    vector<SQLiteMigration> migrations = {
        SQLiteMigration("BEGIN;"
                        "CREATE TABLE data(id INTEGER PRIMARY KEY);"
                        "INSERT INTO data (id) VALUES(0);"
                        "INSERT INTO data (id) VALUES(1);"
                        "INSERT INTO data (id) VALUES(2);"
                        "COMMIT;"),
        SQLiteMigration("INSERT INTO data (id) VALUES(3);")};

    applyMigrations(database, loggerName, migrations);
    EXPECT_EQ(2, getLoggerVersion(database, loggerName));

    migrations.push_back(SQLiteMigration("INSERT INTO data (id) VALUES(4);"));
    applyMigrations(database, loggerName, migrations);
    EXPECT_EQ(3, getLoggerVersion(database, loggerName));

    SQLite::Statement query(database, "SELECT COUNT(*) FROM data");
    ASSERT_TRUE(query.executeStep());
    EXPECT_EQ(static_cast<int>(query.getColumn(0)), 5);
}
