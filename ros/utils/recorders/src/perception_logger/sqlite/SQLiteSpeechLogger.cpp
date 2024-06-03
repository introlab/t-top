#include <perception_logger/sqlite/SQLiteSpeechLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

SQLiteSpeechLogger::SQLiteSpeechLogger(SQLite::Database& database) : SQLitePerceptionLogger(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE speech("
                                                       "    perception_id INTEGER PRIMARY KEY,"
                                                       "    source TEXT,"
                                                       "    text TEXT"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "speech", migrations);
}

SQLiteSpeechLogger::~SQLiteSpeechLogger() {}

int64_t SQLiteSpeechLogger::log(const Speech& speech)
{
    string sourceString = speechSourceToString(speech.source);

    int64_t id = insertPerception(speech.timestamp, nullopt, nullopt);
    SQLite::Statement insert(m_database, "INSERT INTO speech(perception_id, source, text) VALUES(?, ?, ?)");

    insert.bind(1, id);
    insert.bindNoCopy(2, sourceString);
    insert.bindNoCopy(3, speech.text);
    insert.exec();
    return id;
}
