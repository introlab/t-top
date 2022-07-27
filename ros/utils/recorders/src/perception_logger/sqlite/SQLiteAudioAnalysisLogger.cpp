#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

SQLiteAudioAnalysisLogger::SQLiteAudioAnalysisLogger(SQLite::Database& database) : SQLitePerceptionLogger(database)
{
    vector<SQLiteMigration> migrations
    {
        SQLiteMigration(
            "BEGIN;"
            "CREATE TABLE audio_analysis("
            "    perception_id INTEGER PRIMARY KEY,"
            "    classes TEXT,"
            "    voice_descriptor BLOB"
            ");"
            "COMMIT;"
        )
    };

    applyMigrations(database, "audio_analysis", migrations);
}

SQLiteAudioAnalysisLogger::~SQLiteAudioAnalysisLogger()
{
}

int64_t SQLiteAudioAnalysisLogger::log(const AudioAnalysis& analysis)
{
    int64_t id = insertPerception(analysis.timestamp, tl::nullopt, analysis.direction);
    SQLite::Statement insert(m_database, "INSERT INTO audio_analysis(perception_id, classes, voice_descriptor) VALUES(?, ?, ?)");
    insert.clearBindings();
    insert.bind(1, id);
    insert.bind(2, analysis.classes);

    if (analysis.voiceDescriptor.has_value())
    {
        insert.bind(3, reinterpret_cast<const void*>(analysis.voiceDescriptor.value().data()), analysis.voiceDescriptor.value().size() * sizeof(float));
    }

    insert.exec();
    return id;
}
