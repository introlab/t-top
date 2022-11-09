#include <perception_logger/sqlite/SQLiteAudioAnalysisLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>
#include <perception_logger/BinarySerialization.h>

using namespace std;

SQLiteAudioAnalysisLogger::SQLiteAudioAnalysisLogger(SQLite::Database& database) : SQLitePerceptionLogger(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE audio_analysis("
                                                       "    perception_id INTEGER PRIMARY KEY,"
                                                       "    classes TEXT,"
                                                       "    voice_descriptor BLOB"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "audio_analysis", migrations);
}

SQLiteAudioAnalysisLogger::~SQLiteAudioAnalysisLogger() {}

int64_t SQLiteAudioAnalysisLogger::log(const AudioAnalysis& analysis)
{
    int64_t id = insertPerception(analysis.timestamp, nullopt, analysis.direction);
    SQLite::Statement insert(
        m_database,
        "INSERT INTO audio_analysis(perception_id, classes, voice_descriptor) VALUES(?, ?, ?)");
    insert.clearBindings();
    insert.bind(1, id);
    insert.bindNoCopy(2, analysis.classes);

    optional<Bytes> voiceDescriptorBytes;
    if (analysis.voiceDescriptor.has_value())
    {
        voiceDescriptorBytes = serializeToBytesNoCopy(analysis.voiceDescriptor.value());
        insert.bindNoCopy(3, voiceDescriptorBytes->data(), voiceDescriptorBytes->size());
    }

    insert.exec();
    return id;
}
