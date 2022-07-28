#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>

using namespace std;

SQLiteVideoAnalysisLogger::SQLiteVideoAnalysisLogger(SQLite::Database& database) : SQLitePerceptionLogger(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE video_analysis("
                                                       "    perception_id INTEGER PRIMARY KEY,"
                                                       "    object_class TEXT,"
                                                       "    person_pose BLOB,"
                                                       "    person_pose_confidence BLOB,"
                                                       "    face_descriptor BLOB"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "video_analysis", migrations);
}

SQLiteVideoAnalysisLogger::~SQLiteVideoAnalysisLogger() {}

int64_t SQLiteVideoAnalysisLogger::log(const VideoAnalysis& analysis)
{
    int64_t id = insertPerception(analysis.timestamp, analysis.position, analysis.direction);
    SQLite::Statement insert(
        m_database,
        "INSERT INTO video_analysis(perception_id, object_class, person_pose, person_pose_confidence, face_descriptor)"
        " VALUES(?, ?, ?, ?, ?)");

    insert.clearBindings();
    insert.bind(1, id);
    insert.bind(2, analysis.objectClass);

    if (analysis.personPose.has_value())
    {
        insert.bind(
            3,
            reinterpret_cast<const void*>(analysis.personPose.value().data()),
            analysis.personPose.value().size() * sizeof(Position));
    }
    if (analysis.personPoseConfidence.has_value())
    {
        insert.bind(
            4,
            reinterpret_cast<const void*>(analysis.personPoseConfidence.value().data()),
            analysis.personPoseConfidence.value().size() * sizeof(float));
    }
    if (analysis.faceDescriptor.has_value())
    {
        insert.bind(
            5,
            reinterpret_cast<const void*>(analysis.faceDescriptor.value().data()),
            analysis.faceDescriptor.value().size() * sizeof(float));
    }

    insert.exec();
    return id;
}
