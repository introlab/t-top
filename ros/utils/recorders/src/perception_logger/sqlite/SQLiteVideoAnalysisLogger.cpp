#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>
#include <perception_logger/BinarySerialization.h>

using namespace std;

SQLiteVideoAnalysisLogger::SQLiteVideoAnalysisLogger(SQLite::Database& database) : SQLitePerceptionLogger(database)
{
    vector<SQLiteMigration> migrations{SQLiteMigration("BEGIN;"
                                                       "CREATE TABLE video_analysis("
                                                       "    perception_id INTEGER PRIMARY KEY,"
                                                       "    object_class TEXT,"
                                                       "    bounding_box_centre_x REAL,"
                                                       "    bounding_box_centre_y REAL,"
                                                       "    bounding_box_width REAL,"
                                                       "    bounding_box_height REAL,"
                                                       "    person_pose_image BLOB,"
                                                       "    person_pose BLOB,"
                                                       "    person_pose_confidence BLOB,"
                                                       "    face_descriptor BLOB"
                                                       ");"
                                                       "COMMIT;")};

    applyMigrations(database, "video_analysis", migrations);
}

SQLiteVideoAnalysisLogger::~SQLiteVideoAnalysisLogger() = default;

int64_t SQLiteVideoAnalysisLogger::log(const VideoAnalysis& analysis)
{
    int64_t id = insertPerception(analysis.timestamp, analysis.position, analysis.direction);
    SQLite::Statement insert(
        m_database,
        "INSERT INTO video_analysis(perception_id, object_class, bounding_box_centre_x, bounding_box_centre_y, "
        "bounding_box_width, bounding_box_height, person_pose_image, person_pose, "
        "person_pose_confidence, face_descriptor)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

    insert.clearBindings();
    insert.bind(1, id);
    insert.bind(2, analysis.objectClass);
    insert.bind(3, analysis.boundingBox.center.x);
    insert.bind(4, analysis.boundingBox.center.y);
    insert.bind(5, analysis.boundingBox.width);
    insert.bind(6, analysis.boundingBox.height);

    tl::optional<Bytes> personPoseImageBytes;
    if (analysis.personPoseImage.has_value())
    {
        personPoseImageBytes = BinarySerializer<vector<ImagePosition>>::serialize(analysis.personPoseImage.value());
        insert.bindNoCopy(7, personPoseImageBytes->data(), personPoseImageBytes->size());
    }

    tl::optional<Bytes> personPoseBytes;
    if (analysis.personPose.has_value())
    {
        personPoseBytes = BinarySerializer<vector<Position>>::serialize(analysis.personPose.value());
        insert.bindNoCopy(8, personPoseBytes->data(), personPoseBytes->size());
    }

    tl::optional<Bytes> personPoseConfidenceBytes;
    if (analysis.personPoseConfidence.has_value())
    {
        personPoseConfidenceBytes = BinarySerializer<vector<float>>::serialize(analysis.personPoseConfidence.value());
        insert.bindNoCopy(9, personPoseConfidenceBytes->data(), personPoseConfidenceBytes->size());
    }

    tl::optional<Bytes> faceDescriptorBytes;
    if (analysis.faceDescriptor.has_value())
    {
        faceDescriptorBytes = BinarySerializer<vector<float>>::serialize(analysis.faceDescriptor.value());
        insert.bindNoCopy(10, faceDescriptorBytes->data(), faceDescriptorBytes->size());
    }

    insert.exec();
    return id;
}
