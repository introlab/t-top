#include <perception_logger/sqlite/SQLiteVideoAnalysisLogger.h>

#include <perception_logger/sqlite/SQLiteMigration.h>
#include <perception_logger/BinarySerialization.h>

using namespace std;

SQLiteVideoAnalysisLogger::SQLiteVideoAnalysisLogger(SQLite::Database& database) : SQLitePerceptionLogger(database)
{
    vector<SQLiteMigration> migrations{
        SQLiteMigration("BEGIN;"
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
                        "COMMIT;"),
        SQLiteMigration("BEGIN;"
                        "ALTER TABLE video_analysis ADD object_confidence REAL;"
                        "ALTER TABLE video_analysis ADD object_class_probability REAL;"
                        "ALTER TABLE video_analysis ADD face_alignment_keypoint_count INTEGER;"
                        "ALTER TABLE video_analysis ADD face_sharpness_score REAL;"
                        "COMMIT;"),
    };

    applyMigrations(database, "video_analysis", migrations);
}

SQLiteVideoAnalysisLogger::~SQLiteVideoAnalysisLogger() = default;

int64_t SQLiteVideoAnalysisLogger::log(const VideoAnalysis& analysis)
{
    int64_t id = insertPerception(analysis.timestamp, analysis.position, analysis.direction);
    SQLite::Statement insert(
        m_database,
        "INSERT INTO video_analysis(perception_id, object_class, object_confidence, object_class_probability, "
        "bounding_box_centre_x, bounding_box_centre_y, bounding_box_width, bounding_box_height,  "
        "person_pose_image, person_pose, person_pose_confidence, "
        "face_descriptor, face_alignment_keypoint_count, face_sharpness_score)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

    insert.clearBindings();
    insert.bind(1, id);
    insert.bind(2, analysis.objectClass);
    insert.bind(3, analysis.objectConfidence);
    insert.bind(4, analysis.objectClassProbability);
    insert.bind(5, analysis.boundingBox.center.x);
    insert.bind(6, analysis.boundingBox.center.y);
    insert.bind(7, analysis.boundingBox.width);
    insert.bind(8, analysis.boundingBox.height);

    optional<Bytes> personPoseImageBytes;
    if (analysis.personPoseImage.has_value())
    {
        personPoseImageBytes = serializeToBytesNoCopy(analysis.personPoseImage.value());
        insert.bindNoCopy(9, personPoseImageBytes->data(), personPoseImageBytes->size());
    }

    optional<Bytes> personPoseBytes;
    if (analysis.personPose.has_value())
    {
        personPoseBytes = serializeToBytesNoCopy(analysis.personPose.value());
        insert.bindNoCopy(10, personPoseBytes->data(), personPoseBytes->size());
    }

    optional<Bytes> personPoseConfidenceBytes;
    if (analysis.personPoseConfidence.has_value())
    {
        personPoseConfidenceBytes = serializeToBytesNoCopy(analysis.personPoseConfidence.value());
        insert.bindNoCopy(11, personPoseConfidenceBytes->data(), personPoseConfidenceBytes->size());
    }

    optional<Bytes> faceDescriptorBytes;
    if (analysis.faceDescriptor.has_value())
    {
        faceDescriptorBytes = serializeToBytesNoCopy(analysis.faceDescriptor.value());
        insert.bindNoCopy(12, faceDescriptorBytes->data(), faceDescriptorBytes->size());
    }

    if (analysis.faceAlignmentKeypointCount.has_value())
    {
        insert.bind(13, analysis.faceAlignmentKeypointCount.value());
    }
    if (analysis.faceSharpnessScore.has_value())
    {
        insert.bind(14, analysis.faceSharpnessScore.value());
    }

    insert.exec();
    return id;
}
