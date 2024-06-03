#include <perception_logger/AudioAnalysisLogger.h>

#include <gtest/gtest.h>

using namespace std;

TEST(AudioAnalysisLoggerTests, audioAnalysis_constructor_shouldSetAttributes)
{
    AudioAnalysis analysis0(Timestamp(1), Direction{2.0, 3.0, 4.0}, 5, "a");
    EXPECT_EQ(analysis0.timestamp.unixEpochMs, 1);
    EXPECT_EQ(analysis0.direction.x, 2.0);
    EXPECT_EQ(analysis0.direction.y, 3.0);
    EXPECT_EQ(analysis0.direction.z, 4.0);
    EXPECT_EQ(analysis0.trackingId, 5);
    EXPECT_EQ(analysis0.classes, "a");
    EXPECT_EQ(analysis0.voiceDescriptor, std::nullopt);

    AudioAnalysis analysis1(Timestamp(5), Direction{6.0, 7.0, 8.0}, 9, "b", {10.f});
    EXPECT_EQ(analysis1.timestamp.unixEpochMs, 5.0);
    EXPECT_EQ(analysis1.direction.x, 6.0);
    EXPECT_EQ(analysis1.direction.y, 7.0);
    EXPECT_EQ(analysis1.direction.z, 8.0);
    EXPECT_EQ(analysis1.trackingId, 9);
    EXPECT_EQ(analysis1.classes, "b");
    EXPECT_EQ(analysis1.voiceDescriptor, vector<float>({10.f}));
}
