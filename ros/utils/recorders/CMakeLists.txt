cmake_minimum_required(VERSION 3.5)
project(recorders)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(audio_utils_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(behavior_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# C++ Library
add_subdirectory(3rd_party/SQLiteCpp)
install(DIRECTORY 3rd_party/SQLiteCpp/include/ DESTINATION include)
install(TARGETS SQLiteCpp DESTINATION lib)

include_directories(
  include
  3rd_party/SQLiteCpp/include
)

## Declare a C++ library
add_library(perception_logger
  src/perception_logger/AudioAnalysisLogger.cpp
  src/perception_logger/VideoAnalysisLogger.cpp
  src/perception_logger/BinarySerialization.cpp
  src/perception_logger/SpeechLogger.cpp
  src/perception_logger/HbbaStrategyStateLogger.cpp
  src/perception_logger/sqlite/SQLiteMigration.cpp
  src/perception_logger/sqlite/SQLitePerceptionLogger.cpp
  src/perception_logger/sqlite/SQLiteAudioAnalysisLogger.cpp
  src/perception_logger/sqlite/SQLiteVideoAnalysisLogger.cpp
  src/perception_logger/sqlite/SQLiteSpeechLogger.cpp
  src/perception_logger/sqlite/SQLiteHbbaStrategyStateLogger.cpp
)
ament_target_dependencies(perception_logger rclcpp)
target_link_libraries(perception_logger SQLiteCpp sqlite3)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS perception_logger DESTINATION lib)

ament_export_include_directories(include)
ament_export_libraries(perception_logger SQLiteCpp sqlite3 dl)

install(
  TARGETS perception_logger
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

add_executable(perception_logger_node src/perception_logger_node.cpp)
ament_target_dependencies(perception_logger_node
  rclcpp
  audio_utils_msgs
  sensor_msgs
  geometry_msgs
  hbba_lite
  hbba_lite_msgs
  perception_msgs
  behavior_msgs
  tf2
  tf2_ros
  tf2_msgs
  tf2_geometry_msgs
)
target_link_libraries(perception_logger_node perception_logger SQLiteCpp sqlite3 dl)
install(TARGETS perception_logger_node DESTINATION lib/${PROJECT_NAME})

# Python Nodes
install(PROGRAMS
  scripts/video_recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Tests
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_test
    test/perception_logger/main.cpp
    test/perception_logger/BinarySerializationTests.cpp
    test/perception_logger/PerceptionLoggerTests.cpp
    test/perception_logger/AudioAnalysisLoggerTests.cpp
    test/perception_logger/VideoAnalysisLoggerTests.cpp
    test/perception_logger/SpeechLoggerTests.cpp
    test/perception_logger/HbbaStrategyStateLoggerTests.cpp
    test/perception_logger/sqlite/SQLiteMigrationTests.cpp
    test/perception_logger/sqlite/SQLitePerceptionLoggerTests.cpp
    test/perception_logger/sqlite/SQLiteAudioAnalysisLoggerTests.cpp
    test/perception_logger/sqlite/SQLiteVideoAnalysisLoggerTests.cpp
    test/perception_logger/sqlite/SQLiteSpeechLoggerTests.cpp
    test/perception_logger/sqlite/SQLiteHbbaStrategyStateLoggerTests.cpp
  )
  ament_target_dependencies(${PROJECT_NAME}_test rclcpp)
  target_link_libraries(${PROJECT_NAME}_test perception_logger SQLiteCpp sqlite3 dl)
endif()

ament_package()
