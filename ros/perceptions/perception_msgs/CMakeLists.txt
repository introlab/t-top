cmake_minimum_required(VERSION 3.5)
project(perception_msgs)


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
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(audio_utils REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioAnalysis.msg"
  "msg/PersonName.msg"
  "msg/PersonNames.msg"
  "msg/PoseClassification.msg"
  "msg/PoseClassifications.msg"
  "msg/SemanticSegmentation.msg"
  "msg/Transcript.msg"
  "msg/VideoAnalysis.msg"
  "msg/VideoAnalysisObject.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs visualization_msgs audio_utils
)

ament_package()
