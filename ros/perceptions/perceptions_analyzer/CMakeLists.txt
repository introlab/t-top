cmake_minimum_required(VERSION 3.5)
project(perceptions_analyzer)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(audio_analyzer REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(video_analyzer REQUIRED)
find_package(std_msgs REQUIRED)
find_package(perception_msgs)
find_package(geometry_msgs REQUIRED)
find_package(person_identification REQUIRED)

# Message files
set(msg_files
  "msg/DetectedAudio.msg"
  "msg/DetectedObjects.msg"
  "msg/IdentifiedPerson.msg"
)


rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  "srv/PerceiveObjects.srv"
  DEPENDENCIES std_msgs
)


# Install executable script
install(PROGRAMS
  scripts/perceptions_analyzer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python support files
install(DIRECTORY src/
  DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

ament_package()
