cmake_minimum_required(VERSION 3.5)
project(pose_classifier)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(video_analyzer REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseClassification.msg"
  "msg/PoseClassifications.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs video_analyzer
)

# Python Nodes
install(PROGRAMS
  scripts/pose_classifier_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
