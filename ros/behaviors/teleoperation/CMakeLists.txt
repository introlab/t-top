cmake_minimum_required(VERSION 3.8)
project(teleoperation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)

find_package(t_top REQUIRED)

install(PROGRAMS
  scripts/teleoperation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
