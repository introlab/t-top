cmake_minimum_required(VERSION 3.5)
project(daemon_ros_client)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)


# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BaseStatus.msg"
  "msg/MotorStatus.msg"
  "msg/LedColor.msg"
  "msg/LedColors.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

include(../../system/common/common.cmake)

# C++ Nodes

set(app_headers
  src/DaemonRosClientNode.h
  src/QtUtils.h
)

set (app_sources
  src/daemon_ros_client_node.cpp
  src/DaemonRosClientNode.cpp
)

qt5_wrap_cpp(app_headers_moc_srcs ${app_headers})

add_executable(daemon_ros_client_node ${app_headers} ${app_sources} ${app_headers_moc_srcs})
ament_target_dependencies(daemon_ros_client_node rclcpp std_msgs geometry_msgs sensor_msgs tf2 tf2_ros)
target_link_libraries(daemon_ros_client_node serial_communication_common)
rosidl_target_interfaces(daemon_ros_client_node ${PROJECT_NAME} "rosidl_typesupport_cpp")
install(TARGETS daemon_ros_client_node DESTINATION lib/${PROJECT_NAME})

ament_package()
