cmake_minimum_required(VERSION 3.5)
project(t_top_hbba_lite)


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

set(MUSIC_BEAT_DETECTOR_ENABLE_TESTS OFF CACHE BOOL "")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(behavior_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(daemon_ros_client REQUIRED)

# C++ Libraries
add_library(t_top_hbba_lite_cpp
  src/Desires.cpp
  src/Strategies.cpp
)
target_include_directories(t_top_hbba_lite_cpp PUBLIC include)
ament_target_dependencies(t_top_hbba_lite_cpp rclcpp hbba_lite nav_msgs sensor_msgs behavior_msgs std_msgs std_msgs daemon_ros_client)
install(TARGETS t_top_hbba_lite_cpp DESTINATION lib)

ament_export_dependencies(rclcpp geometry_msgs hbba_lite nav_msgs sensor_msgs behavior_msgs std_msgs daemon_ros_client)
ament_export_include_directories(include)
ament_export_libraries(t_top_hbba_lite_cpp)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS t_top_hbba_lite_cpp
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# C++ Nodes
add_executable(set_head_pose_arbitration_node src/set_head_pose_arbitration_node.cpp)
ament_target_dependencies(set_head_pose_arbitration_node rclcpp geometry_msgs daemon_ros_client tf2)
install(TARGETS set_head_pose_arbitration_node DESTINATION lib/${PROJECT_NAME})

ament_package()
