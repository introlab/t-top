cmake_minimum_required(VERSION 3.8)
project(connect4)

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

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(daemon_ros_client REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(t_top_hbba_lite REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)

set(connect4_components Core Widgets Gui WebSockets Network)
find_package(Qt5 COMPONENTS ${connect4_components} REQUIRED)

# C++ Nodes
set(moc_headers
  src/Connect4Widget.h
  src/ImageDisplay.h
)

set(other_headers
  src/QtUtils.h
)

set(srcs
  src/connect4_node.cpp
  src/Connect4Widget.cpp
  src/ImageDisplay.cpp
)

qt5_wrap_cpp(project_moc_srcs ${moc_headers})

add_executable(${PROJECT_NAME}_node ${srcs} ${uis} ${moc_headers} ${other_headers} ${project_moc_uis} ${project_moc_srcs})
qt5_use_modules(${PROJECT_NAME}_node ${connect4_components})
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  t_top_hbba_lite
  perception_msgs
  opentera_webrtc_ros_msgs
)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})

# Python Nodes
install(PROGRAMS
  scripts/nearest_face_following_orientation_replication_node.py
  scripts/nearest_face_orientation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
