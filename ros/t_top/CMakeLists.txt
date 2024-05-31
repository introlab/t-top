cmake_minimum_required(VERSION 3.5)
project(t_top)

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
find_package(rclpy REQUIRED)
find_package(daemon_ros_client REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(hbba_lite REQUIRED)

# Python Librairies
ament_python_install_package(${PROJECT_NAME})

# Python Nodes
install(PROGRAMS
  scripts/head_roll_image_rotation_node.py
  scripts/robot_status.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install the config
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
