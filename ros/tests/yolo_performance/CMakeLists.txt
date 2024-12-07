cmake_minimum_required(VERSION 3.5)
project(yolo_performance)


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
find_package(cv_bridge REQUIRED)
find_package(dnn_utils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Python Nodes
install(PROGRAMS
  scripts/image_gathering_node.py
  scripts/processing_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
