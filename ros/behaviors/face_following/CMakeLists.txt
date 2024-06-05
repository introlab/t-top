cmake_minimum_required(VERSION 3.5)
project(face_following)


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
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(t_top REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(person_identification REQUIRED)

# Python Librairies
ament_python_install_package(${PROJECT_NAME})

# Python Nodes
install(PROGRAMS
  scripts/nearest_face_following_node.py
  scripts/specific_face_following_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
