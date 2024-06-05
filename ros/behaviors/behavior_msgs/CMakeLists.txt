cmake_minimum_required(VERSION 3.5)
project(behavior_msgs)


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
find_package(daemon_ros_client REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Done.msg"
  "msg/GestureName.msg"
  "msg/LedAnimation.msg"
  "msg/SoundFile.msg"
  "msg/SoundStarted.msg"
  "msg/Statistics.msg"
  "msg/Text.msg"
  DEPENDENCIES daemon_ros_client
)

ament_package()
