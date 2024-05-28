cmake_minimum_required(VERSION 3.5)
project(cloud_data)

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
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/CurrentLocalWeather.srv"
  "srv/LocalWeatherForecast.srv"
  "srv/Location.srv"
)

# Python Nodes
install(PROGRAMS
  scripts/cloud_data_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
