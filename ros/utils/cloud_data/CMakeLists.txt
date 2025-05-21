cmake_minimum_required(VERSION 3.5)
project(cloud_data)

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
find_package(rosidl_default_generators REQUIRED)

# Generate services
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/CurrentLocalWeather.srv"
  "srv/LocalWeatherForecast.srv"
  "srv/CurrentLocalWeather2.srv"
  "srv/LocalWeatherForecast2.srv"
  "srv/Location.srv"
)

# Python Nodes
install(PROGRAMS
  scripts/cloud_data_node.py
  scripts/cloud_data_node_2.py
  scripts/weather_tool_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
