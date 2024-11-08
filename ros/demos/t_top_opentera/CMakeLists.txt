cmake_minimum_required(VERSION 3.8)
project(t_top_opentera)

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

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)

find_package(std_srvs REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)

find_package(t_top_hbba_lite REQUIRED)

add_executable(${PROJECT_NAME}_node
  src/${PROJECT_NAME}_node.cpp
)
target_include_directories(${PROJECT_NAME}_node PRIVATE src)
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  std_srvs
  t_top_hbba_lite
  opentera_webrtc_ros_msgs
)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})

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
