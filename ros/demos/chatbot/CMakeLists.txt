cmake_minimum_required(VERSION 3.8)
project(chatbot)

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
find_package(rclcpp REQUIRED)
find_package(t_top_hbba_lite REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(behavior_srvs REQUIRED)
find_package(behavior_msgs REQUIRED)
find_package(cloud_data REQUIRED)
find_package(perceptions_analyzer REQUIRED)

find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

add_library(chatbot_lib
  src/tools/chatbot_tools.cpp
)

target_include_directories(chatbot_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(chatbot_lib
  rclcpp
  t_top_hbba_lite
  behavior_srvs
  behavior_msgs
  perception_msgs
  cloud_data
  perceptions_analyzer
  nlohmann_json
  fmt
)

add_executable(${PROJECT_NAME}_node src/chatbot_node.cpp)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  t_top_hbba_lite
  perception_msgs
  behavior_srvs
  behavior_msgs
  cloud_data
  perceptions_analyzer
  nlohmann_json
  fmt
)

install(TARGETS ${PROJECT_NAME}_node chatbot_lib DESTINATION lib/${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}_node
  chatbot_lib
  fmt::fmt
  nlohmann_json::nlohmann_json
)

ament_export_targets(chatbot_lib HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

install(
  TARGETS chatbot_lib
  EXPORT chatbot_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)

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
