cmake_minimum_required(VERSION 3.5)
project(home_logger)


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
find_package(rclcpp REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(cloud_data REQUIRED)
find_package(gesture REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(t_top_hbba_lite REQUIRED)
find_package(talk REQUIRED)
find_package(recorders REQUIRED)
find_package(led_animations REQUIRED)
find_package(daemon_ros_client REQUIRED)
find_package(sound_player REQUIRED)
find_package(tf2 REQUIRED)

# C++ Library
add_subdirectory(3rd_party/fmt)

include_directories(
  include
  3rd_party/fmt/include
)

## Declare a C++ library
add_library(home_logger_common
  src/home_logger_common/DateTime.cpp
  src/home_logger_common/Properties.cpp
  src/home_logger_common/commands/Commands.cpp
  src/home_logger_common/commands/CommandParsers.cpp
  src/home_logger_common/commands/AllCommandParser.cpp
  src/home_logger_common/parameters/AllCommandParametersParser.cpp
  src/home_logger_common/parameters/CommandParametersParser.cpp
  src/home_logger_common/parameters/WeatherCommandParametersParser.cpp
  src/home_logger_common/parameters/SetVolumeCommandParametersParser.cpp
  src/home_logger_common/parameters/AlarmCommandParametersParsers.cpp
  src/home_logger_common/parameters/ReminderCommandParametersParsers.cpp
  src/home_logger_common/language/StringResources.cpp
  src/home_logger_common/language/Formatter.cpp
  src/home_logger_common/managers/AlarmManager.cpp
  src/home_logger_common/managers/ReminderManager.cpp
)
ament_target_dependencies(home_logger_common
  rclcpp
  std_msgs
  hbba_lite
  recorders
)
target_link_libraries(home_logger_common ${recorders_LIBRARIES})

add_executable(${PROJECT_NAME}_node
  src/home_logger_node.cpp
  src/managers/VolumeManager.cpp
  src/executors/CommandExecutor.cpp
  src/executors/AllCommandExecutor.cpp
  src/executors/WeatherCommandExecutor.cpp
  src/executors/VolumeCommandExecutors.cpp
  src/executors/SleepCommandExecutor.cpp
  src/executors/DateTimeCommandExecutors.cpp
  src/executors/AlarmCommandExecutors.cpp
  src/executors/ReminderCommandExecutors.cpp
  src/executors/ListCommandsCommandExecutor.cpp
  src/parameters/AllCommandParametersAsker.cpp
  src/parameters/CommandParametersAsker.cpp
  src/parameters/WeatherCommandParametersAsker.cpp
  src/parameters/SetVolumeCommandParametersAsker.cpp
  src/parameters/AlarmCommandParametersAskers.cpp
  src/parameters/ReminderCommandParametersAskers.cpp
  src/states/State.cpp
  src/states/StateManager.cpp
  src/states/common/SoundFaceFollowingState.cpp
  src/states/common/TalkState.cpp
  src/states/specific/IdleState.cpp
  src/states/specific/SleepState.cpp
  src/states/specific/WaitCommandState.cpp
  src/states/specific/ExecuteCommandState.cpp
  src/states/specific/WaitCommandParameterState.cpp
  src/states/specific/WaitFaceDescriptorCommandParameterState.cpp
  src/states/specific/AlarmState.cpp
  src/states/specific/TellReminderState.cpp
)
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  perception_msgs
  cloud_data
  gesture
  hbba_lite
  sensor_msgs
  geometry_msgs
  std_msgs
  talk
  t_top_hbba_lite
  led_animations
  daemon_ros_client
  sound_player
  tf2
)
target_link_libraries(${PROJECT_NAME}_node
  ${hbba_lite_LIBRARIES}
  ${led_animations_LIBRARIES}
  ${talk_LIBRARIES}
  ${gesture_LIBRARIES}
  ${recorders_LIBRARIES}
  ${sound_player_LIBRARIES}
  home_logger_common
  fmt
)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Resources files
install(DIRECTORY resources DESTINATION share/${PROJECT_NAME})

# Tests
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_test
    test/home_logger_common/main.cpp
    test/home_logger_common/loadStringResources.cpp
    test/home_logger_common/StringUtilsTests.cpp
    test/home_logger_common/DateTimeTests.cpp
    test/home_logger_common/PropertiesTests.cpp
    test/home_logger_common/commands/CommandsTests.cpp
    test/home_logger_common/commands/CommandParsersTests.cpp
    test/home_logger_common/commands/AllCommandParserTests.cpp
    test/home_logger_common/parameters/CommandParametersParserTests.cpp
    test/home_logger_common/parameters/WeatherCommandParametersParserTests.cpp
    test/home_logger_common/parameters/SetVolumeCommandParametersParserTests.cpp
    test/home_logger_common/parameters/AlarmCommandParametersParsersTests.cpp
    test/home_logger_common/parameters/ReminderCommandParametersParsersTests.cpp
    test/home_logger_common/language/LanguageTests.cpp
    test/home_logger_common/language/StringResourcesTests.cpp
    test/home_logger_common/language/FormatterTests.cpp
    test/home_logger_common/managers/AlarmManagerTests.cpp
    test/home_logger_common/managers/ReminderManagerTests.cpp
  )
  ament_target_dependencies(${PROJECT_NAME}_test
    rclcpp
    std_msgs
    hbba_lite
  )
  target_link_libraries(${PROJECT_NAME}_test ${recorders_LIBRARIES} home_logger_common fmt)
endif()


ament_package()
