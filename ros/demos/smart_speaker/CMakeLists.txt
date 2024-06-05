cmake_minimum_required(VERSION 3.5)
project(smart_speaker)


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
find_package(talk REQUIRED)
find_package(t_top_hbba_lite REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sound_player REQUIRED)
find_package(led_animations REQUIRED)
find_package(daemon_ros_client REQUIRED)

add_executable(${PROJECT_NAME}_rss_node
  src/${PROJECT_NAME}_rss_node.cpp
  src/StringUtils.cpp
  src/states/State.cpp
  src/states/StateManager.cpp
  src/states/common/TalkState.cpp
  src/states/common/WaitAnswerState.cpp
  src/states/common/ValidTaskState.cpp
  src/states/common/InvalidTaskState.cpp
  src/states/common/AfterTaskDelayState.cpp
  src/states/task/CurrentWeatherState.cpp
  src/states/task/WeatherForecastState.cpp
  src/states/task/DanceState.cpp
  src/states/task/DancePlayedSongState.cpp
  src/states/rss/RssIdleState.cpp
  src/states/rss/RssWaitPersonIdentificationState.cpp
  src/states/rss/RssAskTaskState.cpp
  src/states/rss/RssWaitAnswerState.cpp
  src/states/rss/RssValidTaskState.cpp
  src/states/rss/RssStoryState.cpp
)
ament_target_dependencies(${PROJECT_NAME}_rss_node
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
  tf2
  tf2_ros
  sound_player
  led_animations
  daemon_ros_client
)
target_link_libraries(${PROJECT_NAME}_rss_node
  ${hbba_lite_LIBRARIES}
  ${led_animations_LIBRARIES}
  ${talk_LIBRARIES}
  ${sound_player_LIBRARIES}
  ${gesture_LIBRARIES}
)
install(TARGETS ${PROJECT_NAME}_rss_node DESTINATION lib/${PROJECT_NAME})

add_executable(${PROJECT_NAME}_smart_node
  src/${PROJECT_NAME}_smart_node.cpp
  src/StringUtils.cpp
  src/states/State.cpp
  src/states/StateManager.cpp
  src/states/common/TalkState.cpp
  src/states/common/WaitAnswerState.cpp
  src/states/common/ValidTaskState.cpp
  src/states/common/InvalidTaskState.cpp
  src/states/common/AfterTaskDelayState.cpp
  src/states/task/CurrentWeatherState.cpp
  src/states/task/DancePlayedSongState.cpp
  src/states/smart/SmartIdleState.cpp
  src/states/smart/SmartAskTaskState.cpp
  src/states/smart/SmartWaitAnswerState.cpp
  src/states/smart/SmartValidTaskState.cpp
  src/states/smart/SmartAskOtherTaskState.cpp
  src/states/smart/SmartThankYouState.cpp
)
ament_target_dependencies(${PROJECT_NAME}_smart_node
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
  tf2
  tf2_ros
  sound_player
  led_animations
  daemon_ros_client
)
target_link_libraries(${PROJECT_NAME}_smart_node
  ${hbba_lite_LIBRARIES}
  ${led_animations_LIBRARIES}
  ${talk_LIBRARIES}
  ${sound_player_LIBRARIES}
  ${gesture_LIBRARIES}
)
install(TARGETS ${PROJECT_NAME}_smart_node DESTINATION lib/${PROJECT_NAME})

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Resources files
install(DIRECTORY resources DESTINATION share/${PROJECT_NAME})

ament_package()
