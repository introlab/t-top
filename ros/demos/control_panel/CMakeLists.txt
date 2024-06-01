cmake_minimum_required(VERSION 3.5)
project(control_panel)


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
find_package(audio_analyzer REQUIRED)
find_package(gesture REQUIRED)
find_package(person_identification REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(talk REQUIRED)
find_package(video_analyzer REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(t_top_hbba_lite REQUIRED)
find_package(speech_to_text REQUIRED)
find_package(led_animations REQUIRED)
find_package(daemon_ros_client REQUIRED)
find_package(sound_player REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)

set(control_panel_components Core Widgets Gui OpenGL WebKitWidgets)
find_package(Qt5 COMPONENTS ${control_panel_components} REQUIRED)

# C++ Nodes
set(moc_headers
  src/widgets/ImageDisplay.h
  src/widgets/ControlPanel.h
  src/widgets/AvatarTab.h
  src/widgets/SpeechTab.h
  src/widgets/GestureTab.h
  src/widgets/BehaviorsTab.h
  src/widgets/LedTab.h
  src/widgets/PerceptionsTab.h
)

set(other_headers
  src/QtUtils.h
)

set(srcs
  src/control_panel_node.cpp
  src/widgets/ImageDisplay.cpp
  src/widgets/ControlPanel.cpp
  src/widgets/AvatarTab.cpp
  src/widgets/SpeechTab.cpp
  src/widgets/GestureTab.cpp
  src/widgets/BehaviorsTab.cpp
  src/widgets/LedTab.cpp
  src/widgets/PerceptionsTab.cpp
)

qt5_wrap_cpp(project_moc_srcs ${moc_headers})

add_executable(${PROJECT_NAME}_node ${srcs} ${uis} ${moc_headers} ${other_headers} ${project_moc_uis} ${project_moc_srcs})
qt5_use_modules(${PROJECT_NAME}_node ${control_panel_components})
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp audio_analyzer
  gesture
  person_identification
  std_msgs
  sensor_msgs
  talk
  video_analyzer
  hbba_lite
  t_top_hbba_lite
  speech_to_text
  led_animations
  daemon_ros_client
  sound_player
)
target_link_libraries(${PROJECT_NAME}_node
  ${control_panel_components_link}
  ${hbba_lite_LIBRARIES}
  ${led_animations_LIBRARIES}
  ${talk_LIBRARIES}
  ${sound_player_LIBRARIES}
  ${gesture_LIBRARIES}
)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})

ament_package()
