cmake_minimum_required(VERSION 3.0.2)
project(home_logger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  audio_analyzer
  cloud_data
  gesture
  hbba_lite
  person_identification
  roscpp
  rospy
  sensor_msgs
  speech_to_text
  std_msgs
  t_top_hbba_lite
  talk
  tf
  video_analyzer
  recorders
  led_animations
  daemon_ros_client
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(_FMT_LIB_NAME fmtd)
else()
  set(_FMT_LIB_NAME fmt)
endif()

catkin_package(
 INCLUDE_DIRS include 3rd_party/fmt/include
 LIBRARIES home_logger_common ${_FMT_LIB_NAME}
#  CATKIN_DEPENDS audio_analyzer cloud_data gesture hbba_lite person_identification roscpp rospy sensor_msgs speech_to_text std_msgs t_top_hbba_lite talk tf video_analyzer recorders
#  DEPENDS system_lib
)

unset(_FMT_LIB_NAME)

add_subdirectory(3rd_party/fmt)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
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

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
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

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(home_logger_common
  ${catkin_LIBRARIES}
  fmt
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  home_logger_common
  fmt
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
catkin_add_gtest(home_logger_common-test
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
if(TARGET home_logger_common-test)
  find_package(Boost REQUIRED COMPONENTS system filesystem)
  target_link_libraries(home_logger_common-test home_logger_common ${Boost_LIBRARIES} ${catkin_LIBRARIES} fmt)
endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
