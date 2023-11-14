cmake_minimum_required(VERSION 3.0.2)
project(piper_ros)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    message_generation
    roscpp
    roslib
)

## System dependencies are found with CMake's conventions
# External Dependencies
include(ExternalProject)


# ******* FMT dependency *******
set(FMT_VERSION "10.0.0")
set(FMT_DIR "${CMAKE_BINARY_DIR}/install/fmt")
ExternalProject_Add(
    fmt_external
    PREFIX "${CMAKE_BINARY_DIR}/fmt"
    URL "https://github.com/fmtlib/fmt/archive/refs/tags/${FMT_VERSION}.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FMT_DIR}
    CMAKE_ARGS -DFMT_TEST:BOOL=OFF  # Don't build all the tests
)
include_directories(${CMAKE_BINARY_DIR}/install/fmt/include)
link_directories(${CMAKE_BINARY_DIR}/install/fmt/lib)


# ******* eSpeak NG dependency *******
set(ESPEAK_NG_DIR "${CMAKE_BINARY_DIR}/install/espeak-ng")
ExternalProject_Add(
    espeak_ng_external
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/espeak-ng"
    URL "https://github.com/rhasspy/espeak-ng/archive/refs/tags/2023.9.7-4.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ESPEAK_NG_DIR}
    CMAKE_ARGS -DUSE_ASYNC:BOOL=OFF
    CMAKE_ARGS -DBUILD_SHARED_LIBS:BOOL=ON
    CMAKE_ARGS -DUSE_MBROLA:BOOL=OFF
    CMAKE_ARGS -DUSE_LIBSONIC:BOOL=OFF
    CMAKE_ARGS -DUSE_LIBPCAUDIO:BOOL=OFF
    CMAKE_ARGS -DUSE_KLATT:BOOL=OFF
    CMAKE_ARGS -DUSE_SPEECHPLAYER:BOOL=OFF
    CMAKE_ARGS -DEXTRA_cmn:BOOL=ON
    CMAKE_ARGS -DEXTRA_ru:BOOL=ON
)
include_directories(${CMAKE_BINARY_DIR}/install/espeak-ng/include)
link_directories(${CMAKE_BINARY_DIR}/install/espeak-ng/lib)
add_compile_definitions(ESPEAK_NG_DATA_PATH="${CMAKE_BINARY_DIR}/install/espeak-ng/share/espeak-ng-data")


# ******* ONNX Runtime dependency *******
option(PIPER_USE_GPU "Use the gpu for piper" OFF)
if (EXISTS "/etc/nv_tegra_release")
    if (NOT EXISTS "/usr/local/lib/libonnxruntime.so")
	message(FATAL_ERROR "The ONNX Runtime is not installed.")
    endif()
    add_compile_definitions(ONNXRUNTIME_CUDA_PROVIDER_ENABLED)
    include_directories(/usr/local/include/onnxruntime/core/session)
    set(ONNXRUNTIME_LIBRARIES "/usr/local/lib/libonnxruntime.so")
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
    if(PIPER_USE_GPU)
        set(ONNXRUNTIME_NAME "onnxruntime-linux-x64-gpu-1.15.1")
        add_compile_definitions(ONNXRUNTIME_CUDA_PROVIDER_ENABLED)
    else()
        set(ONNXRUNTIME_NAME "onnxruntime-linux-x64-1.15.1")
    endif()
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    if(PIPER_USE_GPU)
        message(FATAL_ERROR "The GPU is not supported on aarch64.")
    else()
        set(ONNXRUNTIME_NAME "onnxruntime-linux-aarch64-1.15.1")
    endif()
else()
    message(FATAL_ERROR "The CPU architecture is not supported.")
endif()

if (ONNXRUNTIME_NAME)
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_NAME}.tgz)
        file(DOWNLOAD https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/${ONNXRUNTIME_NAME}.tgz ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_NAME}.tgz SHOW_PROGRESS)
    endif()
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_NAME}/VERSION_NUMBER)
        execute_process(COMMAND tar -xvf ${ONNXRUNTIME_NAME}.tgz WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
    include_directories(${CMAKE_BINARY_DIR}/${ONNXRUNTIME_NAME}/include)
    set(ONNXRUNTIME_LIBRARIES ${CMAKE_BINARY_DIR}/${ONNXRUNTIME_NAME}/lib/libonnxruntime.so)
endif()


# ******* Piper Phonemize Dependency *******
set(PIPER_PHONEMIZE_TAG 5ae075a8f9163e9e92682abd4bcfd9fad112ab7d)
if (NOT EXISTS ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}.zip)
    file(DOWNLOAD https://github.com/rhasspy/piper-phonemize/archive/${PIPER_PHONEMIZE_TAG}.zip ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}.zip SHOW_PROGRESS)
endif()
if (NOT EXISTS ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/CMakeLists.txt)
    execute_process(COMMAND unzip -o piper-phonemize-${PIPER_PHONEMIZE_TAG}.zip WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    execute_process(COMMAND mv ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/src ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/piper-phonemize)
endif()

add_library(
    piper_phonemize STATIC
    ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/piper-phonemize/phonemize.cpp
    ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/piper-phonemize/phoneme_ids.cpp
    ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/piper-phonemize/tashkeel.cpp
    ${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG}/piper-phonemize/shared.cpp
)
target_link_libraries(piper_phonemize espeak-ng ${ONNXRUNTIME_LIBRARIES})
add_dependencies(piper_phonemize espeak_ng_external)
include_directories(${CMAKE_BINARY_DIR}/piper-phonemize-${PIPER_PHONEMIZE_TAG})

# ******* Piper Dependency *******
set(PIPER_TAG ce3d92955bf09322dd3cfe7df2c570edbf2a8648)
if (NOT EXISTS ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}.zip)
    file(DOWNLOAD https://github.com/rhasspy/piper/archive/${PIPER_TAG}.zip ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}.zip SHOW_PROGRESS)
endif()
if (NOT EXISTS ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}/CMakeLists.txt)
    execute_process(COMMAND unzip -o piper-${PIPER_TAG}.zip WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Replace new spdlog functions
    file(READ ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}/src/cpp/piper.cpp FILE_CONTENTS)
    string(REPLACE "spdlog::should_log(spdlog::level::debug)" "false" FILE_CONTENTS "${FILE_CONTENTS}")
    file(WRITE ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}/src/cpp/piper.cpp "${FILE_CONTENTS}")
endif()
include_directories(${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}/src/cpp)


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
add_service_files(
    FILES
    GenerateSpeechFromText.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages()

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
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES piper_ros
#  CATKIN_DEPENDS message_generation roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
    ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/piper_ros.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(piper_node src/piper_node.cpp ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}/src/cpp/piper.cpp)
add_dependencies(piper_node fmt_external)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(piper_node
    ${catkin_LIBRARIES}
    piper_phonemize
    fmt
    spdlog
    ${ONNXRUNTIME_LIBRARIES}
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
# catkin_add_gtest(${PROJECT_NAME}-test test/test_piper_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
