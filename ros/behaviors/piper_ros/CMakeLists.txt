cmake_minimum_required(VERSION 3.5)
project(piper_ros)


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
ExternalProject_Add(
    espeak_ng_external
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/espeak-ng"
    URL "https://github.com/rhasspy/espeak-ng/archive/refs/tags/2023.9.7-4.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
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
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
add_compile_definitions(ESPEAK_NG_DATA_PATH="${CMAKE_INSTALL_PREFIX}/share/espeak-ng-data")


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

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate services
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GenerateSpeechFromText.srv"
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# C++ Nodes
add_executable(piper_node src/piper_node.cpp ${CMAKE_BINARY_DIR}/piper-${PIPER_TAG}/src/cpp/piper.cpp)
ament_target_dependencies(piper_node rclcpp ament_index_cpp)
target_link_libraries(piper_node piper_phonemize fmt spdlog ${ONNXRUNTIME_LIBRARIES})
rosidl_target_interfaces(piper_node ${PROJECT_NAME} "rosidl_typesupport_cpp")
install(TARGETS piper_node DESTINATION lib/${PROJECT_NAME})
add_dependencies(piper_node fmt_external espeak_ng_external)

# Install models
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
