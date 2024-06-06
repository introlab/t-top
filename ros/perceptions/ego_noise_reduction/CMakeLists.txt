cmake_minimum_required(VERSION 3.5)
project(ego_noise_reduction)


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
find_package(rclpy REQUIRED)
find_package(audio_utils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(hbba_lite REQUIRED)
find_package(t_top REQUIRED)
find_package(daemon_ros_client REQUIRED)

# C++ Library
include_directories(
  include
)

## Declare a C++ library
add_library(${PROJECT_NAME} STATIC
  src/${PROJECT_NAME}/StftNoiseRemover.cpp
  src/${PROJECT_NAME}/SpectralSubtractionNoiseRemover.cpp
  src/${PROJECT_NAME}/LogMmseNoiseRemover.cpp
  src/${PROJECT_NAME}/NoiseEstimator.cpp
  src/${PROJECT_NAME}/WeightedAverageNoiseEstimator.cpp
  src/${PROJECT_NAME}/WeightedAverageWithAPrioriNoiseEstimator.cpp
  src/${PROJECT_NAME}/NoiseMagnitudeDatabase.cpp
)
ament_target_dependencies(${PROJECT_NAME} audio_utils)
target_link_libraries(${PROJECT_NAME} ${audio_utils_LIBRARIES})
install(DIRECTORY include DESTINATION include)
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

ament_export_dependencies(rclcpp audio_utils)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

# C++ Nodes
add_executable(ego_noise_reduction_node src/ego_noise_reduction_node.cpp)
ament_target_dependencies(ego_noise_reduction_node rclcpp std_msgs daemon_ros_client hbba_lite hbba_lite_srvs audio_utils)
target_link_libraries(ego_noise_reduction_node
  ${PROJECT_NAME}
  ${audio_utils_LIBRARIES}
  ${hbba_lite_LIBRARIES}
  ${hbba_lite_srvs_LIBRARIES}
)
install(TARGETS ego_noise_reduction_node DESTINATION lib/${PROJECT_NAME})

# Python Nodes
install(PROGRAMS
  scripts/data_gathering.py
  scripts/test.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Noise files
install(DIRECTORY noise_data DESTINATION share/${PROJECT_NAME})

# Tests
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_test
    test/ego_noise_reduction/MathTests.cpp
    test/ego_noise_reduction/StftNoiseRemoverTests.cpp
    test/ego_noise_reduction/SpectralSubtractionNoiseRemoverTests.cpp
    test/ego_noise_reduction/LogMmseNoiseRemoverTests.cpp
    test/ego_noise_reduction/NoiseEstimatorTests.cpp
    test/ego_noise_reduction/WeightedAverageNoiseEstimatorTests.cpp
    test/ego_noise_reduction/WeightedAverageWithAPrioriNoiseEstimatorTests.cpp
    test/ego_noise_reduction/NoiseMagnitudeDatabaseTests.cpp
    test/ego_noise_reduction/TestUtils.cpp
    test/ego_noise_reduction/main.cpp
  )
  ament_target_dependencies(${PROJECT_NAME}_test rclcpp audio_utils)
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${audio_utils_LIBRARIES})

  add_custom_command(TARGET ${PROJECT_NAME}_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/test/ego_noise_reduction/resources $<TARGET_FILE_DIR:${PROJECT_NAME}_test>/resources
  )
endif()

ament_package()
