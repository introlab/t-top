cmake_minimum_required(VERSION 3.5)
project(dnn_utils)

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
find_package(rclpy REQUIRED)

# Python Librairies
ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/dnn_training DESTINATION ${PYTHON_INSTALL_DIR})


# Python Nodes
install(PROGRAMS
  scripts/test.py
  DESTINATION lib/${PROJECT_NAME}
)

# DNN Models
add_custom_target(
        dnn_utils_export_models ALL
        COMMAND scripts/export_models.bash
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
