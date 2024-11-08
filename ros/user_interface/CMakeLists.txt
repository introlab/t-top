cmake_minimum_required(VERSION 3.5)
project(user_interface)


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

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/package.json
        COMMAND npm install
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
)

add_custom_target(
        user_interface-install ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp
        VERBATIM
)

add_custom_target(
    user_interface-build ALL
    DEPENDS user_interface-install
    COMMAND npm run build
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
)

# Shell scripts
install(PROGRAMS
  scripts/web.sh
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY node_modules dist DESTINATION lib/${PROJECT_NAME})

#######################
## Required programs ##
#######################
function(assert_program_installed PROGRAM)
    find_program(prog-${PROGRAM} ${PROGRAM})
    if (${prog-${PROGRAM}} STREQUAL "prog-${PROGRAM}-NOTFOUND")
        message(FATAL_ERROR "'${PROGRAM}' needs to be installed")
    endif ()
endfunction()

assert_program_installed("npm")

ament_package()
