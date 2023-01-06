cmake_minimum_required(VERSION 3.8.2)

project(TTOP_Daemon)


## Set compile options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Required Qt5 Modules
find_package(Qt5 REQUIRED COMPONENTS Core Network WebSockets SerialPort)

## Build a static library with common serial communication files and tools
include(../common/common.cmake)


set (daemon_headers
    src/DaemonApp.h
    src/DaemonWebSocketServer.h
    src/DaemonSerialPortWrapper.h
    src/DaemonSerialManager.h
)

set (daemon_srcs
    src/DaemonApp.cpp
    src/DaemonWebSocketServer.cpp
    src/DaemonSerialPortWrapper.cpp
    src/DaemonSerialManager.cpp
    src/main.cpp
)

set (daemon_other_files
    README.md
)

qt5_wrap_cpp(daemon_headers_moc ${daemon_headers})

add_executable(daemon ${daemon_srcs} ${daemon_headers} ${daemon_headers_moc} ${daemon_other_files})
target_include_directories(daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(daemon Qt5::Core Qt5::Network Qt5::WebSockets Qt5::SerialPort serial_communication_common)
