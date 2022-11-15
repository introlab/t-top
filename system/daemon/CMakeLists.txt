cmake_minimum_required(VERSION 3.8.2)

project(TTOP_Deamon)


## Set compile options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Required Qt5 Modules
find_package(Qt5 REQUIRED COMPONENTS Core Network WebSockets SerialPort)

## Build a static library with common serial communication files
file(GLOB serial_communication_common_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/SerialCommunication/src/*.cpp)
file(GLOB serial_communication_common_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/SerialCommunication/src/*.h)
add_library(serial_communication_common STATIC ${serial_communication_common_srcs} ${serial_communication_common_headers})
target_include_directories(serial_communication_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../firmwares/common/lib/SerialCommunication/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ros/utils/recorders/3rd_party/optional/include)


set (deamon_headers
    src/DaemonApp.h
    src/DaemonWebSocketServer.h
    src/DaemonSerialManager.h
)

set (daemon_srcs
    src/DaemonApp.cpp
    src/DaemonWebSocketServer.cpp
    src/DaemonSerialManager.cpp
    src/main.cpp
)

set (daemon_other_files
    README.md
)

qt5_wrap_cpp(daemon_headers_moc ${deamon_headers})

add_executable(daemon ${daemon_srcs} ${deamon_headers} ${daemon_headers_moc} ${serial_message_common_srcs} ${serial_message_common_headers} ${daemon_other_files})
target_include_directories(daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(daemon Qt5::Core Qt5::Network Qt5::WebSockets Qt5::SerialPort serial_communication_common)
