cmake_minimum_required(VERSION 3.8.2)

project(TTOP_SystemTray)


## Set compile options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Required Qt5 Modules
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Network WebSockets)

## Build a static library with common serial communication files
file(GLOB serial_communication_common_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/SerialCommunication/src/*.cpp)
file(GLOB serial_communication_common_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/SerialCommunication/src/*.h)
add_library(serial_communication_common STATIC ${serial_communication_common_srcs} ${serial_communication_common_headers})
target_include_directories(serial_communication_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../firmwares/common/lib/SerialCommunication/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ros/utils/recorders/3rd_party/optional/include)


set (system_tray_headers
    src/SystemTrayIcon.h
    src/SystemTrayApp.h
    src/WebSocketDaemonClient.h
)

set (system_tray_srcs
    src/SystemTrayIcon.cpp
    src/SystemTrayApp.cpp
    src/WebSocketDaemonClient.cpp
    src/main.cpp
)

set(system_tray_qrcs
    SystemTrayApp.qrc
)

set(readme_file
    README.md
)

qt5_wrap_cpp(system_tray_headers_moc ${system_tray_headers})

qt5_add_resources(system_tray_qrc_src ${system_tray_qrcs})

add_executable(ttop-system-tray ${system_tray_srcs} ${system_tray_headers} ${system_tray_headers_moc} ${serial_message_common_srcs}
    ${serial_message_common_headers} ${system_tray_qrc_src} ${system_tray_qrcs} ${readme_file})
target_include_directories(ttop-system-tray PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ttop-system-tray Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Network Qt5::WebSockets serial_communication_common)