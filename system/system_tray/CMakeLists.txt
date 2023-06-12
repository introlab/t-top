cmake_minimum_required(VERSION 3.8.2)

project(TTOP_SystemTray)


## Set compile options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX /opt/t-top/)
if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "")
endif ()

## Required Qt5 Modules
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Network WebSockets)

## Build a static library with common serial communication files and tools
include(../common/common.cmake)


set (system_tray_headers
    src/SystemTrayIcon.h
    src/SystemTrayApp.h
)

set (system_tray_srcs
    src/SystemTrayIcon.cpp
    src/SystemTrayApp.cpp
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

add_executable(ttop-system-tray ${system_tray_srcs} ${system_tray_headers} ${system_tray_headers_moc} ${system_tray_qrc_src} ${system_tray_qrcs} ${readme_file})
target_include_directories(ttop-system-tray PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ttop-system-tray Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Network Qt5::WebSockets serial_communication_common)

install(TARGETS ttop-system-tray DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/ttop_system_tray.desktop DESTINATION /etc/xdg/autostart)
install(DIRECTORY DESTINATION working_directory/system_tray)
