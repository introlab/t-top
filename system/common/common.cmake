## Build a static library with common serial communication files and tools
find_package(Qt5 REQUIRED COMPONENTS Core WebSockets)


file(GLOB serial_communication_srcs  ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/SerialCommunication/src/*.cpp)
file(GLOB serial_communication_headers  ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/SerialCommunication/src/*.h)
file(GLOB crc8_srcs ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/Crc8/src/*.cpp)
file(GLOB crc8_headers ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/Crc8/src/*.h)
file(GLOB class_macros_headers  ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/ClassMacro/src/*.h)
file(GLOB qt_common_files_srcs  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB qt_common_files_headers  ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

qt5_wrap_cpp(qt_common_files_headers_moc ${qt_common_files_headers})

add_library(serial_communication_common STATIC
    ${serial_communication_srcs}
    ${serial_communication_headers}
    ${crc8_srcs}
    ${crc8_headers}
    ${class_macros_headers}
    # Qt specific
    ${qt_common_files_srcs}
    ${qt_common_files_headers}
    ${qt_common_files_headers_moc}
)

target_link_libraries(serial_communication_common Qt5::Core Qt5::Network Qt5::WebSockets)

target_include_directories(serial_communication_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/SerialCommunication/src
    ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/Crc8/src
    ${CMAKE_CURRENT_LIST_DIR}/../../firmwares/common/lib/ClassMacro/src
    ${CMAKE_CURRENT_LIST_DIR}/../optional/include
    ${CMAKE_CURRENT_LIST_DIR}/src
)

target_compile_definitions(serial_communication_common PUBLIC
    -DSERIAL_COMMUNICATION_MANAGER_USE_STD_FUNCTION
)



