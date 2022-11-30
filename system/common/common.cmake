## Build a static library with common serial communication files and tools
file(GLOB serial_communication_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/SerialCommunication/src/*.cpp)
file(GLOB serial_communication_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/SerialCommunication/src/*.h)
file(GLOB crc8_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/Crc8/src/*.cpp)
file(GLOB crc8_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/Crc8/src/*.h)
file(GLOB class_macros_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../firmwares/common/lib/ClassMacro/src/*.h)

add_library(serial_communication_common STATIC
    ${serial_communication_srcs}
    ${serial_communication_headers}
    ${crc8_srcs}
    ${crc8_headers}
    ${class_macros_headers}
)

target_include_directories(serial_communication_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../firmwares/common/lib/SerialCommunication/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../firmwares/common/lib/Crc8/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../firmwares/common/lib/ClassMacro/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ros/utils/recorders/3rd_party/optional/include
)

target_compile_definitions(serial_communication_common PUBLIC
    -DSERIAL_COMMUNICATION_MANAGER_USE_STD_FUNCTION
)
