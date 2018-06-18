cmake_minimum_required(VERSION 3.10)
include_directories(${CMAKE_CURRENT_SRC_DIR}/paddle)
add_subdirectory(paddle)
add_subdirectory(src)
