cmake_minimum_required(VERSION 3.17)

project(neural-network)
set(CMAKE_CXX_STANDARD 17)

find_package(fmt CONFIG REQUIRED)

add_subdirectory(lib)

file(GLOB_RECURSE SOURCES src/*.*)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE renderer fmt::fmt)