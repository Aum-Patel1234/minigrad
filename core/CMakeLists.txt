set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.14)

project(minigrad
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE SRC_FILES *.cpp)

message(STATUS "Source files: ${SRC_FILES}")

pybind11_add_module(minigrad
    ${SRC_FILES}
)

target_link_libraries(minigrad PUBLIC pybind11::module)

target_compile_definitions(minigrad PRIVATE VERSION_INFO="${PROJECT_VERSION}")
