cmake_minimum_required(VERSION 3.21)
project(16833_HW2_EKF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# find python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

include(FetchContent)

#Fmt lib
FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
        )
FetchContent_MakeAvailable(fmt)

#NumCpp contents
FetchContent_Declare(
        NumCpp
        GIT_REPOSITORY "https://github.com/dpilger26/NumCpp.git"
        GIT_TAG "Version_2.7.0"
)
FetchContent_MakeAvailable(NumCpp)

FetchContent_Declare(
        matplotlib
        GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
        GIT_TAG        ef0383f1315d32e0156335e10b82e90b334f6d9f
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()
include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

file(COPY data/data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(16833_HW2_EKF src/main.cpp)
target_link_libraries(${PROJECT_NAME} NumCpp fmt::fmt
        ${PYTHON_LIBRARIES}
        Python3::NumPy
        )
