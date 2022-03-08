cmake_minimum_required(VERSION 3.21)
project(16833_HW2_EKF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)

#NumCpp contents
FetchContent_Declare(
        NumCpp
        GIT_REPOSITORY "https://github.com/dpilger26/NumCpp.git"
        GIT_TAG "Version_2.7.0"
)
FetchContent_Declare(sciplot_content
        GIT_REPOSITORY https://github.com/sciplot/sciplot.git
        GIT_TAG master)
FetchContent_MakeAvailable(NumCpp)

file(COPY data/data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#Sci-plot contents
FetchContent_GetProperties(sciplot_content)
if(NOT sciplot_content_POPULATED)
    FetchContent_Populate(sciplot_content)
endif()
include_directories(${sciplot_content_SOURCE_DIR} data)

add_executable(16833_HW2_EKF src/main.cpp)
target_link_libraries(${PROJECT_NAME} NumCpp)
