cmake_minimum_required(VERSION 3.10)

# set the project name in a variable
set(project_name yolov10_cpp)
project(${project_name})
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)

add_library(${project_name}-lib
        src/ia/inference.cpp
        src/ia/inference.h
)

target_include_directories(${project_name}-lib PUBLIC src)

target_link_libraries(${project_name}-lib
        PUBLIC ${OpenCV_LIBS}
)

# add the executable
add_executable(${project_name} 
        src/main.cpp
)
target_link_libraries(${project_name} ${project_name}-lib)

add_dependencies(${project_name} ${project_name}-lib)


