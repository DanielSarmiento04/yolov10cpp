cmake_minimum_required(VERSION 3.10)

# set the project name in a variable
set(project_name yolov10_cpp)
project(${project_name})
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)

# Find ONNX Runtime package
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime/core/session/onnxruntime_c_api.h
    HINTS /usr/local/include/onnxruntime
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS /usr/local/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime include directory not found")
endif()
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()


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
        # src/main.cpp
        run.cpp
)
target_link_libraries(${project_name} ${project_name}-lib)

add_dependencies(${project_name} ${project_name}-lib)


