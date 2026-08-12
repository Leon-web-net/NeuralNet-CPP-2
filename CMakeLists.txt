cmake_minimum_required(VERSION 3.20) 
project(NeuralNet LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenMP)
add_executable(neuralnet
    src/main.cpp
)
target_include_directories(neuralnet PRIVATE include)
target_compile_definitions(neuralnet PRIVATE
    DATA_DIR="${CMAKE_SOURCE_DIR}/data")
if(OpenMP_CXX_FOUND)
    target_link_libraries(neuralnet PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    target_compile_options(neuralnet PRIVATE /W4)
endif() 