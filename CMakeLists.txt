cmake_minimum_required(VERSION 3.20) 
project(NeuralNet LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_library(nn_core
    src/matrix.cpp
)

target_include_directories(nn_core PUBLIC include)
target_compile_definitions(nn_core PUBLIC
    DATA_DIR="${CMAKE_SOURCE_DIR}/data")

 add_executable(NeuralNet src/main.cpp)
 target_link_libraries(NeuralNet PRIVATE nn_core)

 add_executable(test_matrix src/test_matrix.cpp)
 target_link_libraries(test_matrix PRIVATE nn_core)


if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(nn_core PRIVATE /W4)
    target_compile_options(NeuralNet PRIVATE /W4)
    target_compile_options(test_matrix PRIVATE /W4)
endif() 