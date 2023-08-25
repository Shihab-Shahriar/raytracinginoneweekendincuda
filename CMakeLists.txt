# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.23)

# Project's name
project(RT CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")


include(FetchContent)

FetchContent_Declare(
  r123
  #SOURCE_DIR /home/shihab/src/rnd
  GIT_REPOSITORY https://github.com/DEShawResearch/random123.git
  GIT_TAG        main
)

FetchContent_MakeAvailable(r123)

# Later in the file, when defining their targets:
add_executable(rt123 main.cu)
#target_link_libraries(rt123 PRIVATE r123)
target_include_directories(rt123 PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/_deps/r123-src/include)

