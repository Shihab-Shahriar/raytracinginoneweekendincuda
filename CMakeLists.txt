# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.23)

# Project's name
project(RND-APPL CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 ")

set(CMAKE_CUDA_STANDARD 17)


include(FetchContent)

FetchContent_Declare(
  crng
  SOURCE_DIR /home/shihab/src/rnd
  # GIT_REPOSITORY https://github.com/Shihab-Shahriar/rnd.git
  # GIT_TAG        main
)

FetchContent_MakeAvailable(crng)

# Later in the file, when defining their targets:
add_executable(rt main.cu)
target_link_libraries(rt PRIVATE crng)

