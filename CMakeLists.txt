cmake_minimum_required(VERSION 3.23)

project(kokkos_rt CXX CUDA)

set (CMAKE_CXX_STANDARD 17)


set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in Kokkos")
set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "lambda")
set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "volta 70")
#set(Kokkos_ARCH_TURING75 ON CACHE BOOL "turing 75")

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        master
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)

add_executable(rt main.cu)
target_link_libraries(rt Kokkos::kokkos)
set_property(TARGET rt PROPERTY CUDA_ARCHITECTURES 70)
