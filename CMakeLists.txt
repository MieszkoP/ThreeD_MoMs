cmake_minimum_required(VERSION 3.18)

project(ThreeD_MoMs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(Python3_EXECUTABLE
#    "/home/mieszko/.config/jupyterlab-desktop/envs/env_1/bin/python"
#)
##TRZEBA NAPISAĆ, ŻE mwymagany jest minimum Python 3.9

#FINDING PYTHON3.9 MODULE 
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
message("Python executable is in " "${Python_EXECUTABLE}")

##Copying pybind path to pybind11_DIR
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_result
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message("Pybind is found in " ${pybind11_DIR})

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ThreeD_MoMs module.cpp L_mn_creator.cu creator.cu E_s_creator.cu custom_operators.cu memory_man.cu V_creator.cu)

set_source_files_properties(
    L_mn_creator_cuda.cu
    PROPERTIES LANGUAGE CUDA
)

set_source_files_properties(
    E_s_creator_cuda.cu
    PROPERTIES LANGUAGE CUDA
)

target_compile_options(ThreeD_MoMs PRIVATE -Wall -Wextra -Wconversion -Wsign-conversion -Wcast-qual -Wswitch-enum -Wshadow -Wfloat-equal -Wnull-dereference)
#target_compile_options(ThreeD_MoMs PRIVATE)
set_target_properties(ThreeD_MoMs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(ThreeD_MoMs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
)
