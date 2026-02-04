cmake_minimum_required(VERSION 3.22.1)

project("multimodal_audio_jni")

add_library(${CMAKE_PROJECT_NAME} SHARED
        multimodal_audio_jni.cpp
)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")

set (MNN_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../")
set (MNN_INSTALL_ROOT "${MNN_SOURCE_ROOT}/project/android/build_64")

message(STATUS "MNN_SOURCE_ROOT: ${MNN_SOURCE_ROOT}")
message(STATUS "MNN_INSTALL_ROOT: ${MNN_INSTALL_ROOT}")

include_directories("${MNN_SOURCE_ROOT}/include/")
include_directories("${MNN_SOURCE_ROOT}/transformers/llm/engine/include")


set(LIB_PATH "${MNN_INSTALL_ROOT}/lib")

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN.so")

add_library(MNN_CL SHARED IMPORTED)
set_target_properties(MNN_CL PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_CL.so")

add_library(MNN_LLM SHARED IMPORTED)
set_target_properties(MNN_LLM PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libllm.so")

add_library(MNN_Express SHARED IMPORTED)
set_target_properties(MNN_Express PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_Express.so")

add_library(MNN_OpenCV SHARED IMPORTED)
set_target_properties(MNN_OpenCV PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNNOpenCV.so")

add_library(MNN_Audio SHARED IMPORTED)
set_target_properties(MNN_Audio PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNNAudio.so")




target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log
        MNN
        MNN_CL
        MNN_LLM
        MNN_Express
        MNN_OpenCV
        MNN_Audio
        atomic
        m
)
