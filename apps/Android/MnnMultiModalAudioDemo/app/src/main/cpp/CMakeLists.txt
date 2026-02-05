cmake_minimum_required(VERSION 3.22.1)

project("multimodal_audio_jni")





set (MNN_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../")
set (MNN_INSTALL_ROOT "${MNN_SOURCE_ROOT}/project/android/build_64")

message(STATUS "MNN_SOURCE_ROOT: ${MNN_SOURCE_ROOT}")
message(STATUS "MNN_INSTALL_ROOT: ${MNN_INSTALL_ROOT}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

include_directories("${MNN_SOURCE_ROOT}/include/")
include_directories("${MNN_SOURCE_ROOT}/transformers/llm/engine/include")
include_directories("${MNN_SOURCE_ROOT}/transformers/llm/engine/src") # For omni.hpp and llmconfig.hpp
include_directories("${MNN_SOURCE_ROOT}/tools/audio/include")
include_directories("${MNN_SOURCE_ROOT}/3rd_party") # For rapidjson
include_directories("${MNN_SOURCE_ROOT}/source") # For core/FileLoader.hpp
include_directories("${MNN_SOURCE_ROOT}/3rd_party/half") # For half.hpp

add_definitions(-DLLM_SUPPORT_AUDIO)
add_definitions(-DLLM_SUPPORT_VISION)
add_definitions(-DMNN_IMGCODECS)
include_directories("${MNN_SOURCE_ROOT}/tools/cv/include")

add_library(${CMAKE_PROJECT_NAME} SHARED
        multimodal_audio_jni.cpp
        "${MNN_SOURCE_ROOT}/transformers/llm/engine/src/omni.cpp"
        "${MNN_SOURCE_ROOT}/transformers/llm/engine/src/tokenizer.cpp"
        "${MNN_SOURCE_ROOT}/transformers/llm/engine/src/sampler.cpp"
        "${MNN_SOURCE_ROOT}/transformers/llm/engine/src/diskembedding.cpp"
)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")


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
