# Download and build kaldi-native-fbank — exact-match Whisper feature extraction
# Used by MNN_BUILD_AUDIO for pixel-level alignment with sherpa-onnx / training pipeline.
function(download_kaldi_native_fbank)
  include(FetchContent)

  set(kaldi_native_fbank_URL  "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.22.3.tar.gz")
  set(kaldi_native_fbank_URL2 "https://hf-mirror.com/csukuangfj/sherpa-ncnn-cmake-deps/resolve/main/kaldi-native-fbank-1.22.3.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=9176cc66fc7ce1edf85cf355b06e320c57db6297df74277f575183468893cf61")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  # Allow pre-downloaded tarball for offline builds
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.22.3.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-native-fbank-1.22.3.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-native-fbank-1.22.3.tar.gz
    /tmp/kaldi-native-fbank-1.22.3.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_fbank_URL}" kaldi_native_fbank_URL)
      message(STATUS "Found local kaldi-native-fbank: ${kaldi_native_fbank_URL}")
      set(kaldi_native_fbank_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_native_fbank
    URL
      ${kaldi_native_fbank_URL}
      ${kaldi_native_fbank_URL2}
    URL_HASH ${kaldi_native_fbank_HASH}
  )

  FetchContent_GetProperties(kaldi_native_fbank)
  if(NOT kaldi_native_fbank_POPULATED)
    message(STATUS "Downloading kaldi-native-fbank from ${kaldi_native_fbank_URL}")
    FetchContent_Populate(kaldi_native_fbank)
  endif()
  message(STATUS "kaldi-native-fbank source dir: ${kaldi_native_fbank_SOURCE_DIR}")

  # Export variables to parent scope so tools/audio/CMakeLists.txt can use them
  set(kaldi_native_fbank_SOURCE_DIR ${kaldi_native_fbank_SOURCE_DIR} PARENT_SCOPE)
  set(kaldi_native_fbank_BINARY_DIR ${kaldi_native_fbank_BINARY_DIR} PARENT_SCOPE)

  # Force static build for kaldi-native-fbank within MNN
  set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)

  set(_cmake_warn_deprecated_bak "${CMAKE_WARN_DEPRECATED}")
  set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
  add_subdirectory(${kaldi_native_fbank_SOURCE_DIR} ${kaldi_native_fbank_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(CMAKE_WARN_DEPRECATED "${_cmake_warn_deprecated_bak}" CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ${_build_shared_libs_bak})

  # Suppress clang warnings on kissfft
  if(TARGET kissfft AND (CMAKE_C_COMPILER_ID MATCHES "Clang"))
    target_compile_options(kissfft PRIVATE -Wno-cast-align)
  endif()

  # Ensure kaldi-native-fbank-core is built as static with hidden visibility
  set_target_properties(kaldi-native-fbank-core
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
  )

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}/
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS kaldi-native-fbank-core kissfft DESTINATION lib)
  endif()
endfunction()

download_kaldi_native_fbank()
