cmake_minimum_required(VERSION 3.22.1)
project("jarvisai")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Silence policy warnings from sub-projects
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

# ──────────────────────────────────────────────────────────────────────────────
# llama.cpp  (use GIT — CMake 3.22 has a bug with zip URL paths)
# ──────────────────────────────────────────────────────────────────────────────
set(LLAMA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(LLAMA_NATIVE           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b4618
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   FALSE
)
FetchContent_MakeAvailable(llama_cpp)

# ──────────────────────────────────────────────────────────────────────────────
# whisper.cpp
# ──────────────────────────────────────────────────────────────────────────────
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2           OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    whisper_cpp
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        v1.7.4
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   FALSE
)
FetchContent_MakeAvailable(whisper_cpp)

# ──────────────────────────────────────────────────────────────────────────────
# JarvisAI JNI shared library
# ──────────────────────────────────────────────────────────────────────────────
add_library(jarvisai SHARED
    llm_inference.cpp
    whisper_inference.cpp
)

target_include_directories(jarvisai PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${whisper_cpp_SOURCE_DIR}
)

target_link_libraries(jarvisai
    llama
    whisper
    android
    log
    aaudio
)
