# Minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Project name
project(llama_cpp)

# Git repository location
set(REPO_URL "https://github.com/ggerganov/llama.cpp")

# Requirements file
set(REQUIREMENTS_FILE "requirements.txt")

# Llama directory
set(LLAMA_DIR "${PROJECT_SOURCE_DIR}/src/llama_cpp")

# Check for Python and Git using CMake's FIND_PACKAGE
find_package(PythonLibs REQUIRED)
find_package(Git REQUIRED)

# Download and clone the llama.cpp repository
execute_process(
  COMMAND git clone ${REPO_URL} ${LLAMA_DIR}
  RESULT_VARIABLE git_result
)

# Error handling for Git clone
if(NOT ${git_result} EQUAL 0)
  message(FATAL_ERROR "Failed to clone llama.cpp repository")
endif()

# Install Python requirements
execute_process(
  COMMAND pip install -r "${LLAMA_DIR}/${REQUIREMENTS_FILE}"
)

file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/src/quantized_model")