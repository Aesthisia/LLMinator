# Makefile to clone llama.cpp repository and install requirements

# Variables
REPO_URL := https://github.com/ggerganov/llama.cpp
REQUIREMENTS_FILE := requirements.txt
LLAMA_DIR := src/llama_cpp

# Determine pip command
PIP := $(shell command -v pip3 2>/dev/null || command -v pip)

# Check if python and git are installed
PYTHON := $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
GIT := $(shell command -v git)

ifeq ($(PYTHON),)
$(error Python is not installed. Please install Python before running this Makefile.)
endif

ifeq ($(GIT),)
$(error Git is not installed. Please install Git before running this Makefile.)
endif

# Targets
.PHONY: all clone install clean quantized_model_dir append_to_configs

all: clone install quantized_model_dir append_to_configs

clone:
	mkdir -p $(LLAMA_DIR)
	git clone $(REPO_URL) $(LLAMA_DIR)

install:
	cd $(LLAMA_DIR) && \
		$(PIP) install -r $(REQUIREMENTS_FILE)

quantized_model_dir:
	mkdir -p src/quantized_model

append_to_configs:
	echo "py_cmd = $(PYTHON)" >> configs/config.ini

clean:
	rm -rf $(LLAMA_DIR)
