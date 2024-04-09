# Makefile to clone llama.cpp repository and install requirements

# Variables
REPO_URL := https://github.com/ggerganov/llama.cpp
REQUIREMENTS_FILE := requirements.txt
LLAMA_DIR := src/llama_cpp

# Determine pip command
PIP := $(shell command -v pip3 || command -v pip)

# Check if python and git are installed
PYTHON := $(shell command -v python3 || command -v python)
GIT := $(shell command -v git)

ifeq ($(PYTHON),)
$(error Python is not installed. Please install Python before running this Makefile.)
endif

ifeq ($(GIT),)
$(error Git is not installed. Please install Git before running this Makefile.)
endif

# Targets
.PHONY: all clone install clean

all: clone install

clone:
	mkdir -p $(LLAMA_DIR)
	git clone $(REPO_URL) $(LLAMA_DIR)

install:
	cd $(LLAMA_DIR) && \
		$(PIP) install -r $(REQUIREMENTS_FILE)
	touch $(LLAMA_DIR)/__init__.py
	cp $(LLAMA_DIR)/convert-hf-to-gguf.py $(LLAMA_DIR)/convert_hf_to_gguf.py

clean:
	rm -rf $(LLAMA_DIR)
