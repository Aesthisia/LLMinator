import os, shutil
from configparser import ConfigParser
import gradio as gr

default_repo_id = "stabilityai/stable-code-instruct-3b"
config_path = "configs/config.ini"
cache_gguf_dir = os.path.join(os.getcwd(), "src/quantized_model")
cache_original_dir = os.path.join(os.getcwd(), "src/original_model")

def format_gguf_model_name(file_name):
    parts = file_name.replace('.gguf', '').split("__")
    return "/".join(parts)

def list_converted_gguf_models(cache_gguf_dir):
    contents = os.listdir(cache_gguf_dir)
    model_files = [format_gguf_model_name(item) for item in contents]
    return model_files

def removeModelFromCache(model_name):
    if model_name == default_repo_id:
        raise gr.Error("Can not delete default model")
    else:
        gguf_model_name = model_name.replace("/", "__") + ".gguf"
        original_model_parts = model_name.split("/")
        original_model_name = f"model--{'--'.join(original_model_parts)}"
        try:
            os.remove(os.path.join(cache_gguf_dir, gguf_model_name))
            shutil.rmtree(os.path.join(cache_original_dir, original_model_name))
        except FileNotFoundError:
            raise gr.Error("Model not found in cache.")

def read_config():
    config = ConfigParser()
    config.read(config_path)
    if config.get('Settings', 'repo_id') == "" and config.get('Settings', 'execution_provider') == "":
        return None, config
    else:
        return config, config

def update_config(config, **kwargs):
    for key, value in kwargs.items():
        config.set('Settings', key, value)
    with open(config_path, 'w') as configfile:
        config.write(configfile)