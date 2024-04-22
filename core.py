import os, shutil
from configparser import ConfigParser

default_repo_id = "stabilityai/stable-code-instruct-3b"
config_path = "configs/config.ini"
default_repo_id_parts = default_repo_id.split("/")
default_model_folder = f"models--{'--'.join(default_repo_id_parts)}"

def format_model_name(directory_name):
    parts = directory_name.split("--")
    return "/".join(parts[1:])

def format_gguf_model_name(file_name):
    parts = file_name.replace('.gguf', '').split("__")
    return "/".join(parts)

def list_download_models(cache_dir):
    contents = os.listdir(cache_dir)
    directories = [format_model_name(item) for item in contents if os.path.isdir(os.path.join(cache_dir, item)) and item.startswith("models")]
    return directories

def list_converted_gguf_models(cache_gguf_dir):
    contents = os.listdir(cache_gguf_dir)
    model_files = [format_gguf_model_name(item) for item in contents]
    return model_files

def remove_dir(path):
    try:
        for model in os.listdir(path):
            if model != default_model_folder:
                model_path = os.path.join(path, model)
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
        print("successfully removed cached models!")
    except OSError as e:
        print(f"Error: {e.strerror}")

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