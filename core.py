import os, shutil

def format_model_name(directory_name):
    parts = directory_name.split("--")
    return "/".join(parts[1:])

def list_download_models(cache_dir):
    contents = os.listdir(cache_dir)
    directories = [format_model_name(item) for item in contents if os.path.isdir(os.path.join(cache_dir, item)) and item.startswith("models")]
    return directories

def remove_dir(path):
    try:
        for folder in os.listdir(path):
            if folder != "models--openai-community--gpt2":
                full_path = os.path.join(path, folder)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
    except OSError as e:
        print(f"Error: {e.strerror}")