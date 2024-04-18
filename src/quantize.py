import subprocess, os
from huggingface_hub import snapshot_download

def quantize_model(repo_id):
    base_model = "./src/original_model/"
    quantized_path = "./src/quantized_model/"
    outfile = quantized_path + repo_id.replace("/", "__") + ".gguf"
    
    if os.path.isfile(outfile):
        return outfile;
    
    snapshot_download(repo_id=repo_id, local_dir=base_model , local_dir_use_symlinks=True)

    command = [
        'python', 
        './src/llama_cpp/convert-hf-to-gguf.py',
        base_model,
        '--outtype', 'f16',
        '--outfile', outfile
    ]

    # Run the command
    subprocess.run(command, check=True)

    return outfile
