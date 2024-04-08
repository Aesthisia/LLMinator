# !git clone https://github.com/ggerganov/llama.cpp

# !cd llama.cpp && LLAMA_CUBLAS=1 make && pip install -r requirements.txt

from huggingface_hub import snapshot_download

model_name = "bigscience/bloom-560m"

methods = ['q4_k_m']

base_model = "./original_model/"
quantized_path = "./quantized_model/"

snapshot_download(repo_id=model_name, local_dir=base_model , local_dir_use_symlinks=False)
original_model = quantized_path+'/bloom-560m.gguf'

# !mkdir ./quantized_model/

#python3 ./../../../llama.cpp/convert-hf-to-gguf.py ./original_model/ --outtype f16 --outfile ./quantized_model/bloom-560m.gguf