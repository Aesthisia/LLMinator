import torch
from fastapi import FastAPI, WebSocket
from src import quantize
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from core import default_repo_id

app = FastAPI()

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_gpu_layers = None
if device == "cuda":
    n_gpu_layers = -1
else:
    n_gpu_layers = 0
n_ctx = 6000
n_batch = 30
n_parts = 1
temperature = 0.9
max_tokens = 500

def snapshot_download_and_convert_to_gguf(repo_id):
    gguf_model_path = quantize.quantize_model(repo_id)
    return gguf_model_path

def init_llm_chain(model_path):    
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        temperature=temperature,
        max_tokens=max_tokens,
        n_parts=n_parts,
        callback_manager=callback_manager, 
        verbose=True
    )    
    
    template = """Question: {question}
        Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm
    return llm_chain, llm

model_path = snapshot_download_and_convert_to_gguf(default_repo_id)

llm_chain, llm = init_llm_chain(model_path)

@app.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        prompt = await websocket.receive_text()
        
        async def bot(prompt):
            print("Question: ", prompt)
            output = llm_chain.stream(prompt)
            print("stream:", output)
            for character in output:
                print(character)
                await websocket.send_text(character)
                
        await bot(prompt)