import os, torch
from threading import Thread
from typing import Optional

import gradio as gr
from llama_cpp import Llama
from src import quantize
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from core import list_download_models, remove_dir, default_repo_id

import sys

sys.path.append('./src/llama_cpp/')
sys.path.append('./src/')

cache_dir = os.path.join(os.getcwd(), "models")
saved_models_list = list_download_models(cache_dir)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#downloaded_file = hf_hub_download(repo_id=default_repo_id, cache_dir=cache_dir)


def initialize_model_and_tokenizer(model_path):
    ##model_path2 = quantize.quantize_model("stabilityai/stable-code-instruct-3b")
    llm = Llama(
        model_path=model_path,
        n_ctx=6000,
        n_batch=30,
  #      temperature=0.9,
   #     max_tokens=4095,
        n_parts=1,
        callback_manager=callback_manager, 
        verbose=True)

    return llm

def init_chain(model):
    class CustomLLM(LLM):

        """Streamer Object"""

        streamer: Optional[str] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            # Removed TextIteratorStreamer and unnecessary inputs
            self.streamer = ""
            #print(prompt)

            response = model(
                prompt, 
                max_tokens=1024, 
                stop=["Q:", "\n"], 
                echo=False, 
                temperature=0.5,
                top_p=0.95,
                top_k=100)

            print("response", response)
            return response
            #return response["choices"][0]["text"]

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()

    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm

model = initialize_model_and_tokenizer("./src/quantized_model/stable-code-3b-q4_k_m.gguf")

with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            title = gr.Button(
                value="LLMinator",
                scale=1,
                variant="primary",
                interactive=True)
            with gr.Group():
                repo_id = gr.Textbox(
                    value=default_repo_id,
                    label="Hugging Face Repo",
                    info="Default: openai-community/gpt2")
                load_model_btn = gr.Button(
                    value="Load Model",
                    variant="secondary",
                    interactive=True,)
            
            with gr.Group():
                execution_provider = gr.Radio(
                    ["cuda", "cpu"], 
                    value=device, 
                    label="Execution providers",
                    info="Select Device")

            with gr.Group():
                saved_models = gr.Dropdown(
                    choices=saved_models_list,
                    max_choices=5, 
                    filterable=True, 
                    label="Saved Models",
                    info="Models available in the disk"
                )
                offload_models = gr.ClearButton(
                    value="Remove Cached Models",
                    variant="Secondary",
                    interactive=True,
                )

        with gr.Column(scale=4):
            with gr.Group():
                chatbot = gr.Chatbot(scale=4)
                msg = gr.Textbox(label="Prompt")
                stop = gr.Button("Stop")

    llm_chain, llm = init_chain(model)


    def user(user_message, history):
        return "", history + [[user_message, None]]

    def removeModelCache():
        remove_dir(cache_dir)
        return gr.update(value=default_repo_id), gr.update(choices=[default_repo_id])
    
    def updateExecutionProvider(provider):
        if provider == "cuda":
            if torch.cuda.is_available():
                model.cuda()
                print("Model loaded in cuda", model)
            else:
                raise gr.Error("Torch not compiled with CUDA enabled. Please make sure cuda is installed.")

        else:
            model.cpu()

    def loadModel(repo_id):
        global llm_chain, llm
        if repo_id:
            model, tokenizer = initialize_model_and_tokenizer(repo_id)
            llm_chain, llm = init_chain(model, tokenizer)
            return gr.update(value=repo_id)
        else:
            raise gr.Error("Repo can not be empty!")

    def bot(history):
        print("Question: ", history[-1][0])
        output = llm_chain.run(question=history[-1][0])
        print("stream:", output)
        history[-1][1] = ""
        for character in output:
            print(character)
            history[-1][1] += character
            yield history

    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    stop.click(None, None, None, cancels=[submit_event], queue=False)
    load_model_btn.click(loadModel, repo_id, repo_id, queue=False, show_progress="full")
    execution_provider.change(fn=updateExecutionProvider, inputs=execution_provider, queue=False, show_progress="full")
    saved_models.change(loadModel, saved_models, repo_id, queue=False, show_progress="full")
    offload_models.click(removeModelCache, None, [repo_id, saved_models], queue=False, show_progress="full")

demo.queue()
demo.launch(server_name="0.0.0.0")