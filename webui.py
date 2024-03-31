import os, torch
from threading import Thread
from typing import Optional

import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoConfig

from core import list_download_models, format_model_name, remove_dir

cache_dir = os.path.join(os.getcwd(), "models")
saved_models = list_download_models(cache_dir)

def initialize_model_and_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True)
    
    model.eval()
    model.cpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def init_chain(model, tokenizer):
    class CustomLLM(LLM):

        """Streamer Object"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device) #uncomment for cuda
            kwargs = dict(input_ids=inputs["input_ids"], streamer=self.streamer, max_new_tokens=1024, temperature=0.5, top_p=0.95, top_k=100, do_sample=True, use_cache=True)
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            return ""

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()

    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm

model, tokenizer = initialize_model_and_tokenizer("stabilityai/stable-code-instruct-3b")

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
                    value="stabilityai/stable-code-instruct-3b",
                    label="Hugging Face Repo",
                    info="Default: stabilityai/stable-code-instruct-3b")
                load_model_btn = gr.Button(
                    value="Load Model",
                    variant="secondary",
                    interactive=True,)
            
            with gr.Group():
                execution_provider = gr.Radio(
                    ["cuda", "cpu"], 
                    value="cpu", 
                    label="Execution providers",
                    info="Select Device")

            with gr.Group():
                saved_models = gr.Dropdown(
                    choices=saved_models,
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
    llm_chain, llm = init_chain(model, tokenizer)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def removeModelCache():
        remove_dir(cache_dir)
    
    def updateExecutionProvider(provider):
        if provider == "cuda":
            model.cuda()
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
        llm_chain.run(question=history[-1][0])
        history[-1][1] = ""
        for character in llm.streamer:
            #print(character)
            history[-1][1] += character
            yield history

    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    stop.click(None, None, None, cancels=[submit_event], queue=False)
    load_model_btn.click(loadModel, repo_id, repo_id, queue=False, show_progress="full")
    execution_provider.change(fn=updateExecutionProvider, inputs=execution_provider, queue=False, show_progress="full")
    saved_models.change(loadModel, saved_models, repo_id, queue=False, show_progress="full")
    offload_models.click(removeModelCache, None, saved_models, queue=False, show_progress="full")

demo.queue()
demo.launch(server_name="0.0.0.0")