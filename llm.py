import torch
from threading import Thread
from typing import Optional

import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

def initialize_model_and_tokenizer(model_name="stabilityai/stable-code-instruct-3b"):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def init_chain(model, tokenizer):
    class CustomLLM(LLM):

        """Streamer Object"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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

model, tokenizer = initialize_model_and_tokenizer()

with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Textbox(
                label="Hugging Face Repo",
                info="Select Model",
                value="stabilityai/stable-code-instruct-3b",
            )
            gr.Radio(["cuda", "cpu"], label="Execution providers", info="Providers")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(scale=4)
            with gr.Group():
                msg = gr.Textbox()
                stop = gr.Button("Stop")
    llm_chain, llm = init_chain(model, tokenizer)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        print("Question: ", history[-1][0])
        llm_chain.run(question=history[-1][0])
        history[-1][1] = ""
        for character in llm.streamer:
            print(character)
            history[-1][1] += character
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    stop.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_name="0.0.0.0")