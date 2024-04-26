## LLMinator: Run & Test LLMs locally

#### Gradio based tool with integrated chatbot to locally run & test LLMs directly from HuggingFace.

An easy-to-use tool made with Gradio, LangChain, and Torch.

![image](https://github.com/Aesthisia/LLMinator/assets/91900622/54cc0b3f-c5a8-4470-bcc5-a22e5fd24707)
![Screenshot from 2024-04-25 11-04-11](https://github.com/Aesthisia/LLMinator/assets/91900622/97881650-b0d5-4487-9a12-a259ec266458)


### ⚡ Features

- Context-aware Chatbot.
- Inbuilt code syntax highlighting.
- Load any LLM repo directly from HuggingFace.
- Supports both CPU & Cuda modes.
- Utilizes llama.cpp(through llama-cpp-python) 🦙.

### What is llama.cpp ?  

llama.cpp is a C++ library that allows you to run large language model on your own hardware.Imagine it as an effective language agent condensed into a C++ library – that’s llama.cpp in a nutshell.

  #### llama-cpp-python :
  
  - llama-cpp-python is a Python binding for llama.cpp.
  - It supports inference for many LLMs models, which can be  accessed on Hugging Face.

## 🚀 How to use

To use LLMinator, follow these simple steps:

- Clone the LLMinator repository from GitHub.
- Navigate to the directory containing the cloned repository.
- Install the required dependencies by running `pip install -r requirements.txt`.
- Build LLMinator with llama.cpp :

    - On Linux or MacOS:
      - Using `make`:

        ```bash
        make
        ```

      - On Windows:

    - Using `CMake`:
      ```bash
      mkdir build
      cd build
      cmake ..
      ```
- Run the LLMinator tool using the command `python webui.py`.
- Access the web interface by opening the provided URL in your browser.
- Start interacting with the chatbot and experimenting with LLMs!

### Command line arguments

| Argument Command | Default   | Description                                                                 |
| ---------------- | --------- | --------------------------------------------------------------------------- |
| --host           | 127.0.0.1 | Host or IP address on which the server will listen for incoming connections |
| --port           | 7860      | Launch gradio with given server port                                        |
| --share          | False     | This generates a public shareable link that you can send to anybody         |


## 🤝 Contributions

We welcome contributions from the community to enhance LLMinator further. If you'd like to contribute, please follow these guidelines:

- Fork the LLMinator repository on GitHub.
- Create a new branch for your feature or bug fix.
- Test your changes thoroughly.
- Submit a pull request, providing a clear description of the changes you've made.

Reach out to us: info@aesthisia.com