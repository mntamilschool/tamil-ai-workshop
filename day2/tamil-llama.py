# pip install --upgrade --no-cache-dir llama-cpp-python huggingface-hub gradio

import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import gradio as gr

# --- Configuration ---
# The model name and a specific quantized file from the Hugging Face repo.
# The Q4_K_M quantization offers a good balance of performance and size.
model_name = "abhinand/tamil-llama-7b-instruct-v0.2-GGUF"
model_file_name = "tamil-llama-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Model Loading ---
# Download the model file if it's not already in the cache.
# This will handle fetching the GGUF file from Hugging Face Hub.
print(f"Downloading model '{model_file_name}' from '{model_name}'...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file_name)

# Initialize the Llama model.
# The n_ctx parameter sets the maximum context length (how much of the conversation
# the model can "remember"). 2048 is a reasonable starting point.
# You can increase this for longer conversations if you have enough RAM.
print("Loading the model...")
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window size
        n_gpu_layers=-1, # -1 to offload all layers to GPU if available (Metal for Apple Silicon, CUDA for NVIDIA)
        verbose=False # Set to True for detailed logging
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# --- Chatbot (Gradio) ---
# The `tamil-llama-7b-instruct-v0.2` model uses the ChatML prompt format.
# It's a structured way of formatting the conversation for the model.
# The system message sets the overall persona or task for the model.
system_prompt = "You are a helpful and friendly chatbot that speaks both Tamil and English. Provide detailed answers, only in Tamil."

def chat_reply(message, history):
    # Build ChatML prompt with system + history + current user message
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for user_msg, assistant_msg in (history or []):
        prompt += (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
        )
    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

    # Stream tokens and yield partial text for Gradio to render incrementally
    stream = llm.create_completion(
        prompt=prompt,
        max_tokens=256,
        stop=["<|im_end|>"],
        stream=True
    )

    partial = ""
    for token in stream:
        text = token["choices"][0]["text"]
        partial += text
        yield partial  # Gradio ChatInterface supports generators for streaming

# Minimal Gradio UI
demo = gr.ChatInterface(
    fn=chat_reply,
    title="Tamil-Llama Chatbot",
    description="Tamil AI Workshop - Day 2: Chatbot using Tamil Llama",
)

if __name__ == "__main__":
    demo.launch()

