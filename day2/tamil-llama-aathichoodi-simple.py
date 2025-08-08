# pip install --upgrade --no-cache-dir llama-cpp-python huggingface-hub gradio datasets sentence-transformers

import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import gradio as gr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# The model name and a specific quantized file from the Hugging Face repo.
model_name = "abhinand/tamil-llama-7b-instruct-v0.2-GGUF"
model_file_name = "tamil-llama-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Model Loading ---
# Download the model file if it's not already in the cache.
print(f"Downloading model '{model_file_name}' from '{model_name}'...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file_name)

# Initialize the Llama model.
print("Loading the Llama model...")
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    print("Llama model loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the Llama model: {e}")
    exit()

# --- RAG Setup: Dataset and Embeddings ---
# Load the Aathichoodi dataset from Hugging Face.
print("Loading Aathichoodi dataset...")
try:
    dataset = load_dataset("Selvakumarduraipandian/Aathichoodi")
    print("Aathichoodi dataset loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Extract the aathichoodi content as a list of strings, including the English translation.
# This enriches the text for better retrieval.
aathichoodi_texts = [
    f"ஆத்திசூடி: {item['தமிழ் வாக்கியம்']} - பொருள்: {item['Tamil Meaning']} - English: {item['English Translation']}"
    for item in dataset['train']
]


# Initialize a sentence-transformer model to create embeddings.
# This model is specifically trained for multilingual tasks, including Tamil.
print("Loading sentence-transformer model for embeddings...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    print("Embedding model loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the embedding model: {e}")
    exit()

# Pre-compute embeddings for all aathichoodi texts.
# This is a one-time process that makes retrieval very fast later on.
print("Creating embeddings for Aathichoodi texts...")
aathichoodi_embeddings = embedding_model.encode(aathichoodi_texts, convert_to_tensor=True).cpu().numpy()
print("Embeddings created successfully!")

# --- Chatbot (Gradio) ---
system_prompt = "You are a helpful and friendly chatbot that speaks both Tamil and English. You are an expert on Aathichoodi and provide detailed answers in Tamil based on the provided context."

def chat_reply(message, history):
    # Step 1: Encode the user's message to get its embedding and convert it to a NumPy array.
    query_embedding = embedding_model.encode(message, convert_to_tensor=True).cpu().numpy()

    # Step 2: Calculate cosine similarity to find the most relevant Aathichoodi entries.
    # This compares the user's query against all the pre-computed aathichoodi embeddings.
    similarities = cosine_similarity(query_embedding.reshape(1, -1), aathichoodi_embeddings)

    # Step 3: Get the indices of the top 3 most similar entries.
    top_indices = np.argsort(similarities[0])[-3:][::-1]

    # Step 4: Retrieve the corresponding Aathichoodi texts to use as context.
    retrieved_context = [aathichoodi_texts[i] for i in top_indices]
    context_str = "\n".join(retrieved_context)

    # Step 5: Augment the prompt with the retrieved context.
    # We create a new prompt that instructs the LLM to use the context.
    rag_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"இங்கே ஆத்திசூடி தொடர்பான சில தகவல்கள் உள்ளன. உங்கள் பதிலுக்கு இந்த தகவலைப் பயன்படுத்தவும்.\n" # "Here is some information related to Aathichoodi. Use this information for your response."
        f"Context: {context_str}\n"
    )

    # Append the history and current message to the augmented prompt.
    for user_msg, assistant_msg in (history or []):
        rag_prompt += (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
        )
    rag_prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"

    # Step 6: Stream the tokens from the LLM based on the new, augmented prompt.
    stream = llm.create_completion(
        prompt=rag_prompt,
        max_tokens=256,
        stop=["<|im_end|>"],
        stream=True
    )

    partial = ""
    for token in stream:
        text = token["choices"][0]["text"]
        partial += text
        yield partial

# Minimal Gradio UI
demo = gr.ChatInterface(
    fn=chat_reply,
    title="Tamil-Llama Aathichoodi RAG Chatbot",
    description="Tamil AI Workshop - Day 2: RAG Chatbot using Tamil Llama and Aathichoodi Dataset",
)

if __name__ == "__main__":
    demo.launch()
