# pip install --upgrade --no-cache-dir llama-cpp-python huggingface-hub gradio datasets sentence-transformers faiss-cpu

import os
import numpy as np
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
import pickle
import json

# --- Configuration ---
model_name = "abhinand/tamil-llama-7b-instruct-v0.2-GGUF"
model_file_name = "tamil-llama-7b-instruct-v0.2.Q4_K_M.gguf"
dataset_name = "Selvakumarduraipandian/Aathichoodi"

# --- Model Loading ---
print(f"Downloading model '{model_file_name}' from '{model_name}'...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file_name)

print("Loading the LLM model...")
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # Increased context for RAG
        n_gpu_layers=-1,
        verbose=False
    )
    print("LLM model loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# --- RAG Setup ---
class AathichoodiRAG:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.setup_rag()
    
    def setup_rag(self):
        """Setup the RAG system with Aathichoodi dataset"""
        print("Setting up RAG system...")
        
        # Load embedding model (multilingual for Tamil support)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load Aathichoodi dataset
        print("Loading Aathichoodi dataset...")
        try:
            dataset = load_dataset(dataset_name, split='train')
            
            # Process dataset into searchable documents
            self.documents = []
            texts_to_embed = []
            
            for item in dataset:
                # Create rich document with Tamil and English content
                doc = {
                    'verse_tamil': item['தமிழ் வாக்கியம்'],
                    'letter': item['தமிழ் எழுத்து'],
                    'meaning_tamil': item['Tamil Meaning'],
                    'meaning_english': item['English Translation'],
                    'transliteration': item['Transliteration'],
                    'number': item['எண்']
                }
                self.documents.append(doc)
                
                # Combine texts for embedding (Tamil verse + meanings)
                combined_text = f"{doc['verse_tamil']} {doc['meaning_tamil']} {doc['meaning_english']}"
                texts_to_embed.append(combined_text)
            
            # Create embeddings
            print("Creating embeddings...")
            embeddings = self.embedder.encode(texts_to_embed)
            
            # Create FAISS index
            print("Building FAISS index...")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            print(f"RAG setup complete! Indexed {len(self.documents)} Aathichoodi verses.")
            
        except Exception as e:
            print(f"Error setting up RAG: {e}")
            self.documents = []
    
    def search_relevant_verses(self, query, k=3):
        """Search for relevant Aathichoodi verses based on query"""
        if not self.index or not self.documents:
            return []
        
        try:
            # Embed the query
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return relevant documents
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['relevance_score'] = float(score)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []

# Initialize RAG system
rag_system = AathichoodiRAG()

# --- Enhanced System Prompt ---
def create_enhanced_system_prompt(relevant_verses):
    """Create system prompt with relevant Aathichoodi context"""
    base_prompt = """நீங்கள் ஒரு அன்பான மற்றும் அறிவுள்ள தமிழ் உதவியாளர். நீங்கள் அவ்வையாரின் ஆத்திச்சூடி பற்றிய அறிவைக் கொண்டு, தமிழ் மக்களுக்கு அறநெறி வழிகாட்டல் மற்றும் ஆலோசனைகள் வழங்குகிறீர்கள்.

You are a helpful Tamil assistant with knowledge of Avvaiyar's Aathichoodi. You provide moral guidance and advice based on these ancient Tamil wisdom verses. Always respond in Tamil."""

    if relevant_verses:
        context_prompt = "\n\nதொடர்புடைய ஆத்திச்சூடி வரிகள்:\n"
        for i, verse in enumerate(relevant_verses, 1):
            context_prompt += f"\n{i}. {verse['letter']} - {verse['verse_tamil']}\n"
            context_prompt += f"   பொருள்: {verse['meaning_tamil']}\n"
            context_prompt += f"   English: {verse['meaning_english']}\n"
        
        return base_prompt + context_prompt + "\n\nமேற்கண்ட ஆத்திச்சூடி வரிகளின் அடிப்படையில் பதிலளிக்கவும்."
    
    return base_prompt

# --- Enhanced Chat Function ---
def enhanced_chat_reply(message, history):
    """Enhanced chat function with RAG"""
    
    # Search for relevant Aathichoodi verses
    relevant_verses = rag_system.search_relevant_verses(message, k=2)
    
    # Create enhanced system prompt
    system_prompt = create_enhanced_system_prompt(relevant_verses)
    
    # Build ChatML prompt
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    # Add conversation history
    for user_msg, assistant_msg in (history or []):
        prompt += (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
        )
    
    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate response
    stream = llm.create_completion(
        prompt=prompt,
        max_tokens=512,  # Increased for richer responses
        stop=["<|im_end|>"],
        stream=True,
        temperature=0.7  # Slight randomness for more natural responses
    )
    
    partial = ""
    for token in stream:
        text = token["choices"][0]["text"]
        partial += text
        yield partial

# --- Aathichoodi Search Function ---
def search_aathichoodi(query):
    """Standalone function to search Aathichoodi verses"""
    if not query.strip():
        return "தேடல் வார்த்தையை உள்ளிடவும் / Please enter a search term"
    
    results = rag_system.search_relevant_verses(query, k=5)
    
    if not results:
        return "தொடர்புடைய வரிகள் கிடைக்கவில்லை / No relevant verses found"
    
    output = f"'{query}' குறித்த ஆத்திச்சூடி வரிகள்:\n\n"
    
    for i, verse in enumerate(results, 1):
        output += f"{i}. {verse['letter']} - {verse['verse_tamil']}\n"
        output += f"   பொருள்: {verse['meaning_tamil']}\n"
        output += f"   English: {verse['meaning_english']}\n"
        output += f"   ஒலிப்பு: {verse['transliteration']}\n\n"
    
    return output

# --- Gradio Interface ---
with gr.Blocks(title="Tamil Llama with Aathichoodi RAG") as demo:
    gr.Markdown("""
    # Tamil Llama Chatbot with Aathichoodi RAG
    
    This chatbot combines the Tamil Llama model with Retrieval-Augmented Generation (RAG) 
    using Avvaiyar's Aathichoodi dataset to provide moral guidance and wisdom.
    """)
    
    with gr.Tab("💬 Chat / உரையாடல்"):
        chatbot = gr.ChatInterface(
            fn=enhanced_chat_reply,
            title="Aathichoodi Wisdom Chat",
            description="அறநெறி வழிகாட்டலுக்கு கேள்விகள் கேட்கவும்",
            examples=[
                "நல்ல நண்பர்களை எப்படி தேர்வு செய்வது?",
                "கோபத்தை எப்படி கட்டுப்படுத்துவது?",
                "How to choose good friends?",
                "What is the importance of learning?",
                "கல்வியின் முக்கியத்துவம் என்ன?",
            ]
        )
    
    with gr.Tab("🔍 Search Aathichoodi / ஆத்திச்சூடி தேடல்"):
        gr.Markdown("### Search specific Aathichoodi verses")
        
        with gr.Row():
            search_input = gr.Textbox(
                label="Search Query / தேடல் வார்த்தை",
                placeholder="Enter topic (e.g., நண்பன், கோபம், learning, anger)",
                scale=3
            )
            search_btn = gr.Button("Search / தேடு", scale=1)
        
        search_output = gr.Textbox(
            label="Search Results / தேடல் முடிவுகள்",
            lines=15,
            max_lines=20
        )
        
        search_btn.click(
            search_aathichoodi,
            inputs=search_input,
            outputs=search_output
        )
        
        search_input.submit(
            search_aathichoodi,
            inputs=search_input,
            outputs=search_output
        )

if __name__ == "__main__":
    demo.launch(share=True)