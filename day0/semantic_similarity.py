"""
Semantic Similarity Demonstration Script

This script demonstrates how to calculate semantic similarity between words/phrases
using pre-trained transformer models. It uses DistilBERT to generate embeddings
and cosine similarity to measure semantic relationships.

Required installations:
pip install transformers torch numpy scikit-learn
"""

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_embeddings(model, tokenizer, texts):
    """
    Generate embeddings from a pre-trained transformer model.
    
    Args:
        model: Pre-trained transformer model
        tokenizer: Corresponding tokenizer for the model
        texts (list): List of text strings to embed
    
    Returns:
        list: List of numpy arrays containing embeddings for each text
    """
    embeddings = []
    
    for text in texts:
        # Tokenize the input text and prepare for model input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate embeddings without computing gradients (inference mode)
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use average pooling to get a single vector representation
            # This averages all token embeddings to create a sentence-level embedding
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            
            # Convert tensor to numpy array and store
            embeddings.append(sentence_embedding.numpy())
    
    return embeddings

def demonstrate_similarity(model, tokenizer, word_pairs):
    """
    Calculate and display semantic similarity scores for word pairs.
    
    Args:
        model: Pre-trained transformer model
        tokenizer: Corresponding tokenizer for the model
        word_pairs (list): List of tuples containing word pairs to compare
    """
    print("\n" + "="*50)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("="*50)
    
    for word1, word2 in word_pairs:
        # Generate embeddings for both words
        emb1 = get_embeddings(model, tokenizer, [word1])[0]
        emb2 = get_embeddings(model, tokenizer, [word2])[0]
        
        # Calculate cosine similarity between the embeddings
        similarity_score = cosine_similarity([emb1], [emb2])[0][0]
        
        # Format and display the result
        print(f"'{word1}' ↔ '{word2}': {similarity_score:.3f}")

def main():
    """
    Main function to orchestrate the semantic similarity demonstration.
    """
    try:
        # Initialize the pre-trained model and tokenizer
        print("Loading DistilBERT model and tokenizer...")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print("Model loaded successfully!")
        
        # Define test word pairs with varying semantic relationships
        word_pairs = [
            ("dog", "puppy"),        # High similarity (same category)
            ("car", "vehicle"),      # High similarity (general/specific)            
            ("king", "queen"),       # High similarity (related concepts)
            ("mars", "yogurt"),      # Low similarity (unrelated)
        ]
        
        # Demonstrate semantic similarity calculation
        demonstrate_similarity(model, tokenizer, word_pairs)
        
        print("\n" + "="*50)
        print("Analysis complete! Try modifying the word_pairs list")
        print("to test semantic similarity with your own word combinations.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install transformers torch numpy scikit-learn")

if __name__ == "__main__":
    main()