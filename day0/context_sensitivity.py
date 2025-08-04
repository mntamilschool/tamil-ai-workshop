# Required installations:
# pip install transformers torch numpy scikit-learn

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def get_embeddings(model, tokenizer, texts):
    """
    Extract sentence embeddings from a pre-trained transformer model.
    
    Args:
        model: Pre-trained transformer model
        tokenizer: Corresponding tokenizer for the model
        texts: List of text strings to encode
    
    Returns:
        List of numpy arrays containing sentence embeddings
    """
    embeddings = []
    
    for text in texts:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate embeddings without computing gradients
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use average pooling to create sentence-level representation
            # Shape: [batch_size, sequence_length, hidden_size] -> [hidden_size]
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            embeddings.append(sentence_embedding.numpy())
    
    return embeddings

def demonstrate_context_sensitivity(model, tokenizer):
    """
    Demonstrate how different contexts affect sentence embeddings.
    Shows that similar sentences about banking have high similarity.
    """
    print("\n" + "="*50)
    print("CONTEXT SENSITIVITY DEMONSTRATION")
    print("="*50)
    
    # Two sentences about banking with similar semantic meaning
    contexts = [
        "I deposited money at the bank.",
        "I got the loan from the bank.",
        # "I sat by the river bank and watched the sunset.",
    ]
    
    print("\nAnalyzing these sentences:")
    for i, context in enumerate(contexts, 1):
        print(f"  {i}. '{context}'")
    
    # Generate embeddings for both sentences
    print("\nGenerating embeddings...")
    embeddings = get_embeddings(model, tokenizer, contexts)
    
    # Calculate cosine similarity between the embeddings
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Display results
    print(f"\nResults:")
    print(f"  Cosine similarity: {similarity_score:.4f}")
    print(f"  Interpretation: {'High similarity' if similarity_score > 0.7 else 'Moderate similarity' if similarity_score > 0.4 else 'Low similarity'}")
    
    return similarity_score

def main():
    """
    Main function to run the context sensitivity demonstration.
    """
    try:
        print("Context Sensitivity Analysis")
        print("This script demonstrates how transformer models capture semantic similarity")
        
        # Load pre-trained model and tokenizer
        model_name = "distilbert-base-uncased"
        print(f"\nLoading model: {model_name}")
        print("This may take a moment on first run...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("Model loaded successfully!")
        
        # Run the demonstration
        similarity_score = demonstrate_context_sensitivity(model, tokenizer)
        
        # Summary
        print(f"\n" + "-"*50)
        print("SUMMARY")
        print("-"*50)
        print(f"The model captured semantic similarity with score: {similarity_score:.4f}")
        print("Higher scores indicate more similar semantic meaning.")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install transformers torch numpy scikit-learn")

if __name__ == "__main__":
    main()
