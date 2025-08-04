# Required installations:
# pip install transformers torch

from transformers import AutoTokenizer
from typing import List

def demonstrate_tokenization(tokenizer: AutoTokenizer, texts: List[str]) -> None:
    """
    Show how a real tokenizer breaks down text into tokens.
    
    Args:
        tokenizer: The Hugging Face tokenizer instance
        texts: List of text strings to tokenize
    """
    print("\n" + "="*50)
    print("TOKENIZATION DEMONSTRATION")
    print("="*50)
    
    for i, text in enumerate(texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Original text: '{text}'")
        
        # Convert text to token IDs (numerical representation)
        token_ids = tokenizer.encode(text)
        
        # Convert token IDs back to readable tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Make tokens more readable by replacing special characters
        # 'Ġ' represents spaces in some tokenizers (like GPT-2/RoBERTa style)
        readable_tokens = [
            token.replace('Ġ', '▁') if 'Ġ' in token else token 
            for token in tokens
        ]
        
        # Display results
        print(f"Tokens:       {readable_tokens}")
        print(f"Token IDs:    {token_ids}")
        print(f"Token count:  {len(tokens)}")

def main() -> None:
    """Main function to run the tokenization demonstration."""
    try:
        print("Initializing tokenizer...")
        
        # Using DistilBERT - a smaller, faster version of BERT
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✓ Successfully loaded tokenizer: {model_name}")
        
        # Sample texts demonstrating different tokenization scenarios
        sample_texts = [
            "Hello, how are you?",                    
            "Are you learning about AI?",             
            "Supercalifragilisticexpialidocious"      
        ]
        
        # Run the demonstration
        demonstrate_tokenization(tokenizer, sample_texts)
        
        print("\n" + "="*50)
        print("Demonstration complete!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install transformers torch")

if __name__ == "__main__":
    main()
