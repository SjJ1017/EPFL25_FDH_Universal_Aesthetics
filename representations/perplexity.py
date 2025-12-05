import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
import argparse

def calculate_perplexity(text, model, tokenizer, device, max_length=512):
    """
    Calculate perplexity for a given text.
    
    Args:
        text: Input text string
        model: Pretrained language model
        tokenizer: Tokenizer for the model
        device: Device to run computation on
        max_length: Maximum sequence length (longer texts will be truncated)
    
    Returns:
        perplexity: Perplexity score (float)
    """
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    
    # Skip very short texts
    if input_ids.shape[1] < 2:
        return float('inf')
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Perplexity is exp(loss)
        perplexity = torch.exp(loss).item()
    
    return perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate perplexity of poems using a pretrained language model.")
    parser.add_argument("--model_name", type=str, default="gpt2-large", help="Pretrained model name or path")
    parser.add_argument("--dataset_name", type=str, default="minhuh/prh", help="Dataset name to load poems from")
    parser.add_argument("--output", type=str, default="results/perplexity_scores_text.npy", help="Path to save output file")
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    print("Loading dataset...")
    if args.dataset_name == "minhuh/prh":
        dataset = load_dataset(args.dataset_name, split="train", revision="wit_1024")
        contents = [str(x['text'][0]) for x in dataset]
    else:
        dataset = load_dataset(args.dataset_name, split="train")
        contents = [str(x['content']) for x in dataset]
    
    print(f"Loaded {len(contents)} poems/texts.")



    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nCalculating perplexity for all /texts...")
    perplexities = []

    for i, content in enumerate(tqdm(contents, desc="Computing perplexity")):
        try:
            ppl = calculate_perplexity(content, model, tokenizer, device, max_length=512)
            perplexities.append(ppl)
        except Exception as e:
            print(f"\nError processing poem/text {i}: {e}")
            perplexities.append(float('inf'))

    perplexities = np.array(perplexities)

    # Save results

    np.save(args.output, perplexities)
    print(f"\nSaved perplexity scores to {args.output}")

    # Print statistics
    valid_ppls = perplexities[perplexities != float('inf')]
    print("\n=== Perplexity Statistics ===")
    print(f"Total poems: {len(perplexities)}")
    print(f"Valid scores: {len(valid_ppls)}")
    print(f"Mean perplexity: {np.mean(valid_ppls):.2f}")
    print(f"Median perplexity: {np.median(valid_ppls):.2f}")
    print(f"Std perplexity: {np.std(valid_ppls):.2f}")
    print(f"Min perplexity: {np.min(valid_ppls):.2f}")
    print(f"Max perplexity: {np.max(valid_ppls):.2f}")
