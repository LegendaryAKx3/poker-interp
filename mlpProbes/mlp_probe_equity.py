#!/usr/bin/env python
"""
Train MLP probe for Monte Carlo equity prediction on preprocessed poker hands.

This script loads a preprocessed poker dataset and trains a PyTorch MLP regression
probe on frozen Poker-GPT features to predict Monte Carlo equity values.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# PyTorch MLP Regression Model
# ---------------------------
class MLPRegressionProbe(nn.Module):
    """Non‑linear probe: Linear → ReLU → Linear mapping to single equity output."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super(MLPRegressionProbe, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=True),  # Single output for regression
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)  # Return shape (batch_size,)

# ---------------------------
# Metrics functions
# ---------------------------
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics using PyTorch."""
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # MSE
    mse = torch.mean((y_true - y_pred) ** 2).item()
    
    # MAE
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    
    # R-squared
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    
    return mse, mae, r2

# ---------------------------
# Batched embedding extraction
# ---------------------------
def get_embeddings(texts, tokenizer, model, batch_size=64, max_length=128):
    """Extract embeddings from poker GPT model in batches."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch_texts:
            # Tokenize using CharTokenizer
            tokens = tokenizer.encode(text)
            
            # Truncate to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Pad if necessary
            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))  # Pad with 0
            
            # Convert to tensor and add batch dimension
            tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                try:
                    # Extract embeddings from the model
                    # Use the transformer layers to get representations
                    x = model.tok_emb(tokens_tensor) + model.pos_emb(torch.arange(len(tokens), device=device).unsqueeze(0))
                    
                    # Pass through transformer layers
                    for layer in model.transformer:
                        x = layer(x)
                    
                    # Apply final layer norm
                    x = model.ln_f(x)
                    
                    # Mean pooling over sequence length
                    embedding = x.mean(dim=1).squeeze(0)  # Shape: (n_embd,)
                    batch_embeddings.append(embedding.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error processing text '{text[:20]}...': {e}")
                    # Use zero embedding as fallback
                    embedding = torch.zeros(model.tok_emb.embedding_dim, device=device)
                    batch_embeddings.append(embedding.cpu().numpy())
        
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# ---------------------------
# Load preprocessed dataset
# ---------------------------
def load_preprocessed_data(file_path, max_samples=None):
    """
    Load preprocessed poker dataset with equity labels, using masked actions.
    
    Returns:
        X_text: List of text representations (masked)
        y: Array of equity values
    """
    print(f"Loading preprocessed data from {file_path}...")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # Always use masked actions for training
    X_text = []
    y = []
    
    for item in data:
        # Try to get masked actions first, fallback to regular actions
        if 'masked_actions' in item and item['masked_actions']:
            # Convert masked actions list to text representation
            masked_text = ' '.join(item['masked_actions'])
            X_text.append(masked_text)
            y.append(item['equity'])
        elif 'text_repr' in item:
            # Fallback to text_repr if no masked_actions
            print(f"Warning: Using unmasked text_repr for item (no masked_actions found)")
            X_text.append(item['text_repr'])
            y.append(item['equity'])
        else:
            print(f"Warning: Skipping item - no masked_actions or text_repr found")
            continue
    
    y = np.array(y)
    
    print(f"Loaded {len(X_text)} samples with masked actions")
    print(f"Equity range: {y.min():.3f} - {y.max():.3f}")
    print(f"Mean equity: {y.mean():.3f} ± {y.std():.3f}")
    
    # DEBUG: Show first few masked examples
    print("=== First 3 masked examples ===")
    for i in range(min(3, len(X_text))):
        print(f"Sample {i+1}: {X_text[i][:100]}..." if len(X_text[i]) > 100 else f"Sample {i+1}: {X_text[i]}")
        print(f"  Equity: {y[i]:.3f}")
    print("=" * 40)
    
    return X_text, y

# ---------------------------
# Main training function
# ---------------------------
def train_mlp_equity_probe(dataset_path, model_artifacts_path="artifacts", max_samples=None, 
                          epochs=1000, learning_rate=0.001, batch_size=64, hidden_dim=256, 
                          huggingface_model=None):
    """
    Train MLP regression probe on poker equity prediction.
    
    Args:
        dataset_path: Path to preprocessed dataset JSON
        model_artifacts_path: Path to model artifacts directory
        max_samples: Maximum number of samples to use
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for embedding extraction
        hidden_dim: Hidden dimension size for MLP
        huggingface_model: Optional Hugging Face model name/path (e.g., "gpt2", "microsoft/DialoGPT-medium")
    """
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    
    # Check if using Hugging Face model
    if huggingface_model is not None:
        print(f"Loading Hugging Face model: {huggingface_model}")
        
        from transformers import AutoModel, AutoTokenizer
        
        try:
            # Load Hugging Face model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
            model = AutoModel.from_pretrained(huggingface_model).to(device)
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            
            print(f"✓ Successfully loaded Hugging Face model: {huggingface_model}")
            
            # Use Hugging Face embedding function
            def get_embeddings_hf(texts, tokenizer, model, batch_size=64, max_length=128):
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    # Tokenize batch
                    tokens = tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length
                    ).to(device)
                    
                    with torch.no_grad():
                        # Get model outputs
                        outputs = model(**tokens)
                        
                        # Use different pooling strategies based on model type
                        if hasattr(outputs, 'last_hidden_state'):
                            # Standard transformer output
                            hidden_states = outputs.last_hidden_state
                        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            # Some models return hidden_states tuple
                            hidden_states = outputs.hidden_states[-1]
                        else:
                            # Fallback to first output tensor
                            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                        
                        # Mean pooling over sequence length (ignoring padding)
                        attention_mask = tokens.get('attention_mask', None)
                        if attention_mask is not None:
                            # Weighted average by attention mask
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings_batch = sum_embeddings / sum_mask
                        else:
                            # Simple mean pooling
                            embeddings_batch = hidden_states.mean(dim=1)
                        
                    embeddings.append(embeddings_batch.cpu().numpy())
                return np.vstack(embeddings)
            
            # Override the get_embeddings function for this session
            get_embeddings = get_embeddings_hf
            
        except Exception as e:
            print(f"Failed to load Hugging Face model {huggingface_model}: {e}")
            print("Falling back to local model loading...")
            huggingface_model = None  # Fall through to local model loading
    
    # If not using Hugging Face model, try local models
    if huggingface_model is None:
        # Check if it's a Transformers model checkpoint directory
        transformers_checkpoint_path = f"{model_artifacts_path}/checkpoints/run1/best"
        pytorch_checkpoint_path = f"{model_artifacts_path}/checkpoints/run1/best.pt"
        
        if os.path.exists(transformers_checkpoint_path) and os.path.isdir(transformers_checkpoint_path):
            # Check if it contains Transformers files
            config_file = os.path.join(transformers_checkpoint_path, "config.json")
            model_file = os.path.join(transformers_checkpoint_path, "model.safetensors")
            
            if os.path.exists(config_file) and os.path.exists(model_file):
                print("Found Transformers model checkpoint, loading with Transformers...")
                
                # Re-import Transformers components
                from transformers import AutoModel, AutoConfig
                from tokenizers import Tokenizer
                from transformers import PreTrainedTokenizerFast
                
                # Load tokenizer
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=Tokenizer.from_file(f"{model_artifacts_path}/tokenizer/tokenizer.json")
                )
                tokenizer.add_special_tokens({
                    "pad_token": "<PAD>",
                    "unk_token": "<UNK>",
                    "bos_token": "<BOS>",
                    "eos_token": "<EOS>"
                })
                
                # Try to load model with different methods
                try:
                    # Try loading normally first
                    model = AutoModel.from_pretrained(transformers_checkpoint_path).to(device)
                    print("✓ Successfully loaded model with safetensors")
                except Exception as e:
                    print(f"Failed to load with safetensors ({e}), trying alternative methods...")
                    
                    try:
                        # Try forcing PyTorch loading instead of safetensors
                        model = AutoModel.from_pretrained(
                            transformers_checkpoint_path, 
                            use_safetensors=False
                        ).to(device)
                        print("✓ Successfully loaded model without safetensors")
                    except Exception as e2:
                        print(f"Failed to load without safetensors ({e2}), trying config-only approach...")
                        
                        try:
                            # Load config and create model, then load weights manually
                            config = AutoConfig.from_pretrained(transformers_checkpoint_path)
                            model = AutoModel.from_config(config).to(device)
                            
                            # Try to load state dict manually
                            import glob
                            pt_files = glob.glob(os.path.join(transformers_checkpoint_path, "*.pt"))
                            bin_files = glob.glob(os.path.join(transformers_checkpoint_path, "*.bin"))
                            
                            if pt_files:
                                state_dict = torch.load(pt_files[0], map_location=device)
                                model.load_state_dict(state_dict)
                                print("✓ Successfully loaded model from .pt file")
                            elif bin_files:
                                state_dict = torch.load(bin_files[0], map_location=device)
                                model.load_state_dict(state_dict)
                                print("✓ Successfully loaded model from .bin file")
                            else:
                                raise Exception("No compatible model files found")
                                
                        except Exception as e3:
                            print(f"All Transformers loading methods failed ({e3})")
                            print("Falling back to random model for testing...")
                            
                            # Create a dummy model for testing
                            config = AutoConfig.from_pretrained(transformers_checkpoint_path)
                            model = AutoModel.from_config(config).to(device)
                            print("⚠️ Using randomly initialized model - results will be meaningless!")
                
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                
                # Use Transformers-style embedding function
                def get_embeddings_transformers(texts, tokenizer, model, batch_size=64, max_length=128):
                    embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        tokens = tokenizer(
                            batch_texts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=max_length
                        ).to(device)
                        
                        with torch.no_grad():
                            out = model(**tokens, output_hidden_states=True)
                            h = out.last_hidden_state.mean(dim=1)  # Mean pooling
                        embeddings.append(h.cpu().numpy())
                    return np.vstack(embeddings)
                
                # Override the get_embeddings function for this session
                get_embeddings = get_embeddings_transformers
                
            else:
                raise FileNotFoundError("Invalid Transformers checkpoint directory")
                
        elif os.path.exists(pytorch_checkpoint_path):
            print("Found PyTorch checkpoint, loading with custom GPT...")
            
            # Import the actual poker GPT classes
            import sys
            sys.path.append('..')  # Add parent directory to import poker_gpt
            from poker_gpt import GPT, GPTConfig, CharTokenizer
            
            # Load checkpoint
            checkpoint = torch.load(pytorch_checkpoint_path, map_location=device)
            
            # Extract model state and config
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                gpt_cfg = checkpoint.get('config', {})
            else:
                # Assume the checkpoint is just the state dict
                model_state_dict = checkpoint
                # Use default config if not available
                gpt_cfg = {
                    'vocab_size': 100,  # Adjust based on your tokenizer
                    'n_layer': 4,
                    'n_head': 4,
                    'n_embd': 256,
                    'block_size': 256
                }
            
            # Create model config
            config = GPTConfig(**gpt_cfg)
            
            # Initialize model
            model = GPT(config).to(device)
            model.load_state_dict(model_state_dict)
            model.eval()
            
            # Freeze model parameters
            for p in model.parameters():
                p.requires_grad = False
            
            # Load tokenizer
            try:
                tokenizer_path = f"{model_artifacts_path}/tokenizer/tokenizer.json"
                if os.path.exists(tokenizer_path):
                    # If you have a saved tokenizer
                    with open(tokenizer_path, 'r') as f:
                        vocab_data = json.load(f)
                        vocab = vocab_data.get('vocab', [])
                        if not vocab:
                            # Try different JSON structure
                            vocab = list(vocab_data.keys()) if isinstance(vocab_data, dict) else []
                else:
                    # Create a simple character tokenizer
                    vocab = [chr(i) for i in range(256)]  # Basic character vocab
                
                tokenizer = CharTokenizer(extra_chars=''.join(vocab) if vocab else "")
            except Exception as e:
                print(f"Warning: Could not load tokenizer ({e}), using simple character tokenizer")
                tokenizer = CharTokenizer()
        
        else:
            # Try alternative paths
            alt_paths = [
                f"{model_artifacts_path}/model.pt",
                "artifacts/model.pt", 
                "model.pt",
                "../model.pt"
            ]
            
            found_checkpoint = None
            for path in alt_paths:
                if os.path.exists(path):
                    found_checkpoint = path
                    break
            
            if found_checkpoint is None:
                raise FileNotFoundError(f"Could not find model checkpoint. Tried: {[transformers_checkpoint_path, pytorch_checkpoint_path] + alt_paths}")
            
            print(f"Loading PyTorch checkpoint from: {found_checkpoint}")
            
            # Import the actual poker GPT classes
            import sys
            sys.path.append('..')  # Add parent directory to import poker_gpt
            from poker_gpt import GPT, GPTConfig, CharTokenizer
            
            # Load checkpoint
            checkpoint = torch.load(found_checkpoint, map_location=device)
            
            # Extract model state and config  
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                gpt_cfg = checkpoint.get('config', {})
            else:
                model_state_dict = checkpoint
                gpt_cfg = {
                    'vocab_size': 100,
                    'n_layer': 4,
                    'n_head': 4,
                    'n_embd': 256,
                    'block_size': 256
                }
            
            config = GPTConfig(**gpt_cfg)
            model = GPT(config).to(device)
            model.load_state_dict(model_state_dict)
            model.eval()
            
            for p in model.parameters():
                p.requires_grad = False
            
            tokenizer = CharTokenizer()

    # Load preprocessed data
    X_text, y = load_preprocessed_data(dataset_path, max_samples)
    
    if len(X_text) == 0:
        raise ValueError("No data found in preprocessed dataset!")

    # Filter out extreme outliers (optional)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (y >= lower_bound) & (y <= upper_bound)
    X_text, y = np.array(X_text)[mask], y[mask]
    
    print(f"After outlier filtering: {len(X_text)} samples")
    print(f"Final equity range: {y.min():.3f} - {y.max():.3f}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )

    # Get embeddings
    print("Extracting embeddings...")
    emb_train = get_embeddings(X_train.tolist(), tokenizer, model, batch_size=batch_size)
    emb_test = get_embeddings(X_test.tolist(), tokenizer, model, batch_size=batch_size)

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    emb_train = scaler.fit_transform(emb_train)
    emb_test = scaler.transform(emb_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(emb_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(emb_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Initialize PyTorch MLP regression probe
    print(f"Training PyTorch MLP regression probe (hidden_dim={hidden_dim})...")
    input_dim = emb_train.shape[1]
    regressor = MLPRegressionProbe(input_dim, hidden_dim).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)
    
    # Training loop
    regressor.train()
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = regressor(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Evaluation
    regressor.eval()
    with torch.no_grad():
        y_pred_train = regressor(X_train_tensor).cpu().numpy()
        y_pred_test = regressor(X_test_tensor).cpu().numpy()
    
    # Calculate metrics
    train_mse, train_mae, train_r2 = calculate_metrics(y_train, y_pred_train)
    test_mse, test_mae, test_r2 = calculate_metrics(y_test, y_pred_test)
    
    print(f"\nMLP Regression Results:")
    print(f"Train MSE: {train_mse:.6f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Test  MSE: {test_mse:.6f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    # Create visualizations
    os.makedirs("confusion_matrices/equityMLP", exist_ok=True)
    
    # Plot 1: Training loss
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title("Training Loss (MLP)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual (Test Set)
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("True Equity")
    plt.ylabel("Predicted Equity")
    plt.title(f"Predicted vs True Equity (Test)\nR² = {test_r2:.3f}")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Equity")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot (Test)\nMAE = {test_mae:.3f}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("confusion_matrices/equityMLP/equity_mlp_plots.png", dpi=300)
    plt.close()
    
    # Plot 4: Equity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Monte Carlo Equity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Monte Carlo Equity Values")
    plt.grid(True, alpha=0.3)
    plt.savefig("confusion_matrices/equityMLP/equity_distribution.png", dpi=300)
    plt.close()
    
    # Save model and results
    results = {
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'train_mse': float(train_mse),
        'train_mae': float(train_mae),
        'train_r2': float(train_r2),
        'n_samples': len(X_text),
        'equity_range': {'min': float(y.min()), 'max': float(y.max())},
        'training_params': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim
        }
    }
    
    # Save probe model
    torch.save({
        'model_state_dict': regressor.state_dict(),
        'scaler': scaler,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'results': results
    }, "confusion_matrices/equityMLP/equity_mlp_model.pth")
    
    with open("confusion_matrices/equityMLP/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to confusion_matrices/equityMLP/")
    print("Plots: equity_mlp_plots.png, equity_distribution.png")
    print("Model: equity_mlp_model.pth")
    print("Metrics: results.json")
    
    return regressor, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MLP probe for poker equity prediction")
    parser.add_argument("--dataset", default="preprocessed_equity_dataset.json", 
                       help="Path to preprocessed dataset")
    parser.add_argument("--artifacts", default="artifacts", 
                       help="Path to model artifacts directory")
    parser.add_argument("--huggingface_model", type=str, default=None,
                       help="Hugging Face model name/path (e.g., 'gpt2', 'microsoft/DialoGPT-medium')")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum samples to use")
    parser.add_argument("--epochs", type=int, default=1000, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, 
                       help="Batch size for embedding extraction")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension size for MLP")
    
    args = parser.parse_args()
    
    train_mlp_equity_probe(
        dataset_path=args.dataset,
        model_artifacts_path=args.artifacts,
        max_samples=args.max_samples,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        huggingface_model=args.huggingface_model
    )
