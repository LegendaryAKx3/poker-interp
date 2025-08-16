#!/usr/bin/env python
"""
Preprocess poker hands dataset with masking and Monte Carlo equity labels.

This script processes NDJSON poker hand histories, applies masking to hide opponent information,
and calculates Monte Carlo equity values for each hand. The output is a preprocessed
dataset ready for training linear probes.
"""

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
from treys import Card, Deck, Evaluator

# ---------------------------
# Card utilities using Treys
# ---------------------------
def parse_cards(cards_str):
    """Convert a string like "Jc9h5c" or "Ac2d" into a list of Treys card ints."""
    if not cards_str or '?' in cards_str:
        return []
    return [Card.new(cards_str[i:i+2]) for i in range(0, len(cards_str), 2)]

def parse_game_log(lines):
    """
    Parse action logs using Treys. Returns:
      - board: list of community card ints
      - hole_cards: mapping player_id -> hole card ints
    """
    board = []
    hole_cards = {}

    for line in lines:
        parts = line.strip().split()
        if parts[:2] == ["d", "dh"] and len(parts) >= 4:
            pid, cs = parts[2], parts[3]
            if "?" not in cs:
                hole_cards[pid] = parse_cards(cs)
        elif parts[:2] == ["d", "db"] and len(parts) >= 3:
            board.extend(parse_cards(parts[2]))
        elif len(parts) >= 3 and parts[1] == "sm":
            pid, cs = parts[0], parts[2]
            if "?" not in cs:
                hole_cards[pid] = parse_cards(cs)
    return board, hole_cards

def calculate_monte_carlo_equity(hole_cards, board_cards, trials=1000):
    """Calculate Monte Carlo equity using Treys library."""
    try:
        # Ensure we have valid hole cards
        if not hole_cards or len(hole_cards) != 2:
            return 0.0
            
        # Ensure board cards are valid
        board_ints = board_cards if board_cards else []
        
        # Monte Carlo simulation using Treys
        evaluator = Evaluator()
        wins = ties = 0
        
        for _ in range(trials):
            # Create a new deck and remove known cards
            deck = Deck()
            for card in hole_cards + board_ints:
                if card in deck.cards:
                    deck.cards.remove(card)
            deck.shuffle()
            
            # Deal opponent cards
            if len(deck.cards) < 2:
                continue
            opp_hole = deck.draw(2)
            
            # Complete the board if needed
            cards_needed = max(0, 5 - len(board_ints))
            if len(deck.cards) < cards_needed:
                continue
            full_board = board_ints + (deck.draw(cards_needed) if cards_needed > 0 else [])
            
            # Evaluate hands
            try:
                player_score = evaluator.evaluate(full_board, hole_cards)
                opponent_score = evaluator.evaluate(full_board, opp_hole)
                
                if player_score < opponent_score:  # Lower is better in Treys
                    wins += 1
                elif player_score == opponent_score:
                    ties += 1
            except:
                continue  # Skip invalid combinations
        
        if trials == 0:
            return 0.0
        return (wins + ties * 0.5) / trials
        
    except Exception as e:
        print(f"Error calculating equity: {e}")
        return 0.0

def apply_masking(actions, model_player=1):
    """
    Apply masking to poker actions to hide opponent information.
    
    Args:
        actions: List of action strings
        model_player: Which player is the model (default 1)
    
    Returns:
        List of masked action strings
    """
    masked_actions = []
    
    for action in actions:
        if 'sm' in action:
            # Mask showdown cards
            masked_actions.append(action[:action.index('sm')+2] + ' ' + ('?' * (len(action)-action.index('sm')-3)))
        elif "d dh p" in action:
            if f"d dh p{model_player}" in action:
                # Keep model player's hole cards
                masked_actions.append(action)
            else:
                # Mask other players' hole cards
                masked_actions.append(action[:8] + "?" * (len(action)-8))
        else:
            # Keep other actions as-is
            masked_actions.append(action)
    
    return masked_actions

def extract_from_phh(phh_path):
    """Extract starting stacks and actions from PHH file."""
    try:
        doc = toml.loads(Path(phh_path).read_text())
        return doc.get('starting_stacks', []), doc.get("actions", [])
    except Exception as e:
        print(f"Error reading {phh_path}: {e}")
        return [], []

def load_ndjson_hands(file_path, max_hands=None):
    """
    Load poker hands from NDJSON file.
    
    Args:
        file_path: Path to NDJSON file
        max_hands: Maximum number of hands to load
    
    Returns:
        List of action arrays
    """
    hands = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_hands and i >= max_hands:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    actions = json.loads(line)
                    if isinstance(actions, list) and actions:
                        hands.append(actions)
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return hands

def process_poker_dataset(ndjson_files=None, output_file="preprocessed_equity_dataset.json", 
                         max_samples=None, model_player=1):
    """
    Process poker hand histories from NDJSON files and create preprocessed dataset with equity labels.
    
    Args:
        ndjson_files: List of NDJSON file paths. If None, uses default paths.
        output_file: Output JSON file path
        max_samples: Maximum number of samples to process (None for all)
        model_player: Which player is the model
    """
    if ndjson_files is None:
        # Default NDJSON files to process
        ndjson_files = [
            "data/hands3.ndjson",
            # Add more NDJSON files here as needed
        ]
    
    dataset = []
    equity_counter = Counter()
    processed_count = 0
    
    print("Processing NDJSON poker hand histories...")
    
    for file_path in ndjson_files:
        if not Path(file_path).exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        print(f"Processing {file_path}...")
        hands = load_ndjson_hands(file_path, max_hands=max_samples)
        
        for i, actions in enumerate(hands):
            if max_samples and processed_count >= max_samples:
                break
                
            try:
                # Apply masking
                masked_actions = apply_masking(actions, model_player)
                
                # Parse for equity calculation
                board, hole_cards = parse_game_log(actions)
                p1_hole = hole_cards.get(f'p{model_player}', [])
                
                if len(p1_hole) == 2:
                    # Calculate Monte Carlo equity
                    equity = calculate_monte_carlo_equity(p1_hole, board, trials=1000)
                    
                    if 0.0 <= equity <= 1.0:
                        # Create text representation
                        hole_text = ''.join([Card.int_to_str(card) for card in p1_hole])
                        board_text = ''.join([Card.int_to_str(card) for card in board]) if board else ""
                        text_repr = f"{hole_text} {board_text}".strip()
                        
                        dataset.append({
                            'original_actions': actions,
                            'masked_actions': masked_actions,
                            'text_repr': text_repr,
                            'equity': equity,
                            'source': f'{Path(file_path).stem}_{i}'
                        })
                        
                        # Track equity distribution
                        equity_bin = round(equity, 1)
                        equity_counter[equity_bin] += 1
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} hands...")
                            
            except Exception as e:
                print(f"Error processing hand {i} from {file_path}: {e}")
                continue
    
    # Save dataset
    print(f"\nProcessed {len(dataset)} total hands")
    print("Equity distribution:", dict(sorted(equity_counter.items())))
    
    if dataset:
        equities = [item['equity'] for item in dataset]
        print(f"Equity range: {min(equities):.3f} - {max(equities):.3f}")
        print(f"Mean equity: {np.mean(equities):.3f} ± {np.std(equities):.3f}")
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess NDJSON poker dataset with masking and equity labels")
    parser.add_argument("--input", nargs="+", default=None, 
                       help="Input NDJSON file paths (default: data/hands3.ndjson)")
    parser.add_argument("--output", default="preprocessed_equity_dataset.json", 
                       help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum samples to process")
    parser.add_argument("--model_player", type=int, default=1, 
                       help="Which player is the model")
    
    args = parser.parse_args()
    
    process_poker_dataset(
        ndjson_files=args.input,
        output_file=args.output,
        max_samples=args.max_samples,
        model_player=args.model_player
    )
