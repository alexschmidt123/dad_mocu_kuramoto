"""
design/train_rl.py - DAD policy training with REINFORCE and progress bars
"""

import torch
import torch.nn as nn
import torch.optim as optim
from design.dad_policy import DADPolicy

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def make_hist_tokens(h, N):
    """Convert history to tokens for LSTM input"""
    toks = []
    for (i,j), y in zip(h.pairs, h.outcomes):
        toks.append([i/(N-1+1e-9), j/(N-1+1e-9), float(y)])
    return toks


def train_dad_rl(env_factory, policy: DADPolicy, surrogate, epochs=10, 
                 episodes_per_epoch=50, lr=1e-3):
    """
    Train DAD policy to minimize MOCU using REINFORCE policy gradient.
    
    Key differences from DAD paper:
    - Objective: Minimize terminal MOCU (not maximize EIG)
    - Reward: -MOCU (negative because we minimize)
    - Uses MPNN surrogate for MOCU prediction
    """
    
    opt = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    
    print("\n" + "="*80)
    print("DAD POLICY TRAINING (Reinforcement Learning)")
    print("="*80)
    print(f"Objective: Minimize final MOCU")
    print(f"Method: REINFORCE policy gradient")
    print(f"Epochs: {epochs}, Episodes/epoch: {episodes_per_epoch}")
    print(f"Learning rate: {lr}")
    print("="*80 + "\n")
    
    # Progress bar for epochs
    if TQDM_AVAILABLE:
        epoch_pbar = tqdm(range(epochs), desc="RL Training", unit="epoch",
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        epoch_iterator = epoch_pbar
    else:
        epoch_iterator = range(epochs)
    
    for epc in epoch_iterator:
        epoch_returns = []
        epoch_loss = 0.0
        
        # Progress bar for episodes within epoch
        if TQDM_AVAILABLE:
            episode_pbar = tqdm(range(episodes_per_epoch), desc=f"  Epoch {epc+1} episodes",
                               unit="ep", leave=False, ncols=100,
                               bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
            episode_iterator = episode_pbar
        else:
            episode_iterator = range(episodes_per_epoch)
        
        for ep_idx in episode_iterator:
            env = env_factory()
            log_probs = []
            
            # Run episode
            for step in range(env.K):
                cands = env.candidate_pairs()
                hist_tokens = make_hist_tokens(env.h, env.N)
                g = env.features()
                
                # Get policy scores
                scores = policy.forward(g, cands, hist_tokens, env.N)
                
                # Convert scores to probabilities (lower score = higher probability)
                temperature = 0.5
                logits = -scores / temperature
                probs = torch.softmax(logits, dim=0)
                
                # Exploration schedule: more exploration early
                exploration_rate = max(0.1, 1.0 - epc / (epochs * 0.5))
                
                if torch.rand(1).item() < exploration_rate:
                    # Exploration: sample from distribution
                    action_dist = torch.distributions.Categorical(probs)
                    action_idx = action_dist.sample()
                    log_prob = action_dist.log_prob(action_idx)
                else:
                    # Exploitation: greedy
                    action_idx = scores.argmin()
                    log_prob = torch.log(probs[action_idx] + 1e-8)
                
                log_probs.append(log_prob)
                
                # Take action
                xi = cands[action_idx.item()]
                env.step(xi)
            
            # Compute final MOCU (reward)
            final_g = env.features()
            final_mocu = surrogate.forward_mocu(final_g).item()
            
            # Return = -MOCU (higher return = lower MOCU = better)
            episode_return = -final_mocu
            epoch_returns.append(final_mocu)
            
            # REINFORCE loss
            loss = 0.0
            for log_prob in log_probs:
                loss += -episode_return * log_prob
            
            loss = loss / len(log_probs)
            
            # Backward pass with gradient clipping
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item()
        
        if TQDM_AVAILABLE:
            episode_pbar.close()
        
        # Epoch statistics
        avg_loss = epoch_loss / episodes_per_epoch
        avg_mocu = sum(epoch_returns) / len(epoch_returns)
        min_mocu = min(epoch_returns)
        max_mocu = max(epoch_returns)
        
        # Update epoch progress bar or print
        if TQDM_AVAILABLE:
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'MOCU': f'{avg_mocu:.4f}',
                'min': f'{min_mocu:.4f}'
            })
        else:
            print(f"Epoch {epc+1}/{epochs}:")
            print(f"  Loss: {avg_loss:.4f}, Avg MOCU: {avg_mocu:.4f}, "
                  f"Min: {min_mocu:.4f}, Max: {max_mocu:.4f}")
        
        # Quality check
        if avg_mocu < 0.15:
            if not TQDM_AVAILABLE:
                print(f"  Excellent performance - MOCU < 0.15")
        elif avg_mocu < 0.20:
            if not TQDM_AVAILABLE:
                print(f"  Good performance - MOCU < 0.20")
    
    if TQDM_AVAILABLE:
        epoch_pbar.close()
    
    print("\n" + "="*80)
    print("DAD POLICY TRAINING COMPLETE")
    print(f"Final Avg MOCU: {avg_mocu:.4f}")
    print("="*80)
    
    return policy