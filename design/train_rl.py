import torch
import torch.nn as nn
import torch.optim as optim
from design.dad_policy import DADPolicy

def make_hist_tokens(h, N):
    toks = []
    for (i,j), y in zip(h.pairs, h.outcomes):
        toks.append([i/(N-1+1e-9), j/(N-1+1e-9), float(y)])
    return toks

def train_dad_rl(env_factory, policy: DADPolicy, surrogate, epochs=10, episodes_per_epoch=50, lr=1e-3):
    """Train DAD policy to minimize MOCU using REINFORCE policy gradient."""
    
    opt = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    
    print("\n" + "="*80)
    print("DAD POLICY TRAINING (Reinforcement Learning)")
    print("="*80)
    print(f"Objective: Minimize final MOCU")
    print(f"Method: REINFORCE policy gradient")
    print(f"Epochs: {epochs}, Episodes/epoch: {episodes_per_epoch}")
    print(f"Learning rate: {lr}")
    print("="*80)
    
    for epc in range(epochs):
        epoch_returns = []
        epoch_loss = 0.0
        
        for ep_idx in range(episodes_per_epoch):
            env = env_factory()
            
            # Storage for episode
            log_probs = []
            
            # Run episode
            for step in range(env.K):
                cands = env.candidate_pairs()
                hist_tokens = make_hist_tokens(env.h, env.N)
                g = env.features()
                
                # Get policy scores
                scores = policy.forward(g, cands, hist_tokens, env.N)
                
                # Convert scores to probabilities (lower score = higher probability)
                # Use softmax on negative scores with temperature
                temperature = 0.5
                logits = -scores / temperature
                probs = torch.softmax(logits, dim=0)
                
                # Sample action with exploration in early epochs
                if epc < epochs // 2:
                    # Exploration: sample from distribution
                    action_dist = torch.distributions.Categorical(probs)
                    action_idx = action_dist.sample()
                    log_prob = action_dist.log_prob(action_idx)
                else:
                    # Exploitation: greedy with small noise
                    if torch.rand(1).item() < 0.1:  # 10% exploration
                        action_dist = torch.distributions.Categorical(probs)
                        action_idx = action_dist.sample()
                        log_prob = action_dist.log_prob(action_idx)
                    else:
                        action_idx = scores.argmin()
                        log_prob = torch.log(probs[action_idx] + 1e-8)
                
                log_probs.append(log_prob)
                
                # Take action in environment
                xi = cands[action_idx.item()]
                env.step(xi)
            
            # Compute final MOCU (this is what we want to minimize)
            final_g = env.features()
            final_mocu = surrogate.forward_mocu(final_g).item()
            
            # Return = -MOCU (negative because we want to minimize MOCU)
            # Higher return = lower MOCU = better
            episode_return = -final_mocu
            epoch_returns.append(final_mocu)  # Store MOCU for logging
            
            # REINFORCE policy gradient loss
            # Maximize: E[Return * log π(a|s)]
            # Minimize: -Return * log π(a|s)
            loss = 0.0
            for log_prob in log_probs:
                loss += -episode_return * log_prob
            
            loss = loss / len(log_probs)  # Average over trajectory
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item()
        
        # Epoch statistics
        avg_loss = epoch_loss / episodes_per_epoch
        avg_mocu = sum(epoch_returns) / len(epoch_returns)
        min_mocu = min(epoch_returns)
        max_mocu = max(epoch_returns)
        
        print(f"[DAD-RL] Epoch {epc+1}/{epochs}:")
        print(f"  Policy Loss: {avg_loss:.4f}")
        print(f"  Avg MOCU: {avg_mocu:.4f}")
        print(f"  Min MOCU: {min_mocu:.4f}")
        print(f"  Max MOCU: {max_mocu:.4f}")
        
        if avg_mocu < 0.15:
            print(f"  ✓ Excellent - MOCU < 0.15")
        elif avg_mocu < 0.20:
            print(f"  ✓ Good - MOCU < 0.20")
    
    print("="*80)
    print("DAD POLICY TRAINING COMPLETE")
    print(f"Final Avg MOCU: {avg_mocu:.4f}")
    print("="*80)
    
    return policy