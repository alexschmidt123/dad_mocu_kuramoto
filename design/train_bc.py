import torch, torch.nn as nn, torch.optim as optim
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy

def make_hist_tokens(h, N):
    toks = []
    for (i,j), y in zip(h.pairs, h.outcomes):
        toks.append([i/(N-1+1e-9), j/(N-1+1e-9), float(y)])
    return toks

def train_behavior_cloning(env_factory, policy: DADPolicy, epochs=3, episodes_per_epoch=30, lr=1e-3):
    opt = optim.Adam(policy.parameters(), lr=lr); ce = nn.CrossEntropyLoss()
    for epc in range(epochs):
        total_loss = 0.0
        for _ in range(episodes_per_epoch):
            env = env_factory()
            for _k in range(env.K):
                cands = env.candidate_pairs()
                xi_star = choose_next_pair_greedy(env, cands)
                y = torch.tensor([cands.index(xi_star)], dtype=torch.long)
                hist_tokens = make_hist_tokens(env.h, env.N)
                g = env.features()
                scores = policy.forward(g, cands, hist_tokens, env.N).unsqueeze(0)
                loss = ce(scores, y)
                opt.zero_grad(); loss.backward(); opt.step()
                env.step(xi_star)
                total_loss += float(loss.item())
        print(f"[BC] epoch {epc+1}: avg loss = {total_loss/(episodes_per_epoch*env.K):.4f}")
    return policy
