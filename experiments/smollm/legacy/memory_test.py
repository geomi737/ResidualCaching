import torch
from torch.nn import functional as F
import gc

torch.manual_seed(0)

# Simulate the shapes
B, S_full, S_half, H, V = 4, 256, 128, 288, 49152


def measure_peak():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# 1. Measure Deep Layers Memory (Transformer layers)
reset_mem()
start_mem = measure_peak()
# dummy params for 4 layers
deep_weights = [torch.nn.Linear(H, H, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
x = torch.randn(B, S_full, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
for w in deep_weights:
    x = w(x)
loss = x.sum()
loss.backward()
print(f"Deep Layers (4 layers, Seq={S_full}) Memory: {measure_peak() - start_mem:.2f} MiB")

# 2. Measure lm_head + Loss Memory (Seq = 256)
reset_mem()
start_mem = measure_peak()
lm_head = torch.nn.Linear(H, V, device='cuda', dtype=torch.bfloat16)
x_full = torch.randn(B, S_full, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
logits_full = lm_head(x_full)
targets_full = torch.randint(0, V, (B, S_full), device='cuda')
loss_full = F.cross_entropy(logits_full.float().view(-1, V), targets_full.view(-1))
loss_full.backward()
print(f"LM Head + Loss (Seq={S_full}) Memory: {measure_peak() - start_mem:.2f} MiB")

# 3. Measure lm_head + Loss Memory (Seq = 128)
reset_mem()
start_mem = measure_peak()
lm_head = torch.nn.Linear(H, V, device='cuda', dtype=torch.bfloat16)
x_half = torch.randn(B, S_half, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
logits_half = lm_head(x_half)
targets_half = torch.randint(0, V, (B, S_half), device='cuda')
loss_half = F.cross_entropy(logits_half.float().view(-1, V), targets_half.view(-1))
loss_half.backward()
print(f"LM Head + Loss (Seq={S_half}) Memory: {measure_peak() - start_mem:.2f} MiB")
