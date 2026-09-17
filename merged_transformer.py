"""
Merged Token Transformer — тестируем идею мердж-токенов + residual cache

Historical non-causal classification prototype. The supported causal language
model is nanoGPT/model.py; see the project README for the current architecture.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Attention & FFN
# ============================================================================


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        # Переходим к форме (B, n_heads, L, d_k) для внимания между токенами L
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # scores: (B, n_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attn_out: (B, L, D)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(attn_out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


# ============================================================================
# 2. Token Merger — объединяем N токенов в 1 + сумма участвовавших эмбеддингов
# ============================================================================

class TokenMerger(nn.Module):
    def __init__(self, d_model, merge_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.merge_ratio = merge_ratio
        self.merge_weight = nn.Parameter(torch.ones(merge_ratio) / merge_ratio)

    def forward(self, x):
        B, L, D = x.shape
        assert L % self.merge_ratio == 0, f"Длина последовательности {L} должна делиться на {self.merge_ratio}"
        num_groups = L // self.merge_ratio
        groups, merged_tokens, group_sums = [], [], []

        for i in range(num_groups):
            start = i * self.merge_ratio
            end = start + self.merge_ratio
            group_tokens = x[:, start:end, :]
            indices = tuple(range(start, end))
            groups.append(indices)

            weights = F.softmax(self.merge_weight, dim=0).unsqueeze(0).unsqueeze(2)  # (1, merge_ratio, 1)
            merged = torch.sum(group_tokens * weights, dim=1)  # (B, D)
            merged_tokens.append(merged)

            # Идея: вместо хранения всех N эмбеддингов сохраняем сумму всех участвовавших
            # merged + 1residual + 2residual = merged + (1residual + 2residual)
            group_sum = torch.sum(group_tokens, dim=1)  # (B, D)
            group_sums.append(group_sum)

        # merged_tokens: (B, num_groups, D), group_sums: (B, num_groups, D)
        return torch.stack(merged_tokens, dim=1), torch.stack(group_sums, dim=1)


# ============================================================================
# 3. ResidualCache — храним суммы участвовавших эмбеддингов для residual
# ============================================================================

@dataclass
class ResidualCache:
    """
    Кэш суммированных residual-эмбеддингов.
    Каждая запись: тензор (B, num_groups, D) — поэлементная сумма оригинальных токенов группы.
    """
    entries: List[torch.Tensor] = field(default_factory=list)

    def add(self, group_residuals: torch.Tensor):
        self.entries.append(group_residuals)

    def clear(self):
        self.entries.clear()

    def __len__(self):
        return len(self.entries)

    def __bool__(self):
        return len(self.entries) > 0


# ============================================================================
# 4. Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================================
# 5. Transformer Blocks
# ============================================================================

class MergedTransformerBlock(nn.Module):
    """
    Pipeline:
      1. Attention + standard residual (Pre-LN)
      2. Merge N→1 (если merge=True) + сохранение суммы оригиналов в ResidualCache
      3. FFN + residual + sum(residuals из cache)
    """

    def __init__(self, d_model, n_heads, d_ff, merge_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.merger = TokenMerger(d_model, merge_ratio)
        self.dropout = nn.Dropout(dropout)
        self.merge_ratio = merge_ratio

    def forward(self, x, residual_cache: Optional[ResidualCache] = None, merge: bool = True):
        B, L, D = x.shape
        was_merged = False

        # Step 1: Attention + standard residual (Pre-LN)
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Step 2: Merge N→1
        if merge and L >= self.merge_ratio:
            merged, group_residuals = self.merger(x)
            x = merged
            if residual_cache is not None:
                residual_cache.add(group_residuals)
            was_merged = True

        # Step 3: FFN + residual с добавлением суммированных оригинальных эмбеддингов
        ffn_out = self.ffn(self.norm2(x))
        residual = ffn_out

        # Добавляем сумму участвовавших эмбеддингов из cache
        if residual_cache and len(residual_cache) > 0:
            for res in residual_cache.entries:
                # res shape: (B, new_L, D) — в точности совпадает с формой x и ffn_out
                residual = residual + res

        x = x + self.dropout(residual)
        return x, was_merged


class StandardTransformerBlock(nn.Module):
    """Стандартный Pre-LN Transformer блок (MHA + FFN) без сжатия токенов."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ============================================================================
# 6. Full Architectures
# ============================================================================

class MergedTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, d_ff=128,
                 n_layers=4, max_seq_len=64, merge_ratio=2, dropout=0.1,
                 num_classes=10, use_cache=True):
        super().__init__()
        self.use_cache = use_cache
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MergedTransformerBlock(d_model, n_heads, d_ff, merge_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        residual_cache = ResidualCache() if self.use_cache else None
        for i, block in enumerate(self.blocks):
            merge = (i + 1) % 2 == 0  # каждый 2-й блок мерджит
            x, was_merged = block(x, residual_cache, merge=merge)
            if was_merged and residual_cache is not None:
                residual_cache.clear()  # очищаем кэш после применения в блоке слияния

        x = self.norm(x)
        x = x[:, 0, :]
        return self.classifier(x)


class MergedTransformerNoCache(MergedTransformer):
    """MergedTransformer с явно отключенным Residual Cache для абляций."""

    def __init__(self, *args, **kwargs):
        kwargs["use_cache"] = False
        super().__init__(*args, **kwargs)


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, d_ff=128,
                 n_layers=4, max_seq_len=64, dropout=0.1, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0, :]
        return self.classifier(x)


# ============================================================================
# 7. Dataset & Training Utilities
# ============================================================================

def create_simple_dataset(num_samples, seq_len=16, vocab_size=50, num_classes=3):
    """
    Простой датасет:
    - Случайные последовательности токенов
    - Метка: сумма всех токенов mod num_classes
    """
    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    y = torch.sum(X, dim=1) % num_classes
    return X, y


# Для обратной совместимости
create_simple_dataset2 = create_simple_dataset


def train_step(model, x, y, optimizer, device):
    optimizer.zero_grad()
    logits = model(x.to(device))
    loss = F.cross_entropy(logits, y.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def train_and_eval(model, X_train, y_train, X_val, y_val, device, epochs=100, lr=1e-3, bs=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), bs):
            idx = indices[i:i + bs]
            optimizer.zero_grad()
            logits = model(X_train[idx].to(device))
            loss = F.cross_entropy(logits, y_train[idx].to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_acc = (val_logits.argmax(dim=1) == y_val.to(device)).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Val Acc: {val_acc:.3f} | Best: {best_val_acc:.3f}")

    return best_val_acc


# ============================================================================
# 8. Benchmarks & Main
# ============================================================================

def run_comprehensive_test():
    """Сравнение: Merged+Cache vs Merged no Cache vs Standard"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE, D_MODEL, N_HEADS, D_FF = 50, 128, 4, 256
    N_LAYERS, SEQ_LEN, MERGE_RATIO, NUM_CLASSES = 4, 16, 2, 3
    EPOCHS, BS, LR = 100, 64, 1e-3

    X_train, y_train = create_simple_dataset(5000, SEQ_LEN, VOCAB_SIZE, NUM_CLASSES)
    X_val, y_val = create_simple_dataset(1000, SEQ_LEN, VOCAB_SIZE, NUM_CLASSES)
    print(f"Dataset: {len(X_train)} train, {len(X_val)} val")
    print(f"d_model={D_MODEL}, layers={N_LAYERS}, seq_len={SEQ_LEN}")
    print()

    # 1. Merged Transformer + Residual Cache (our method)
    print("=" * 60)
    print("1. Merged Transformer + Residual Cache")
    print("=" * 60)
    model_merged = MergedTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, max_seq_len=SEQ_LEN, merge_ratio=MERGE_RATIO,
        dropout=0.1, num_classes=NUM_CLASSES, use_cache=True
    ).to(device)
    params = sum(p.numel() for p in model_merged.parameters())
    print(f"Params: {params:,}")
    acc_merged = train_and_eval(model_merged, X_train, y_train, X_val, y_val, device, EPOCHS, LR, BS)

    # 2. Standard Transformer
    print()
    print("=" * 60)
    print("2. Standard Transformer (baseline)")
    print("=" * 60)
    model_std = StandardTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, max_seq_len=SEQ_LEN, dropout=0.1, num_classes=NUM_CLASSES
    ).to(device)
    params_std = sum(p.numel() for p in model_std.parameters())
    print(f"Params: {params_std:,}")
    acc_std = train_and_eval(model_std, X_train, y_train, X_val, y_val, device, EPOCHS, LR, BS)

    # 3. Merged Transformer БЕЗ Residual Cache
    print()
    print("=" * 60)
    print("3. Merged Transformer БЕЗ Residual Cache (control)")
    print("=" * 60)
    model_no_cache = MergedTransformerNoCache(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, max_seq_len=SEQ_LEN, merge_ratio=MERGE_RATIO,
        dropout=0.1, num_classes=NUM_CLASSES
    ).to(device)
    params_nc = sum(p.numel() for p in model_no_cache.parameters())
    print(f"Params: {params_nc:,}")
    acc_no_cache = train_and_eval(model_no_cache, X_train, y_train, X_val, y_val, device, EPOCHS, LR, BS)

    # Summary
    print()
    print("=" * 60)
    print("ИТОГИ:")
    print("=" * 60)
    print(f"Merged Transformer + Residual Cache: {acc_merged:.3f}  (params: {params:,})")
    print(f"Standard Transformer:                {acc_std:.3f}  (params: {params_std:,})")
    print(f"Merged Transformer БЕЗ Cache:        {acc_no_cache:.3f}  (params: {params_nc:,})")
    print()
    print(f"Residual Cache эффект:   {acc_merged - acc_no_cache:+.3f}")
    print(f"Merge vs Standard:       {acc_merged - acc_std:+.3f}")
    print("=" * 60)


def main():
    print(f"Training on: {device}")
    print('=' * 60)
    print("Merged Token Transformer — testing the theory!")
    print('=' * 60)

    BATCH_SIZE = 64
    NUM_SAMPLES = 2000
    SEQ_LEN = 16
    VOCAB_SIZE = 50
    D_MODEL = 64
    N_HEADS = 4
    D_FF = 128
    N_LAYERS = 4
    MERGE_RATIO = 2
    NUM_CLASSES = 3
    LEARNING_RATE = 1e-3
    EPOCHS = 50

    X_train, y_train = create_simple_dataset(NUM_SAMPLES, SEQ_LEN, VOCAB_SIZE, NUM_CLASSES)
    X_val, y_val = create_simple_dataset(500, SEQ_LEN, VOCAB_SIZE, NUM_CLASSES)

    print(f"Dataset: {NUM_SAMPLES} samples, seq_len={SEQ_LEN}, classes={NUM_CLASSES}")
    print(f"d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")

    # Merged Transformer
    print("\n🧠 Merged Transformer...")
    model = MergedTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, max_seq_len=SEQ_LEN, merge_ratio=MERGE_RATIO,
        dropout=0.1, num_classes=NUM_CLASSES, use_cache=True
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        indices = torch.randperm(NUM_SAMPLES)
        train_loss = 0.0
        train_correct = 0
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            loss = train_step(model, X_train[batch_idx], y_train[batch_idx], optimizer, device)
            train_loss += loss * len(batch_idx)
            with torch.no_grad():
                logits = model(X_train[batch_idx].to(device))
                preds = logits.argmax(dim=1)
                train_correct += (preds == y_train[batch_idx].to(device)).sum().item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_acc = (val_logits.argmax(dim=1) == y_val.to(device)).float().mean().item()
        train_acc = train_correct / NUM_SAMPLES
        avg_train_loss = train_loss / NUM_SAMPLES

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_train_loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")

    with torch.no_grad():
        val_logits = model(X_val.to(device))
        final_val_acc = (val_logits.argmax(dim=1) == y_val.to(device)).float().mean().item()
    print(f"\nMerged Transformer Val Acc: {final_val_acc:.3f}")

    # Standard Transformer comparison
    print("\n📊 Standard Transformer (baseline)...")
    std_model = StandardTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, max_seq_len=SEQ_LEN, dropout=0.1, num_classes=NUM_CLASSES
    ).to(device)
    std_params = sum(p.numel() for p in std_model.parameters())
    print(f"Params: {std_params:,}")

    std_optim = torch.optim.Adam(std_model.parameters(), lr=LEARNING_RATE)
    std_sched = torch.optim.lr_scheduler.CosineAnnealingLR(std_optim, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        std_model.train()
        indices = torch.randperm(NUM_SAMPLES)
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            train_step(std_model, X_train[batch_idx], y_train[batch_idx], std_optim, device)
        std_sched.step()

    std_model.eval()
    with torch.no_grad():
        val_logits_std = std_model(X_val.to(device))
        std_val_acc = (val_logits_std.argmax(dim=1) == y_val.to(device)).float().mean().item()

    print(f"Standard Transformer Val Acc: {std_val_acc:.3f}")
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Merged Token Transformer: {final_val_acc:.3f}  (params: {total_params:,})")
    print(f"  Standard Transformer:     {std_val_acc:.3f}  (params: {std_params:,})")
    print(f"  Diff: {final_val_acc - std_val_acc:+.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_comprehensive_test()
