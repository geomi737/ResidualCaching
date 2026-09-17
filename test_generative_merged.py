"""
Прототип генеративной языковой модели с буферным сжатием (Sliding Window Token Merger)
и Residual Cache Amplification.

HISTORICAL PROTOTYPE: unmerging leaks future tokens into earlier predictions.
Do not use its training loss as evidence of causal LM quality. The supported
one-way causal implementation is nanoGPT/model.py, with regressions in tests/.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Causal Attention & FeedForward
# ============================================================================


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 2. Token Merger & Residual Cache Amplification
# ============================================================================

class WindowTokenMerger(nn.Module):
    """Сжимает группы токенов размера window_size в один вектор."""

    def __init__(self, d_model, window_size=2):
        super().__init__()
        self.window_size = window_size
        self.merge_weight = nn.Parameter(torch.ones(window_size) / window_size)

    def forward(self, x):
        B, L, D = x.shape
        assert L % self.window_size == 0
        num_windows = L // self.window_size

        # Сворачиваем в форму (B, num_windows, window_size, D)
        x_reshaped = x.view(B, num_windows, self.window_size, D)
        weights = F.softmax(self.merge_weight, dim=0).view(1, 1, self.window_size, 1)

        # 1. Взвешенный сжатый токен:
        merged = (x_reshaped * weights).sum(dim=2)  # (B, num_windows, D)

        # 2. Сумма участвовавших оригиналов для Residual Cache:
        group_sums = x_reshaped.sum(dim=2)  # (B, num_windows, D)

        # Также возвращаем сами исходные кусочки для по-позиционного unmerging
        return merged, group_sums, x_reshaped


# ============================================================================
# 3. Hourglass Generative Model с Residual Amplification
# ============================================================================

class GenerativeMergedLM(nn.Module):
    """
    Иерархическая генеративная модель:
    - Слой 1 (L): локальная обработка
    - Сжатие N -> 1 (L/N): глубокие слои думают на сжатом контексте!
    - Разжатие 1 -> N с прибавлением Residual Cache: генерация точных токенов
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=128, n_latent_layers=3, window_size=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)

        # Входной блок высокого разрешения (размер L)
        self.in_block = TransformerBlock(d_model, n_heads, d_ff)

        # Модуль сжатия
        self.merger = WindowTokenMerger(d_model, window_size)

        # Глубокие скрытые блоки (работают на сжатой в window_size раз длине!)
        self.latent_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_latent_layers)
        ])

        # Выходной блок разжатия (обратно размер L) с проекцией и Residual Cache
        self.out_block = TransformerBlock(d_model, n_heads, d_ff)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, L = idx.shape
        pad_len = 0
        if L % self.window_size != 0:
            pad_len = self.window_size - (L % self.window_size)
            idx = F.pad(idx, (0, pad_len), value=0)
            L = L + pad_len

        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # 1. Локальная обработка
        x = self.in_block(x)

        # 2. Сжатие в окно window_size (L -> L / window_size)
        merged, group_residual_sum, orig_pieces = self.merger(x)

        # 3. Глубокие слои на сжатом представлении + добавление Residual Cache
        # Твоя идея: прибавляем Residual Cache к сжатому вектору (Residual Amplification)
        latent = merged + group_residual_sum
        for block in self.latent_blocks:
            latent = block(latent)

        # 4. Развертка (Unmerging): дублируем сжатый контекст по размеру окна
        # и добавляем оригинальные residual кусочки каждого токена
        num_windows = L // self.window_size
        # (B, num_windows, 1, D) -> (B, num_windows, window_size, D)
        expanded_latent = latent.unsqueeze(2).repeat(1, 1, self.window_size, 1)
        unmerged = expanded_latent + orig_pieces  # Точное восстановление индивидуальности токенов!
        unmerged = unmerged.view(B, L, self.d_model)

        # 5. Выходной блок и логиты
        out = self.out_block(unmerged)
        logits = self.head(out)

        if pad_len > 0:
            logits = logits[:, :-pad_len, :]

        return logits


# ============================================================================
# 4. Демонстрация генерации с буфером (Sliding Buffer Merge)
# ============================================================================

def generate_with_buffer_demo(model, prompt_text, char2idx, idx2char, max_new_tokens=30, window_size=2):
    model.eval()
    print(f"\n--- Генерация с буферным сжатием (Window size = {window_size}) ---")
    print(f"Начальный промпт: '{prompt_text}'")
    print("=" * 65)

    input_ids = [char2idx[c] for c in prompt_text]
    generated_ids = list(input_ids)
    buffer = []

    for step in range(1, max_new_tokens + 1):
        x = torch.tensor([generated_ids], device=device)
        with torch.no_grad():
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits / 0.8, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        next_char = idx2char[next_id]
        generated_ids.append(next_id)
        buffer.append(next_char)

        # Логика буфера
        if len(buffer) < window_size:
            print(f"Шаг {step:2d}: Токен '{next_char}' -> Буфер не полон: {buffer} (ожидает {window_size - len(buffer)} токен)")
        else:
            print(f"Шаг {step:2d}: Токен '{next_char}' -> БУФЕР ЗАПОЛНЕН: {buffer} -> ⚡ Сжатие {window_size} токенов в 1 сжатый вектор!")
            buffer.clear()

    full_text = "".join([idx2char[i] for i in generated_ids])
    print("=" * 65)
    print(f"Итоговый сгенерированный текст:\n'{full_text}'\n")


# ============================================================================
# 5. Обучение на коротком тексте и запуск
# ============================================================================

def main():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Language models compress context to reason efficiently and accurately. "
        "Token merging reduces memory usage while residual cache keeps fine details. "
    ) * 35

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}

    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    print(f"Датасет: {len(text)} символов, размер словаря: {vocab_size}")

    model = GenerativeMergedLM(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_latent_layers=3,
        window_size=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    seq_len = 32
    batch_size = 32

    print("Обучение генеративной модели...")
    model.train()
    for ep in range(1, 151):
        # Случайные батчи
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 30 == 0:
            print(f"  Эпоха {ep:3d}/150 | Loss: {loss.item():.4f}")

    # Запуск генерации с демонстрацией работы окна/буфера
    generate_with_buffer_demo(model, "The quick ", char2idx, idx2char, max_new_tokens=24, window_size=2)


if __name__ == "__main__":
    main()
