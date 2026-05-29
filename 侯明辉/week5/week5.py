"""
基于 Transformer Decoder-Only 的单向语言模型
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ─────────────────────── 配置 ───────────────────────

CONFIG = {
    # 训练参数
    "mode": "train",           # "train" / "generate"
    "corpus": __file__.replace("\\", "/").rsplit("/", 1)[0] + "/corpus.txt",   # 语料文件路径
    "epochs": 30,
    "seq_len": 128,
    "batch_size": 64,
    "d_model": 256,
    "num_heads": 8,
    "d_ff": 1024,
    "num_layers": 4,
    "dropout": 0.1,
    "lr": 3e-4,
    "val_ratio": 0.05,
    "save": "gpt_lm.pt",
    "grad_accum": 1,
    # 生成参数
    "model_path": "gpt_lm.pt",
    "prompt": "春天来了",
    "length": 200,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.0,
    "compare": False,
}


# ─────────────────────── 1. 因果多头自注意力 ───────────────────────

class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力，每个位置只能看到自己和之前的位置"""

    def __init__(self, d_model, num_heads, max_len=512, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # 因果掩码：下三角矩阵
        mask = torch.tril(torch.ones(max_len, max_len)).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        causal_mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.resid_drop(self.out_proj(out))
        return out


# ─────────────────────── 2. Transformer Block ───────────────────────

class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, d_model, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ─────────────────────── 3. GPT 语言模型 ───────────────────────

class GPTLanguageModel(nn.Module):
    """单向语言模型"""

    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 d_ff=1024, num_layers=4, max_len=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_len, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享
        self.token_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_len

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ─────────────────────── 4. 数据处理 ───────────────────────

def load_corpus(file_path):
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────── 5. 训练 ───────────────────────

def train(cfg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")

    text = load_corpus(cfg["corpus"])
    if not text:
        raise FileNotFoundError(f"未找到语料文件: {cfg['corpus']}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - cfg["val_ratio"]))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, cfg["seq_len"])
    val_ds = CharDataset(val_text, char2idx, cfg["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                               shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                             shuffle=False, drop_last=True, pin_memory=True)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        d_model=cfg["d_model"], num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"], num_layers=cfg["num_layers"],
        max_len=cfg["seq_len"] + 1, dropout=cfg["dropout"],
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)

    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = min(500, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    accum_steps = cfg["grad_accum"]
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}  {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_tokens = 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{cfg['epochs']}", unit="batch")
        for step_idx, (x, y) in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss = loss / accum_steps

            loss.backward()

            if (step_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps * y.numel()
            train_tokens += y.numel()

            pbar.set_postfix({
                "loss": f"{loss.item() * accum_steps:.4f}",
                "ppl": f"{math.exp(min(loss.item() * accum_steps, 20)):.2f}"
            })

        avg_train_loss = train_loss / max(train_tokens, 1)
        train_ppl = math.exp(min(avg_train_loss, 20))

        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val {epoch}/{cfg['epochs']}", unit="batch"):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item() * y.numel()
                val_tokens += y.numel()

        avg_val_loss = val_loss / max(val_tokens, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))

        current_lr = scheduler.get_last_lr()[0]
        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "config": cfg,
            }, cfg["save"])

        print(f"{epoch:>6}  {avg_train_loss:>10.4f}  {train_ppl:>10.2f}  "
              f"{avg_val_loss:>10.4f}  {val_ppl:>10.2f}  {current_lr:>10.6f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {cfg['save']}")


# ─────────────────────── 6. 文本生成 ───────────────────────

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.0):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.max_len else idx[:, -model.max_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs - sorted_probs > top_p
            sorted_logits[remove_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


def generate_text(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg["model_path"], map_location=device, weights_only=False)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    saved_cfg = ckpt["config"]

    model = GPTLanguageModel(
        vocab_size=len(char2idx),
        d_model=saved_cfg["d_model"], num_heads=saved_cfg["num_heads"],
        d_ff=saved_cfg["d_ff"], num_layers=saved_cfg["num_layers"],
        max_len=saved_cfg.get("seq_len", 128) + 1, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    prompt_chars = [c for c in cfg["prompt"] if c in char2idx]
    if not prompt_chars:
        print(f"提示词 '{cfg['prompt']}' 中的字符不在词表中")
        return

    idx = torch.tensor([[char2idx[c] for c in prompt_chars]], dtype=torch.long, device=device)

    print(f"\n{'=' * 60}")
    print(f"  提示词: {''.join(prompt_chars)}")
    print(f"  生成长度: {cfg['length']} | temperature: {cfg['temperature']} | top_k: {cfg['top_k']}")
    print(f"{'=' * 60}")

    output_ids = generate(model, idx, max_new_tokens=cfg["length"],
                          temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=cfg["top_p"])

    generated = "".join([idx2char[i] for i in output_ids[0].cpu().tolist()])
    print(f"\n{generated}\n")

    if cfg["compare"]:
        print(f"\n{'=' * 60}")
        print("  不同 temperature 对比:")
        print(f"{'=' * 60}")
        for temp in [0.3, 0.5, 0.8, 1.0, 1.2]:
            output_ids = generate(model, idx, max_new_tokens=cfg["length"],
                                  temperature=temp, top_k=cfg["top_k"], top_p=cfg["top_p"])
            text = "".join([idx2char[i] for i in output_ids[0].cpu().tolist()])
            print(f"\n[temperature={temp}]")
            print(text)


# ─────────────────────── 主函数 ───────────────────────

if __name__ == "__main__":
    mode = CONFIG["mode"]

    if mode == "train":
        train(CONFIG)
    elif mode == "generate":
        generate_text(CONFIG)