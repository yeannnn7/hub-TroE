"""
Week05：训练一个基于 Transformer 的单向语言模型,并完成文本生成。
要求：
1) 训练基于 transformer 的单向语言模型，并完成文本生成
2) 训练数据可以从网上下载（本脚本默认下载 tinyshakespeare）

运行示例：
python week05.py --steps 200 --generate_len 200
"""

import argparse  # 导入命令行参数解析库，用于从命令行配置训练参数。
import os  # 导入操作系统接口库，用于路径与文件判断。
import random  # 导入随机库，用于设置随机种子与随机采样。
import time  # 导入时间库，用于统计训练耗时。
import urllib.request  # 导入标准库下载工具，用于从网络下载训练数据。
from dataclasses import dataclass  # 导入 dataclass，用于写更清晰的配置类。

import torch  # 导入 PyTorch 主库。
import torch.nn as nn  # 导入 PyTorch 神经网络模块。
import torch.nn.functional as F  # 导入函数式 API（softmax、交叉熵等）。
from torch.utils.data import DataLoader, Dataset  # 导入数据集与数据加载器工具。


def set_seed(seed: int) -> None:  # 定义设置随机种子的函数，保证可复现性。
    random.seed(seed)  # 设置 Python random 的种子。
    torch.manual_seed(seed)  # 设置 PyTorch CPU 的随机种子。
    torch.cuda.manual_seed_all(seed)  # 设置 PyTorch GPU 的随机种子（如果有 GPU）。


def download_text(url: str, save_path: str) -> str:  # 定义下载文本数据的函数，返回保存后的文件路径。
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 若目录不存在则创建目录。
    if not os.path.exists(save_path):  # 如果文件不存在则执行下载。
        urllib.request.urlretrieve(url, save_path)  # 从 url 下载文件并保存到本地。
    return save_path  # 返回本地文件路径。


def read_text(path: str) -> str:  # 定义读取文本文件的函数。
    with open(path, "r", encoding="utf-8") as f:  # 以 UTF-8 编码打开文本文件。
        text = f.read()  # 读取全部文本内容。
    return text  # 返回文本字符串。


def build_vocab(text: str):  # 定义构建字符级词表的函数（最简单，便于学习）。
    chars = sorted(list(set(text)))  # 取出文本出现过的所有字符并排序，得到字符表。
    stoi = {ch: i for i, ch in enumerate(chars)}  # 构建字符到整数 id 的映射（string-to-int）。
    itos = {i: ch for i, ch in enumerate(chars)}  # 构建整数 id 到字符的映射（int-to-string）。
    return stoi, itos  # 返回两张映射表。


def encode(text: str, stoi: dict) -> torch.Tensor:  # 定义把字符串编码成整数张量的函数。
    ids = [stoi[ch] for ch in text]  # 把每个字符映射为对应的整数 id。
    return torch.tensor(ids, dtype=torch.long)  # 返回 long 类型张量（embedding 需要 long）。


def decode(ids: torch.Tensor, itos: dict) -> str:  # 定义把整数 id 张量解码回字符串的函数。
    return "".join(itos[int(i)] for i in ids)  # 把每个 id 映射回字符并拼接成字符串。


class CharDataset(Dataset):  # 定义字符级数据集：给定长序列，切出 (x, y) 训练对。
    def __init__(self, data: torch.Tensor, block_size: int):  # 初始化数据集，data 是编码后的整段文本。
        self.data = data  # 保存数据张量（1D，长度为 N）。
        self.block_size = block_size  # 保存上下文长度（模型每次看到的 token 数）。

    def __len__(self) -> int:  # 定义数据集长度。
        return max(0, self.data.numel() - self.block_size - 1)  # 可切片的起点数，避免越界。

    def __getitem__(self, idx: int):  # 定义取出第 idx 个样本的方法。
        x = self.data[idx : idx + self.block_size]  # 输入序列：长度为 block_size。
        y = self.data[idx + 1 : idx + self.block_size + 1]  # 目标序列：右移一位（预测下一个 token）。
        return x, y  # 返回 (输入, 目标)。


@dataclass  # 使用 dataclass 让配置更清晰。
class ModelConfig:  # 定义模型配置类。
    vocab_size: int  # 词表大小（字符种类数）。
    block_size: int  # 上下文长度。
    n_layer: int = 4  # Transformer block 的层数。
    n_head: int = 4  # 注意力头数。
    n_embd: int = 128  # 隐藏维度（embedding 维度）。
    dropout: float = 0.1  # dropout 比例。


class CausalSelfAttention(nn.Module):  # 定义带因果掩码的自注意力（单向语言模型核心）。
    def __init__(self, cfg: ModelConfig):  # 初始化注意力层。
        super().__init__()  # 调用父类初始化。
        assert cfg.n_embd % cfg.n_head == 0  # 保证隐藏维度可被头数整除。
        self.n_head = cfg.n_head  # 保存头数。
        self.head_dim = cfg.n_embd // cfg.n_head  # 计算每个头的维度。
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # 一次性投影出 Q、K、V。
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)  # 输出投影层，把多头拼接结果映射回 hidden。
        self.attn_drop = nn.Dropout(cfg.dropout)  # 注意力权重 dropout。
        self.resid_drop = nn.Dropout(cfg.dropout)  # 残差路径 dropout。
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))  # 构造下三角矩阵：只看当前位置及之前。
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))  # 注册为 buffer，随设备移动。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播：输入 x 为 [B, T, C]。
        B, T, C = x.shape  # 取出 batch、序列长度、通道维度。
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 线性映射并切分得到 Q、K、V。
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # 变形为 [B, nh, T, hd]。
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # 变形为 [B, nh, T, hd]。
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # 变形为 [B, nh, T, hd]。
        att = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # 计算缩放点积注意力分数 [B, nh, T, T]。
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))  # 应用因果 mask：未来位置设为 -inf。
        att = F.softmax(att, dim=-1)  # 对最后一维做 softmax 得到注意力权重。
        att = self.attn_drop(att)  # 对注意力权重做 dropout。
        y = att @ v  # 使用注意力权重对 V 加权求和，得到输出 [B, nh, T, hd]。
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 把多头拼接回 [B, T, C]。
        y = self.resid_drop(self.proj(y))  # 输出投影并在残差路径上 dropout。
        return y  # 返回注意力层输出。


class MLP(nn.Module):  # 定义前馈网络（Transformer 的 FFN 部分）。
    def __init__(self, cfg: ModelConfig):  # 初始化 FFN。
        super().__init__()  # 调用父类初始化。
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)  # 第一层线性：扩展维度（常用 4 倍）。
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)  # 第二层线性：映射回隐藏维度。
        self.drop = nn.Dropout(cfg.dropout)  # dropout 层。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播。
        x = self.fc1(x)  # 线性变换。
        x = F.gelu(x)  # GELU 激活（GPT 常用）。
        x = self.fc2(x)  # 再线性变换回原维度。
        x = self.drop(x)  # dropout 正则化。
        return x  # 返回 FFN 输出。


class Block(nn.Module):  # 定义一个 Transformer Block：LN + Attention + LN + MLP（含残差）。
    def __init__(self, cfg: ModelConfig):  # 初始化 block。
        super().__init__()  # 调用父类初始化。
        self.ln1 = nn.LayerNorm(cfg.n_embd)  # 第一处 LayerNorm（pre-norm 结构）。
        self.attn = CausalSelfAttention(cfg)  # 因果自注意力子层。
        self.ln2 = nn.LayerNorm(cfg.n_embd)  # 第二处 LayerNorm。
        self.mlp = MLP(cfg)  # 前馈网络子层。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播。
        x = x + self.attn(self.ln1(x))  # 先 LN，再注意力，再做残差相加。
        x = x + self.mlp(self.ln2(x))  # 再 LN，再 MLP，再做残差相加。
        return x  # 返回 block 输出。


class GPTLikeLM(nn.Module):  # 定义一个最小可训练的 GPT 风格单向语言模型。
    def __init__(self, cfg: ModelConfig):  # 初始化语言模型。
        super().__init__()  # 调用父类初始化。
        self.cfg = cfg  # 保存配置。
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embedding：把 token id 映射为向量。
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # 位置 embedding：为 0..T-1 的位置编码。
        self.drop = nn.Dropout(cfg.dropout)  # embedding 后的 dropout。
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])  # 堆叠多个 Transformer block。
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # 最后的 LayerNorm。
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # 语言模型头：映射到词表 logits。

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):  # 前向：idx [B,T]，targets [B,T] 可选。
        B, T = idx.shape  # 获取 batch 与序列长度。
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)  # 构造位置索引 [T]。
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]  # token embedding + position embedding 相加。
        x = self.drop(x)  # dropout。
        for block in self.blocks:  # 依次通过每个 Transformer block。
            x = block(x)  # 更新隐藏状态。
        x = self.ln_f(x)  # 最后做 LayerNorm。
        logits = self.head(x)  # 得到每个位置对下一个 token 的预测 logits：[B,T,V]。
        loss = None  # 默认不计算 loss。
        if targets is not None:  # 如果给了 targets，就计算交叉熵损失用于训练。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # 把 [B,T,V] 展平成二维计算 loss。
        return logits, loss  # 返回 logits 与 loss。

    @torch.no_grad()  # 生成时不需要梯度，减少显存与加速。
    def generate(  # 定义文本生成函数：给定起始上下文，自回归生成后续 token。
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()  # 切换到 eval 模式（关闭 dropout）。
        for _ in range(max_new_tokens):  # 循环生成 max_new_tokens 个 token。
            idx_cond = idx[:, -self.cfg.block_size :]  # 只保留最后 block_size 个 token 作为上下文。
            logits, _ = self(idx_cond, targets=None)  # 前向得到 logits。
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # 取最后一个位置的 logits，并做温度缩放。
            if top_k is not None:  # 如果启用 top-k 截断采样。
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))  # 取出每行 top-k 的阈值。
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))  # 把非 top-k 的 logits 置为 -inf。
            probs = F.softmax(logits, dim=-1)  # softmax 得到概率分布。
            next_id = torch.multinomial(probs, num_samples=1)  # 从分布中采样下一个 token id。
            idx = torch.cat([idx, next_id], dim=1)  # 将新 token 拼接到序列末尾。
        return idx  # 返回生成后的 token 序列。


def main() -> None:  # 定义主函数。
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器。
    parser.add_argument("--data_url", type=str, default="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")  # 数据下载地址。
    parser.add_argument("--data_path", type=str, default=os.path.join("data", "tinyshakespeare.txt"))  # 数据本地保存路径。
    parser.add_argument("--seed", type=int, default=1337)  # 随机种子。
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")  # 训练设备。
    parser.add_argument("--block_size", type=int, default=128)  # 上下文长度。
    parser.add_argument("--batch_size", type=int, default=64)  # batch 大小。
    parser.add_argument("--n_layer", type=int, default=4)  # Transformer 层数。
    parser.add_argument("--n_head", type=int, default=4)  # 注意力头数。
    parser.add_argument("--n_embd", type=int, default=128)  # embedding/隐藏维度。
    parser.add_argument("--dropout", type=float, default=0.1)  # dropout。
    parser.add_argument("--lr", type=float, default=3e-4)  # 学习率。
    parser.add_argument("--steps", type=int, default=200)  # 训练步数（默认设置较小，便于本机快速跑通）。
    parser.add_argument("--log_interval", type=int, default=50)  # 打印日志间隔（步）。
    parser.add_argument("--generate_len", type=int, default=200)  # 训练过程中生成文本的长度。
    parser.add_argument("--temperature", type=float, default=1.0)  # 采样温度。
    parser.add_argument("--top_k", type=int, default=50)  # top-k 采样参数。
    args = parser.parse_args()  # 解析命令行参数。

    set_seed(args.seed)  # 设置随机种子，保证复现。
    device = torch.device(args.device)  # 构造设备对象。

    data_file = download_text(args.data_url, args.data_path)  # 下载数据（如果本地没有）。
    text = read_text(data_file)  # 读取文本。
    stoi, itos = build_vocab(text)  # 构建字符词表。
    data = encode(text, stoi)  # 把整段文本编码成整数序列。

    split = int(0.9 * data.numel())  # 计算训练/验证切分点（90% 训练）。
    train_data = data[:split]  # 取训练部分。
    val_data = data[split:]  # 取验证部分（本脚本主要用于演示，可以不严格评估）。

    train_ds = CharDataset(train_data, block_size=args.block_size)  # 构建训练数据集。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)  # 构建训练 DataLoader。

    cfg = ModelConfig(  # 组装模型配置。
        vocab_size=len(stoi),  # 设置词表大小。
        block_size=args.block_size,  # 设置上下文长度。
        n_layer=args.n_layer,  # 设置层数。
        n_head=args.n_head,  # 设置头数。
        n_embd=args.n_embd,  # 设置隐藏维度。
        dropout=args.dropout,  # 设置 dropout。
    )  # 配置定义结束。

    model = GPTLikeLM(cfg).to(device)  # 创建模型并移动到指定设备。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 使用 AdamW 优化器（Transformer 常用）。

    model.train()  # 切换到训练模式。
    t0 = time.time()  # 记录开始时间。

    it = iter(train_loader)  # 构造一个迭代器，用于按 step 取 batch。
    for step in range(1, args.steps + 1):  # 训练主循环：从 1 到 steps。
        try:  # 尝试从迭代器取一个 batch。
            x, y = next(it)  # 获取输入序列和目标序列。
        except StopIteration:  # 如果迭代器走完一轮数据。
            it = iter(train_loader)  # 重新创建迭代器开始下一轮。
            x, y = next(it)  # 再取一个 batch。

        x = x.to(device)  # 把输入移动到设备。
        y = y.to(device)  # 把目标移动到设备。

        logits, loss = model(x, targets=y)  # 前向传播得到 loss。
        optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none 更快更省显存）。
        loss.backward()  # 反向传播计算梯度。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸。
        optimizer.step()  # 更新参数。

        if step % args.log_interval == 0 or step == 1:  # 按间隔打印训练信息。
            elapsed = time.time() - t0  # 计算耗时。
            print(f"[step {step:>5d}/{args.steps}] loss={loss.item():.4f} time={elapsed:.1f}s")  # 打印 loss 与耗时。
            prompt = "To be, or not to be"  # 设置一个简单的起始 prompt。
            prompt_ids = encode(prompt, stoi).unsqueeze(0).to(device)  # 编码 prompt 并加上 batch 维度。
            out_ids = model.generate(  # 调用模型生成函数。
                prompt_ids,  # 输入 prompt token。
                max_new_tokens=args.generate_len,  # 生成长度。
                temperature=args.temperature,  # 温度。
                top_k=args.top_k,  # top-k。
            )  # 生成结束。
            out_text = decode(out_ids[0].cpu(), itos)  # 解码回文本。
            print("-" * 80)  # 打印分隔线。
            print(out_text)  # 打印生成的文本。
            print("-" * 80)  # 打印分隔线。

    model.eval()  # 训练结束后切换到 eval 模式。
    with torch.no_grad():  # 关闭梯度计算。
        x0 = val_data[: args.block_size].unsqueeze(0).to(device)  # 取验证集前 block_size 个 token 作为 prompt。
        y0 = model.generate(x0, max_new_tokens=200, temperature=0.9, top_k=args.top_k)  # 生成一些文本用于查看效果。
        print("Final sample:")  # 打印提示语。
        print(decode(y0[0].cpu(), itos))  # 打印最终生成样例。

    summary = (  # 归纳总结：用字符串集中输出，便于学习复盘。
        "\n学习总结（抓住主线）：\n"
        "1) 单向语言模型（GPT 风格）训练目标：给定 x[0..t] 预测下一个 token y[t]，所以数据是 (x, x右移一位)。\n"
        "2) Causal Mask：自注意力里用下三角矩阵，保证位置 t 只能看 <=t 的信息，不能偷看未来。\n"
        "3) Transformer Block（pre-norm）：x = x + Attn(LN(x))；x = x + MLP(LN(x))。\n"
        "4) 输出层：把每个位置的隐藏状态映射成 vocab logits，用 cross_entropy 训练。\n"
        "5) 生成（generate）：循环做 '前向 -> 取最后位置 logits -> softmax -> 采样 -> 拼接'。\n"
        "6) Char-level 优点：实现最简单；缺点：序列更长、效果有限；真实项目常用 BPE/WordPiece。\n"
        "建议练习：调大 steps / n_layer / n_embd；加入验证 loss；保存/加载模型；换成词级或 BPE 分词。\n"
    )  # 总结文本结束。
    print(summary)  # 打印总结。


if __name__ == "__main__":  # Python 入口：直接运行此文件会执行 main。
    main()  # 调用主函数。
