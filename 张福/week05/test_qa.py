"""
问答模型测试脚本

加载训练好的模型进行问答测试。
"""

import torch
from transformer import Transformer
from train_qa import Vocabulary


def load_model(model_path="ask_answer.pt"):
    """加载训练好的模型

    Args:
        model_path (str): 模型文件路径

    Returns:
        tuple: (模型, 词汇表, 配置字典)
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    vocab = checkpoint["vocab"]
    model = Transformer(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        ff_dim=checkpoint["ff_dim"],
        num_layers=checkpoint["num_layers"],
        max_len=checkpoint["max_len"],
        dropout=0.1
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = {
        "vocab_size": checkpoint["vocab_size"],
        "embed_dim": checkpoint["embed_dim"],
        "num_heads": checkpoint["num_heads"],
        "ff_dim": checkpoint["ff_dim"],
        "num_layers": checkpoint["num_layers"],
        "max_len": checkpoint["max_len"]
    }

    return model, vocab, config


def beam_search(model, vocab, input_text, max_len=100, beam_size=3, device='cpu'):
    """使用Beam Search生成回答

    Beam Search核心原理：
    1. 维护一个大小为beam_size的候选序列集合
    2. 每一步对每个候选序列扩展所有可能的下一个token
    3. 计算每个扩展序列的得分（对数概率）
    4. 选择得分最高的beam_size个序列作为下一步的候选
    5. 重复直到遇到<EOS>或达到最大长度

    Args:
        model: 训练好的模型
        vocab: 词汇表
        input_text: 输入问题
        max_len: 最大生成长度
        beam_size: beam宽度
        device: 设备

    Returns:
        str: 生成的回答
    """
    model = model.to(device)
    model.eval()

    end_token_idx = vocab.char2idx["<EOS>"]
    pad_token_idx = vocab.char2idx["<PAD>"]

    input_indices = vocab.encode(input_text) + [vocab.char2idx["<SEP>"]]
    
    beams = [(input_indices, 0.0)]
    completed = []

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                if seq and seq[-1] == end_token_idx:
                    completed.append((seq, score))
                    continue
                
                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                output = model(input_tensor)
                
                last_token_logits = output[0, -1, :]
                probs = torch.softmax(last_token_logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                
                top_k = torch.topk(log_probs, beam_size)
                top_indices = top_k.indices.tolist()
                top_scores = top_k.values.tolist()
                
                for idx, s in zip(top_indices, top_scores):
                    if idx != pad_token_idx:
                        new_seq = seq + [idx]
                        new_score = score + s
                        new_beams.append((new_seq, new_score))
            
            new_beams.sort(key=lambda x: -x[1])
            beams = new_beams[:beam_size]
            
            if not beams:
                break
            
            if all(seq[-1] == end_token_idx for seq, _ in beams):
                break

        completed.extend(beams)
        completed.sort(key=lambda x: -x[1])
        
        if completed:
            best_seq = completed[0][0][len(input_indices):]
            if best_seq and best_seq[-1] == end_token_idx:
                best_seq = best_seq[:-1]
            answer = vocab.decode(best_seq)
        else:
            answer = ""

    return answer


def generate_answer(model, vocab, input_text, max_len=100, beam_size=3, device='cpu'):
    """生成回答（使用Beam Search）

    Args:
        model: 训练好的模型
        vocab: 词汇表
        input_text: 输入问题
        max_len: 最大生成长度
        beam_size: beam宽度（默认为3）
        device: 设备

    Returns:
        str: 生成的回答
    """
    return beam_search(model, vocab, input_text, max_len, beam_size, device)


def interactive_test(model, vocab, device='cpu'):
    """交互式测试

    Args:
        model: 训练好的模型
        vocab: 词汇表
        device: 设备
    """
    print("\n" + "=" * 60)
    print("问答系统测试 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入问题: ").strip()

            if question.lower() == 'quit':
                print("感谢使用，再见！")
                break

            if not question:
                print("问题不能为空，请重新输入！")
                continue

            answer = generate_answer(model, vocab, question, device=device)

            print(f"回答: {answer}")

        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def batch_test(model, vocab, test_questions, device='cpu'):
    """批量测试

    Args:
        model: 训练好的模型
        vocab: 词汇表
        test_questions: 测试问题列表
        device: 设备
    """
    print("\n" + "=" * 60)
    print("批量测试结果")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        answer = generate_answer(model, vocab, question, device=device)
        print(f"\n问题 {i}: {question}")
        print(f"回答:   {answer}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_path = "ask_answer.pt"

    try:
        model, vocab, config = load_model(model_path)
        print(f"\n模型加载成功！")
        print(f"词汇表大小: {config['vocab_size']}")
        print(f"嵌入维度: {config['embed_dim']}")
        print(f"注意力头数: {config['num_heads']}")
        print(f"Transformer层数: {config['num_layers']}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{model_path}'")
        print("请先运行 train_qa.py 训练模型")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    test_questions = [
        "避险资金青睐依旧 黄金牛市难言终结",
        "□本报记者 熊锋",
        "谁为欧债背书？",
        "曹忠忠：如果希腊真的退出欧元区"
    ]

    batch_test(model, vocab, test_questions, device)

    print("\n" + "=" * 60)
    print("是否进入交互式测试？")
    print("=" * 60)
    choice = input("输入 'y' 进入交互式测试，其他键退出: ").strip().lower()

    if choice == 'y':
        interactive_test(model, vocab, device)


if __name__ == "__main__":
    main()