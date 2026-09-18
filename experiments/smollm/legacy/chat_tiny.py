
import argparse
from pathlib import Path
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser()
    from experiments.smollm.smollm_model import MergedSmolLM
    parser.add_argument('--variant', choices=['baseline', 'plain', 'residual'], required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--dir', default='out-tiny-smollm')
    parser.add_argument('--length', type=int, default=50, help='Max tokens to generate')
    args = parser.parse_args()

    ckpt_path = Path(args.dir) / f"{args.variant}-{args.seed}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading {args.variant} (seed {args.seed})...")

    # Initialize tiny config just like in pretrain_tiny.py
    config = AutoConfig.from_pretrained('models/SmolLM2-135M', local_files_only=True)
    config.num_hidden_layers = 8
    config.hidden_size = 288
    config.intermediate_size = 768
    config.num_attention_heads = 6
    config.num_key_value_heads = 3
    base = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa')

    # Create wrapper model
    merge_layer = 4
    model = MergedSmolLM(base, args.variant, merge_layer).cuda()

    # Load weights
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('models/SmolLM2-135M', local_files_only=True)

    print("\nModel loaded! Note: This model is very small and only trained on WikiText-2 for a few minutes.")
    print("It doesn't know how to chat, but it will try to autocomplete your sentences.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue

            input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

            with torch.autocast('cuda', dtype=torch.bfloat16):
                generated_ids = model.generate_cached(input_ids, args.length)

            output_text = tokenizer.decode(generated_ids[0].cpu().tolist())
            print(f"\n{output_text}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
