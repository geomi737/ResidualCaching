# isort: skip_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if True:
    from experiments.smollm.sliding_model import SlidingSmolLM
    from experiments.smollm.smollm_model import endpoints



@torch.no_grad()
def run_generation(args):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        args.device = 'cuda'

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load Models
    print("Loading Baseline Model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16).to(device)
    base_model.eval()

    print("Loading Compressed Student Model...")
    student_base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    if args.merge_layer == -1:
        args.merge_layer = student_base.config.num_hidden_layers // 2
    student_model = SlidingSmolLM(student_base, variant=args.variant, merge_layer=args.merge_layer).to(device)

    # Optionally load adapted weights
    ckpt_path = Path(args.output) / f'checkpoint_{args.variant}.pt'
    if ckpt_path.exists():
        print(f"Loading weights from {ckpt_path}")
        student_model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False)['model_state_dict'])
    student_model.eval()

    prompts = [
        "The most important lesson in history is that",
        "Once upon a time in a digital kingdom,",
        "def quicksort(arr):",
    ]

    def generate_baseline(model, prompt, max_new=50):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0])

    def generate_sliding(model, prompt, max_new=50):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate_cached(inputs.input_ids, max_new)
        return tokenizer.decode(out[0])

    for p in prompts:
        print("=" * 50)
        print("PROMPT:", p)
        print("-" * 50)
        b_out = generate_baseline(base_model, p)
        print("BASELINE:\n", b_out)
        print("-" * 50)
        s_out = generate_sliding(student_model, p)
        print("SLIDING:\n", s_out)
        print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/SmolLM2-135M')
    parser.add_argument('--variant', choices=['plain', 'residual'], default='plain')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--merge-layer', type=int, default=-1)
    parser.add_argument('--output', default='experiments/smollm-retrain-for-context-compression/results')
    args = parser.parse_args()
    run_generation(args)
