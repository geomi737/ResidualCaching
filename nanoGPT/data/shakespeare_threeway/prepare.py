"""Prepare disjoint 80/10/10 Tiny Shakespeare BPE splits before tokenization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
import tiktoken

SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).parents[1] / 'shakespeare/input.txt')
    args = parser.parse_args()
    if args.source.exists():
        raw = args.source.read_bytes()
    else:
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
        raw = response.content
    text = raw.decode('utf-8')
    cut1, cut2 = int(len(text) * 0.8), int(len(text) * 0.9)
    bounds = [(0, cut1), (cut1, cut2), (cut2, len(text))]
    enc = tiktoken.get_encoding('gpt2')
    output = Path(__file__).parent
    manifest = {'source_url': SOURCE, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'tokenizer': 'tiktoken/gpt2', 'split_method': 'contiguous raw-text 80/10/10 before tokenization',
                'splits': {}}
    for split, (start, end) in zip(('train', 'val', 'test'), bounds):
        tokens = np.asarray(enc.encode_ordinary(text[start:end]), dtype=np.uint16)
        tokens.tofile(output / f'{split}.bin')
        manifest['splits'][split] = {'character_start': start, 'character_end': end,
                                     'tokens': len(tokens), 'sha256': hashlib.sha256(tokens).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
