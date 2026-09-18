"""Tokenize official WikiText-2 splits without mixing held-out articles."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path('data/smollm')
    raw = root / 'wikitext'
    tokenizer = AutoTokenizer.from_pretrained('models/SmolLM2-135M', local_files_only=True)
    manifest = {'dataset': 'Salesforce/wikitext', 'configuration': 'wikitext-2-raw-v1',
                'tokenizer': 'HuggingFaceTB/SmolLM2-135M',
                'tokenizer_revision': '93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
                'note': 'Official held-out splits with exact-article deduplication. Scratch runs load no pretrained model weights.',
                'splits': {}}
    seen = set()
    for split in ('test', 'validation', 'train'):
        path = raw / 'wikitext-2-raw-v1' / f'{split}-00000-of-00001.parquet'
        rows = pq.read_table(path, columns=['text'])['text'].to_pylist()
        articles, current = [], []
        for row in rows:
            if re.fullmatch(r'\s*= [^=\n]+ =\s*', row) and current:
                articles.append(''.join(current))
                current = []
            current.append(row)
        if current:
            articles.append(''.join(current))
        tokens, removed = [], 0
        for article in articles:
            digest = sha(article.strip().encode())
            if not article.strip() or digest in seen:
                removed += 1
                continue
            seen.add(digest)
            tokens.extend(tokenizer.encode(article, add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
        array = np.asarray(tokens, dtype=np.uint16)
        array.tofile(root / f'{split}.bin')
        metadata = raw / '.cache/huggingface/download/wikitext-2-raw-v1' / f'{split}-00000-of-00001.parquet.metadata'
        manifest['splits'][split] = {
            'tokens': len(tokens), 'source_sha256': sha(path.read_bytes()),
            'tokens_sha256': sha(array.tobytes()),
            'source_revision': metadata.read_text().splitlines()[0],
            'article_count_before_dedup': len(articles), 'removed_articles': removed,
        }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
