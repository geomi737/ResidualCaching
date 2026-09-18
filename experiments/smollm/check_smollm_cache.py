"""Compare incremental and full-prefix predictions on actual pretrained weights."""
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.smollm.smollm_model import MergedSmolLM

torch.set_num_threads(4)
data = np.memmap('data/smollm/validation.bin', dtype=np.uint16, mode='r')
rows = []
for variant in ('baseline', 'plain', 'residual'):
    model = MergedSmolLM.load('models/SmolLM2-135M', variant, 15).cuda().eval()
    for dtype in (torch.float32, torch.bfloat16):
        ids = torch.tensor(np.asarray(data[500:531]).astype(np.int64)[None], device='cuda')
        errors, agreements, kl = [], [], []
        context = torch.autocast('cuda', dtype=dtype) if dtype == torch.bfloat16 else contextlib.nullcontext()
        with torch.no_grad(), context:
            logits, state = model.prefill(ids)
            for step in range(8):
                full, _ = model(ids, last_only=True)
                errors.append((full.float() - logits.float()).abs().max().item())
                agreements.append(bool(full.argmax(-1).eq(logits.argmax(-1)).all()))
                kl.append(F.kl_div(logits.float().log_softmax(-1),
                                   full.float().softmax(-1), reduction='batchmean').item())
                if dtype == torch.float32:
                    torch.testing.assert_close(full, logits, atol=3e-4, rtol=3e-4)
                else:
                    # Different GEMM shapes cause rounding differences even in
                    # the baseline. Bound distribution error, not bit equality.
                    if kl[-1] > .02:
                        raise AssertionError(f'Unexpected BF16 cache divergence: {kl[-1]}')
                token = torch.tensor([[int(data[531 + step])]], device='cuda')
                ids = torch.cat((ids, token), 1)
                logits, state = model.decode(token, state)
        rows.append({'variant': variant, 'dtype': str(dtype), 'comparisons': len(errors),
                     'max_absolute_logit_error': max(errors), 'max_kl_full_to_cached': max(kl),
                     'top1_agreements': sum(agreements)})
    del model, state, logits, full
    torch.cuda.empty_cache()
Path('results/smollm_cuda_cache_check.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
