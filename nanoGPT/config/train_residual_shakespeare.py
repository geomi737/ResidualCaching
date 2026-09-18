# Start from Karpathy's compact character-level Shakespeare configuration.
exec(open('config/train_shakespeare_char.py').read())
out_dir = 'out-residual-shakespeare'
merge_ratio = 2
merge_layer = 2
use_residual_cache = True
residual_tail = False
unmerged_prob = 0.0
