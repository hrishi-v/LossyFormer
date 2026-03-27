import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections.abc import Mapping

class HeadProfiler:
    def __init__(self, num_heads, head_size, active_heads):
        self.num_heads = num_heads
        self.head_size = head_size
        self.active_heads = active_heads  # Tracks original absolute indices
        self.imp_scores = []
        self.collecting = False

    def get_scores(self):
        if not self.imp_scores:
            return None
        total_sum = sum(s[0] for s in self.imp_scores)
        total_samples = sum(s[1] for s in self.imp_scores)
        if total_samples == 0:
            return None
        return (total_sum / total_samples).tolist()

