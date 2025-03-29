from typing import Optional

import torch.nn.functional as F
from torch import nn
import torch

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


class KronALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=(64, 4), network_alpha=None, device=None, dtype=None):
        super().__init__()
        
        self.a1 = rank[0]
        self.a2 = rank[1]
        """ 
        This is krona implementation
            A: (a1 * a2)
            B: (b1 * b2)
            a1 * b1 = d_out = out_features
            a2 * b2 = d_in = in_features
        
        Note: Supported a1 and a2 must be in multiplier of 2
        """
        assert in_features%self.a2==0 and out_features%self.a1==0
        self.b2 = int(in_features/self.a2); self.b1 = int(out_features/self.a1)
        self.down = nn.Linear(self.a1, self.a2, bias=False, device=device, dtype=dtype) # A
        self.up = nn.Linear(self.b2, self.b1, bias=False, device=device, dtype=dtype) # B
        # self.down = nn.Parameter(torch.FloatTensor(self.a2, self.a1).to(dtype).to(device), requires_grad=True)
        # self.up = nn.Parameter(torch.FloatTensor(self.b1, self.b2).to(dtype).to(device), requires_grad=True)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha

        nn.init.normal_(self.down.weight, std=1 / rank[0])
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        if len(hidden_states.shape) == 3:
            B1, C, D = hidden_states.size() # get the matrix shape
            hidden_states = hidden_states.view(-1, self.b2, self.a2).contiguous().view(-1, self.a2, self.b2).transpose(1, 2)
            
            up_hidden_states = self.up.weight@(hidden_states.to(dtype)@self.down.weight)
            up_hidden_states = up_hidden_states.view(B1, C, self.a1*self.b1)

        else: 
            B1, C = hidden_states.size() # get the matrix shape
            hidden_states = hidden_states.view(B1, self.b2, self.a2)
            hidden_states = hidden_states.view(B1*self.b2, self.a2)
            up_hidden_states = self.up.weight@(self.down(hidden_states.to(dtype)))
            up_hidden_states = up_hidden_states.view(B1, self.b1 * self.a1)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
