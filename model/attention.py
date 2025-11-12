#implementation of attention mechanism


import torch
import torch.nn.functional
from torch import nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V)
    return output, attention_weights



#implementation of self attention
class SelfAttention(nn.Module):


    def __init__(self, embed_size):

        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)



    def forward(self, x, mask=None):
        
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        
        
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return out