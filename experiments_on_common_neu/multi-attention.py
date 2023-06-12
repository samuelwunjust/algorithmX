import math
import torch
from torch import nn
import d2l
class MultiHeadAttention(d2l.Module):
    def __init__(self,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super().__init__()
        self.num_heads=num_heads
        self.attention=d2l.DotProductAttention(dropout)
        self.W_q=nn.LazyLinear(num_hiddens,bias)
        self.W_k=nn.LazyLinear(num_hiddens,bias)
        self.W_v=nn.LazyLinear(num_hiddens,bias)
        self.W_o=nn.LazyLinear(num_hiddens,bias)

    def tranpose_qkv(self, X):
        # X(batch_size,num of k-q,num_hiddens)-->(batchsize,num of k-q,num_heads,num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)  # swap(shape[1],shape[3])-->(ba_s,num_heads,num of k-q,num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])  # -->(ba_s*num_heads,num of k-q,num_hiddens/num_heads)

    def tranpose_output(self, X):
        # reverse tranpose_qkv
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self,queries,keys,values,valid_lens):
        queries=self.tranpose_qkv(self.W_q(queries))
        keys=self.tranpose_qkv(self.W_k(keys))
        values=self.tranpose_qkv(self.W_v(values))
        if valid_lens is not None:
            valid_lens=torch.repeat_interleave( valid_lens, repeats=self.num_heads, dim=0)
        output=self.attention(queries,keys,values,valid_lens)
        output_cocat=self.tranpose_output(output)
        return self.W_o(output_cocat)




num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))

