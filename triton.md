denominator of softmax in triton is different from torch
```python
# torch
 x_exp.sum(1, keepdim=True)
# triton
qk = q * k
m_prev = m
m = tl.maximum(m, tl.max(qk, axis=1, keep_dims=True))
s = s*tl.exp2(log2_e*(m_prev-m)) + tl.sum(tl.exp2(log2_e*(qk-m)), axis=1, keep_dims=True)
```
