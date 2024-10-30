baseline softmax gating, eplevel aux_loss
sigmoid gating + auxiliary loss free training is more stable, mmlu score is better than baseline
only switch softmax to sigmoid, the result is similar, sigmoid gating + auxiliary loss free perform differently


the reason why mla is better than mha is that when go back to MHA stage, the num_head or head_dim is increased.
