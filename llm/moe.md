baseline softmax gating, eplevel aux_loss
sigmoid gating + auxiliary loss free training is more stable, mmlu score is better than baseline
only switch softmax to sigmoid, the result is similar, sigmoid gating + auxiliary loss free perform differently


the reason why mla is better than mha is that when go back to MHA stage, the num_head or head_dim is increased.

aria 0.4b vision, 3.5b text
initialize the weights of our ViT using the SigLIP-SO400M model
aria 22A/247B, max pictures 195, 2s one frame can accept a 6m30s video
