## Why DPO instead of PPO after SFT (≈ 30 seconds)

We chose DPO because it avoids the reward-hacking loop that often destabilizes PPO.

## Structured JSON output from the VLM (≈ 30 seconds)

To guarantee schema-valid JSON, we used constrained decoding with a context-free grammar compiled into a finite-state machine. At each token step we mask the logits so only legal continuations are possible. The overhead is about 1 millisecond, but the downstream parser never fails, which is critical for automatic video editing.

## Compensation expectation (≈ 20 seconds)

What matters most is the scope of the problems I can solve and the caliber of the team.
