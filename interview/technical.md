# Intro

Hi, I'm Jiasheng Gu. At 01.AI, I built the end-to-end training pipeline for proprietary VLMs and LLMs. These include the 25B-A3.5B and 247B-A22B MoE models. All this was used to create a generative AI video editing and mixing product from scratch.

First, I did continued pre-training on a 20-billion-token dataset using Megatron-LM. Then, I set up a two-stage fine-tuning process: SFT and DPO. This made the models better at understanding and clipping video, which is critical for the product.

I’m excited about this opportunity. I’m really interested in robotic foundation models. My background in training models for video perception and reasoning fits well with the challenges in robotics. I believe my experience can contribute to developing foundation models for Project GR00T.

## Why DPO instead of PPO after SFT (≈ 30 seconds)

We chose DPO because it avoids the reward-hacking loop that often destabilizes PPO.

## Structured JSON output from the VLM (≈ 30 seconds)

To guarantee schema-valid JSON, we used constrained decoding with a context-free grammar compiled into a finite-state machine. At each token step we mask the logits so only legal continuations are possible. The overhead is about 1 millisecond, but the downstream parser never fails, which is critical for automatic video editing.

## Compensation expectation (≈ 20 seconds)

What matters most is the scope of the problems I can solve and the caliber of the team.

## Question: "What do you think of NVIDIA's approach with Isaac and Project GR00T?"

Your Potential Answer: "I've been following it closely, and I think NVIDIA's full-stack strategy is precisely what the field needs. It's incredibly ambitious and correct, in my opinion. You're not just building a model; you're building the entire ecosystem required for success. This includes the specialized compute hardware like the Jetson Thor, the hyper-realistic Isaac Sim platform for scalable training, and finally, the GR00T foundation model that ties it all together. This vertical integration allows for optimizations at every level. The concept of GR00T as a 'generalist blueprint' that understands multimodal instructions and can be fine-tuned for different embodiments is the most promising path towards scalable robotics, and it aligns perfectly with my own experience in building and adapting large foundation models."

# Questions Should Ask
## balance real data and simulated data
I would like to know how you handle the balance between real data and simulated data?


