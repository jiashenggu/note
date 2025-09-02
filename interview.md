# Intro

Hi, it's a pleasure to meet you. I'm Jiasheng Gu.

At 01.AI, I was an engineer on a generative AI video product, building it from scratch. My main responsibility was the end-to-end training pipeline for our VLMs and LLMs, including 25-billion and 247-billion parameter MoE models.

My process had two key stages. First, I handled the continued pre-training on a 20-billion-token domain-specific dataset, using the Megatron-LM framework. Second, I designed and implemented our alignment pipeline using both SFT and DPO. This was crucial for teaching the model the complex nuances of video understanding and intelligent clipping.

While I've greatly valued the challenges and growth of my previous roles, after researching the potential of physical AI and being inspired by NVIDIA's vision, I came to believe that applying this technology to create physical AI is the essential next step, and this conviction was a key driver in my decision to seek new opportunities. The reason I'm so excited about this role is that I see a direct bridge between my work and the challenges of this role. My experience in teaching models to perceive and reason about the world through video can directly contribute to Project GR00T.

## Why DPO instead of PPO after SFT (≈ 30 seconds)

We chose DPO because it avoids the reward-hacking loop that often destabilizes PPO.

## Structured JSON output from the VLM (≈ 30 seconds)

To guarantee schema-valid JSON, we used constrained decoding with a context-free grammar compiled into a finite-state machine. At each token step we mask the logits so only legal continuations are possible. The overhead is about 1 millisecond, but the downstream parser never fails, which is critical for automatic video editing.

## Compensation expectation (≈ 20 seconds)

What matters most is the scope of the problems I can solve and the caliber of the team.

## Question: "What do you think of NVIDIA's approach with Isaac and Project GR00T?"

Your Potential Answer: "I've been following it closely, and I think NVIDIA's full-stack strategy is precisely what the field needs. It's incredibly ambitious. You're not just building a model; you're building the entire ecosystem required for success. This includes the specialized compute hardware like the Jetson Thor, Isaac Sim platform for scalable training, and finally, the GR00T foundation model that ties it all together. This vertical integration allows for optimizations at every level. The concept of GR00T as a 'generalist blueprint' that understands multimodal instructions and can be fine-tuned for different embodiments is the most promising path towards scalable robotics, and it aligns perfectly with my own experience in building and adapting large foundation models."

## Question: "Our strategy is to provide the platform and tools like Isaac and GR00T to enable the entire robotics ecosystem. How does this compare to more vertically integrated approaches, like Tesla's Optimus? What are the pros and cons in your view?"

"Tesla's vertical integration gives them tight control and the ability to optimize for a specific set of tasks within their own factories—a huge advantage for rapid, focused deployment. However, it's a closed garden. NVIDIA's platform strategy is fundamentally about enabling an entire market, much like Windows did for PCs or Android for mobile. The pro is massive scale and capturing a wider range of innovation from countless partners. The con is that it can be slower to get off the ground and requires managing a more complex ecosystem. I believe the platform approach will win in the long run because the diversity of real-world robotics applications is too vast for any single company to address alone."

## 1. On Distributed Training Nuances: 
## Question: "You mentioned using Megatron-LM. Could you discuss the trade-offs between different parallelism strategies like data, tensor, and pipeline parallelism? In your 247B MoE model training, how did you likely combine these, and what was the main bottleneck you were trying to solve with that specific combination?"

How to Answer:

Define: Briefly explain each. Data Parallelism (simple, scales batch size, but memory per GPU is a limit). Tensor Parallelism (splits model weights, solves memory, but high communication overhead). Pipeline Parallelism (splits layers, helps with throughput, but can lead to idle 'bubbles').

Synthesize:  "For a large MoE model, you'd typically use Data Parallelism to distribute the experts across GPUs, as each input only routes to a few experts. For the very large, non-expert layers (like the attention blocks), you'd apply Tensor Parallelism to make them fit in memory. Pipeline Parallelism could be layered on top to keep the pipeline full. The main bottleneck we're solving is GPU memory capacity, followed by communication overhead." 

Computational efficiency: EP has an advantage in expert's computational efficiency, which reduces the number of kernel launches and increases the workload of each launch (without using the cutlass grouped gemm).

Communication overhead: In the case of topk < parallel size, EP has less communication than TP.

Memory usage: TP has lower and more stable memory usage, which is suitable for long sequence training or memory-sensitive scenarios, while EP may cause memory overflow problem when the allocation is not balanced.

Of course. Since you are meeting with a senior leader, the conversation can easily pivot into deeper technical trade-offs, behavioral scenarios, and forward-looking strategic topics. Here are more potential questions and areas to prepare for, designed to test your depth, strategic thinking, and team fit.

Category 1: Deeper Technical Probing & Trade-offs
These questions go beyond "what you did" to "why you did it that way" and test the depth of your understanding.

## 1. On Distributed Training Nuances:

## Question: "You mentioned using Megatron-LM. Could you discuss the trade-offs between different parallelism strategies like data, tensor, and pipeline parallelism? In your 247B MoE model training, how did you likely combine these, and what was the main bottleneck you were trying to solve with that specific combination?"

How to Answer:

Define: Briefly explain each. Data Parallelism (simple, scales batch size, but memory per GPU is a limit). Tensor Parallelism (splits model weights, solves memory, but high communication overhead). Pipeline Parallelism (splits layers, helps with throughput, but can lead to idle 'bubbles').

Synthesize: "For a large MoE model, Expert Parallelism is used to distribute the experts across GPUs, as each input only routes to a few experts. For the very large, non-expert layers (like the attention blocks), you'd apply Tensor Parallelism to make them fit in memory. Pipeline Parallelism could be layered on top to keep the pipeline full. The main bottleneck we're solving is GPU memory capacity, followed by communication overhead." 

## 2. On Data Quality & Challenges:

## Question: "You mentioned building a 'robust data quality loop' and a scoring model. Can you get more specific? What features did the scoring model use? And what was the most challenging or surprising data quality issue you encountered when training VLMs on video data?"

How to Answer:

Be Specific: "The scoring model was a fine-tuned LLM that took a generated caption and a video script as input and predicted a quality score from 1-5. Features included token number, object consistency (did the caption mention objects actually detected in the video?), and semantic relevance."

Share a Story: "The most surprising challenge was 'temporal hallucination.' The model would generate perfectly fluent descriptions of events that could logically happen next but didn't actually occur in the video clip. This forced us to create negative samples in our DPO training to penalize this behavior." 

3. On Systems-Level Optimization:

Question: "This role is about optimizing the full stack. Can you describe a time when a performance bottleneck wasn't in the model architecture itself, but in the data loading, CPU preprocessing, or network communication? How did you diagnose and solve it?"

How to Answer: This is a chance to show you don't just think about the model.

"Yes, we frequently encountered situations where our GPUs were underutilized due to CPU-bound preprocessing. I identified this bottleneck using tools like Weights & Biases (wandb), which clearly visualized periods of GPU idle time. The solution was straightforward: we increased the number of CPU cores in the machine and adjusted the num_workers parameter of the dataloader. On another occasion, we pinpointed internet speed as the limiting factor—loading video files strained the available bandwidth. To resolve this, we pre-encoded the videos and migrated them to a storage drive with higher bandwidth capacity."

# Questions Should Ask
## balance real data and simulated data
I would like to know how you handle the balance between real data and simulated data?


