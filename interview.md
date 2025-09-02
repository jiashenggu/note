# Intro

Hi, it's a pleasure to meet you. I'm Jiasheng Gu.

At 01.AI, my work was focused on the entire lifecycle of VLMs and LLMs, including 25B-A3.5B and 247B-A22B MoE models. My process had two key stages. First, I handled the continued pre-training on a 20-billion-token domain-specific dataset, using the Megatron-LM framework. Second, I designed and implemented our alignment pipeline using SFT and DPO. This played a key role in teaching the model to improve the accuracy of video captions, increase the output density of LLMs, and support intelligent video clipping.

At Alibaba, my work was to drive efficient model iteration by selecting high-value samples based on VLM scores, which is the critical step of a closed-loop system: "selecting data -> training models -> evaluate models -> get benchmark feedback." Besides, I enhanced and validated the model's ability to recognize knowledge entities by constructing high-quality training and validation datasets. Here, a knowledge entity refers to specific entities, such as a Siamese cat or Jackie Chan.

While I've greatly valued the growth of my previous roles, after researching the potential of physical AI and being inspired by NVIDIA's vision, I came to believe that physical AI is the essential next step of AI, and this conviction was a key driver in my decision to seek new opportunities. The reason I'm so excited about this role is that I see a direct bridge between my work and the challenges of this role. My experience in teaching models to perceive and reason through video can directly contribute to Project GR00T.

## Why DPO instead of PPO after SFT (≈ 30 seconds)

We chose DPO because it avoids the reward-hacking loop that often destabilizes PPO.

## Structured JSON output from the VLM (≈ 30 seconds)

To guarantee schema-valid JSON, we used constrained decoding with a context-free grammar compiled into a finite-state machine. At each token step we mask the logits so only legal tokens are possible. The overhead is about 1 millisecond, but the downstream parser never fails, which is critical for automatic video editing.



## Question: "What do you think of NVIDIA's approach with Isaac and Project GR00T?"

Your Potential Answer: "I've been following it closely, and I think NVIDIA's full-stack strategy is what the field needs. It's incredibly ambitious. You're not just building a model; you're building the entire ecosystem required for physical ai. This includes the specialized compute hardware like the Jetson Thor, Isaac Sim platform for scalable training, and finally, the GR00T foundation model that ties it all together. The concept of GR00T as a 'generalist foundation model' that understands multimodal instructions and can be fine-tuned for different embodiments is the most promising path towards scalable robotics."

## Question: "Our strategy is to provide the platform and tools like Isaac and GR00T to enable the entire robotics ecosystem. How does this compare to more vertically integrated approaches, like Tesla's Optimus? What are the pros and cons in your view?"

"Tesla's vertical integration gives them the ability to optimize for their own factories—a huge advantage for rapid, focused deployment. However, it's a closed garden. NVIDIA's platform strategy is fundamentally about enabling the entire market, much like Windows did for PC. The pro is massive scale and capturing a wider range of innovation from countless partners. The con is that it requires managing a more complex ecosystem. I believe the platform approach will win in the long run because the diversity of real-world robotics applications is too big for any single company to address."


# Category 1: Deeper Technical Probing & Trade-offs

## 1. On Distributed Training Nuances:

## Question: "You mentioned using Megatron-LM. Could you discuss the trade-offs between different parallelism strategies like data, tensor, and pipeline parallelism? In your 247B MoE model training, how did you likely combine these, and what was the main bottleneck you were trying to solve with that specific combination?"

### 5 parallel
Data Parallelism is a straightforward approach where we replicate the model across multiple GPUs, each processing a portion of the input data. It's great for scaling batch sizes, though it's constrained by the memory capacity of individual GPUs since each holds a full copy of the model.

Tensor Parallelism addresses memory limits by splitting model weights across GPUs, with each handling a part of the tensor computations. This solves large model memory issues but introduces significant communication overhead between devices to synchronize partial results.

Pipeline Parallelism divides the model into layers, assigning different layer segments to separate GPUs that process data in a sequential pipeline. It boosts throughput for long sequences but can create idle "bubbles" when GPUs wait for inputs from preceding layers.

Expert Parallelism, often used with MoE (Mixture of Experts) models, distributes different expert sub-networks across GPUs. Each input activates only a subset of experts, enabling model scaling without full replication, though it requires smart routing and load balancing.

Context Parallelism splits input sequences (like long texts) across GPUs, with each handling a segment of the context. It's particularly useful for processing extremely long sequences, allowing models to handle context lengths beyond what a single GPU's memory can accommodate.

### EP vs TP in MoE：

Computational efficiency: EP has an advantage in expert's computational efficiency, which reduces the number of kernel launches and increases the workload of each launch (without using the cutlass grouped gemm).

Communication overhead: In the case of topk < parallel size, EP has less communication than TP.

Memory usage: TP has lower and more stable memory usage, which is suitable for long sequence training or memory-sensitive scenarios, while EP may cause memory overflow problem when the allocation is not balanced.

Of course. Since you are meeting with a senior leader, the conversation can easily pivot into deeper technical trade-offs, behavioral scenarios, and forward-looking strategic topics. Here are more potential questions and areas to prepare for, designed to test your depth, strategic thinking, and team fit.

Category 1: Deeper Technical Probing & Trade-offs
These questions go beyond "what you did" to "why you did it that way" and test the depth of your understanding.

## 2. On Data Quality & Challenges:

## Question: "You mentioned building a 'robust data quality loop' and a scoring model. Can you get more specific? What features did the scoring model use? And what was the most challenging or surprising data quality issue you encountered when training VLMs on video data?"

The scoring model was a fine-tuned LLM that took a generated caption and a video script as input and predicted a quality score from 1-5. The score considered token number, semantic relevance and scripts' quality."

The most surprising challenge was 'temporal hallucination.' The model would generate fluent descriptions of events that could logically happen next but didn't actually occur in the video clip. This forced us to create negative samples in our DPO training to penalize this behavior." 

## 3. On Systems-Level Optimization:

Question: "This role is about optimizing the full stack. Can you describe a time when a performance bottleneck wasn't in the model architecture itself, but in the data loading, CPU preprocessing, or network communication? How did you diagnose and solve it?"

"Yes, we frequently encountered situations where our GPUs were underutilized due to CPU-bound preprocessing. I identified this bottleneck using tools like Weights & Biases (wandb), which clearly visualized periods of GPU idle time. The solution was straightforward: we increased the number of CPU cores in the machine and adjusted the num_workers parameter of the dataloader. On another occasion, we pinpointed internet speed as the limiting factor—loading video files strained the available bandwidth. To resolve this, we pre-encoded the videos and migrated them to a storage drive with higher bandwidth capacity."

# Category 2: Behavioral & Situational Questions


## 1. Handling Ambiguity & Research Collaboration:

Question: "Imagine a researcher gives you a novel model architecture in a messy, single-GPU script. It shows promise, but it's far from scalable. Walk me through your process as a Solutions Architect to take this from a research prototype to a robust, multi-node training pipeline. How do you handle disagreements about technical trade-offs with the researcher?"

1. Understand & Profile: I'd first focus on understanding the core innovation. Then, I'd profile the code to identify the allocated memory and compute bottlenecks.
2. Modularize & Refactor: I'd work with the researcher to refactor the code into modular components (data loading, model definition, training loop). 
3. Introduce Parallelism: Based on the profile, I'd introduce the appropriate parallelism strategy, starting with Data Parallelism and adding other Parallelisms if needed.
4. Scale & Test: We'd test on 2, then 4, then 16 nodes, validating that the convergence behavior matches the single-GPU baseline."

Collaboration: "For disagreements, my approach is data-driven. For example, if a researcher prefers a custom operator that is slow, I would profile it against a standard, optimized one and present the performance data. The goal isn't to say 'no,' but to say 'Here's the performance cost of this approach. Can we work together to find a solution that preserves your research goal while also being scalable?'"

## 2. Failure & Learning:

Question: "Tell me about your most significant failure in a large-scale training project. What went wrong, what was the impact, and what did you learn from it that changed how you work today?"

My most significant failure was a silent data mismatch during the training of a VLM.

What happened: We had a complex data preprocessing pipeline involving separate models for object detection and ASR. An upstream update to the object detection model changed its class label definitions. Our pipeline continued to run without errors, but it was feeding the VLM visual data that was inconsistent with the textual prompts being generated.

Impact: The model didn't crash. The loss went down, and it seemed to be training. However, for about two weeks, we were chasing a "ghost" performance degradation in the model's visual reasoning abilities. We wasted significant time and GPU cycles on hyperparameter tuning and architectural tweaks, assuming the problem was the VLM itself, not the data.

My key learning was that in a complex DL system, silent data corruption is far more dangerous than a loud crash.

As a result, I implemented a versioning system. Before any training run, we run a check that uses content versions to ensure that all data artifacts—from the visual features to the text tokenizers—are perfectly compatible with the specific model checkpoint we are training. This has become a standard step in our pipeline and has prevented those subtle, expensive bugs from recurring.

## Compensation expectation (≈ 20 seconds)

What matters most is the scope of the problems I can solve and the caliber of the team.

I believe that, in the short term, scaling up computing hardware is simpler than scaling up robotic hardware.

# Questions Should Ask
## balance real data and simulated data
"Regarding the data for GR00T, what is your team's current strategy on the balance between simulation data from Isaac and real-world demonstration data?"

"As you look to scale GR00T's capabilities, what do you foresee as the biggest challenge in the post-training alignment process?"

"What would meeting expectations look like for someone in this role in their first 6 months?"

