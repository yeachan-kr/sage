# Saliency-Guided Memory-Efficient Fine-tuning of Large Language Models (ACL 2025)

### Abstract

Fine-tuning is widely recognized as a crucial process for aligning large language models (LLMs) with human intentions. However, the substantial memory requirements associated with fine-tuning pose a significant barrier to extending the applicability of LLMs. While parameter-efficient fine-tuning can be a promising approach by reducing trainable parameters, intermediate activations still need to be cached to compute gradients during the backward pass, thereby limiting overall memory efficiency. In this work, we propose Saliency-Guided Gradient Flow (Sage), a memory-efficient fine-tuning method designed to minimize the memory specifically associated with cached intermediate activations. The key strategy is to selectively cache activations based on their saliency during the forward pass and then use these activations for the backward pass. This process transforms the dense backward pass into a sparse one, thereby enhancing memory efficiency. To verify whether Sage can serve as an efficient alternative for fine-tuning, we conduct comprehensive experiments across diverse fine-tuning scenarios and setups. The experimental results show that Sage substantially improves memory efficiency without a significant loss in accuracy, highlighting its broad value in real-world applications

### TODO
- [ ] Code update and release
