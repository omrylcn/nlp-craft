# NLP & Large Language Models: Comprehensive Guide

This repository is a comprehensive collection of resources covering the latest advancements in Natural Language Processing (NLP), Large Language Models (LLMs), and cutting-edge techniques for training, fine-tuning, and deploying AI systems.

## 📚 Learning Resources & Fundamentals

### Video Lectures & Courses
- [Yann Dubois: Scalable Evaluation of Large Language Models - video](https://www.youtube.com/watch?v=ZaQYM-YF1rM&ab_channel=MayurNaik)
- [Stanford CS229 I Machine Learning I Building Large Language Models - video](https://www.youtube.com/watch?v=9vM4p9NN0Ts&ab_channel=StanfordOnline)

### Essential Reading
- [Building AI for Production](https://towardsai.net/book)
- [No Hype DeepSeek-R1 Reading List](https://www.oxen.ai/blog/no-hype-deepseek-r1-reading-list)

## 🔧 Training & Fine-tuning Techniques

### Parameter-Efficient Fine-tuning (PEFT)
Optimizing the finetuning process of large language models by reducing the number of parameters that need to be updated during training.

#### Comprehensive Surveys
- [Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey](https://arxiv.org/abs/2403.14608) - *Covers algorithmic design, computational efficiency, applications, and system implementation*
- [Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies](https://arxiv.org/abs/2410.19878) - *Latest comprehensive survey covering 100+ research articles from 2019-2024*

#### Practical Guides & Tutorials
- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)
- [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)

#### Research Papers
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

### Reinforcement Learning for LLMs
Applying RL techniques to improve reasoning and decision-making capabilities in language models.

#### Theory & Understanding
- [The State of Reinforcement Learning for LLM Reasoning](https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html)
- [Understanding Reasoning LLMs](https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html)
- [Demystifying Reasoning Models ** also related with rl](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models?sort=top)

#### Frameworks & Tools
- [verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl)
- [Search-R1: An Efficient, Scalable RL Training Framework for Reasoning & Search Engine Calling interleaved LLM based on veRL](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file)

## 🚀 Advanced Applications & Use Cases

### Retrieval Augmented Generation (RAG)
Integration of semantic search and retrieval capabilities into the LLM generation process.

#### Tutorials & Implementation Guides
- [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch/tree/main?tab=readme-ov-file)
- [Tutorial: Building your own retrieval-augmented generation system - llms deep dive](https://github.com/springer-llms-deep-dive/llms-deep-dive-tutorials/tree/main/tutorials/chapter7)
- [Hands-On-Large-Language-Models - Chapter 8 - Semantic Search and Retrieval-Augmented Generation](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08)
- [Building an LLM open source search engine in 100 lines using LangChain and Ray](https://www.anyscale.com/blog/llm-open-source-search-engine-langchain-ray)

### AI Agents & Multimodal Systems

#### Agent Frameworks & Resources
- [Awesome Agents](https://github.com/kyrolabs/awesome-agents)
- [Agents : Build real-time multimodal AI applications](https://github.com/livekit/agents)

#### Multimodal Agents
- [Awesome Large Multimodal Agents](https://github.com/jun0wanan/awesome-large-multimodal-agents?tab=readme-ov-file)
- [Large Multimodal Agents: A Survey](https://arxiv.org/pdf/2402.15116)

## ⚙️ Production & Operations

### LLMOps
- [Why you need LLMOps](https://www.artefact.com/blog/why-you-need-llmops/#:~:text=Model%20adaptation%20to%20changing%20data,when%20significant%20changes%20are%20detected.)

## ⚠️ Challenges & Solutions

### Hallucination
Understanding and mitigating false or misleading outputs from language models.

- [2311 - A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions - v2](https://arxiv.org/abs/2311.05232)
- [2409 - LLMs Will Always Hallucinate, and We Need to Live With This - article](https://arxiv.org/html/2409.05746v1)

## 💻 Implementation Resources

### PEFT & Fine-tuning Code
- [Low-Rank Adaptation Blog Code](https://github.com/rasbt/low-rank-adaptation-blog/tree/main/code)
- [Blog: Finetuning LLaMA Adapters Code](https://github.com/rasbt/blog-finetuning-llama-adapters/tree/main/three-conventional-methods)
- [LLMs from Scratch Appendix E Notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb)

### RAG Implementation Examples
- [Advanced_RAG : Advanced Retrieval-Augmented Generation (RAG) through practical notebooks, using the power of the Langchain, OpenAI GPTs ,META LLAMA3 ,Agents](https://github.com/NisaarAgharia/Advanced_RAG)
- [RagBook Notebooks](https://github.com/towardsai/ragbook-notebooks?tab=readme-ov-file)

---

## 🗂️ Quick Navigation
- **New to LLMs?** → Start with [Learning Resources](#-learning-resources--fundamentals)
- **Want to fine-tune efficiently?** → Check out [PEFT](#parameter-efficient-fine-tuning-peft)
- **Building RAG systems?** → Go to [RAG section](#retrieval-augmented-generation-rag)
- **Working on agents?** → Visit [AI Agents](#ai-agents--multimodal-systems)
- **Production deployment?** → See [LLMOps](#llmops)
- **Need code examples?** → Browse [Implementation Resources](#-implementation-resources)
