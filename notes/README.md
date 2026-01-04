# NLP & LLM Notes

Bu klasör, NLP ve LLM konularında kapsamlı Türkçe ve İngilizce notlar içermektedir.

---

## Table of Contents

### Transformer & Model Architectures

| File | Description | Language |
|------|-------------|----------|
| [bert(tr).md](bert(tr).md) | BERT: Kapsamlı teorik ve pratik inceleme, varyantlar (DeBERTa-v3, ModernBERT) | TR |
| [gpt(tr).md](gpt(tr).md) | GPT modelleri: Teorik analiz, GPT-4o, o1 serisi, açık kaynak modeller | TR |
| [glm(tr).md](glm(tr).md) | Generative Language Models: Kapsamlı araştırma rehberi | TR |
| [moe(tr).md](moe(tr).md) | Mixture of Experts: Mimari, routing, güncel modeller (Mixtral, DeepSeek-MoE) | TR |
| [encoder&decoder(tr).md](encoder&decoder(tr).md) | Encoder-Only vs Decoder-Only: Teknik analiz ve karşılaştırma | TR |
| [encoder&decoders_usage(tr).md](encoder&decoders_usage(tr).md) | Encoder vs Decoder: Pratik kullanım rehberi | TR |
| [emergent_abilities.md](emergent_abilities.md) | Emergent Abilities: Literature review, ICL, CoT, controversies | EN |

### Sentence Embeddings & Semantic Similarity

| File | Description | Language |
|------|-------------|----------|
| [sbert(tr).md](sbert(tr).md) | BERT, RoBERTa, Sentence-BERT: Kapsamlı rehber | TR |
| [semantic(tr).md](semantic(tr).md) | Transformer tabanlı semantik modelleme | TR |
| [semantic_similarity(tr).md](semantic_similarity(tr).md) | BERT ile metin embedding ve benzerlik analizi | TR |

### NLI & STS

| File | Description | Language |
|------|-------------|----------|
| [nli.md](nli.md) | Natural Language Inference: Comprehensive guide | EN |
| [nli&sts(tr).md](nli&sts(tr).md) | NLI ve STS: Derin teorik ve pratik analiz | TR |
| [nli&sts_loss(tr).md](nli&sts_loss(tr).md) | STS ve NLI: Loss fonksiyonları ve değerlendirme | TR |

### Fine-Tuning & PEFT

| File | Description | Language |
|------|-------------|----------|
| [lora(tr).md](lora(tr).md) | LoRA: Zero to Hero - QLoRA, DoRA, LoRA+, VeRA | TR |
| [peft(tr).md](peft(tr).md) | Parameter-Efficient Fine-Tuning: Teorik temeller | TR |

### RAG & Retrieval

| File | Description | Language |
|------|-------------|----------|
| [rag(tr).md](rag(tr).md) | RAG: Teorik temeller, güncel yaklaşımlar (GraphRAG, CRAG, Self-RAG) | TR |

### Prompt Engineering

| File | Description | Language |
|------|-------------|----------|
| [prompt.md](prompt.md) | Context learning, hard/soft prompt tuning, indexing | EN |
| [prompt/](prompt/) | Comprehensive Prompt Engineering Guide (11 parts) | EN |

---

## Algorithm References

### Paged Attention
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [PagedAttention - Blog](https://www.hopsworks.ai/dictionary/pagedattention)

### Flash Attention
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

### Adapters & Fine-Tuning
- [Finetuning LLMs with Adapters - Sebastian Raschka](https://magazine.sebastianraschka.com/p/finetuning-llms-with-adapters)

### LLM Evaluation
Key evaluation approaches:
- Functional correctness
- Similarity measures (lexical & semantic)
- Exact match limitations

---

## External Resources

### RAG Resources
- [RAG Project Files](https://drive.google.com/drive/folders/16GjGMDFRCyWTOj-teFYCutQ5gCdeIO4d)
- [RAGAS Pipeline Evaluation](https://github.com/stanghong/ragas_pipeline_eval/tree/main)
- [End-to-End RAG Pipeline with Model Monitoring](https://medium.com/@Stan_DS/build-an-end-to-end-rag-pipeline-with-model-monitoring-pipeline-c8af35ed731b)

### ZenML & LLMOps
- [ZenML LLMOps Guide](https://docs.zenml.io/user-guide/llmops-guide)
- [RAG with ZenML](https://docs.zenml.io/user-guide/llmops-guide/rag-with-zenml)
- [From RAG to Riches - ZenML Blog](https://www.zenml.io/blog/from-rag-to-riches-the-llmops-pipelines-you-didnt-know-you-needed)

### Vector Databases & Embeddings
- [Pinecone RAG Series](https://www.pinecone.io/learn/series/rag/)
- [Pinecone FAISS Series](https://www.pinecone.io/learn/series/faiss/)
- [Pinecone LangChain Series](https://www.pinecone.io/learn/series/langchain/)

---

## File Statistics

| Category | Count |
|----------|-------|
| Turkish (TR) | 15 files |
| English (EN) | 2 files + prompt/|

---

*Last updated: 2026-01*
