# A README for Context Engineering

## Introduction

Context Engineering is the discipline of designing and building dynamic systems that provide the right information, tools, in the right format, at the right time, to give an LLM everything it needs to accomplish a task.

## Why Context Engineering Matters

- **Key to Agent Success**: The main factor determining whether AI agents succeed or fail is the quality of context provided to them
- **Limited Working Memory**: LLMs have limited context windows, making effective utilization critical
- **Beyond Static Approaches**: Context engineering requires dynamic systems, not just static prompts

## Core Characteristics of Context Engineering

### 1. System Approach

- Context is not just a string, but the output of a system that runs before the main LLM call
- Much more complex than static prompt templates

### 2. Dynamic Nature

- Created on the fly, tailored to the immediate task
- Could be calendar data for one request, emails for another, or web search results

### 3. Right Information and Tools

- Ensures the model isn't missing crucial details (following "Garbage In, Garbage Out" principle)
- Provides both knowledge (information) and capabilities (tools) only when required and helpful

### 4. Format Matters

- How information is presented is critical
- Concise summaries are better than raw data dumps
- Clear tool schemas are better than vague instructions

## Context Engineering Patterns

### Writing Context

Saving context outside the context window to help an agent perform a task

### Selecting Context

Pulling context into the context window to help an agent perform a task

### Compressing Context

Retaining only the tokens required to perform a task

### Isolating Context

Splitting context up to help an agent perform a task

## Context Problems

### Context Poisoning

When a hallucination makes it into the context

### Context Distraction

When the context overwhelms the training

### Context Confusion

When superfluous context influences the response

### Context Clash

When parts of the context disagree

## Alternative to Multi-Agent Approaches

### Principle 1: Share Context Fully

- Share full agent traces, not just individual messages
- Share complete context across agents

### Principle 2: Actions Carry Implicit Decisions

- Every action contains implicit decisions
- Use specialized models to compress action history and conversation into key details

## LLM and Context Relationship

As Andrej Karpathy puts it, LLMs are like a new kind of operating system:

- **LLM = CPU**: Processing unit
- **Context Window = RAM**: Limited capacity working memory
- **Context Engineering = Operating System**: System that curates what fits into the CPU's RAM

## Resources

### Academic Papers

- [A Survey of Context Engineering for Large Language Models](https://arxiv.org/abs/2507.13334)

### Blog Posts

- [How Contexts Fail and How to Fix Them](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com)
- [How to Fix Your Context](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html)
- [Built Multi-Agent Research System](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://research.trychroma.com/context-rot)
- [The New Skill in AI is Not Prompting, It's Context Engineering](https://www.philschmid.de/context-engineering)
- [The Rise of Context Engineering](https://blog.langchain.com/the-rise-of-context-engineering/)
- [Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)

### GitHub Repositories

- [langchain-ai/context_engineering](https://github.com/langchain-ai/context_engineering)
- [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering)
- [coleam00/context-engineering-intro](https://github.com/coleam00/context-engineering-intro)
- [humanlayer/12-factor-agents](https://github.com/humanlayer/12-factor-agents?tab=readme-ov-file)

## Key Insights

### Prompt Engineering vs Context Engineering

- Prompt engineering is a subset of context engineering
- While prompt phrasing matters, providing complete and structured context is far more important
- Context engineering involves architecting prompts to work with dynamic data, not just single input sets

### Agent Performance Optimization

- Agent failures aren't just model failures; they are context failures
- The magic isn't in smarter models or clever algorithms, but in providing the right context for the right task
- Long-running tasks and tool call feedback can cause token bloat, leading to performance degradation

## Best Practices

1. **Design for Dynamics**: Build systems that adapt context based on immediate needs
2. **Optimize Format**: Structure information for clarity and relevance
3. **Manage Token Economy**: Balance completeness with efficiency
4. **Monitor Context Quality**: Watch for poisoning, distraction, confusion, and clash issues
5. **Share Full Context**: When using multiple agents, share complete traces rather than partial information

## Conclusion

Context Engineering represents a fundamental shift from static prompt optimization to dynamic information management systems. As AI applications become more complex, mastering context engineering becomes essential for building reliable, effective AI agents and systems.
