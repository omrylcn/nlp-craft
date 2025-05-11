# Prompt Engineering Notes

## Elements of a Prompt

1. **Instruction**: The specific task or instruction you want the model to perform.
2. **Context**: External information or additional context that guides the model to better responses.
3. **Input Data**: The input or question for which we seek a response.
4. **Output Indicator**: The type or format of the desired output.

## Prompt Engineering Techniques

### Zero-Shot Prompting

- **Definition**: The model is given a task without any examples or demonstrations.
- **Usage**: Directly instruct the model to perform a task without additional context.

### Few-Shot Prompting

- **Definition**: The model is provided with a few examples or demonstrations to guide its responses.
- **Usage**: Enables in-context learning by conditioning the model with examples.
- **Reference**: [2202 - Fine-Tuned Language Models Are Zero-Shot Learners](https://arxiv.org/pdf/2109.01652)

### Chain of Thought (CoT) Prompting

- **Definition**: A technique that encourages the model to generate intermediate reasoning steps before arriving at the final answer.
- **Reference**: [2201 - Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

## Useful Links

- [Best Practices for Prompt Engineering with the OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
- [Comprehensive Guide to Chain-of-Thought Prompting](https://www.mercity.ai/blog-post/guide-to-chain-of-thought-prompting)

## Prompt Engineering Parameters

### Temperature

- **Definition**: A parameter that controls the randomness of the model's outputs. Typically ranges from 0 to 1.
- **Effect**:
  - **Low Temperature**: Produces more deterministic and predictable outputs.
  - **High Temperature**: Encourages creativity and diversity in outputs.
- **References**:
  - [LLM Temperature](https://www.hopsworks.ai/dictionary/llm-temperature#:~:text=The%20LLM%20temperature%20serves%20as,exploration%2C%20fostering%20diversity%20and%20innovation.)
  - [A Comprehensive Guide to LLM Temperature](https://towardsdatascience.com/a-comprehensive-guide-to-llm-temperature/)
  - [Mastering LLM Parameters: A Deep Dive into Temperature, Top-K, and Top-P](https://plainenglish.io/blog/mastering-llm-parameters-a-deep-dive-into-temperature-top-k-and-top-p)

## Articles and References

- [2201 - Chain of Thought Prompting Elicits - v6](https://arxiv.org/abs/2201.11903)
- [Understanding Reasoning LLMs](https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html)

