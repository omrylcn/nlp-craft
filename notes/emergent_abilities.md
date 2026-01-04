## Emergent Behavior Literature Review

### Key Papers

**1. "Emergent Abilities of Large Language Models" (Wei et al., 2022)**
- The seminal paper that defined the concept
- Collaboration between Google Research, Stanford, and DeepMind
- For an ability to be considered "emergent," it must be absent in smaller models but present in larger models; it cannot be predicted by extrapolating from smaller models' performance
- Catalogued 137 different emergent abilities

**2. "Are Emergent Abilities of Large Language Models a Mirage?" (Schaeffer et al., 2023)**
- Counter-argument from Stanford
- Claims that emergent abilities stem from the researcher's choice of metric, not from fundamental changes in model behavior
- When continuous metrics are used instead of discrete metrics (correct/incorrect), the "jump" disappears

**3. "Emergent Abilities in Large Language Models: A Survey" (Berti et al., 2025)**
- The most recent comprehensive survey
- Whether abilities like reasoning, in-context learning, coding, and problem-solving are truly emergent or depend on training dynamics, problem type, or metric choice remains controversial

---

### Theoretical Foundations

**Physics Analogy: Phase Transitions**
- When water is cooled below 0°C, it undergoes an abrupt transition from fluid mechanics to solid mechanics — LLMs exhibit similar dramatic qualitative changes as scale increases
- Philip Anderson's 1972 Nobel Prize-winning paper "More Is Different" provides the conceptual foundation

**Grokking Phenomenon**
- A connection is being established between grokking (models first memorizing data then suddenly acquiring generalization ability) and emergent abilities
- When memorization dominates, the emergence of generalization ability is delayed

---

### Key Emergent Abilities

**In-Context Learning (ICL)**
- One of the earliest and most celebrated emergent behaviors: the model's ability to learn from a few examples in the prompt and generalize to new questions, without any gradient updates
- Discovered with GPT-3 (2020): "Language Models are Few-Shot Learners"
- ICL is an emergent property of model scale — it develops at different rates in larger models compared to smaller ones

**Chain-of-Thought Reasoning**
- Step-by-step reasoning ability emerges after approximately 10²² FLOPs
- Chain-of-thought prompting decreases performance in smaller models while providing dramatic improvements in larger models

**Arithmetic and Mathematics**
- Abilities like multi-digit addition and multiplication "unlock" after reaching a certain scale

---

### Controversial Topics

| Perspective | Proponents | Argument |
|-------------|------------|----------|
| **Scale Hypothesis** | Wei et al., Google | Abilities genuinely emerge through sudden jumps |
| **Mirage Hypothesis** | Schaeffer et al., Stanford | Discrete metrics create an illusion |
| **Nonlinear Dynamics** | Havlík (2025) | Emergent abilities arise from complex dynamics of highly sensitive nonlinear systems, not simply from parameter scaling |

---

### Current Research Directions

1. **Predicting Emergence**: Forecasting which abilities will emerge at which scale
2. **Large Reasoning Models (LRMs)**: Emergent reasoning enhanced by RL in models like DeepSeek-R1 and OpenAI o3
3. **Safety Concerns**: Systems that acquire autonomous reasoning capabilities can also develop harmful behaviors such as deception, manipulation, and reward hacking

---

### Recommended Reading List

1. Wei et al. (2022) - Emergent Abilities of Large Language Models
2. Brown et al. (2020) - Language Models are Few-Shot Learners
3. Schaeffer et al. (2023) - Are Emergent Abilities a Mirage?
4. Berti et al. (2025) - Emergent Abilities Survey
5. Anderson (1972) - More Is Different