# Did Pre-training Do Its Job?

*The question nobody asks before fine-tuning*

---

There are roughly two stages to training a language model.

- **The first stage, *pre-training*:** The model learns language, the world, and concepts by predicting the next word (*next token prediction*) over trillions of tokens. Think of the model as a blank slate (*tabula rasa*). Intuitively, the words that come before and after a word shape its meaning. Suppose there's a completely made-up word in our language: "**kasdJ**." You can't even pronounce it, but look at these sentences:

  > *I peeled the "kasdJ" and ate it.*
  > *Get a "kasdJ" instead of an apple.*
  > *The "kasdJ" doesn't taste bad, a bit like a pear.*
  > *The easily peelable "kasdJ" reminds me of a tangerine.*

  After reading all these, even without an explicit definition, "kasdJ" has settled somewhere in your mind (probably as a fruit). "Settled somewhere in your mind" is actually a perfect phrase — because mathematically, that's exactly what we do. We map words into a high-dimensional vector space (*embedding space*) and position them close to related concepts. During *pre-training*, the model learns what "kasdJ" means, which concepts it resembles, where it diverges, and what it relates to — all through this process.

- **The second stage, *fine-tuning*:** Here, the pre-trained model is given a specific behavior — adapting to a question-answer format, following instructions (*instruction following*), or specializing in a particular domain. In short: *pre-training* determines **"what the model knows,"** *fine-tuning* determines **"how it presents it."** It's like teaching manners. The model already knows what "kasdJ" is; *fine-tuning* teaches it how to respond when asked "What is kasdJ?", what format to use, and how to address questions. The knowledge is already there — *fine-tuning* just shapes its presentation. Like the difference between a child learning about the world (content/knowledge) and learning how to speak in different situations (behavior/manners).

I hope we've established some common ground or refreshed our memory on the basics so far. It's important that we're on the same page for the rest of this piece :)

---

## Motivation

Let's say you need to build a Turkish finance chatbot. You go to [Hugging Face](https://huggingface.co/) looking for a model with Turkish proficiency. You find one; the model card says "Turkish support," and there's even an *instruction-tuned* version. To make things easier, you grab the ready-made *instruct* (already fine-tuned) version.

You check the metrics; PPL (Perplexity) isn't bad. *(Quick note: Perplexity measures how "surprised" or confused the model is when it encounters new text. The lower the score, the more proficient the model is in that language — it predicts the next word with greater confidence.)* MMLU scores are decent too.

You sit down and *fine-tune*. You have your own Finance QA dataset — a few thousand examples. Training finishes without issues, you deploy it with excitement.

Then weird things start happening.

The model produces grammatically flawless, fluent sentences about "interest rate hikes" — but doesn't know that when interest rates go up, currency tends to fall. Or it doesn't recognize the word for "foreign exchange" at all — not knowing what foreign currencies are called means not knowing the ABCs of the domain. Sometimes the terminology is fine, the relationships are fine, but then you see it writing "from the houses" as "from the house-s-dan" — stumbling on Turkish morphology.

So what happened?

You trusted the numbers on the model card and the benchmark scores, but you never looked at **what the model actually knows and doesn't know** — what was actually etched into its mind during that foundational training.

---

## So What Should We Have Looked At?

"What does this model know?" is actually a very broad and vague question. Ask it that way, and the answer comes back equally vague — "it's generally good" or "not bad." But if we truly want to understand the model, we need to break this question into pieces.

When I evaluate models, I always think in 3 separate layers. Let me be clear — this isn't some arbitrary categorization I made up; the NLP *probing* literature (studies that examine a model's inner world) already has numerous measurement methods along these lines. But given our specific problem, we focus on these three — because what we expect from *pre-training* is precisely these three layers being in place.

*Side note: I'll share the papers and studies I drew on while preparing this piece at the end, for those who want to dig deeper.*

- **Knowing the language** — how well the model has internalized the rules of the target language. Linzen et al. measured this through subject-verb agreement tests.
- **Knowing the world** — what factual knowledge has been etched into the model's mind. Petroni et al.'s LAMA dataset tests exactly this: with questions like "What is the capital of Turkey?"
- **Knowing the relationships between concepts** — how consistently the model can make "if A then B" type inferences. Elazar et al. measured this by checking whether the model gives the same answer under different phrasings.

Understanding that these three layers are independent of each other is crucial. Because the way each forms during training is different, their measurement signatures are different, and the remedies when they're lacking are completely different. For instance, if the model doesn't know the language, *fine-tuning* won't fix it — you need a different model. But if it lacks world knowledge, you might be able to bridge that gap with RAG. We'll touch on each of these later.

This is exactly why aggregate metrics like PPL or MMLU aren't sufficient on their own. When you compress three independent problems into a single score, you can't see which layer is weak and which is strong. The model comes out as "generally 7/10" but that 7 might be grammar at 10 and concept relationships at 2. You don't notice this when you start *fine-tuning* — not until the model starts butchering the grammar.

Let's look at these three layers one by one.

---

### First: Knowing the Language

The most fundamental thing. Does the model know the target language?

"From the houses" or "from the house-s-dan"? "The children are playing" or "the children am playing"? Does it understand word order, morphological rules, tense agreement?

What "knowing a language" means for a human is what I mean for a model. Not just having memorized the rules — but knowing them and being able to use them fluently. The two are inseparable. A person who theoretically knows English subject-verb agreement but writes "the children plays" — you wouldn't say they "know English." The same applies to models. Knowing the language means these rules have settled in and can be applied naturally.

What happens if this is missing? Say the model puts a first-person singular suffix after a plural subject — it means subject-verb agreement hasn't taken hold. This might seem like a small detail at first, but it isn't — it's the foundation of the whole picture. Without morphology and agreement, the model constantly stumbles when generating text, and the output feels off. And worst of all: these errors persist after *fine-tuning*. Because *fine-tuning* doesn't teach you the language — *fine-tuning* teaches the model "how to respond when asked," not the language itself. The language was supposed to have been learned during *pre-training*.

There's an interesting aspect to this layer: it makes no claims about the world. "The children are playing" and "the children are running" should be equally plausible to the model. What we're testing here isn't what the children are doing, just the internal rules of the language. After "the children," a first-person singular suffix can't come — that's it. This is why this layer, when properly established, typically settles very early in *pre-training*. The model can pick up the rhythm of the language before it even knows about the world.

In practice, measuring this is quite simple. You give the model two alternative sentences and check which one it assigns a higher probability to.

```python
compare("the houses", "from", "frum")                    # morphology
compare("The children in the garden", " play.", " plays.")  # agreement
compare("Yesterday evening I", " arrived late.", " will arrive.")  # tense
```

---

### Second: Knowing Factual Knowledge

Let's start with a direct question: "Even if the model knows the language, does it know **what things are**?" (By "know" we mean that this knowledge has been permanently etched as a statistical weight among the model's billions of parameters.)

```python
"The capital of Turkey is ___"              → "Ankara"
"Foreign currencies are called ___"         → "forex"
"The market where stocks are traded is ___" → "stock exchange"
```

Notice these aren't about making connections or reasoning. They're straight-up encyclopedic knowledge — the model has directly "memorized" what something is, what it's called. "The capital of Turkey is Ankara" is a fact. "Foreign currencies are called forex" is terminological knowledge. No reasoning required — it either knows it or it doesn't. If the model saw this often enough during *pre-training*, it knows it. If not, it doesn't. That simple.

This layer operates quite differently from the language layer we just discussed. Chang et al. (NeurIPS 2024) have a nice paper on this: they examined how LLMs learn facts during *pre-training* and found something interesting. Each time the model encounters a fact, its probability of correctly predicting that fact increases a bit. But in subsequent training steps, this increase partially decays — the model forgets some of it (*forgetting*). Factual knowledge isn't etched in one shot; it's reinforced through repetition — much like studying, really. Something you read once gets forgotten; what you review sticks.

The practical consequence: the model knows facts that appear frequently in the *pre-training* corpus well, but doesn't know rare ones. In the literature, this is called the *long-tail knowledge* problem. "The capital of Turkey" appears millions of times across the internet — the model knows that easily. But a niche finance term like "EBITDA"? That's a different story. How often the model saw this term in the corpus, how repeatedly — its performance directly depends on this.

This is exactly why, when building a Turkish finance chatbot, you need to test upfront whether domain terms have settled in the model's mind. Don't let fluent sentences fool you.

```python
complete("The capital of Turkey is")    # "Ankara" should come
complete("Foreign currencies are")      # "forex" should come

compare("The company's EBITDA compared to last year",
        " increased.", " was baked.")
```

---

### Third: Knowing the Relationships Between Concepts

Let's say the model knows the language and the terminology. But does it also know the **relationships** between them?

```python
"We lit the stove, the room ___"
```

Does the model write "warmed up" or "cooled down"? It should say "warmed up," right? This isn't complex reasoning — it's just the most basic association: *"when a stove burns, a room warms up."* Moreover, the "cooled down" alternative violates physical laws, so the model's preference gap should be very clear. Something like 95% to 5%. If it comes out as 52% to 48%, the model doesn't actually know this relationship — it just happened to guess one of two options.

A few more examples:

```python
"It got dark, the children went ___"      → "home" or "to the park"?
"When winter comes, trees ___ their leaves" → "shed" or "sprout"?
```

Now you might ask what's different from the second layer (factual knowledge). There, we tested single associations — "Turkey ↔ Ankara," "foreign currency ↔ forex." One word directly linked to another word. Here, there's a *pattern* across multiple concepts: "when situation A happens, outcome B follows." This is no longer a single fact but a **schema** — a pattern of expectations across concepts settling in the mind. The model might recognize the word "forex" but not know the answer to "what happens when exchange rates rise." The reverse can also be true — it senses the relationship but doesn't recognize the term.

This distinction matters enormously in practice. Why? Because you can largely bridge second-layer gaps with RAG — you put the unknown term in the *context*, the model reads and uses it. But schemas don't work that way. Schemas aren't built at *runtime*; they need to be written into the model's parameters during *pre-training*. You can put "when interest rates rise, currency falls" in the *context*, but if the model can't connect that sentence to "therefore, after today's rate decision, USD/TRY will fall," then having that text there doesn't help. If the schema is missing from the *weights*, context alone doesn't fill that gap.

In the finance domain, we can test this like so:

```python
compare("When interest rates fall, bond prices",
        " rise.", " fall.")

compare("Oil prices rose, increasing production costs, which",
        " raised inflation.", " lowered inflation.")
```

Here, we need to look not just at which answer comes but **how confident** it is. A 90% to 10% result → the model is confident, it knows the relationship. 52% to 48% → it doesn't actually know, just randomly picked one of two options.

---

## Why Should We Measure These Before Fine-tuning?

"Okay, but can't I fix these gaps with fine-tuning?" you might ask. Fair question. For the answer, we need to look at what *fine-tuning* (more technically *SFT* or *instruction tuning*) does — or more precisely, what it was designed to do. Let's look from three angles:

**First, look at the data format.** What's the input to *fine-tuning*? **(question, answer)** pairs. For example:

> *"Question: What is inflation? → Answer: Inflation is the rise in the general level of prices."*

Now think for a moment — is this data pair teaching the model the **concept** of "inflation," or is it teaching the **behavior** of "when you get a 'what is' question, respond in definition format"? Clearly the latter. The model already knows (or doesn't know, separate issue) what "inflation" is; *fine-tuning* only teaches it how to serve that knowledge. When Wei et al. (2022) first defined *instruction tuning*, this is exactly what they said: converting tasks into an instruction format and getting the model accustomed to responding in that format.

**Second, look at the training objective.** What is the model minimizing during training? The *loss* on the answer portion. It's learning "to produce correctly formatted responses" — not "to learn correct information." These are very different things. The first is behavior, the second is content.

**Third, look at the data preparation process.** When preparing a *fine-tuning* dataset, what's on your mind? Collecting diverse questions and quality answers. Even when talking about data quality, you're looking at "How well-formatted and consistent are the answers?" — not "Do the answers impart new knowledge to the model?" The person preparing the data is focused on defining behavior, not injecting knowledge.

The design, purpose, data format, training objective, and data preparation all tell the same story. None of them are designed to impart knowledge to the model. None are designed to teach grammar. None are designed to build relationships between concepts. So where will these capabilities come from? One place only: *Pre-training*.

This isn't a new idea either. The LIMA paper (Zhou et al., NeurIPS 2023) formalized this as the "Superficial Alignment Hypothesis": *"A model's knowledge and capabilities are learned almost entirely during pre-training; alignment teaches the model which format sub-distribution to use when interacting with users."* This hypothesis has been debated since then; subsequent work refined, critiqued, and nuanced certain aspects. For instance, a recent study (Vergara-Browne et al., 2026) placed the same claim on an information-theoretic foundation and showed: *Pre-training* dramatically reduces the cost of achieving good performance on a task. That is, *fine-tuning* doesn't create capabilities from scratch — it opens easy access to capabilities that are already there.

So the real question isn't *"Can fine-tuning do this?"* The real question is: ***Fine-tuning* wasn't designed for this.** Teaching grammar isn't *fine-tuning*'s job. Teaching "what is forex" isn't *fine-tuning*'s job. Teaching "when rates rise, currency falls" isn't *fine-tuning*'s job.

These capabilities come from *pre-training*. That's why before starting *fine-tuning*, you need to check whether these capabilities are already in place. Otherwise, you're trying to build behavior on top of something that doesn't exist — and that doesn't work.

> **A small but bitter note:** This isn't just a theoretical concern — it's been empirically proven. Gekhman et al. (EMNLP 2024) demonstrated that when you try to inject new facts into a model via *fine-tuning*, the model's tendency to hallucinate increases linearly. So *fine-tuning* not only fails to impart these capabilities — when forced to do so, it actively damages the model.

What do we do if one or more of these three layers is lacking? Briefly: if the language is missing, fine-tuning won't save you — you either find a different model or do a Turkish CPT (continual pre-training). If factual knowledge is missing, at least for narrow terminology, RAG is usually sufficient; for broad domain knowledge, CPT again. If conceptual relationships are missing — that's the hardest case. RAG isn't enough; domain CPT is essential. And there's one dangerous combination: facts are fine but relationships are missing. The model uses the right terms but draws wrong inferences. In serious domains like finance, this is the worst failure mode — surface-level fluency with hollow relational ground underneath.

---

## Do Current Metrics Show This?

Now you'll rightfully ask: "If this is the case, why does everyone look at PPL and benchmark scores to choose models? Aren't those sufficient?" Let me be direct: They're not.

PPL, benchmark scores, BPC (Bits Per Character) — these are all *aggregate* metrics. They answer the question "Is the model generally good?" But we've already seen above that there's no such thing as "general" — there are three completely independent layers. These metrics take all three layers, blend them into a single score, and present it to you. You look at that single score and say "Okay, this seems like a good model."

Think of it this way: imagine two different models both with a PPL score of 15. The first has flawless grammar but near-zero factual knowledge. The second has excellent terminology but grammar that constantly falls apart. These two models can have exactly the same PPL score, yet one works perfectly for your finance chatbot while the other is completely useless. Someone looking only at that single score would never see this critical difference.

And believe me, this isn't just a theoretical concern either. Ankner et al. (ICLR 2025), in their paper "Perplexed by Perplexity," demonstrated this very concretely: models trained with *perplexity-based pruning* show worse PPL scores on pre-training test data; yet despite this, they show improvements of up to 2 points in *downstream task* accuracy! That is, PPL improving and actual performance improving are not just unrelated (uncorrelated) — they sometimes move in exactly opposite directions. PPL can worsen while task performance improves, or vice versa.

I summarize this situation like so: a PPL score answers *"Does the patient have a fever?"* It's useful information, yes, but insufficient on its own. Because a fever can have many causes — is the problem in the lungs, the kidneys, or the heart? The layer tests we've been discussing throughout this piece are searching for exactly the answer to: **Which organ?**

---

## Summary: What Measures What?

Let's compile the tests we've discussed throughout this piece and which layer each one measures in a quick table:

| Test | What It Asks | Which Layer |
|------|-------------|-------------|
| Vowel harmony, subject-verb agreement | "Does it know the fundamental rules of the language?" | Knowing the language |
| Word order tests | "Does it recognize natural sentence structure?" | Knowing the language |
| Fact completion ("The capital of X is ___") | "Does it have encyclopedic knowledge?" | Knowing the world |
| Term recognition ("EBITDA", "forex", etc.) | "Does it recognize domain terms?" | Knowing the world |
| Contrastive pairs ("When A happens, B follows") | "Can it establish relationships between concepts?" | Knowing the relationships |
| Cloze completion (contextual gap-filling) | "Can it select contextually appropriate relationships?" | Knowing the relationships |

And a reminder: aggregate metrics like PPL and MMLU are certainly used too, but they're not sufficient on their own. Because they compress three layers into a single score, they don't tell you which is weak and which is strong. We run these tests **in addition to** those metrics, not instead of them.

---

*This piece was the first in a trilogy I had in mind. First, we discussed what we expect from pre-training and how to diagnose those expectations — how to ask "which of the patient's organs is healthy?"*

*In the next piece, we'll dive into the details of each test in the table: which probing tradition it comes from, how to implement it, which thresholds are meaningful, and how to interpret the model's responses. Think of it as a "Turkish LLM evaluation handbook."*

*In the third piece, we move from theory to practice: we'll compare Kumru-2B (base) with Kara-Kumru (fine-tuned). We'll run 150+ tests and see whether these three layers actually changed after fine-tuning.*

---

## References

[1] Wei, J., et al. (2022). Finetuned Language Models Are Zero-Shot Learners. *ICLR 2022.*

[2] Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. *NeurIPS 2023.*

[3] Vergara-Browne, T., Patil, D., Titov, I., Reddy, S., Pimentel, T., Mosbach, M. (2026). Operationalising the Superficial Alignment Hypothesis via Task Complexity. *arXiv:2602.15829.*

[4] Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., Seo, M. (2024). How Do Large Language Models Acquire Factual Knowledge During Pretraining? *NeurIPS 2024.*

[5] Gekhman, Z., et al. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? *EMNLP 2024.*

[6] Ankner, Z., Blakeney, C., Sreenivasan, K., Marion, M., Leavitt, M. L., Paul, M. (2025). Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models. *ICLR 2025.*

[7] Linzen, T., Dupoux, E., Goldberg, Y. (2016). Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies. *TACL.*

[8] Petroni, F., et al. (2019). Language Models as Knowledge Bases? *EMNLP.*

[9] Elazar, Y., et al. (2021). Measuring and Improving Consistency in Pretrained Language Models. *TACL.*

---
