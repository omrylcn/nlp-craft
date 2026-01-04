# BERT Text Embeddings and Similarity Analysis: Comprehensive Guide

## Table of Contents
- [1. Introduction and Theoretical Foundation](#1-introduction-and-theoretical-foundation)
  - [1.1. What are Text Embeddings?](#11-what-are-text-embeddings)
  - [1.2. Semantic Similarity Concept](#12-semantic-similarity-concept)
  - [1.3. BERT Embedding Types](#13-bert-embedding-types)
- [2. Creating Text Embeddings with BERT](#2-creating-text-embeddings-with-bert)
  - [2.1. Types of BERT Models](#21-types-of-bert-models)
  - [2.2. Pooling Strategies](#22-pooling-strategies)
  - [2.3. SentenceTransformers Library](#23-sentencetransformers-library)
- [3. Text Similarity Computation](#3-text-similarity-computation)
  - [3.1. Similarity Metrics](#31-similarity-metrics)
  - [3.2. Normalization Techniques](#32-normalization-techniques)
  - [3.3. Semantic Search and Retrieval](#33-semantic-search-and-retrieval)

## 1. Introduction and Theoretical Foundation

### 1.1. What are Text Embeddings?

Text embeddings are techniques that convert text data into numerical vectors. These vectors represent the semantic content of texts in numerical space, enabling computers to "understand" texts.

**Properties of Text Embeddings:**

- **Semantic Representation**: Words/sentences with similar meanings are positioned close together in vector space.
- **Dimensionality**: Generally vectors with hundreds or thousands of dimensions (e.g., BERT-base: 768 dimensions).
- **Dense Vectors**: Unlike sparse vectors like one-hot encoding, they encode information more efficiently.
- **Contextual Information**: Especially models like BERT produce different embeddings for a word depending on its context within a sentence.

**Types of Embeddings:**

1. **Word Embeddings**:
   - Models like Word2Vec, GloVe, FastText create word-level representations
   - A word has a single fixed representation
   - Limited ability to capture context

2. **Contextual Embeddings**:
   - Created by models like BERT, RoBERTa, GPT
   - A word's representation changes based on the sentence it's in
   - Contains much richer semantic information

3. **Sentence Embeddings**:
   - Represents an entire sentence/paragraph with a single vector
   - Generally obtained through some type of averaging or pooling of token embeddings
   - Suitable for document comparison, similarity, search tasks

**Use Cases for Embeddings:**
- Semantic search
- Document classification and clustering
- Recommendation systems
- Feature extraction for language models
- Similarity and relationship analysis
- Information extraction and question answering systems

### 1.2. Semantic Similarity Concept

Semantic similarity is a concept that measures how close two pieces of text are in meaning. Unlike traditional lexical matching approaches, semantic similarity targets meaning.

**Semantic vs. Lexical Similarity:**

- **Lexical Similarity**: Based on surface features like common word count, n-gram overlap
  ```
  "The dog is barking" vs "The dog is barking" → High lexical similarity
  "The dog is barking" vs "The cat is meowing" → Low lexical similarity
  ```

- **Semantic Similarity**: Based on meaning proximity
  ```
  "The dog is barking" vs "The canine is making sounds" → High semantic similarity
  "I opened a bank account" vs "I sat on the river bank" → Low semantic similarity (different meanings)
  ```

**Types of Semantic Similarity:**

1. **Paragraph/Document Similarity**: Comparing long texts, finding similar documents

2. **Sentence Similarity**: Semantic proximity of two sentences, paraphrase detection, semantic equivalence

3. **Word-Sentence Similarity**: Measuring the relevance of a word to a sentence

4. **Cross-lingual Similarity**: Measuring meaning similarity of texts in different languages

**Semantic Similarity Measures:**

- **Cosine Similarity**: Measures the angle between two vectors (most common method)
- **Euclidean Distance**: Measures physical distance between two vectors
- **Manhattan Distance**: Measures horizontal+vertical distance between two vectors
- **Dot Product**: Dot product of two vectors (usually after normalization)

**Challenges of Semantic Similarity:**

- **Context Sensitivity**: Distinguishing different meanings of "bank" in different contexts
- **Polysemy**: Different meanings of the same word
- **Synonymy**: Same meaning of different words
- **Domain-Specific Language**: Word meanings can change in different domains
- **Cultural and Linguistic Nuances**: Idioms, metaphors, cultural references

### 1.3. BERT Embedding Types

BERT (Bidirectional Encoder Representations from Transformers) produces rich embeddings that can represent text at different levels. These embeddings capture complex semantic and syntactic relationships within text data.

**Embedding Types Produced by BERT:**

1. **Token Embeddings**:
   - A vector for each token (typically 768-dimensional)
   - Context-sensitive (same word has different embeddings in different sentences)
   - Captures syntactic and semantic information
   - Usually taken from BERT's last layer (or different layers for specific tasks)

2. **[CLS] Token Embedding**:
   - Representation of the special [CLS] token added to the beginning of every BERT input
   - Specifically trained to carry the holistic representation of the entire sentence/paragraph
   - Generally used for classification tasks
   - Can also be used as sentence embedding in some cases

3. **Sentence Embeddings**:
   - Specifically designed to create a complete representation of a sentence
   - Can be obtained by different methods:
     - Using [CLS] token representation
     - Taking average of all token representations (mean pooling)
     - Taking maximum of all token representations (max pooling)
     - Combining first and last tokens
     - Weighted averaging with attention mechanisms

4. **Cross-Layer BERT Embeddings**:
   - Different layers of BERT capture different linguistic features:
   - Lower layers: Syntactic information, phrase structures, basic semantic relationships
   - Middle layers: Complex semantic structures, relationships
   - Upper layers: Task-specific information, more abstract representations

**Embedding Example - BERT Output for a Sentence:**

When BERT model processes the sentence "The weather warmed up, let's go on a picnic":
```
[CLS] The weather warmed up, let's go on a picnic [SEP]
```

It produces embeddings like:
- [CLS] token embedding: A 768-dimensional vector containing the representation of the entire sentence
- "The" token embedding: A 768-dimensional vector
- "weather" token embedding: A 768-dimensional vector
- "warmed" token embedding: A 768-dimensional vector
- ...etc.

**Important Note**: In BERT's original architecture, there is no direct "sentence embedding" concept. Generally, [CLS] token or some type of averaging (pooling) of token embeddings is applied to obtain sentence-level representations. To solve this problem, models specifically fine-tuned to produce sentence embeddings have been developed (e.g., Sentence-BERT).

## 2. Creating Text Embeddings with BERT

### 2.1. Types of BERT Models

BERT models come in different sizes and optimized forms for various tasks. The main BERT variations you can use for creating embeddings:

**Base BERT Models:**

- **BERT-base**:
  - 12 transformer layers, 768 hidden dimension, 12 attention heads
  - ~110M parameters
  - Good balance for general-purpose use

- **BERT-large**:
  - 24 transformer layers, 1024 hidden dimension, 16 attention heads
  - ~340M parameters
  - Offers higher quality embeddings but requires more resources

**Optimized BERT Variations:**

- **DistilBERT**:
  - Distilled version of BERT, 40% smaller, 60% faster
  - 6 transformer layers
  - Retains ~97% of BERT-base's performance
  - Ideal for limited resources

- **ALBERT (A Lite BERT)**:
  - Provides memory efficiency through parameter sharing technique
  - Similar performance with fewer parameters
  - Especially effective for large models

**Semantic Similarity-Focused Models:**

- **Sentence-BERT / SentenceTransformers**:
  - Fine-tuned specifically to produce meaningful sentence embeddings
  - Uses Siamese or triplet network architectures
  - Produces high-quality sentence embeddings
  - Cosine similarity can directly measure sentence similarity

- **MPNet**:
  - Masked and permuted pre-training
  - Combines advantages of BERT and permuted language modeling
  - Shows superior performance as sentence embedding

**Domain-Specific and Multilingual BERT Models:**

- **mBERT (multilingual BERT)**:
  - Trained on 104 languages
  - Suitable for cross-lingual transfer learning
  - Can produce multilingual embeddings

- **Domain-Specific BERTs**:
  - BioBERT: Optimized for biomedical texts
  - SciBERT: Optimized for scientific publications
  - FinBERT: Optimized for financial texts
  - LegalBERT: Optimized for legal texts

**Models That Improve BERT:**

- **RoBERTa**:
  - Uses BERT architecture but developed with larger data and better training strategy
  - Provides better performance and more reliable embeddings

- **DeBERTa**:
  - Adds disentangled attention mechanisms
  - Separately encodes content and position
  - Offers state-of-the-art performance

**Criteria for Model Selection:**

- **Task suitability**: Sentence-BERT/SentenceTransformers for semantic similarity
- **Language support**: mBERT or XLM-RoBERTa for multilingual tasks
- **Speed-quality balance**: DistilBERT or MiniLM for limited resources
- **Domain**: Domain-specific models preferred for specialized areas

### 2.2. Pooling Strategies

BERT produces token-level embeddings, but to get sentence or document-level representations, these token embeddings need to be converted into a single vector. Various pooling strategies are used for this transformation.

**Basic Pooling Strategies:**

1. **CLS Token Pooling**:
   - Uses the representation of [CLS] token added to the beginning of every BERT input
   - Designed to contain the complete representation of the sentence
   - Advantages:
     - Simple and direct implementation
     - Uses the part of the model specifically trained for sentence representation
   - Disadvantages:
     - May be less effective than other pooling strategies for some tasks
     - Information loss may occur with very long texts

2. **Mean Pooling (Average Pooling)**:
   - Takes the average of all token representations (usually excluding padding and special tokens)
   - Advantages:
     - Uses information from all tokens
     - Generally gives best performance for sentence similarity
     - Provides fixed-size representation even for long texts
   - Disadvantages:
     - Important information may be diluted in long texts

3. **Max Pooling**:
   - Takes the maximum value of all token representations for each dimension
   - Advantages:
     - Captures the most prominent features
     - Length-independent representation
   - Disadvantages:
     - May miss overall semantic structure
     - Generally shows lower performance than mean pooling

4. **Weighted Pooling**:
   - Takes weighted average by assigning different weights to tokens
   - Types of weighting:
     - IDF weighting: Higher weight to rarer words
     - Attention weighting: Using self-attention scores
     - Learnable weights: Fine-tuning related to task
   - Advantages:
     - Can give more importance to more meaningful words
     - Can better capture semantic content of sentence
   - Disadvantages:
     - Complex implementation
     - May require additional computation

5. **Concat Pooling**:
   - Combines results of multiple pooling strategies (e.g., CLS + Mean + Max)
   - Advantages:
     - Combines advantages of different pooling strategies
     - Richer representation
   - Disadvantages:
     - Higher dimensional vectors
     - Higher computational cost

**Code Example - Different Pooling Strategies:**

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Prepare and tokenize text
sentences = ["This is an example sentence.", "Let's see pooling strategies."]
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Get model output
with torch.no_grad():
    model_output = model(**encoded_input)

# Get last layer token representations
token_embeddings = model_output.last_hidden_state  # [batch_size, sequence_length, hidden_size]
input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

# 1. CLS Token Pooling
cls_embeddings = token_embeddings[:, 0, :]
print("CLS Token Embedding size:", cls_embeddings.shape)

# 2. Mean Pooling
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = input_mask_expanded.sum(1)
mean_embeddings = sum_embeddings / sum_mask
print("Mean Pooling Embedding size:", mean_embeddings.shape)

# 3. Max Pooling - applying attention mask
# First assign very small value (-1e9) for unmasked tokens
token_embeddings_masked = token_embeddings * input_mask_expanded + (1 - input_mask_expanded) * -1e9
max_embeddings = torch.max(token_embeddings_masked, 1)[0]
print("Max Pooling Embedding size:", max_embeddings.shape)

# 4. Weighted Mean Pooling (simple example - more weight to last words)
weights = torch.linspace(0.1, 1.0, steps=token_embeddings.size(1)).unsqueeze(0).unsqueeze(-1)
weights = weights.expand_as(token_embeddings) * input_mask_expanded
weighted_sum = torch.sum(token_embeddings * weights, 1)
weighted_mean_embeddings = weighted_sum / weights.sum(1)
print("Weighted Mean Pooling Embedding size:", weighted_mean_embeddings.shape)

# 5. Concat Pooling (CLS + Mean)
concat_embeddings = torch.cat([cls_embeddings, mean_embeddings], dim=-1)
print("Concat Pooling Embedding size:", concat_embeddings.shape)

# Normalize embeddings (for cosine similarity)
cls_embeddings_normalized = F.normalize(cls_embeddings, p=2, dim=1)
mean_embeddings_normalized = F.normalize(mean_embeddings, p=2, dim=1)
```

**Which Pooling Strategy to Choose:**

- **For semantic similarity**: Mean Pooling generally gives best results
- **For classification tasks**: CLS Token Pooling or Mean Pooling
- **For information extraction**: Weighted Pooling or Max Pooling can focus on informative words
- **For best performance**: Try different strategies and compare on validation set

### 2.3. SentenceTransformers Library

SentenceTransformers is a library containing BERT-based models specifically designed to produce meaningful sentence embeddings. Unlike standard BERT models, SentenceTransformers models are optimized directly for tasks like semantic similarity, information extraction, and search.

**Features of SentenceTransformers:**

1. **Purpose-Specific Training**:
   - Fine-tuned on NLI (Natural Language Inference) and STS (Semantic Textual Similarity) datasets
   - Trained using Siamese and triplet network architectures
   - Optimized to produce meaningful sentence embeddings

2. **Ease of Use**:
   - Easy usage with simple API
   - Can directly produce sentence embeddings
   - Various similarity functions integrated

3. **Model Variety**:
   - 100+ specialized models in 30+ different languages
   - Different model sizes (small, medium, large)
   - Models optimized for different tasks

4. **Performance**:
   - Much higher performance on semantic similarity tasks compared to standard BERT models
   - Faster operation with optimized architecture

**Creating Sentence Embeddings with SentenceTransformers:**

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Small and fast model

# Embedding for single sentence
sentence = "This is an example sentence."
embedding = model.encode(sentence)
print(f"Embedding size: {embedding.shape}")  # (384,) - 384 dimensional vector

# Embeddings for multiple sentences (batch processing)
sentences = [
    "SentenceTransformers is great for sentence embeddings.",
    "It is an ideal solution for semantic search.",
    "Works with BERT-based models."
]

# Compute embeddings - default mean pooling is used
embeddings = model.encode(sentences)
print(f"Batch embeddings size: {embeddings.shape}")  # (3, 384) - 3 sentences, each 384 dimensional
```

**Different SentenceTransformer Models and Use Cases:**

1. **General Purpose Models**:
   - `paraphrase-MiniLM-L6-v2`: Fast and compact (384 dimensions)
   - `all-mpnet-base-v2`: Best performance (768 dimensions)
   - `all-MiniLM-L12-v2`: Good performance/speed balance (384 dimensions)

2. **Multilingual Models**:
   - `paraphrase-multilingual-MiniLM-L12-v2`: 50+ language support
   - `distiluse-base-multilingual-cased-v1`: Distilled model for 15 languages

3. **Specialized Task Models**:
   - `msmarco-distilbert-base-v4`: Optimized for document search
   - `multi-qa-MiniLM-L6-cos-v1`: Optimized for question-answering systems
   - `nli-distilroberta-base-v2`: Optimized for natural language inference

**Advanced Usage with SentenceTransformers:**

```python
from sentence_transformers import SentenceTransformer, util
import torch

# Load more advanced model
model = SentenceTransformer('all-mpnet-base-v2')  # For better performance

# Custom encoding parameters
embeddings = model.encode(
    sentences,
    batch_size=32,           # Batch size
    show_progress_bar=True,  # Show progress bar
    convert_to_tensor=True,  # Return as PyTorch tensor
    normalize_embeddings=True  # Apply L2 normalization (for cosine similarity)
)

# Compute cosine similarity between embeddings
cosine_scores = util.cos_sim(embeddings, embeddings)
print("Cosine similarity matrix:")
print(cosine_scores)

# Find most similar sentence pairs
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j].item()})

# Sort by similarity score
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

# Print most similar pairs
for pair in pairs:
    i, j = pair['index']
    print(f"Similarity score: {pair['score']:.4f}")
    print(f"Sentence 1: {sentences[i]}")
    print(f"Sentence 2: {sentences[j]}\n")
```

**Custom Model Training with SentenceTransformers:**

SentenceTransformers library also supports training models on your own custom dataset.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Training examples
train_examples = [
    InputExample(texts=['This is an apple.', 'This is a fruit.'], label=0.8),
    InputExample(texts=['The dog is barking.', 'The cat is meowing.'], label=0.3),
    # More examples...
]

# Create training dataloader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Select base model for training
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Define loss function (CosineSimilarityLoss - labels should be between -1 and 1,
# since 0-1 range is used here, scaling is done)
train_loss = losses.CosineSimilarityLoss(model)

# Train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    evaluation_steps=1000,
    output_path='my-custom-model'
)
```

**Advantages of SentenceTransformers:**

1. Embeddings directly usable for semantic search, similarity computation, and information extraction
2. Much better performance for sentence representations compared to standard BERT
3. Easy-to-use API
4. Models optimized for various tasks and languages
5. Computational efficiency - pre-computed embeddings enable high-speed similarity computation

## 3. Text Similarity Computation

### 3.1. Similarity Metrics

Various metrics are used to calculate the semantic similarity of two pieces of text. These metrics convert the relationship between text embeddings into a numerical value.

**Common Similarity Metrics:**

1. **Cosine Similarity**:
   - Measures the cosine of the angle between two vectors
   - Takes values between -1 (completely opposite) and 1 (completely same)
   - Formula: $\text{cosine}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||}$
   - Advantages:
     - Independent of vector magnitude (only considers direction)
     - Works well in high-dimensional spaces
     - Most commonly used metric

   ```python
   def cosine_similarity(vec1, vec2):
       dot_product = np.dot(vec1, vec2)
       norm_vec1 = np.linalg.norm(vec1)
       norm_vec2 = np.linalg.norm(vec2)
       return dot_product / (norm_vec1 * norm_vec2)
   ```

2. **Euclidean Distance**:
   - Measures direct distance between two vectors
   - Takes values between 0 (completely same) and ∞ (infinitely different)
   - Formula: $\text{euclidean}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$
   - Advantages:
     - Intuitive and easy to understand
     - Has direct meaning in physical space
   - Disadvantages:
     - Sensitive to vector magnitude
     - "Curse of dimensionality" problem in high-dimensional spaces

   ```python
   def euclidean_distance(vec1, vec2):
       return np.sqrt(np.sum((vec1 - vec2) ** 2))
   ```

3. **Manhattan Distance**:
   - Measures "city block" distance between two vectors
   - Sum of absolute values of differences in each dimension
   - Formula: $\text{manhattan}(A, B) = \sum_{i=1}^{n} |A_i - B_i|$
   - Advantages:
     - Can be more robust with noisy data
     - Gives more meaningful results in some applications
     - Faster to compute

   ```python
   def manhattan_distance(vec1, vec2):
       return np.sum(np.abs(vec1 - vec2))
   ```

4. **Dot Product**:
   - Sum of products between two vectors
   - Sensitive to vector magnitude
   - Formula: $\text{dot}(A, B) = \sum_{i=1}^{n} A_i \cdot B_i$
   - Usually used with normalized vectors (in which case it equals cosine similarity)

   ```python
   def dot_product(vec1, vec2):
       return np.dot(vec1, vec2)
   ```

5. **Jaccard Similarity**:
   - Measures the ratio of intersection to union of two sets
   - Takes values between 0 (completely different) and 1 (completely same)
   - Formula: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
   - Usually used for sparse vectors or token sets

   ```python
   def jaccard_similarity(set1, set2):
       intersection = len(set1.intersection(set2))
       union = len(set1.union(set2))
       return intersection / union if union != 0 else 0
   ```

**Metric Selection Strategies:**

- **Cosine Similarity**: General-purpose semantic similarity, especially used with normalized vectors
- **Euclidean Distance**: When vector magnitude is also meaningful
- **Manhattan Distance**: When working with noisy data or feature vectors
- **Dot Product**: For fast computation with normalized vectors
- **Jaccard Similarity**: When working with token sets, sparse vectors, or binary features

**Code Example - Comparing Different Similarity Metrics:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean, cityblock
import torch
import torch.nn.functional as F

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Test sentences
sentences = [
    "Artificial intelligence is developing rapidly today.",
    "Machine learning technologies have advanced a lot in recent years.",
    "Cats are generally independent animals.",
    "Cats have an independent nature."
]

# Compute embeddings
embeddings = model.encode(sentences, convert_to_numpy=True)

# Compare different metrics
print("Sentence pairs and similarity scores:\n")

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        # Cosine similarity: 1 - cosine_distance (close to 1 = similar)
        cosine_sim = 1 - cosine(embeddings[i], embeddings[j])

        # Euclidean distance (close to 0 = similar)
        euc_distance = euclidean(embeddings[i], embeddings[j])

        # Manhattan distance (close to 0 = similar)
        manhattan_dist = cityblock(embeddings[i], embeddings[j])

        # Average length for normalization
        avg_length = (np.linalg.norm(embeddings[i]) + np.linalg.norm(embeddings[j])) / 2

        # Normalized distances (between 0 and 1, 1 = similar)
        norm_euc_sim = 1 / (1 + euc_distance / avg_length)
        norm_manhattan_sim = 1 / (1 + manhattan_dist / avg_length)

        print(f"Sentence 1: {sentences[i]}")
        print(f"Sentence 2: {sentences[j]}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"Normalized Euclidean Similarity: {norm_euc_sim:.4f}")
        print(f"Normalized Manhattan Similarity: {norm_manhattan_sim:.4f}")
        print("-" * 50)
```

### 3.2. Normalization Techniques

Normalization of embedding vectors is a critical step to improve the accuracy and efficiency of similarity computations. Normalization reduces the effect of different magnitudes of vectors and helps obtain more consistent similarity scores.

**Reasons for Normalization:**

1. **Reducing Scale Effects**: Eliminates differences caused by vector magnitudes
2. **Speeding Up Cosine Similarity Calculations**: With normalized vectors, cosine similarity can be found with just dot product calculation
3. **Numerical Stability**: Prevents numerical issues caused by very large or very small values
4. **Improving Model Performance**: Generally better similarity scores are obtained with normalized vectors

**Common Normalization Techniques:**

1. **L2 Normalization (Unit Vector Normalization)**:
   - Scales vector to unit length (1)
   - Formula: $v_{norm} = \frac{v}{||v||_2}$, where $||v||_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$
   - This preserves the direction of the vector but equalizes its magnitude to 1
   - Ideal especially for cosine similarity

   ```python
   def l2_normalize(vector):
       norm = np.linalg.norm(vector)
       return vector / norm if norm > 0 else vector
   ```

2. **L1 Normalization**:
   - Normalization according to sum of absolute values of vector elements
   - Formula: $v_{norm} = \frac{v}{||v||_1}$, where $||v||_1 = \sum_{i=1}^{n} |v_i|$
   - Can be useful when working with Manhattan distance

   ```python
   def l1_normalize(vector):
       norm = np.sum(np.abs(vector))
       return vector / norm if norm > 0 else vector
   ```

3. **Min-Max Normalization**:
   - Scales vector to a specific range (usually [0,1])
   - Formula: $v_{norm} = \frac{v - \min(v)}{\max(v) - \min(v)}$
   - Brings each feature to the same scale
   - Sensitive to outliers

   ```python
   def min_max_normalize(vector):
       min_val = np.min(vector)
       max_val = np.max(vector)
       range_val = max_val - min_val
       return (vector - min_val) / range_val if range_val > 0 else vector
   ```

4. **Z-Score Normalization**:
   - Transforms vector to have mean 0, standard deviation 1
   - Formula: $v_{norm} = \frac{v - \mu}{\sigma}$
   - Works well with normally distributed data
   - Preserves outliers but reduces their effects

   ```python
   def z_score_normalize(vector):
       mean = np.mean(vector)
       std = np.std(vector)
       return (vector - mean) / std if std > 0 else vector
   ```

**Normalization Techniques with Different Libraries:**

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sk_normalize

# Example vector
vector = np.array([0.2, 0.5, -0.3, 1.2, 0.8])

# L2 normalization with NumPy
np_l2_norm = vector / np.linalg.norm(vector)
print("NumPy L2 norm:", np_l2_norm)

# L2 normalization with PyTorch
pt_vector = torch.tensor(vector, dtype=torch.float32)
pt_l2_norm = F.normalize(pt_vector, p=2, dim=0)
print("PyTorch L2 norm:", pt_l2_norm.numpy())

# Normalization with Scikit-learn
sk_l2_norm = sk_normalize(vector.reshape(1, -1), norm='l2')[0]
sk_l1_norm = sk_normalize(vector.reshape(1, -1), norm='l1')[0]
print("Scikit-learn L2 norm:", sk_l2_norm)
print("Scikit-learn L1 norm:", sk_l1_norm)

# Normalization with SentenceTransformers
from sentence_transformers import util
st_vector = torch.tensor(vector, dtype=torch.float32).reshape(1, -1)
st_l2_norm = util.normalize_embeddings(st_vector)
print("SentenceTransformers norm:", st_l2_norm.numpy()[0])
```

**Embedding Normalization with SentenceTransformers:**

```python
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sentences
sentences = [
    "This is an example sentence.",
    "This is another example."
]

# Automatic normalization during encoding
embeddings_normalized = model.encode(
    sentences,
    normalize_embeddings=True  # Applies L2 normalization
)

# Manual normalization
embeddings_raw = model.encode(sentences, normalize_embeddings=False)
embeddings_manual_norm = util.normalize_embeddings(torch.tensor(embeddings_raw))

# Checking L2 norms
l2_norms_auto = np.linalg.norm(embeddings_normalized, axis=1)
l2_norms_manual = np.linalg.norm(embeddings_manual_norm, axis=1)

print("L2 norms of auto-normalized embeddings:", l2_norms_auto)
print("L2 norms of manually normalized embeddings:", l2_norms_manual)
```

**Normalization Selection Criteria:**

- **For Cosine Similarity**: L2 normalization (most common usage)
- **For Manhattan Distance**: L1 normalization
- **For data at different scales**: Min-Max or Z-score normalization
- **If having numerical stability issues**: Min-Max normalization
- **If performance is priority**: Cosine similarity + L2 normalization combination

### 3.3. Semantic Search and Retrieval

Semantic search is the process of finding semantically relevant content using text embeddings. Unlike traditional keyword-based search, semantic search considers the semantic similarity of words and thus can achieve more relevant results.

**Advantages of Semantic Search:**

1. **Understanding Synonyms**: Can understand synonymous words like "automobile", "car", "vehicle"
2. **Contextual Understanding**: Considers the context of words
3. **Concept-Based Search**: Focuses on conceptual matching rather than exact word matching
4. **Better User Experience**: Better understands user's query intent

**Semantic Search Process:**

1. **Preprocessing (Offline)**:
   - Compute embeddings of all documents in corpus
   - Save embeddings to an index for efficient search

2. **Search (Online)**:
   - Compute embedding of user query
   - Compare embedding with document embeddings (compute similarity)
   - Sort and return most similar documents by similarity scores

**Simple Semantic Search Implementation:**

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Prepare corpus - documents to search
corpus = [
    "Artificial intelligence is systems that mimic human intelligence.",
    "Deep learning is a subfield in the area of artificial intelligence.",
    "BERT is a model that revolutionized natural language processing.",
    "Transformers architecture is based on attention mechanisms.",
    "Python is a popular programming language for AI applications.",
    "Machine learning learns patterns from data.",
    "NLP enables computers to understand human language.",
    "GPT models are widely used for text generation.",
    "Embedding vectors numerically represent words or sentences.",
    "Semantic search is a search method based on semantic similarity."
]

# Compute corpus embeddings - preprocessing
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# User queries
queries = [
    "What is artificial intelligence?",
    "Information about BERT and transformer models",
    "How do vector-based text representations work?",
    "Machine learning with Python"
]

# Semantic search for each query
top_k = 3  # How many results to show for each query

for query in queries:
    # Compute query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine-similarity with corpus
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # Get documents with highest similarity
    top_results = torch.topk(cos_scores, k=top_k)

    print(f"\nQuery: {query}")
    print(f"Top {top_k} most relevant results:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Score: {score:.4f} - {corpus[idx]}")
```

**Optimization for Large-Scale Semantic Search:**

For large datasets, efficient Approximate Nearest Neighbor (ANN) algorithms are used instead of brute-force similarity computation.

```python
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create larger corpus (as example)
corpus = [
    "Artificial intelligence is systems that mimic human intelligence." + str(i)
    for i in range(10000)  # 10,000 documents
]

# Compute corpus embeddings
corpus_embeddings = model.encode(corpus, show_progress_bar=True)
embedding_size = corpus_embeddings.shape[1]  # Embedding dimension

# Normalize embeddings (optional, for inner product search)
faiss.normalize_L2(corpus_embeddings)

# Create FAISS index
index = faiss.IndexFlatIP(embedding_size)  # IP = Inner Product (for cosine sim)
# Alternative: For acceleration
# index = faiss.IndexIVFFlat(quantizer, embedding_size, 100, faiss.METRIC_INNER_PRODUCT)
# index.train(corpus_embeddings)

# Add vectors to index
index.add(corpus_embeddings)

# Search query
query = "What is artificial intelligence?"
query_embedding = model.encode([query])[0]
faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize

# Find k most similar documents
k = 5
distances, indices = index.search(query_embedding.reshape(1, -1), k)

# Show results
print(f"Query: {query}")
for i in range(k):
    print(f"Score: {distances[0][i]:.4f} - {corpus[indices[0][i]][:50]}...")
```

**Other Advanced Semantic Search Techniques:**

1. **Hybrid Search (BM25 + Semantic Search)**:
   - Combines advantages of lexical search and semantic search
   - Provides higher accuracy and coverage

   ```python
   from sentence_transformers import SentenceTransformer, util
   from rank_bm25 import BM25Okapi
   import numpy as np

   # Load model
   model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

   # Prepare corpus
   corpus = [
       "Artificial intelligence is systems that mimic human intelligence.",
       "Deep learning is a subfield in the area of artificial intelligence.",
       # ...other documents
   ]

   # Tokenize corpus for BM25
   tokenized_corpus = [doc.lower().split() for doc in corpus]
   bm25 = BM25Okapi(tokenized_corpus)

   # Embeddings for semantic search
   corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

   # Hybrid search function
   def hybrid_search(query, top_k=3, alpha=0.5):
       # Compute BM25 score
       tokenized_query = query.lower().split()
       bm25_scores = bm25.get_scores(tokenized_query)

       # Min-max normalization
       bm25_scores = (bm25_scores - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-6)

       # Compute semantic scores
       query_embedding = model.encode(query, convert_to_tensor=True)
       semantic_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

       # Compute hybrid score
       hybrid_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

       # Get documents with highest scores
       top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

       return [(hybrid_scores[idx], corpus[idx]) for idx in top_indices]
   ```

2. **Dense Passage Retrieval (DPR)**:
   - Uses separate encoder models for query and document
   - Optimized for question-answering systems

3. **Cross-Encoder Reranking**:
   - First stage finds broad candidate set with Bi-encoder
   - Second stage reranks candidates with Cross-encoder
   - Higher accuracy but slower

   ```python
   from sentence_transformers import SentenceTransformer, CrossEncoder, util

   # Bi-encoder (for first stage retrieval)
   bi_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

   # Cross-encoder (for second stage reranking)
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

   # Corpus embeddings
   corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True)

   # Two-stage search function
   def two_stage_search(query, top_k_retrieval=100, top_k_rerank=10):
       # Stage 1: Find broad candidate set with Bi-encoder
       query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
       hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k_retrieval)[0]

       # Prepare candidates
       candidates = [(hit['corpus_id'], corpus[hit['corpus_id']]) for hit in hits]

       # Stage 2: Rerank with Cross-encoder
       cross_inp = [[query, cand[1]] for cand in candidates]
       cross_scores = cross_encoder.predict(cross_inp)

       # Sort results
       for idx in range(len(cross_scores)):
           candidates[idx] = (candidates[idx][0], candidates[idx][1], cross_scores[idx])

       candidates.sort(key=lambda x: x[2], reverse=True)
       return candidates[:top_k_rerank]
   ```

**Practical Semantic Search Applications:**

1. **Document Archive Search**: Semantic search in large document collections
2. **Question-Answering Systems**: Finding appropriate answers to user questions
3. **Recommendation Systems**: Recommending semantically similar content based on user interests
4. **Customer Support**: Finding most relevant answers from FAQs semantically
5. **Content Filtering**: Semantically detecting unwanted content
