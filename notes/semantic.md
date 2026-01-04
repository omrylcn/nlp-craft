# Transformer-Based Semantic Modeling: Theoretical Foundations and Applications


## 1. Theoretical Foundations of Semantic Modeling

### 1.1 Mathematical Foundation of Transformer Architecture

The Transformer architecture is a neural network model that uses attention mechanisms to create semantic representations of text data. The mathematical foundation can be expressed as:

$$\mathcal{T}: \mathcal{S} \rightarrow \mathcal{H}$$

This formula shows how the transformer model ($\mathcal{T}$) transforms the word space ($\mathcal{S}$) into the hidden representation space ($\mathcal{H}$). Simply put, this transformation converts text into numerical vectors.

At each layer, a new representation is created for each token (word piece):

$$\mathbf{h}_i^{(l)} = \text{TransformerLayer}_l(\mathbf{h}_i^{(l-1)}, \{\mathbf{h}_j^{(l-1)}\}_{j=1}^n)$$

This formula explains:
- $\mathbf{h}_i^{(l)}$: Representation of the $i$-th token at layer $l$
- $\mathbf{h}_i^{(l-1)}$: Representation of the same token from the previous layer
- $\{\mathbf{h}_j^{(l-1)}\}_{j=1}^n$: Representations of all tokens from the previous layer (context)

The Transformer layer updates each token not only based on itself but also considering the entire text context. This enables modeling word meanings according to their context.

### 1.2 Understanding the Attention Mechanism

The heart of the Transformer architecture is the attention mechanism. The attention mechanism is computed using:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's explain this formula step by step:

1. **Query, Key, Value Matrices:**
   - $Q$ (Query): Represents "What am I looking for?"
   - $K$ (Key): Contains "Where can I search?" information
   - $V$ (Value): Contains "What will I get when I find it?" information

2. **Similarity Computation:**
   - $QK^T$: Dot product of each query vector with each key vector measures similarity
   - $\frac{QK^T}{\sqrt{d_k}}$: Division by scaling factor ($\sqrt{d_k}$) prevents gradient explosion

3. **Weight Generation:**
   - $\text{softmax}(...)$: Converts results to probability distribution in [0,1] range
   - Determines how much "attention" each token should pay to all other tokens

4. **Weighted Sum:**
   - $\text{softmax}(...) \times V$: Takes the weighted sum of value vectors

Simple example:
In the sentence "John went to the bank", when the model computes attention weights for "bank", it attends to other parts of the sentence to determine whether it means "financial institution" or "river bank".

```python
def attention_mechanism(query, key, value):
    """
    Simple attention mechanism computation

    Args:
        query: Query tensor [batch_size, seq_len, d_model]
        key: Key tensor [batch_size, seq_len, d_model]
        value: Value tensor [batch_size, seq_len, d_model]

    Returns:
        Attention-applied values [batch_size, seq_len, d_model]
    """
    # Compute similarity between query and keys
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scaling - prevents gradient explosion
    d_k = query.size()[-1]
    scaled_scores = scores / math.sqrt(d_k)

    # Apply softmax - get weights that sum to 1
    weights = F.softmax(scaled_scores, dim=-1)

    # Take weighted sum of values
    output = torch.matmul(weights, value)

    return output
```

### 1.3 Contextual Representations and Meaning Vectors

Models like BERT make word embeddings context-sensitive. For example, the word "bank" can have different meanings in different sentences:

- "I went to the bank for my money." (Financial institution)
- "I sat on the bank in the park." (Seat/bench)

Contextual embeddings can capture this difference. Mathematically:

$$\phi(w|c) \neq \phi(w|c')$$

This notation expresses:
- $\phi(w|c)$: Representation of word $w$ in context $c$
- $\phi(w|c')$: Representation of the same word in different context $c'$

These contextual representations help model the fact that word meanings can change based on context.

## 2. BERT Model Operation and Semantic Features

### 2.1 BERT's Mathematical Model (Simplified)

BERT is a bidirectional Transformer model. We can formulate its operation as:

$$\mathbf{H} = \text{TransformerEncoder}(\mathbf{E} + \mathbf{P})$$

The meaning of this formula:
- $\mathbf{E}$: Token embedding matrix (a vector for each word)
- $\mathbf{P}$: Position encoding matrix (encodes word order)
- $\mathbf{H}$: Output representations (contextual embeddings)

More explicitly:
1. Each token is converted to an embedding vector
2. Position encodings indicating word positions in the sentence are added to these embeddings
3. This sum is passed through Transformer encoder layers
4. The output is a contextual embedding vector for each token

BERT is specifically trained with the Masked Language Modeling (MLM) task:

$$P(w_i | w_1, ..., w_{i-1}, [MASK], w_{i+1}, ..., w_n)$$

This helps the model develop bidirectional context understanding by replacing some words in the sentence with "[MASK]" token and having the model predict these masked words.

```python
class SimplifiedBERT(nn.Module):
    def __init__(self, vocab_size=30000, hidden_size=768, num_layers=12):
        super().__init__()

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

        # Position embeddings (for maximum 512 positions)
        self.position_embeddings = nn.Embedding(512, hidden_size)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, input_ids, attention_mask):
        # Get token embeddings
        embeddings = self.token_embeddings(input_ids)

        # Add position information
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeddings = self.position_embeddings(positions)

        # Combine token and position embeddings
        hidden_states = embeddings + position_embeddings

        # Pass through Transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states
```

### 2.2 Mean Pooling: Extracting Semantic Summaries

BERT produces a separate representation for each token, but often a single representation of the entire sentence is needed. Mean pooling creates a representation of the entire sentence by taking a weighted average of token representations:

$$\text{MeanPooling}(\mathbf{H}, \mathbf{M}) = \frac{\sum_{i=1}^{n} \mathbf{h}_i \cdot \mathbf{m}_i}{\sum_{i=1}^{n} \mathbf{m}_i}$$

Where:
- $\mathbf{H}$ = All token representations produced by BERT ($\mathbf{h}_i$ each token representation)
- $\mathbf{M}$ = Attention mask ($\mathbf{m}_i$ is 1 if token $i$ is real content, 0 if padding)

Why is mean pooling important?
1. **Variable Length Problem**: Creates fixed-size representations for sentences of different lengths
2. **Padding Tokens**: Only considers real content tokens thanks to attention mask
3. **Using All Information**: Uses semantic information from all sentence components compared to [CLS] token

```python
def mean_pooling(token_embeddings, attention_mask):
    """
    Takes average of token embeddings using attention mask

    Args:
        token_embeddings: Model output [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        Sentence embeddings [batch_size, hidden_size]
    """
    # Expand mask to embedding dimension
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Take sum of masked token embeddings
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    # Get mask sum (real token count)
    sum_mask = input_mask_expanded.sum(1)

    # Prevent division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    # Calculate average
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings
```

Mean pooling operation explained step by step:

1. **Expanding the Mask**: The attention mask is typically [batch_size, seq_len] dimensional and indicates whether each token is real (1) or padding (0). We expand this mask to [batch_size, seq_len, hidden_size] dimension to match our token embeddings.

2. **Masking Operation**: By multiplying token embeddings with the mask, we zero out padding token values. Only real content token values remain.

3. **Summation**: Masked embeddings are summed over the sequence dimension (1st dimension) to reduce each example to a single vector.

4. **Normalization**: We divide the sum by the number of real tokens in the sentence to get the average. This creates comparable representations for sentences of different lengths.

5. **Division by Zero Protection**: In very rare cases, there might be a completely masked example. torch.clamp is used to prevent division by zero error.

### 2.3 Semantic Content of Token Representations

BERT and similar models encode different types of information at different layers. This hierarchical structure can be thought of as:

- **Lower Layers (1-4)**: Encode more syntactic features and basic grammatical relationships
- **Middle Layers (5-8)**: Process word meanings and local context information
- **Upper Layers (9-12)**: Encode more complex semantic relationships and contextual meaning

If we express this information hierarchy mathematically:

$$I(H^{(l)}; \text{Syntax}) > I(H^{(l+1)}; \text{Syntax})$$
$$I(H^{(l)}; \text{Semantics}) < I(H^{(l+1)}; \text{Semantics})$$

These formulas express:
- $I(X; Y)$: Amount of mutual information between X and Y
- $H^{(l)}$: Embeddings at layer $l$
- Lower layers have more correlation with syntactic information, less with semantic information
- As layers progress upward, semantic information increases, syntactic information decreases

Therefore, representations from different layers may be more suitable for different NLP tasks:
- Lower layers may be more useful for syntax analysis
- Upper layers may be preferred for semantic similarity tasks

## 3. Effective Sentence Representations for Semantic Similarity

### 3.1 Cosine Similarity and Semantic Space

The most commonly used metric for measuring semantic similarity between two texts is cosine similarity:

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

This formula calculates the cosine of the angle between two vectors:
- $\mathbf{a} \cdot \mathbf{b}$: Dot product of two vectors
- $||\mathbf{a}||$: L2 norm (length) of vector a
- $||\mathbf{b}||$: L2 norm of vector b

Why is cosine similarity suitable for semantic similarity?
1. **Direction-Oriented**: Focuses on vector directions rather than magnitudes
2. **Normalization**: Produces values in [-1, 1] range, 1 means perfect similarity, -1 perfect opposition, 0 no relationship
3. **Dimension Independence**: Gives comparable results for texts of different lengths

```python
def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Computes cosine similarity between two embedding sets

    Args:
        embeddings1: First embedding set [n_samples1, embedding_dim]
        embeddings2: Second embedding set [n_samples2, embedding_dim]

    Returns:
        Similarity matrix [n_samples1, n_samples2]
    """
    # Normalize (optional, if already normalized)
    norm_emb1 = F.normalize(embeddings1, p=2, dim=1)
    norm_emb2 = F.normalize(embeddings2, p=2, dim=1)

    # Cosine similarity = dot product (for normalized vectors)
    return torch.matmul(norm_emb1, norm_emb2.transpose(0, 1))
```

In semantic space, semantically similar sentences are represented by vectors close to each other. Mathematically:

$$d(\phi(s_1), \phi(s_2)) \approx d_{semantic}(s_1, s_2)$$

Where:
- $\phi(s)$: Semantic embedding function for sentence $s$
- $d$: Distance function in vector space (usually cosine distance)
- $d_{semantic}$: Actual semantic distance between two sentences

### 3.2 Normalization and Calibration

Embeddings produced by models like BERT are typically anisotropic - meaning vectors are not uniformly distributed in space but cluster in certain directions. This situation can cause problems in cosine similarity calculations.

Normalization and calibration techniques are used to solve this problem:

1. **L2 Normalization**: Normalizing each embedding vector to unit length:

$$\hat{\mathbf{h}} = \frac{\mathbf{h}}{||\mathbf{h}||}$$

2. **Whitening**: Reducing anisotropy by transforming the embedding distribution:

$$\mathbf{h}_{whitened} = \mathbf{\Sigma}^{-1/2}(\mathbf{h} - \boldsymbol{\mu})$$

Where:
- $\boldsymbol{\mu}$: Mean vector of embeddings
- $\mathbf{\Sigma}$: Covariance matrix
- $\mathbf{\Sigma}^{-1/2}$: Square root inverse of covariance matrix

The whitening operation removes correlations between embeddings and normalizes variance, thus providing better calibration of the semantic space.

```python
def normalize_embeddings(embeddings):
    """Normalizes embedding vectors to unit length with L2 normalization"""
    return F.normalize(embeddings, p=2, dim=1)

def whiten_embeddings(embeddings):
    """
    Applies whitening transformation to embeddings
    """
    # Subtract mean
    mean = embeddings.mean(dim=0, keepdim=True)
    centered_embeddings = embeddings - mean

    # Compute covariance matrix
    n_samples = embeddings.size(0)
    cov = torch.mm(centered_embeddings.t(), centered_embeddings) / (n_samples - 1)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Create whitening matrix
    epsilon = 1e-10  # For numerical stability
    whitening_matrix = torch.mm(
        eigenvectors,
        torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
    )
    whitening_matrix = torch.mm(whitening_matrix, eigenvectors.t())

    # Apply whitening
    whitened_embeddings = torch.mm(centered_embeddings, whitening_matrix)

    return whitened_embeddings
```

### 3.3 Improving Semantic Space with Contrastive Learning

Contrastive learning is a powerful technique used to improve embeddings for semantic similarity tasks. This approach aims to position sentences with similar meanings close together in vector space, and those with different meanings far apart.

Approaches like SimCSE use this loss function:

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}$$

To understand this formula more clearly:
- $\mathbf{h}_i$: Embedding of anchor sentence
- $\mathbf{h}_i^+$: Embedding of semantically similar sentence (positive example)
- $\mathbf{h}_j$: Embeddings of all other sentences in batch (negative examples)
- $\text{sim}$: Similarity function (usually cosine similarity)
- $\tau$: Temperature parameter (typically 0.05 - 0.1)

This loss function maximizes similarity to positive examples while minimizing similarity to negative examples. The temperature parameter $\tau$ controls the distribution of similarity scores.

What's particularly interesting about SimCSE is that it uses two different forward passes of the same sentence (with different dropout patterns) as positive pairs. This achieves good results even without labeled data.

```python
def contrastive_loss(similarities, temperature=0.05):
    """
    Computes contrastive loss (InfoNCE)

    Args:
        similarities: Similarity matrix [batch_size, batch_size]
        temperature: Temperature parameter

    Returns:
        Average contrastive loss
    """
    batch_size = similarities.size(0)

    # Labels: Self-similarity (diagonal)
    labels = torch.arange(batch_size, device=similarities.device)

    # Scale with temperature
    similarities = similarities / temperature

    # Cross entropy loss
    loss = F.cross_entropy(similarities, labels)

    return loss

def train_simcse_step(model, tokenizer, sentences, optimizer):
    """
    Single step for SimCSE training
    """
    # Tokenize texts
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward same batch twice (with different dropout patterns)
    model.train()  # Enable dropout

    # First pass
    embeddings1 = model(**inputs)

    # Second pass (different dropout mask)
    embeddings2 = model(**inputs)

    # Normalize
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Compute similarity matrix between all pairs
    similarities = torch.matmul(embeddings1, embeddings2.t())

    # Compute contrastive loss
    loss = contrastive_loss(similarities)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

## 4. Semantic Model Training and Fine-Tuning

### 4.1 Two-Stage Training: NLI and STS

Semantic similarity models are typically trained through a two-stage process:

1. **NLI (Natural Language Inference) Training**:
   - Natural language inference datasets are used (e.g., SNLI, MNLI)
   - These datasets contain "entailment", "contradiction", and "neutral" relationships
   - "Entailment" pairs are used as positive examples, "contradiction" pairs as negative examples

2. **STS (Semantic Textual Similarity) Fine-Tuning**:
   - STS datasets contain sentence pairs and their similarity scores (typically 0-5)
   - These scores are normalized to [0,1] range
   - Model is trained to minimize difference between predicted similarity and actual similarity

The mathematics of this two-stage approach:

NLI Loss (contrastive):
$$\mathcal{L}_{NLI} = -\log \frac{\exp(\text{sim}(\mathbf{h}_p, \mathbf{h}_h^+) / \tau)}{\exp(\text{sim}(\mathbf{h}_p, \mathbf{h}_h^+) / \tau) + \exp(\text{sim}(\mathbf{h}_p, \mathbf{h}_h^-) / \tau)}$$

Where:
- $\mathbf{h}_p$: Embedding of premise sentence
- $\mathbf{h}_h^+$: Embedding of hypothesis sentence in entailment relationship
- $\mathbf{h}_h^-$: Embedding of hypothesis sentence in contradiction relationship

STS Loss (regression):
$$\mathcal{L}_{STS} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \text{sim}(\mathbf{h}_{i1}, \mathbf{h}_{i2}))^2$$

Where:
- $y_i$: Actual similarity score for pair $i$ (normalized between 0-1)
- $\mathbf{h}_{i1}, \mathbf{h}_{i2}$: Embeddings of sentences in pair $i$

```python
def train_nli_step(model, batch, optimizer):
    """
    Training step with NLI task
    """
    # Extract items from batch
    premises, hypotheses_pos, hypotheses_neg = batch

    # Compute embeddings
    premise_embeddings = model.encode(premises)
    pos_hypothesis_embeddings = model.encode(hypotheses_pos)
    neg_hypothesis_embeddings = model.encode(hypotheses_neg)

    # Normalize
    premise_embeddings = F.normalize(premise_embeddings, p=2, dim=1)
    pos_hypothesis_embeddings = F.normalize(pos_hypothesis_embeddings, p=2, dim=1)
    neg_hypothesis_embeddings = F.normalize(neg_hypothesis_embeddings, p=2, dim=1)

    # Similarities between positive and negative pairs
    pos_similarities = torch.sum(premise_embeddings * pos_hypothesis_embeddings, dim=1)
    neg_similarities = torch.sum(premise_embeddings * neg_hypothesis_embeddings, dim=1)

    # Compute loss (simplified version of InfoNCE formulation)
    logits = torch.stack([pos_similarities, neg_similarities], dim=1) / 0.05
    labels = torch.zeros(len(premises), dtype=torch.long, device=model.device)
    loss = F.cross_entropy(logits, labels)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_sts_step(model, batch, optimizer):
    """
    Fine-tuning step with STS task
    """
    # Extract items from batch
    sentences1, sentences2, similarity_scores = batch

    # Compute embeddings
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # Normalize
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Cosine similarity
    pred_similarities = torch.sum(embeddings1 * embeddings2, dim=1)

    # MSE loss
    loss = F.mse_loss(pred_similarities, similarity_scores)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### 4.2 Multilingual Semantic Modeling

Multilingual semantic models aim to represent similar meanings in different languages in the same vector space. These models are typically trained with two approaches:

1. **Joint Multilingual Pre-training**:
   - Models like mBERT or XLM-R are trained together with multilingual data
   - Sentences carrying the same meaning in different languages have similar vector representations

2. **Cross-lingual Transfer Learning**:
   - Using parallel corpora, equivalent sentences in different languages are positioned close together
   - Parallelism loss is expressed as:

$$\mathcal{L}_{parallel} = \frac{1}{|P|} \sum_{(s_i^{L_1}, s_i^{L_2}) \in P} (1 - \text{sim}(E(s_i^{L_1}), E(s_i^{L_2})))$$

Where:
- $P$: Set of parallel sentence pairs
- $s_i^{L_1}$: Version of $i$-th sentence in first language
- $s_i^{L_2}$: Version of $i$-th sentence in second language
- $E$: Sentence encoder model
- $\text{sim}$: Similarity function

This approach helps create a multilingual vector space, positioning semantically similar expressions in different languages close to each other.

```python
def train_multilingual_step(model, batch, optimizer):
    """
    Multilingual semantic model training step
    """
    # Get parallel sentences from batch
    sentences_lang1, sentences_lang2 = batch

    # Encode sentences in both languages
    embeddings_lang1 = model.encode(sentences_lang1)
    embeddings_lang2 = model.encode(sentences_lang2)

    # Normalize
    embeddings_lang1 = F.normalize(embeddings_lang1, p=2, dim=1)
    embeddings_lang2 = F.normalize(embeddings_lang2, p=2, dim=1)

    # Parallel similarity loss
    # Sentences at same index should be parallel (translations of each other)
    # Cosine similarity should be close to 1, so we minimize (1 - sim)
    similarities = torch.sum(embeddings_lang1 * embeddings_lang2, dim=1)
    loss = torch.mean(1 - similarities)

    # Additionally, contrastive loss can also be added
    # Here non-parallel pairs' similarity is reduced

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

## 5. Semantic Model Evaluation and Analysis

### 5.1 Semantic Similarity Evaluation Metrics

The success of semantic models is typically measured by their performance on STS (Semantic Textual Similarity) benchmarks. The main evaluation metrics are:

1. **Pearson Correlation Coefficient**:

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

Where:
- $x_i$: Similarity score predicted by model
- $y_i$: Similarity score given by humans
- $\bar{x}, \bar{y}$: Means of x and y

2. **Spearman Rank Correlation Coefficient**:

$$\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

Where $d_i$ is the difference between x and y rankings for the $i$-th example.

These correlation metrics measure how well model predictions match human evaluations. Values close to 1 indicate perfect correlation, values close to 0 indicate weak correlation.

```python
def evaluate_semantic_model(model, eval_pairs, human_scores):
    """
    Evaluate semantic model

    Args:
        model: Model to evaluate
        eval_pairs: Evaluation pairs as (sentence1, sentence2)
        human_scores: Human evaluation scores

    Returns:
        dict: Evaluation metrics
    """
    # Compute model similarity scores
    predicted_scores = []

    for sent1, sent2 in eval_pairs:
        # Encode sentences
        emb1 = model.encode([sent1])[0]
        emb2 = model.encode([sent2])[0]

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        predicted_scores.append(similarity)

    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(human_scores, predicted_scores)

    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(human_scores, predicted_scores)

    return {
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p
    }
```

### 5.2 Embedding Space Visualization and Analysis

Various dimensionality reduction and visualization techniques are used to analyze and understand the semantic embedding space:

1. **PCA (Principal Component Analysis)**:
   - Finds components that best explain data variance
   - Mathematically uses eigenvalue decomposition of covariance matrix:

   $$\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$

   Where $\mathbf{\Sigma}$ is covariance matrix, $\mathbf{V}$ is eigenvector matrix, $\mathbf{\Lambda}$ is eigenvalue matrix.

2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
   - Creates low-dimensional visualization while preserving similarity relationships in high-dimensional space
   - Minimizes probability distributions with KL divergence:

   $$KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

   Where $p_{ij}$ is similarity in original space, $q_{ij}$ is similarity in low-dimensional space.

3. **UMAP (Uniform Manifold Approximation and Projection)**:
   - Performs dimensionality reduction while preserving topological structure
   - Faster and more scalable than t-SNE

These techniques are used to understand the structure of semantic space, see clusters, and visualize what kind of relationships the model captures.

```python
def visualize_embeddings(embeddings, labels=None, method='tsne'):
    """
    Visualize embedding space

    Args:
        embeddings: Embeddings to visualize
        labels: Point labels (optional)
        method: Visualization method ('pca', 'tsne', 'umap')
    """
    # Dimensionality reduction
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    # Apply dimensionality reduction
    embeddings_2d = reducer.fit_transform(embeddings)

    # Visualize
    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Colored display
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='Label')
    else:
        # Single color display
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    plt.title(f'Embedding {method.upper()} Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(alpha=0.3)
    plt.show()
```

## 6. Conclusion and Practical Application Recommendations

Transformer-based semantic modeling provides a powerful foundation for semantic similarity, text classification, information retrieval, and many other NLP applications. Some practical recommendations for effective use of these models:

1. **Choosing the Right Pooling Strategy**:
   - Mean pooling generally shows best performance for general semantic similarity
   - [CLS] token or max pooling can be useful for classification tasks
   - Compare different strategies and choose the most suitable one for your task

2. **Normalization and Preprocessing**:
   - Always normalize embeddings (L2 normalization)
   - Use additional normalization techniques like whitening if needed
   - Apply appropriate preprocessing steps for your text (normalize, lowercase, etc.)

3. **Model Selection and Fine-Tuning**:
   - Choose models specifically trained for your language (e.g., XLM-R for multilingual)
   - Fine-tune appropriately for your task (with NLI, STS, or domain-specific data)
   - Use the two-stage training approach (NLI + STS)

4. **Evaluation and Analysis**:
   - Compare model performance with standard STS benchmarks
   - Visualize embedding space to understand model behavior
   - Perform error analysis to identify areas where the model can be improved
