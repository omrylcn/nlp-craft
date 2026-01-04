# STS and NLI: Loss Functions and Evaluation Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundations](#2-theoretical-foundations)
   - 2.1 [Semantic Textual Similarity (STS)](#21-semantic-textual-similarity-sts)
   - 2.2 [Natural Language Inference (NLI)](#22-natural-language-inference-nli)
   - 2.3 [Vector Representations and Semantic Space](#23-vector-representations-and-semantic-space)
3. [Loss Functions](#3-loss-functions)
   - 3.1 [Loss Functions for STS](#31-loss-functions-for-sts)
     - 3.1.1 [Mean Squared Error (MSE)](#311-mean-squared-error-mse)
     - 3.1.2 [Cosine Embedding Loss](#312-cosine-embedding-loss)
     - 3.1.3 [Contrastive Loss](#313-contrastive-loss)
     - 3.1.4 [Triplet Loss](#314-triplet-loss)
     - 3.1.5 [Multiple Negatives Ranking Loss](#315-multiple-negatives-ranking-loss)
   - 3.2 [Loss Functions for NLI](#32-loss-functions-for-nli)
     - 3.2.1 [Cross-Entropy Loss](#321-cross-entropy-loss)
     - 3.2.2 [Focal Loss](#322-focal-loss)
     - 3.2.3 [Label Smoothing](#323-label-smoothing)
   - 3.3 [Loss Functions for Multi-Task Learning](#33-loss-functions-for-multi-task-learning)
4. [Evaluation Metrics](#4-evaluation-metrics)
   - 4.1 [STS Evaluation Metrics](#41-sts-evaluation-metrics)
     - 4.1.1 [Pearson and Spearman Correlation](#411-pearson-and-spearman-correlation)
     - 4.1.2 [Cosine Similarity](#412-cosine-similarity)
     - 4.1.3 [Manhattan and Euclidean Distances](#413-manhattan-and-euclidean-distances)
   - 4.2 [NLI Evaluation Metrics](#42-nli-evaluation-metrics)
     - 4.2.1 [Accuracy](#421-accuracy)
     - 4.2.2 [Precision, Recall, F1 Score](#422-precision-recall-f1-score)
     - 4.2.3 [Confusion Matrix Analysis](#423-confusion-matrix-analysis)
   - 4.3 [SentEval and GLUE Benchmark](#43-senteval-and-glue-benchmark)
5. [Practical Applications](#5-practical-applications)
   - 5.1 [Training and Evaluation for STS Models](#51-training-and-evaluation-for-sts-models)
   - 5.2 [Training and Evaluation for NLI Models](#52-training-and-evaluation-for-nli-models)
   - 5.3 [Hyperparameter Optimization](#53-hyperparameter-optimization)

## 1. Introduction

Semantic Textual Similarity (STS) and Natural Language Inference (NLI) are among the fundamental tasks in natural language processing (NLP). These tasks play an important role in testing the ability of modern language models to understand and represent semantic relationships between texts.

This guide will cover loss functions and evaluation metrics used for STS and NLI tasks with both theoretical and practical aspects. It aims to provide information covering a wide range from basic concepts to advanced techniques.

## 2. Theoretical Foundations

### 2.1 Semantic Textual Similarity (STS)

Semantic Textual Similarity (STS) is an NLP task that measures the semantic similarity between two texts. STS typically scores the similarity between two sentences from 0 (completely different) to 5 (completely synonymous).

**Basic STS Concepts:**

- **Lexical Similarity**: Word-level similarity based on common words
- **Syntactic Similarity**: Similarity of syntactic structures
- **Semantic Similarity**: Meaning-level similarity that considers context and concepts

**Mathematical Formulation of STS:**

STS can be modeled as a problem of computing similarity between two sentence vectors:

$$STS(s_1, s_2) = f(Emb(s_1), Emb(s_2))$$

Where:
- $s_1, s_2$ are two sentences
- $Emb(s)$ is the function that creates the vector representation (embedding) of the sentence
- $f$ is a similarity function that measures the similarity between two vectors

**STS Benchmarks and Datasets:**
- STS Benchmark (STS-B)
- SICK (Sentences Involving Compositional Knowledge)
- SemEval STS tasks

### 2.2 Natural Language Inference (NLI)

Natural Language Inference (NLI) is an NLP task aimed at determining the logical relationship between a premise sentence and a hypothesis sentence. NLI is typically classified into three classes:

- **Entailment**: If the premise is true, the hypothesis is definitely true.
- **Contradiction**: If the premise is true, the hypothesis is definitely false.
- **Neutral**: The truth of the premise does not determine the truth of the hypothesis.

**Mathematical Formulation of NLI:**

NLI can be modeled as a classification problem of the logical relationship between two sentences:

$$NLI(p, h) = \text{argmax}_c P(c | p, h)$$

Where:
- $p$ is the premise sentence
- $h$ is the hypothesis sentence
- $c \in \{\text{entailment}, \text{contradiction}, \text{neutral}\}$ is one of the classes
- $P(c | p, h)$ is the probability of class $c$ given the premise and hypothesis

**NLI Benchmarks and Datasets:**
- SNLI (Stanford Natural Language Inference)
- MultiNLI
- XNLI (Cross-lingual Natural Language Inference)
- ANLI (Adversarial NLI)

### 2.3 Vector Representations and Semantic Space

At the foundation of STS and NLI tasks lies transforming texts into meaningful vector representations. These vector representations reside in a multi-dimensional semantic space that captures the semantic features of text content.

**Vector Representation Methods:**

1. **Word Vectors:**
   - Word2Vec (CBOW and Skip-gram)
   - GloVe (Global Vectors for Word Representation)
   - FastText (models using character n-grams)

2. **Sentence Vectors:**
   - Simple averaging methods (average of word vectors)
   - TF-IDF weighted averages
   - Doc2Vec
   - Skip-Thought Vectors

3. **Contextual Embeddings:**
   - ELMo (Embeddings from Language Models)
   - BERT (Bidirectional Encoder Representations from Transformers)
   - RoBERTa, XLNet, ALBERT, etc.
   - S-BERT (Sentence-BERT)

**Mathematical Foundations of Semantic Space:**

Semantic space can generally be modeled as a $d$-dimensional vector space in the form $\mathbb{R}^d$. Here each text is represented as a point in this space. Semantic similarity between two texts is measured by the distance or angle between these points.

For example, cosine similarity is calculated as follows:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

Where:
- $\mathbf{a}, \mathbf{b}$ are two text vectors
- $a_i, b_i$ are the $i$th components of these vectors
- $|\mathbf{a}|, |\mathbf{b}|$ are the norms of the vectors

## 3. Loss Functions

Loss functions are at the center of model training and measure the difference between model predictions and actual target values. Different loss functions are used for STS and NLI tasks because the nature and goals of these tasks are different.

### 3.1 Loss Functions for STS

#### 3.1.1 Mean Squared Error (MSE)

MSE is the most commonly used loss function for regression problems and is frequently used in STS. It measures the square of the difference between the predicted similarity score and the actual similarity score between two sentences.

**Mathematical Formulation:**

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Where:
- $N$ is the number of samples
- $y_i$ is the actual similarity score for the $i$th sample
- $\hat{y}_i$ is the similarity score predicted by the model

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss()
similarity_pred = model(sentence1, sentence2)
loss = criterion(similarity_pred, true_similarity)
```

**Advantages:**
- Easy to implement
- Continuous and differentiable
- Penalizes large errors disproportionately

**Disadvantages:**
- Sensitive to outliers
- Does not consider the geometric structure of vector space

#### 3.1.2 Cosine Embedding Loss

Cosine Embedding Loss tries to minimize the angle between two vectors. This ensures that semantically similar sentences are close to each other in the representation space.

**Mathematical Formulation:**

$$
\begin{cases}
1 - \cos(a, b), & \text{if } y = 1 \\
\max(0, \cos(a, b) - \text{margin}), & \text{if } y = -1
\end{cases}
$$

Where:
- $a, b$ are the embedding vectors of two sentences
- $y \in \{1, -1\}$ is a label indicating whether sentences are similar or not
- $\cos(a, b)$ is the cosine similarity between $a$ and $b$
- margin is a parameter determining the minimum difference non-similar sentences should have

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

criterion = nn.CosineEmbeddingLoss(margin=0.2)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
# y=1 for similar sentences, y=-1 for different sentences
loss = criterion(embedding1, embedding2, target)
```

**Advantages:**
- Focuses on the direction of vectors, not magnitude
- More suitable for semantic similarity
- Works with normalized vectors

**Disadvantages:**
- Designed for binary similarity (similar/different), requires additional processing for fine-grained similarity scores
- Margin parameter needs to be tuned

#### 3.1.3 Contrastive Loss

Contrastive Loss ensures that similar sentence pairs are close to each other in the representation space, while different sentence pairs are farther than a certain margin.

**Mathematical Formulation:**

$$\mathcal{L}_{contrastive}(a, b, y) = (1-y) \cdot \frac{1}{2} d(a, b)^2 + y \cdot \frac{1}{2} \max(0, \text{margin} - d(a, b))^2$$

Where:
- $a, b$ are the embedding vectors of two sentences
- $y \in \{0, 1\}$ is a label indicating whether sentences are similar or not (1: similar, 0: different)
- $d(a, b)$ is the Euclidean distance between $a$ and $b$
- margin is a parameter determining the minimum distance different sentences should have

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        # Calculate Euclidean distance
        dist = F.pairwise_distance(x1, x2, p=2)
        # Calculate contrastive loss
        loss = 0.5 * (y * dist.pow(2) + (1-y) * F.relu(self.margin - dist).pow(2))
        return loss.mean()

criterion = ContrastiveLoss(margin=2.0)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
# y=1 for similar sentences, y=0 for different sentences
loss = criterion(embedding1, embedding2, target)
```

#### 3.1.4 Triplet Loss

Triplet Loss uses triplet examples (anchor, positive, negative) to ensure the anchor example is closer to the positive example than to the negative example.

**Mathematical Formulation:**

$$\mathcal{L}_{triplet} = \max(0, d(a, p) - d(a, n) + \text{margin})$$

Where:
- $a$ is the embedding vector of the anchor sentence
- $p$ is the embedding vector of the positive sentence (semantically similar to anchor)
- $n$ is the embedding vector of the negative sentence (semantically different from anchor)
- $d(x, y)$ is the distance function between $x$ and $y$ (usually Euclidean distance)
- margin determines the minimum difference between anchor-positive and anchor-negative distances

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        # Calculate triplet loss
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()

criterion = TripletLoss(margin=1.0)
anchor_emb = model.encode(anchor_sentence)
positive_emb = model.encode(positive_sentence)
negative_emb = model.encode(negative_sentence)
loss = criterion(anchor_emb, positive_emb, negative_emb)
```

#### 3.1.5 Multiple Negatives Ranking Loss

Multiple Negatives Ranking Loss (MNRL) uses all examples in a batch to try to distinguish positive pairs from negative pairs. This loss is commonly used especially in models like S-BERT.

**Mathematical Formulation:**

$$\mathcal{L}_{MNRL} = -\log\frac{\exp(\text{sim}(a_i, p_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(a_i, b_j)/\tau)}$$

Where:
- $a_i$ and $p_i$ are the embedding vectors of a positive pair
- $b_j$ are the embedding vectors of all other sentences in the batch
- $\text{sim}(x, y)$ is the similarity measure between $x$ and $y$ (usually cosine similarity)
- $\tau$ is the temperature parameter (controls the softmax function)
- $N$ is the batch size

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale=20.0):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale  # Inverse of temperature

    def forward(self, embeddings_a, embeddings_b):
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        # Calculate cosine similarity matrix
        scores = torch.matmul(embeddings_a, embeddings_b.transpose(0, 1)) * self.scale

        # Diagonal elements are positive pairs
        labels = torch.arange(len(scores), device=scores.device)

        # Calculate cross-entropy loss
        loss = F.cross_entropy(scores, labels)
        return loss
```

### 3.2 Loss Functions for NLI

#### 3.2.1 Cross-Entropy Loss

Cross-Entropy Loss is the most commonly used loss function for classification problems and is ideal for multi-class problems like NLI.

**Mathematical Formulation:**

$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$$

Where:
- $N$ is the number of samples
- $C$ is the number of classes (typically 3 for NLI: entailment, contradiction, neutral)
- $y_{i,c}$ is an indicator for whether the $i$th sample belongs to class $c$ (one-hot encoding)
- $p_{i,c}$ is the model's probability prediction for the $i$th sample belonging to class $c$

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
logits = model(premise, hypothesis)  # Shape: [batch_size, 3]
# labels: 0=entailment, 1=contradiction, 2=neutral
loss = criterion(logits, labels)
```

#### 3.2.2 Focal Loss

Focal Loss is a loss function that is resistant to class imbalance problems. It gives more weight to difficult examples and reduces easy examples.

**Mathematical Formulation:**

$$\mathcal{L}_{focal} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} (1 - p_{i,c})^{\gamma} \log(p_{i,c})$$

Where:
- $\gamma$ is the focusing parameter (usually set to 2)
- Other symbols are the same as Cross-Entropy Loss

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # Apply softmax
        probs = F.softmax(logits, dim=1)
        # Get probability of correct class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Calculate focal weight
        focal_weight = (1 - pt).pow(self.gamma)

        # Apply alpha weight (optional)
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_weight

        # Calculate loss
        loss = -focal_weight * torch.log(pt)
        return loss.mean()
```

#### 3.2.3 Label Smoothing

Label Smoothing is a regularization technique used to prevent the model from making overly confident predictions. It uses smoothed labels instead of one-hot labels.

**Mathematical Formulation:**

$$y_{i,c}^{smooth} = (1 - \alpha) \cdot y_{i,c} + \alpha \cdot \frac{1}{C}$$

Where:
- $\alpha$ is the smoothing parameter (usually set to 0.1)
- $C$ is the number of classes
- $y_{i,c}$ is the original one-hot label
- $y_{i,c}^{smooth}$ is the smoothed label

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=3):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)

        # Create smoothed labels
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / self.num_classes

        # Calculate loss
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        return loss.mean()
```

### 3.3 Loss Functions for Multi-Task Learning

STS and NLI tasks are often trained together because these tasks are related in terms of semantic understanding skills. Loss functions used for multi-task learning combine the losses of different tasks.

**Weighted Sum Loss:**

$$\mathcal{L}_{total} = \lambda_{STS} \cdot \mathcal{L}_{STS} + \lambda_{NLI} \cdot \mathcal{L}_{NLI}$$

Where:
- $\lambda_{STS}$ and $\lambda_{NLI}$ are hyperparameters determining the weight of each task
- $\mathcal{L}_{STS}$ is the loss function used for the STS task
- $\mathcal{L}_{NLI}$ is the loss function used for the NLI task

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

# Loss functions
sts_criterion = nn.MSELoss()
nli_criterion = nn.CrossEntropyLoss()

# Hyperparameters
lambda_sts = 0.5
lambda_nli = 0.5

# Training step
def train_step(model, batch):
    premise, hypothesis, sts_labels, nli_labels = batch

    # Forward pass
    sts_scores, nli_logits = model(premise, hypothesis)

    # Loss calculations
    sts_loss = sts_criterion(sts_scores, sts_labels)
    nli_loss = nli_criterion(nli_logits, nli_labels)

    # Total loss
    total_loss = lambda_sts * sts_loss + lambda_nli * nli_loss

    # Backpropagation and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return sts_loss.item(), nli_loss.item(), total_loss.item()
```

## 4. Evaluation Metrics

Evaluation metrics are used to measure and compare the performance of trained models. Different evaluation metrics are used for STS and NLI tasks.

### 4.1 STS Evaluation Metrics

#### 4.1.1 Pearson and Spearman Correlation

Pearson and Spearman correlation coefficients measure the relationship between predicted similarity scores and actual similarity scores. These metrics are the most commonly used metrics in STS evaluation.

**Pearson Correlation:**

Pearson correlation coefficient measures the linear relationship between two variables. It takes values between -1 and 1. 1 indicates perfect positive correlation; -1 indicates perfect negative correlation; 0 indicates no relationship.

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**Spearman Correlation:**

Spearman correlation coefficient measures the ranking relationship between two variables. It can also capture nonlinear relationships.

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

**Python Implementation:**

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaluate_sts(pred_scores, true_scores):
    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(pred_scores, true_scores)

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(pred_scores, true_scores)

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }
```

#### 4.1.2 Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors and takes values between -1 and 1. It is commonly used in STS tasks to measure similarity between embeddings produced by the model.

**Mathematical Formulation:**

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

**Python Implementation:**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_sim(embeddings1, embeddings2):
    """
    Computes cosine similarity between two embedding sets
    """
    # embeddings1 and embeddings2 are 2D numpy arrays
    sim_matrix = cosine_similarity(embeddings1, embeddings2)
    # Diagonal elements are the similarity of corresponding pairs
    sim_scores = np.diag(sim_matrix)
    return sim_scores
```

#### 4.1.3 Manhattan and Euclidean Distances

Manhattan and Euclidean distances are metrics that measure the distance between two vectors. In STS tasks, distance between embeddings can be interpreted as the inverse of similarity.

**Manhattan Distance (L1 Norm):**

$$d_{manhattan}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{d} |a_i - b_i|$$

**Euclidean Distance (L2 Norm):**

$$d_{euclidean}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}$$

### 4.2 NLI Evaluation Metrics

#### 4.2.1 Accuracy

Accuracy is the most basic evaluation metric for classification tasks. It measures the ratio of correctly predicted samples to all samples.

**Mathematical Formulation:**

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Samples}} = \frac{TP + TN}{TP + TN + FP + FN}$$

Where:
- TP (True Positive): Correct positive predictions
- TN (True Negative): Correct negative predictions
- FP (False Positive): Wrong positive predictions
- FN (False Negative): Wrong negative predictions

**Python Implementation:**

```python
from sklearn.metrics import accuracy_score

def evaluate_nli(pred_labels, true_labels):
    """
    Calculates the accuracy of NLI predictions
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy
```

#### 4.2.2 Precision, Recall, F1 Score

Precision, Recall, and F1 Score are metrics that evaluate classification performance in more detail. In multi-class problems like NLI, they can be calculated separately for each class.

**Mathematical Formulation:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Python Implementation:**

```python
from sklearn.metrics import precision_recall_fscore_support

def evaluate_nli_detailed(pred_labels, true_labels):
    """
    Calculates detailed evaluation metrics for NLI predictions
    """
    # Class labels: 0=entailment, 1=contradiction, 2=neutral
    class_names = ['entailment', 'contradiction', 'neutral']

    # Calculate metrics for all classes
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )

    # Report class-based metrics
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }

    # Calculate average metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted'
    )

    results['weighted_avg'] = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }

    return results
```

#### 4.2.3 Confusion Matrix Analysis

Confusion matrix is a table that visualizes the performance of a classification model in detail. In multi-class problems like NLI, it shows which classes the model confuses with each other.

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(pred_labels, true_labels):
    """
    Draws confusion matrix for NLI predictions
    """
    # Class labels
    class_names = ['entailment', 'contradiction', 'neutral']

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Normalize (optional)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
```

### 4.3 SentEval and GLUE Benchmark

SentEval and GLUE are comprehensive benchmark tools used to evaluate sentence embedding models and general NLP models.

**SentEval:**

SentEval is a tool that evaluates sentence embedding models on various tasks. These tasks include STS, NLI, sentiment analysis, and text classification.

**GLUE Benchmark:**

GLUE (General Language Understanding Evaluation) is a benchmark collection containing various natural language understanding tasks. GLUE is used to evaluate the performance of general NLP models.

GLUE includes nine different tasks: MNLI (MultiNLI), QQP (Quora Question Pairs), QNLI (Question NLI), SST-2 (Stanford Sentiment Treebank), CoLA (Corpus of Linguistic Acceptability), STS-B (STS Benchmark), MRPC (Microsoft Research Paraphrase Corpus), RTE (Recognizing Textual Entailment), and WNLI (Winograd NLI).

## 5. Practical Applications

### 5.1 Training and Evaluation for STS Models

BERT and S-BERT (Sentence-BERT) are transformer-based models frequently used for STS tasks. S-BERT is specifically optimized to produce sentence-level embeddings.

**STS with BERT:**

```python
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AdamW

class BertForSTS(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super(BertForSTS, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        score = self.regressor(pooled_output)
        return score * 5.0  # Scale to 0-5 range for STS
```

**STS with S-BERT:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load S-BERT model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Prepare training data
train_examples = []
for i in range(len(train_data[0])):
    train_examples.append(InputExample(
        texts=[train_data[0][i][0], train_data[0][i][1]],
        label=train_data[1][i] / 5.0  # Normalize to 0-1 range
    ))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Use cosine similarity loss
train_loss = losses.CosineSimilarityLoss(model)

# Train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    evaluation_steps=1000,
    warmup_steps=100
)
```

### 5.2 Training and Evaluation for NLI Models

Transformer models, especially BERT and its variants, are highly effective for NLI tasks. These models can capture contextual relationships by encoding premise and hypothesis sentences together.

```python
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score, classification_report

# BERT model for NLI
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Data preprocessing function
def preprocess_nli_batch(premise_hypothesis_pairs, labels=None):
    premises = [pair[0] for pair in premise_hypothesis_pairs]
    hypotheses = [pair[1] for pair in premise_hypothesis_pairs]

    encodings = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )

    if labels is not None:
        return encodings, torch.tensor(labels)
    else:
        return encodings
```

### 5.3 Hyperparameter Optimization

Hyperparameter optimization is a critical step to improve model performance. Important hyperparameters for STS and NLI models include learning rate, batch size, dropout rate, and model architecture.

**Grid Search:**

```python
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

# Hyperparameters for Grid Search
param_grid = {
    'lr': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32, 64],
    'warmup_ratio': [0.1, 0.2],
    'weight_decay': [0.01, 0.1]
}

# Grid Search structure
def grid_search_sts(train_data, dev_data, model_name="bert-base-uncased", param_grid=param_grid):
    best_score = -1
    best_params = None

    # Try all hyperparameter combinations
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            for warmup_ratio in param_grid['warmup_ratio']:
                for weight_decay in param_grid['weight_decay']:
                    # Create and train model
                    # ...

                    # Evaluate
                    # ...

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'lr': lr,
                            'batch_size': batch_size,
                            'warmup_ratio': warmup_ratio,
                            'weight_decay': weight_decay
                        }

    return best_params, best_score
```

**Bayesian Optimization:**

```python
import optuna

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    pooling_strategy = trial.suggest_categorical("pooling_strategy", ["mean", "max", "cls"])

    # Create and train model
    # ...

    # Evaluate
    # ...

    return score  # Value to be maximized

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best parameters:", study.best_params)
print("Best score:", study.best_value)
```
