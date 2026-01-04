# STS ve NLI: Loss Fonksiyonları ve Değerlendirme Rehberi

## İçindekiler

1. [Giriş](#1-giriş)
2. [Teorik Temeller](#2-teorik-temeller)
   - 2.1 [Semantic Textual Similarity (STS)](#21-semantic-textual-similarity-sts)
   - 2.2 [Natural Language Inference (NLI)](#22-natural-language-inference-nli)
   - 2.3 [Vektör Temsilleri ve Semantik Uzay](#23-vektör-temsilleri-ve-semantik-uzay)
3. [Loss Fonksiyonları](#3-loss-fonksiyonları)
   - 3.1 [STS için Loss Fonksiyonları](#31-sts-için-loss-fonksiyonları)
     - 3.1.1 [Mean Squared Error (MSE)](#311-mean-squared-error-mse)
     - 3.1.2 [Cosine Embedding Loss](#312-cosine-embedding-loss)
     - 3.1.3 [Contrastive Loss](#313-contrastive-loss)
     - 3.1.4 [Triplet Loss](#314-triplet-loss)
     - 3.1.5 [Multiple Negatives Ranking Loss](#315-multiple-negatives-ranking-loss)
   - 3.2 [NLI için Loss Fonksiyonları](#32-nli-için-loss-fonksiyonları)
     - 3.2.1 [Cross-Entropy Loss](#321-cross-entropy-loss)
     - 3.2.2 [Focal Loss](#322-focal-loss)
     - 3.2.3 [Label Smoothing](#323-label-smoothing)
   - 3.3 [Multi-Task Learning için Loss Fonksiyonları](#33-multi-task-learning-için-loss-fonksiyonları)
4. [Değerlendirme Metrikleri](#4-değerlendirme-metrikleri)
   - 4.1 [STS Değerlendirme Metrikleri](#41-sts-değerlendirme-metrikleri)
     - 4.1.1 [Pearson ve Spearman Korelasyonu](#411-pearson-ve-spearman-korelasyonu)
     - 4.1.2 [Cosine Similarity](#412-cosine-similarity)
     - 4.1.3 [Manhattan ve Euclidean Mesafeleri](#413-manhattan-ve-euclidean-mesafeleri)
   - 4.2 [NLI Değerlendirme Metrikleri](#42-nli-değerlendirme-metrikleri)
     - 4.2.1 [Accuracy](#421-accuracy)
     - 4.2.2 [Precision, Recall, F1 Score](#422-precision-recall-f1-score)
     - 4.2.3 [Confusion Matrix Analizi](#423-confusion-matrix-analizi)
   - 4.3 [SentEval ve GLUE Benchmark](#43-senteval-ve-glue-benchmark)
5. [Pratik Uygulamalar](#5-pratik-uygulamalar)
   - 5.1 [STS Modelleri için Eğitim ve Değerlendirme](#51-sts-modelleri-için-eğitim-ve-değerlendirme)
   - 5.2 [NLI Modelleri için Eğitim ve Değerlendirme](#52-nli-modelleri-için-eğitim-ve-değerlendirme)
   - 5.3 [Hyperparameter Optimizasyonu](#53-hyperparameter-optimizasyonu)

## 1. Giriş

Semantic Textual Similarity (STS) ve Natural Language Inference (NLI), doğal dil işleme (NLP) alanında temel görevler arasında yer alır. Bu görevler, modern dil modellerinin metinler arasındaki anlamsal ilişkileri anlama ve temsil etme yeteneklerini test etmekte önemli rol oynar.

Bu rehber, STS ve NLI görevleri için kullanılan loss fonksiyonlarını ve değerlendirme metriklerini teorik ve pratik yönleriyle ele alacaktır. Temel kavramlardan başlayarak, ileri seviye tekniklere kadar geniş bir yelpazede bilgi sunmayı hedeflemektedir.

## 2. Teorik Temeller

### 2.1 Semantic Textual Similarity (STS)

Semantic Textual Similarity (STS), iki metin arasındaki anlamsal benzerliği ölçen bir NLP görevidir. STS, tipik olarak iki cümle arasındaki benzerliği 0 (tamamen farklı) ile 5 (tamamen eş anlamlı) arasında puanlar.

**Temel STS Kavramları:**

- **Lexical Similarity**: Kelime düzeyinde benzerlik, ortak kelimelere dayalı
- **Syntactic Similarity**: Sözdizimsel yapıların benzerliği
- **Semantic Similarity**: Anlam düzeyinde benzerlik, bağlam ve kavramları dikkate alır

**STS'nin Matematiksel Formülasyonu:**

STS, iki cümle vektörü arasındaki benzerliği hesaplama problemi olarak modellenebilir:

$$STS(s_1, s_2) = f(Emb(s_1), Emb(s_2))$$

Burada:
- $s_1, s_2$ iki cümledir
- $Emb(s)$ cümlenin vektör temsilini (embedding) oluşturan fonksiyondur
- $f$ iki vektör arasındaki benzerliği ölçen bir benzerlik fonksiyonudur

**STS Benchmark ve Veri Kümeleri:**
- STS Benchmark (STS-B)
- SICK (Sentences Involving Compositional Knowledge)
- SemEval STS görevleri

### 2.2 Natural Language Inference (NLI)

Natural Language Inference (NLI), bir öncül cümle (premise) ile bir hipotez cümlesi (hypothesis) arasındaki mantıksal ilişkiyi belirlemeyi amaçlayan bir NLP görevidir. NLI, tipik olarak üç sınıfa ayrılır:

- **Entailment (Çıkarım)**: Öncül doğruysa, hipotez de kesinlikle doğrudur.
- **Contradiction (Çelişki)**: Öncül doğruysa, hipotez kesinlikle yanlıştır.
- **Neutral (Nötr)**: Öncülün doğruluğu, hipotezin doğruluğunu belirlemez.

**NLI'nin Matematiksel Formülasyonu:**

NLI, iki cümle arasındaki mantıksal ilişkiyi sınıflandırma problemi olarak modellenebilir:

$$NLI(p, h) = \text{argmax}_c P(c | p, h)$$

Burada:
- $p$ öncül cümledir
- $h$ hipotez cümledir
- $c \in \{\text{entailment}, \text{contradiction}, \text{neutral}\}$ sınıflardan biridir
- $P(c | p, h)$ öncül ve hipotez verildiğinde $c$ sınıfının olasılığıdır

**NLI Benchmark ve Veri Kümeleri:**
- SNLI (Stanford Natural Language Inference)
- MultiNLI
- XNLI (Cross-lingual Natural Language Inference)
- ANLI (Adversarial NLI)

### 2.3 Vektör Temsilleri ve Semantik Uzay

STS ve NLI görevlerinin temelinde, metinleri anlamlı vektör temsillerine dönüştürmek yer alır. Bu vektör temsilleri, metin içeriğinin anlamsal özelliklerini yakalayan çok boyutlu bir semantik uzayda yer alır.

**Vektör Temsil Yöntemleri:**

1. **Sözcük Vektörleri:**
   - Word2Vec (CBOW ve Skip-gram)
   - GloVe (Global Vectors for Word Representation)
   - FastText (Karakter n-gram'larını kullanan modeller)

2. **Cümle Vektörleri:**
   - Basit ortalama yöntemleri (sözcük vektörlerinin ortalaması)
   - TF-IDF ağırlıklı ortalamalar
   - Doc2Vec
   - Skip-Thought Vectors

3. **Contextual Embeddings:**
   - ELMo (Embeddings from Language Models)
   - BERT (Bidirectional Encoder Representations from Transformers)
   - RoBERTa, XLNet, ALBERT vb.
   - S-BERT (Sentence-BERT)

**Semantik Uzayın Matematiksel Temelleri:**

Semantik uzay, genellikle $\mathbb{R}^d$ şeklinde $d$-boyutlu bir vektör uzayı olarak modellenebilir. Burada her metin, bu uzayda bir nokta olarak temsil edilir. İki metin arasındaki anlamsal benzerlik, bu noktalar arasındaki mesafe veya açı ile ölçülür.

Örneğin, cosine benzerliği şu şekilde hesaplanır:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

Burada:
- $\mathbf{a}, \mathbf{b}$ iki metin vektörüdür
- $a_i, b_i$ bu vektörlerin $i$. bileşenleridir
- $|\mathbf{a}|, |\mathbf{b}|$ vektörlerin normlarıdır

## 3. Loss Fonksiyonları

Loss fonksiyonları, model eğitiminin merkezinde yer alır ve modelin tahminleri ile gerçek hedef değerler arasındaki farkı ölçer. STS ve NLI görevleri için farklı loss fonksiyonları kullanılır, çünkü bu görevlerin doğası ve hedefleri farklıdır.

### 3.1 STS için Loss Fonksiyonları

#### 3.1.1 Mean Squared Error (MSE)

MSE, regresyon problemleri için en yaygın kullanılan loss fonksiyonudur ve STS'de de sıklıkla kullanılır. İki cümle arasındaki tahmin edilen benzerlik skoru ile gerçek benzerlik skoru arasındaki farkın karesini ölçer.

**Matematiksel Formülasyon:**

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Burada:
- $N$ örneklerin sayısıdır
- $y_i$ $i$. örnek için gerçek benzerlik skorudur
- $\hat{y}_i$ model tarafından tahmin edilen benzerlik skorudur

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss()
similarity_pred = model(sentence1, sentence2)
loss = criterion(similarity_pred, true_similarity)
```

**Avantajları:**
- Uygulaması kolaydır
- Sürekli ve türevlenebilirdir
- Büyük hataları orantısız şekilde cezalandırır

**Dezavantajları:**
- Aykırı değerlere karşı hassastır
- Vektör boşluğunun geometrik yapısını dikkate almaz

#### 3.1.2 Cosine Embedding Loss

Cosine Embedding Loss, iki vektör arasındaki açıyı minimize etmeye çalışır. Bu, anlamsal olarak benzer cümlelerin temsil uzayında birbirine yakın olmasını sağlar.

**Matematiksel Formülasyon:**

$$
\begin{cases}
1 - \cos(a, b), & \text{if } y = 1 \\
\max(0, \cos(a, b) - \text{margin}), & \text{if } y = -1
\end{cases}
$$

Burada:
- $a, b$ iki cümlenin embedding vektörleridir
- $y \in \{1, -1\}$ cümlelerin benzer olup olmadığını gösteren bir etikettir
- $\cos(a, b)$ $a$ ve $b$ arasındaki cosine benzerliğidir
- margin, benzer olmayan cümlelerin en az ne kadar farklı olması gerektiğini belirleyen bir parametredir

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn

criterion = nn.CosineEmbeddingLoss(margin=0.2)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
# y=1 benzer cümleler için, y=-1 farklı cümleler için
loss = criterion(embedding1, embedding2, target)
```

**Avantajları:**
- Vektörlerin yönü üzerine odaklanır, büyüklüğü değil
- Anlamsal benzerlik için daha uygundur
- Normalize edilmiş vektörlerle çalışır

**Dezavantajları:**
- İkili benzerlik (benzer/farklı) için tasarlanmıştır, ince granüler benzerlik skorları için ek işleme gerektirir
- Margin parametresinin ayarlanması gerekir

#### 3.1.3 Contrastive Loss

Contrastive Loss, benzer cümle çiftlerinin temsil uzayında birbirine yakın, farklı cümle çiftlerinin ise belirli bir marjinden daha uzak olmasını sağlar.

**Matematiksel Formülasyon:**

$$\mathcal{L}_{contrastive}(a, b, y) = (1-y) \cdot \frac{1}{2} d(a, b)^2 + y \cdot \frac{1}{2} \max(0, \text{margin} - d(a, b))^2$$

Burada:
- $a, b$ iki cümlenin embedding vektörleridir
- $y \in \{0, 1\}$ cümlelerin benzer olup olmadığını gösteren bir etikettir (1: benzer, 0: farklı)
- $d(a, b)$ $a$ ve $b$ arasındaki Euclidean mesafedir
- margin, farklı cümlelerin en az ne kadar uzak olması gerektiğini belirleyen bir parametredir

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, x1, x2, y):
        # Euclidean mesafeyi hesapla
        dist = F.pairwise_distance(x1, x2, p=2)
        # Contrastive loss hesapla
        loss = 0.5 * (y * dist.pow(2) + (1-y) * F.relu(self.margin - dist).pow(2))
        return loss.mean()
        
criterion = ContrastiveLoss(margin=2.0)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
# y=1 benzer cümleler için, y=0 farklı cümleler için
loss = criterion(embedding1, embedding2, target)
```

#### 3.1.4 Triplet Loss

Triplet Loss, üçlü örnekler (anchor, positive, negative) kullanarak, anchor örneğinin positive örneğine negative örneğinden daha yakın olmasını sağlar.

**Matematiksel Formülasyon:**

$$\mathcal{L}_{triplet} = \max(0, d(a, p) - d(a, n) + \text{margin})$$

Burada:
- $a$ anchor cümlesinin embedding vektörüdür
- $p$ positive cümlesinin embedding vektörüdür (anchor ile anlamsal olarak benzer)
- $n$ negative cümlesinin embedding vektörüdür (anchor ile anlamsal olarak farklı)
- $d(x, y)$ $x$ ve $y$ arasındaki mesafe fonksiyonudur (genellikle Euclidean mesafe)
- margin, anchor-positive mesafesi ile anchor-negative mesafesi arasındaki minimum farkı belirler

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Mesafeleri hesapla
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        # Triplet loss hesapla
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()
        
criterion = TripletLoss(margin=1.0)
anchor_emb = model.encode(anchor_sentence)
positive_emb = model.encode(positive_sentence)
negative_emb = model.encode(negative_sentence)
loss = criterion(anchor_emb, positive_emb, negative_emb)
```

#### 3.1.5 Multiple Negatives Ranking Loss

Multiple Negatives Ranking Loss (MNRL), bir batch içindeki tüm örnekleri kullanarak, pozitif çiftleri negatif çiftlerden ayırt etmeye çalışır. Bu loss, özellikle S-BERT gibi modellerde yaygın olarak kullanılır.

**Matematiksel Formülasyon:**

$$\mathcal{L}_{MNRL} = -\log\frac{\exp(\text{sim}(a_i, p_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(a_i, b_j)/\tau)}$$

Burada:
- $a_i$ ve $p_i$ pozitif bir çiftin embedding vektörleridir
- $b_j$ batch içindeki tüm diğer cümlelerin embedding vektörleridir
- $\text{sim}(x, y)$ $x$ ve $y$ arasındaki benzerlik ölçüsüdür (genellikle cosine benzerliği)
- $\tau$ sıcaklık parametresidir (softmax fonksiyonunu kontrol eder)
- $N$ batch boyutudur

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale=20.0):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale  # Sıcaklığın tersi
        
    def forward(self, embeddings_a, embeddings_b):
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Cosine similarity matrisini hesapla
        scores = torch.matmul(embeddings_a, embeddings_b.transpose(0, 1)) * self.scale
        
        # Diagonal elementler positive çiftlerdir
        labels = torch.arange(len(scores), device=scores.device)
        
        # Cross-entropy loss hesapla
        loss = F.cross_entropy(scores, labels)
        return loss
```

### 3.2 NLI için Loss Fonksiyonları

#### 3.2.1 Cross-Entropy Loss

Cross-Entropy Loss, sınıflandırma problemleri için en yaygın kullanılan loss fonksiyonudur ve NLI gibi çok sınıflı problemler için idealdir.

**Matematiksel Formülasyon:**

$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$$

Burada:
- $N$ örneklerin sayısıdır
- $C$ sınıfların sayısıdır (NLI için genellikle 3: entailment, contradiction, neutral)
- $y_{i,c}$ $i$. örneğin $c$ sınıfına ait olup olmadığını belirten bir göstergedir (one-hot encoding)
- $p_{i,c}$ modelin $i$. örneğin $c$ sınıfına ait olma olasılığı tahminidir

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
logits = model(premise, hypothesis)  # Shape: [batch_size, 3]
# labels: 0=entailment, 1=contradiction, 2=neutral
loss = criterion(logits, labels)
```

#### 3.2.2 Focal Loss

Focal Loss, sınıf dengesizliği problemlerine karşı dayanıklı bir loss fonksiyonudur. Zor örneklere daha fazla ağırlık verir ve kolay örnekleri azaltır.

**Matematiksel Formülasyon:**

$$\mathcal{L}_{focal} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} (1 - p_{i,c})^{\gamma} \log(p_{i,c})$$

Burada:
- $\gamma$ odaklanma parametresidir (genellikle 2 olarak ayarlanır)
- Diğer semboller Cross-Entropy Loss ile aynıdır

**PyTorch Uygulaması:**

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
        # Softmax uygula
        probs = F.softmax(logits, dim=1)
        # Doğru sınıfın olasılığını al
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Focal weight hesapla
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Alpha weight uygula (opsiyonel)
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_weight
            
        # Loss hesapla
        loss = -focal_weight * torch.log(pt)
        return loss.mean()
```

#### 3.2.3 Label Smoothing

Label Smoothing, modelin aşırı güvenli tahminler yapmasını önlemek için kullanılan bir regularizasyon tekniğidir. One-hot etiketler yerine yumuşatılmış etiketler kullanır.

**Matematiksel Formülasyon:**

$$y_{i,c}^{smooth} = (1 - \alpha) \cdot y_{i,c} + \alpha \cdot \frac{1}{C}$$

Burada:
- $\alpha$ yumuşatma parametresidir (genellikle 0.1 olarak ayarlanır)
- $C$ sınıfların sayısıdır
- $y_{i,c}$ orijinal one-hot etikettir
- $y_{i,c}^{smooth}$ yumuşatılmış etikettir

**PyTorch Uygulaması:**

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
        
        # Yumuşatılmış etiketleri oluştur
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / self.num_classes
        
        # Loss hesapla
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        return loss.mean()
```

### 3.3 Multi-Task Learning için Loss Fonksiyonları

STS ve NLI görevleri genellikle birlikte eğitilir, çünkü bu görevler anlamsal anlama becerileri açısından birbiriyle ilişkilidir. Multi-task learning için kullanılan loss fonksiyonları, farklı görevlerin loss'larını birleştirir.

**Ağırlıklı Toplam Loss:**

$$\mathcal{L}_{total} = \lambda_{STS} \cdot \mathcal{L}_{STS} + \lambda_{NLI} \cdot \mathcal{L}_{NLI}$$

Burada:
- $\lambda_{STS}$ ve $\lambda_{NLI}$ her görevin ağırlığını belirleyen hiperparametrelerdir
- $\mathcal{L}_{STS}$ STS görevi için kullanılan loss fonksiyonudur
- $\mathcal{L}_{NLI}$ NLI görevi için kullanılan loss fonksiyonudur

**PyTorch Uygulaması:**

```python
import torch
import torch.nn as nn

# Loss fonksiyonları
sts_criterion = nn.MSELoss()
nli_criterion = nn.CrossEntropyLoss()

# Hiperparametreler
lambda_sts = 0.5
lambda_nli = 0.5

# Eğitim adımı
def train_step(model, batch):
    premise, hypothesis, sts_labels, nli_labels = batch
    
    # İleri geçiş
    sts_scores, nli_logits = model(premise, hypothesis)
    
    # Loss hesaplamaları
    sts_loss = sts_criterion(sts_scores, sts_labels)
    nli_loss = nli_criterion(nli_logits, nli_labels)
    
    # Toplam loss
    total_loss = lambda_sts * sts_loss + lambda_nli * nli_loss
    
    # Geri yayılım ve optimizasyon
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return sts_loss.item(), nli_loss.item(), total_loss.item()
```

## 4. Değerlendirme Metrikleri

Değerlendirme metrikleri, eğitilen modellerin performansını ölçmek ve karşılaştırmak için kullanılır. STS ve NLI görevleri için farklı değerlendirme metrikleri kullanılır.

### 4.1 STS Değerlendirme Metrikleri

#### 4.1.1 Pearson ve Spearman Korelasyonu

Pearson ve Spearman korelasyon katsayıları, tahmin edilen benzerlik skorları ile gerçek benzerlik skorları arasındaki ilişkiyi ölçer. Bu metrikler, STS değerlendirmesinde en yaygın kullanılan metriklerdir.

**Pearson Korelasyonu:**

Pearson korelasyon katsayısı, iki değişken arasındaki doğrusal ilişkiyi ölçer. -1 ile 1 arasında değer alır. 1, mükemmel pozitif korelasyonu; -1, mükemmel negatif korelasyonu; 0 ise ilişki olmadığını gösterir.

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**Spearman Korelasyonu:**

Spearman korelasyon katsayısı, iki değişken arasındaki sıralama ilişkisini ölçer. Doğrusal olmayan ilişkileri de yakalayabilir.

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

**Python Uygulaması:**

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaluate_sts(pred_scores, true_scores):
    # Pearson korelasyonu hesapla
    pearson_corr, _ = pearsonr(pred_scores, true_scores)
    
    # Spearman korelasyonu hesapla
    spearman_corr, _ = spearmanr(pred_scores, true_scores)
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }
```

#### 4.1.2 Cosine Similarity

Cosine similarity, iki vektör arasındaki açının kosinüsünü ölçer ve -1 ile 1 arasında değer alır. STS görevlerinde, model tarafından üretilen embeddingler arasındaki benzerliği ölçmek için yaygın olarak kullanılır.

**Matematiksel Formülasyon:**

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

**Python Uygulaması:**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_sim(embeddings1, embeddings2):
    """
    İki embedding seti arasındaki cosine benzerliğini hesaplar
    """
    # embeddings1 ve embeddings2 2D numpy dizileridir
    sim_matrix = cosine_similarity(embeddings1, embeddings2)
    # Diagonal elementler karşılık gelen çiftlerin benzerliğidir
    sim_scores = np.diag(sim_matrix)
    return sim_scores
```

#### 4.1.3 Manhattan ve Euclidean Mesafeleri

Manhattan ve Euclidean mesafeleri, iki vektör arasındaki uzaklığı ölçen metriklerdir. STS görevlerinde, embeddingler arasındaki uzaklık, benzerliğin tersi olarak yorumlanabilir.

**Manhattan Mesafesi (L1 Normu):**

$$d_{manhattan}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{d} |a_i - b_i|$$

**Euclidean Mesafesi (L2 Normu):**

$$d_{euclidean}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}$$

### 4.2 NLI Değerlendirme Metrikleri

#### 4.2.1 Accuracy

Accuracy (doğruluk), sınıflandırma görevleri için en temel değerlendirme metriğidir. Doğru tahmin edilen örneklerin tüm örneklere oranını ölçer.

**Matematiksel Formülasyon:**

$$\text{Accuracy} = \frac{\text{Doğru Tahmin Sayısı}}{\text{Toplam Örnek Sayısı}} = \frac{TP + TN}{TP + TN + FP + FN}$$

Burada:
- TP (True Positive): Doğru pozitif tahminler
- TN (True Negative): Doğru negatif tahminler
- FP (False Positive): Yanlış pozitif tahminler
- FN (False Negative): Yanlış negatif tahminler

**Python Uygulaması:**

```python
from sklearn.metrics import accuracy_score

def evaluate_nli(pred_labels, true_labels):
    """
    NLI tahminlerinin doğruluğunu hesaplar
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy
```

#### 4.2.2 Precision, Recall, F1 Score

Precision (kesinlik), Recall (duyarlılık) ve F1 Score, sınıflandırma performansını daha detaylı değerlendiren metriklerdir. NLI gibi çok sınıflı problemlerde, her sınıf için ayrı ayrı hesaplanabilir.

**Matematiksel Formülasyon:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Python Uygulaması:**

```python
from sklearn.metrics import precision_recall_fscore_support

def evaluate_nli_detailed(pred_labels, true_labels):
    """
    NLI tahminleri için detaylı değerlendirme metrikleri hesaplar
    """
    # Sınıf etiketleri: 0=entailment, 1=contradiction, 2=neutral
    class_names = ['entailment', 'contradiction', 'neutral']
    
    # Tüm sınıflar için metrikleri hesapla
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )
    
    # Sınıf bazında metrikleri raporla
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # Ortalama metrikleri hesapla
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

#### 4.2.3 Confusion Matrix Analizi

Confusion matrix (karışıklık matrisi), bir sınıflandırma modelinin performansını detaylı olarak görselleştiren bir tablodur. NLI gibi çok sınıflı problemlerde, modelin hangi sınıfları birbiriyle karıştırdığını gösterir.

**Python Uygulaması:**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(pred_labels, true_labels):
    """
    NLI tahminleri için karışıklık matrisini çizer
    """
    # Sınıf etiketleri
    class_names = ['entailment', 'contradiction', 'neutral']
    
    # Karışıklık matrisini hesapla
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize et (opsiyonel)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
```

### 4.3 SentEval ve GLUE Benchmark

SentEval ve GLUE, cümle embedding modellerini ve genel NLP modellerini değerlendirmek için kullanılan kapsamlı benchmark araçlarıdır.

**SentEval:**

SentEval, cümle embedding modellerini çeşitli görevler üzerinde değerlendiren bir araçtır. Bu görevler arasında STS, NLI, duygu analizi ve metin sınıflandırma yer alır.

**GLUE Benchmark:**

GLUE (General Language Understanding Evaluation), çeşitli doğal dil anlama görevlerini içeren bir benchmark koleksiyonudur. GLUE, genel NLP modellerinin performansını değerlendirmek için kullanılır.

GLUE, aralarında MNLI (MultiNLI), QQP (Quora Question Pairs), QNLI (Question NLI), SST-2 (Stanford Sentiment Treebank), CoLA (Corpus of Linguistic Acceptability), STS-B (STS Benchmark), MRPC (Microsoft Research Paraphrase Corpus), RTE (Recognizing Textual Entailment) ve WNLI (Winograd NLI) görevlerinin bulunduğu dokuz farklı görevi içerir.

## 5. Pratik Uygulamalar

### 5.1 STS Modelleri için Eğitim ve Değerlendirme

BERT ve S-BERT (Sentence-BERT), STS görevleri için sıklıkla kullanılan transformer tabanlı modellerdir. S-BERT, özellikle cümle seviyesinde embedding üretmek için optimize edilmiştir.

**BERT ile STS:**

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

**S-BERT ile STS:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# S-BERT modeli yükle
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Eğitim verilerini hazırla
train_examples = []
for i in range(len(train_data[0])):
    train_examples.append(InputExample(
        texts=[train_data[0][i][0], train_data[0][i][1]],
        label=train_data[1][i] / 5.0  # 0-1 aralığına normalize et
    ))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Cosine similarity loss kullan
train_loss = losses.CosineSimilarityLoss(model)

# Modeli eğit
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    evaluation_steps=1000,
    warmup_steps=100
)
```

### 5.2 NLI Modelleri için Eğitim ve Değerlendirme

Transformer modelleri, özellikle BERT ve türevleri, NLI görevleri için son derece etkilidir. Bu modeller, öncül ve hipotez cümlelerini birlikte kodlayarak bağlamsal ilişkileri yakalayabilir.

```python
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score, classification_report

# NLI için BERT modeli
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Veri ön işleme fonksiyonu
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

### 5.3 Hyperparameter Optimizasyonu

Hiperparametre optimizasyonu, model performansını artırmak için kritik bir adımdır. STS ve NLI modelleri için önemli hiperparametreler arasında öğrenme oranı, batch boyutu, dropout oranı ve model mimarisi yer alır.

**Grid Search:**

```python
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

# Grid Search için hiperparametreler
param_grid = {
    'lr': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32, 64],
    'warmup_ratio': [0.1, 0.2],
    'weight_decay': [0.01, 0.1]
}

# Grid Search yapısı
def grid_search_sts(train_data, dev_data, model_name="bert-base-uncased", param_grid=param_grid):
    best_score = -1
    best_params = None
    
    # Tüm hiperparametre kombinasyonlarını dene
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            for warmup_ratio in param_grid['warmup_ratio']:
                for weight_decay in param_grid['weight_decay']:
                    # Model oluştur ve eğit
                    # ...
                    
                    # Değerlendir
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
    # Hiperparametreleri örnekle
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    pooling_strategy = trial.suggest_categorical("pooling_strategy", ["mean", "max", "cls"])
    
    # Model oluştur ve eğit
    # ...
    
    # Değerlendir
    # ...
    
    return score  # Maksimize edilecek değer

# Optimizasyon çalıştır
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# En iyi parametreleri yazdır
print("Best parameters:", study.best_params)
print("Best score:", study.best_value)
```