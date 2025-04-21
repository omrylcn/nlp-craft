# BERT ile Metin Embeddingi ve Benzerlik Analizi: Kapsamlı Rehber

## İçindekiler
- [1. Giriş ve Teorik Temel](#1-giriş-ve-teorik-temel)
  - [1.1. Metin Embedding'leri Nedir?](#11-metin-embeddingleri-nedir)
  - [1.2. Semantik Benzerlik Kavramı](#12-semantik-benzerlik-kavramı)
  - [1.3. BERT Embedding Tipleri](#13-bert-embedding-tipleri)
- [2. BERT ile Metin Embedding Oluşturma](#2-bert-ile-metin-embedding-oluşturma)
  - [2.1. BERT Modellerinin Çeşitleri](#21-bert-modellerinin-çeşitleri)
  - [2.2. Pooling Stratejileri](#22-pooling-stratejileri)
  - [2.3. SentenceTransformers Kütüphanesi](#23-sentencetransformers-kütüphanesi)
- [3. Metin Benzerlik Hesaplama](#3-metin-benzerlik-hesaplama)
  - [3.1. Benzerlik Metrikleri](#31-benzerlik-metrikleri)
  - [3.2. Normalizasyon Teknikleri](#32-normalizasyon-teknikleri)
  - [3.3. Semantik Arama ve Retrieval](#33-semantik-arama-ve-retrieval)

## 1. Giriş ve Teorik Temel

### 1.1. Metin Embedding'leri Nedir?

Metin embedding'leri, metin verilerini sayısal vektörlere dönüştüren tekniklerdir. Bu vektörler, metinlerin semantik (anlamsal) içeriğini sayısal uzayda temsil eder ve böylece bilgisayarların metinleri "anlamasını" sağlar.

**Metin Embedding'lerinin Özellikleri:**

- **Anlamsal Temsil**: Benzer anlamlı kelimeler/cümleler, vektör uzayında birbirine yakın konumlandırılır.
- **Boyutsallık**: Genellikle yüzlerce veya binlerce boyutlu vektörlerdir (örn. BERT-base: 768 boyut).
- **Yoğun (Dense) Vektörler**: One-hot encoding gibi seyrek vektörlerin aksine, bilgiyi daha verimli kodlar.
- **Bağlamsal Bilgi**: Özellikle BERT gibi modeller, bir kelimenin cümle içindeki bağlamına göre farklı embedding'ler üretir.

**Embedding Türleri:**

1. **Kelime Embedding'leri (Word Embeddings)**: 
   - Word2Vec, GloVe, FastText gibi modeller kelime düzeyinde temsiller oluşturur
   - Bir kelimenin tek bir sabit temsili vardır
   - Bağlamı yakalama yeteneği sınırlıdır

2. **Bağlamsal Embedding'ler (Contextual Embeddings)**:
   - BERT, RoBERTa, GPT gibi modeller tarafından oluşturulur
   - Bir kelimenin temsili, içinde bulunduğu cümleye göre değişir
   - Çok daha zengin semantik bilgi içerir

3. **Cümle Embedding'leri (Sentence Embeddings)**:
   - Tüm bir cümleyi/paragrafı tek bir vektörle temsil eder
   - Genellikle token embedding'lerinin bir tür ortalama veya havuzlaması (pooling) ile elde edilir
   - Doküman karşılaştırma, benzerlik, arama gibi görevler için uygundur

**Embedding'lerin Kullanım Alanları:**
- Semantik arama
- Doküman sınıflandırma ve kümeleme
- Öneri sistemleri
- Dil modelleri için özellik çıkarımı
- Benzerlik ve ilişki analizi
- Bilgi çıkarımı ve soruya cevap verme sistemleri

### 1.2. Semantik Benzerlik Kavramı

Semantik benzerlik, iki metin parçasının anlamsal olarak ne kadar yakın olduğunu ölçen bir kavramdır. Geleneksel kelime eşleştirme (lexical matching) yaklaşımlarının aksine, semantik benzerlik anlamı hedefler.

**Semantik vs. Leksikal Benzerlik:**

- **Leksikal Benzerlik**: Ortak kelime sayısı, n-gram örtüşmesi gibi yüzeysel özelliklere dayanır
  ```
  "Köpek havlıyor" vs "Köpek havlıyor" → Yüksek leksikal benzerlik
  "Köpek havlıyor" vs "Kedi miyavlıyor" → Düşük leksikal benzerlik
  ```

- **Semantik Benzerlik**: Anlamsal yakınlığa dayanır
  ```
  "Köpek havlıyor" vs "Canine ses çıkarıyor" → Yüksek semantik benzerlik
  "Banka hesabı açtım" vs "Nehir bankında oturdum" → Düşük semantik benzerlik (anlam farklı)
  ```

**Semantik Benzerlik Türleri:**

1. **Paragraflar/Dokümanlar Arası Benzerlik**: Uzun metinlerin karşılaştırılması, benzer belgelerin bulunması

2. **Cümleler Arası Benzerlik**: İki cümlenin anlamsal yakınlığı, parafraz tespiti, semantik eşdeğerlik

3. **Kelime-Cümle Benzerliği**: Bir kelimenin bir cümleyle ilgisini ölçme

4. **Çapraz Dil Benzerlik**: Farklı dillerdeki metinlerin anlam benzerliğini ölçme

**Semantik Benzerlik Ölçümleri:**

- **Cosine Similarity**: İki vektör arasındaki açıyı ölçer (en yaygın yöntem)
- **Euclidean Distance**: İki vektör arasındaki fiziksel mesafeyi ölçer
- **Manhattan Distance**: İki vektör arasındaki yatay+dikey mesafeyi ölçer
- **Dot Product**: İki vektörün nokta çarpımı (genellikle normalizasyon sonrası)

**Semantik Benzerliğin Zorlukları:**

- **Bağlam Duyarlılığı**: "Banka" kelimesinin farklı bağlamlardaki anlamını ayırt etme
- **Çok Anlamlılık (Polysemy)**: Aynı kelimenin farklı anlamları
- **Eşanlamlılık (Synonymy)**: Farklı kelimelerin aynı anlamı
- **Domain-Specifik Dil**: Farklı alanlarda kelimelerin anlamı değişebilir
- **Kültürel ve Dilbilimsel Nüanslar**: Deyimler, mecazlar, kültürel referanslar

### 1.3. BERT Embedding Tipleri

BERT (Bidirectional Encoder Representations from Transformers), metni farklı seviyelerinde temsil edebilen zengin embedding'ler üretir. Bu embedding'ler metin verilerinin içindeki kompleks anlamsal ve sözdizimsel ilişkileri yakalar.

**BERT'in Ürettiği Embedding Tipleri:**

1. **Token Embedding'leri**:
   - Her token için bir vektör (genellikle 768-boyutlu)
   - Bağlama duyarlı (yani aynı kelime farklı cümlelerde farklı embedding'lere sahip olur)
   - Sözdizimsel ve anlamsal bilgileri yakalar
   - Genellikle BERT'in son katmanından alınır (veya belirli görevler için farklı katmanlardan)

2. **[CLS] Token Embedding'i**:
   - BERT'in her girdisinin başına eklenen özel [CLS] token'ının temsili
   - Tüm cümlenin/paragrafın bütünsel temsilini taşıması için özel olarak eğitilmiştir
   - Genellikle sınıflandırma görevleri için kullanılır
   - Bazı durumlarda cümle embedding'i olarak da kullanılabilir

3. **Cümle Embedding'leri**:
   - Spesifik olarak bir cümlenin tam temsilini oluşturmak için tasarlanmıştır
   - Farklı yöntemlerle elde edilebilir:
     - [CLS] token temsilini kullanma
     - Tüm token temsillerinin ortalamasını alma (mean pooling)
     - Tüm token temsillerinin maksimumunu alma (max pooling)
     - İlk ve son token'ların birleştirilmesi
     - Attention mekanizmaları ile ağırlıklı ortalama alma
 
4. **BERT Katmanları Arası Embedding'ler**:
   - BERT'in farklı katmanları farklı dilbilimsel özellikleri yakalar:
   - Alt katmanlar: Sözdizimsel bilgi, sözcük grupları, temel anlamsal ilişkiler
   - Orta katmanlar: Karmaşık semantik yapılar, ilişkiler
   - Üst katmanlar: Task-specific bilgiler, daha soyut temsiller

**Embedding Örneği - Bir Cümle İçin BERT Çıktısı:**

BERT modeli, "Havalar ısındı, pikniğe gidelim" cümlesini işlediğinde:
```
[CLS] Havalar ısındı, pikniğe gidelim [SEP]
```

Şu şekilde embeddingler üretir:
- [CLS] token embedding'i: Tüm cümlenin temsilini içeren 768-boyutlu bir vektör
- "Havalar" token embedding'i: 768-boyutlu bir vektör
- "ısındı" token embedding'i: 768-boyutlu bir vektör
- "," token embedding'i: 768-boyutlu bir vektör
- ...vb.

**Önemli Not**: BERT'in original yapısında, doğrudan "cümle embedding'i" kavramı yoktur. Genellikle [CLS] token'ı veya token embedding'lerinin bir tür ortalama alınması (pooling) uygulanarak cümle-düzeyinde temsiller elde edilir. Bu sorunu çözmek için özel olarak cümle embedding'leri üretmek üzere ince ayar (fine-tune) yapılmış modeller geliştirilmiştir (örn. Sentence-BERT).

## 2. BERT ile Metin Embedding Oluşturma

### 2.1. BERT Modellerinin Çeşitleri

BERT modelleri, farklı boyutlarda ve çeşitli görevler için optimize edilmiş şekillerde bulunur. Embedding oluşturmak için kullanabileceğiniz başlıca BERT varyasyonları şunlardır:

**Temel BERT Modelleri:**

- **BERT-base**: 
  - 12 transformer katmanı, 768 gizli boyut, 12 attention head
  - ~110M parametre
  - Genel amaçlı kullanım için uygun denge

- **BERT-large**: 
  - 24 transformer katmanı, 1024 gizli boyut, 16 attention head
  - ~340M parametre
  - Daha yüksek kaliteli embeddingler sunar, ancak daha fazla kaynak gerektirir

**Optimize Edilmiş BERT Varyasyonları:**

- **DistilBERT**: 
  - BERT'in damıtılmış versiyonu, %40 daha küçük, %60 daha hızlı
  - 6 transformer katmanı
  - BERT-base'in performansının ~%97'sini korur
  - Sınırlı kaynaklar için idealdir

- **ALBERT (A Lite BERT)**:
  - Parametre paylaşımı tekniği ile bellek verimliliği sağlar
  - Daha az parametre ile benzer performans
  - Özellikle büyük modeller için etkilidir

**Semantik Benzerlik Odaklı Modeller:**

- **Sentence-BERT / SentenceTransformers**:
  - Özellikle anlamlı cümle embedding'leri oluşturmak için fine-tune edilmiş
  - Siamese veya triplet network mimarileri kullanır
  - Yüksek kaliteli cümle embedding'leri üretir
  - Cosine similarity doğrudan cümle benzerliğini ölçebilir

- **MPNet**:
  - Masked and permuted pre-training
  - BERT ve permuted language modeling avantajlarını birleştirir
  - Sentence embedding olarak üstün performans gösterir

**Domain-Specific ve Çok Dilli BERT Modelleri:**

- **mBERT (multilingual BERT)**:
  - 104 dilde eğitilmiş
  - Diller arası transfer öğrenme için uygun
  - Çok dilli embeddingler üretebilir

- **Domain-Specific BERTs**:
  - BioBERT: Biyomedikal metinler için optimize edilmiş
  - SciBERT: Bilimsel yayınlar için optimize edilmiş
  - FinBERT: Finansal metinler için optimize edilmiş
  - LegalBERT: Hukuki metinler için optimize edilmiş

**BERT'i Geliştiren Modeller:**

- **RoBERTa**: 
  - BERT mimarisini kullanan ancak daha büyük veri ve daha iyi eğitim stratejisiyle geliştirilmiş
  - Daha iyi performans ve daha güvenilir embeddingler sağlar

- **DeBERTa**:
  - Disentangled attention mekanizmaları ekler
  - İçerik ve pozisyonu ayrı kodlar
  - State-of-the-art performans sunar

**Model Seçimi için Kriterler:**

- **Task uygunluğu**: Semantik benzerlik için Sentence-BERT/SentenceTransformers
- **Dil desteği**: Çok dilli görevler için mBERT veya XLM-RoBERTa
- **Hız-kalite dengesi**: Sınırlı kaynaklarda DistilBERT veya MiniLM
- **Domain**: Özel alanlarda domain-specific modeller tercih edilir

### 2.2. Pooling Stratejileri

BERT token-seviyesinde embeddingler üretir, ancak cümle veya doküman seviyesinde temsilller için bu token embeddingler'ini tek bir vektöre dönüştürmek gerekir. Bu dönüşüm için çeşitli pooling (havuzlama) stratejileri kullanılır.

**Temel Pooling Stratejileri:**

1. **CLS Token Pooling**:
   - BERT'in her girdisinin başına eklenen [CLS] token'ının temsilini kullanır
   - Cümlenin tam temsilini içermesi için tasarlanmıştır
   - Avantajlar:
     - Basit ve doğrudan implementasyon
     - Modelin cümle temsili için özel olarak eğitilmiş bölümünü kullanır
   - Dezavantajlar:
     - Bazı görevlerde diğer pooling stratejilerine göre daha az etkili olabilir
     - Çok uzun metinlerde bilgi kaybı yaşanabilir

2. **Mean Pooling (Ortalama Havuzlama)**:
   - Tüm token temsillerinin ortalamasını alır (genellikle padding ve özel tokenlar hariç)
   - Avantajlar:
     - Tüm token'ların bilgisini kullanır
     - Genellikle cümle benzerliği için en iyi performans verir
     - Uzun metinlerde bile sabit boyutlu temsil sağlar
   - Dezavantajlar:
     - Uzun metinlerde önemli bilgiler dilüsyona uğrayabilir

3. **Max Pooling (Maksimum Havuzlama)**:
   - Her boyut için tüm token temsillerinin maksimum değerini alır
   - Avantajlar:
     - En belirgin özellikleri yakalar
     - Uzunluktan bağımsız temsil
   - Dezavantajlar:
     - Genel anlamsal yapıyı kaçırabilir
     - Mean pooling'den genellikle daha düşük performans gösterir

4. **Weighted Pooling (Ağırlıklı Havuzlama)**:
   - Token'lara farklı ağırlıklar atayarak ağırlıklı ortalama alır
   - Ağırlıklandırma çeşitleri:
     - IDF ağırlıklandırma: Daha nadir kelimelere daha yüksek ağırlık
     - Attention ağırlıklandırma: Self-attention skorlarını kullanma
     - Öğrenilebilir ağırlıklar: Görevle ilgili ince ayar yapma
   - Avantajlar:
     - Daha anlamlı kelimelere daha fazla önem verebilir
     - Cümlenin semantik içeriğini daha iyi yakalayabilir
   - Dezavantajlar:
     - Karmaşık implementasyon
     - Ek hesaplama gerektirebilir

5. **Concat Pooling (Birleştirme Havuzlaması)**:
   - Birden fazla havuzlama stratejisinin sonuçlarını birleştirir (örn. CLS + Mean + Max)
   - Avantajlar:
     - Farklı havuzlama stratejilerinin avantajlarını birleştirir
     - Daha zengin temsil
   - Dezavantajlar:
     - Daha yüksek boyutlu vektörler
     - Hesaplama maliyeti daha yüksek

**Kod Örneği - Farklı Pooling Stratejileri:**

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Tokenizer ve model yükleme
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Metni hazırlama ve tokenize etme
sentences = ["Bu bir örnek cümledir.", "Pooling stratejilerini görelim."]
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Model çıktısını elde etme
with torch.no_grad():
    model_output = model(**encoded_input)

# Son katman token temsillerini alma
token_embeddings = model_output.last_hidden_state  # [batch_size, sequence_length, hidden_size]
input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

# 1. CLS Token Pooling
cls_embeddings = token_embeddings[:, 0, :]
print("CLS Token Embedding boyutu:", cls_embeddings.shape)

# 2. Mean Pooling
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = input_mask_expanded.sum(1)
mean_embeddings = sum_embeddings / sum_mask
print("Mean Pooling Embedding boyutu:", mean_embeddings.shape)

# 3. Max Pooling - attention mask uygulayarak
# Önce maskelenmemiş token'lar için çok küçük bir değer (-1e9) atıyoruz
token_embeddings_masked = token_embeddings * input_mask_expanded + (1 - input_mask_expanded) * -1e9
max_embeddings = torch.max(token_embeddings_masked, 1)[0]
print("Max Pooling Embedding boyutu:", max_embeddings.shape)

# 4. Weighted Mean Pooling (basit bir örnek - son kelimeye daha fazla ağırlık)
weights = torch.linspace(0.1, 1.0, steps=token_embeddings.size(1)).unsqueeze(0).unsqueeze(-1)
weights = weights.expand_as(token_embeddings) * input_mask_expanded
weighted_sum = torch.sum(token_embeddings * weights, 1)
weighted_mean_embeddings = weighted_sum / weights.sum(1)
print("Weighted Mean Pooling Embedding boyutu:", weighted_mean_embeddings.shape)

# 5. Concat Pooling (CLS + Mean)
concat_embeddings = torch.cat([cls_embeddings, mean_embeddings], dim=-1)
print("Concat Pooling Embedding boyutu:", concat_embeddings.shape)

# Embeddinglari normalize et (cosine similarity için)
cls_embeddings_normalized = F.normalize(cls_embeddings, p=2, dim=1)
mean_embeddings_normalized = F.normalize(mean_embeddings, p=2, dim=1)
```

**Hangi Pooling Stratejisini Seçmeli:**

- **Semantik benzerlik için**: Genellikle Mean Pooling en iyi sonuçları verir
- **Sınıflandırma görevleri için**: CLS Token Pooling veya Mean Pooling
- **Bilgi çıkarımı için**: Weighted Pooling veya Max Pooling bilgi içeren kelimelere odaklanabilir
- **En iyi performans için**: Farklı stratejileri deneyin ve validasyon setinde karşılaştırın

### 2.3. SentenceTransformers Kütüphanesi

SentenceTransformers, anlamlı cümle embeddinglari üretmek için özel olarak tasarlanmış BERT-tabanlı modelleri içeren bir kütüphanedir. Standart BERT modellerinin aksine, SentenceTransformers modelleri doğrudan semantik benzerlik, bilgi çıkarımı ve arama gibi görevler için optimize edilmiştir.

**SentenceTransformers'ın Özellikleri:**

1. **Amaca Özel Eğitim**:
   - NLI (Natural Language Inference) ve STS (Semantic Textual Similarity) veri setleri üzerinde fine-tune edilmiş
   - Siamese ve triplet network mimarileri kullanarak eğitilmiş
   - Anlamlı cümle embeddinglari üretmek için optimize edilmiş

2. **Kullanım Kolaylığı**:
   - Basit bir API ile kolay kullanım
   - Doğrudan cümle embeddinglari üretebilme
   - Çeşitli benzerlik fonksiyonları entegre edilmiş

3. **Model Çeşitliliği**:
   - 30+ farklı dilde, 100+ özel model
   - Farklı model boyutları (küçük, orta, büyük)
   - Farklı görevler için optimize edilmiş modeller

4. **Performans**:
   - Standart BERT modellerine göre semantik benzerlik görevlerinde çok daha yüksek performans
   - Optimize edilmiş mimari ile daha hızlı çalışma

**SentenceTransformers ile Cümle Embeddingleri Oluşturma:**

```python
from sentence_transformers import SentenceTransformer

# Model yükleme
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Küçük ve hızlı bir model

# Tekli cümle için embedding
sentence = "Bu bir örnek cümledir."
embedding = model.encode(sentence)
print(f"Embedding boyutu: {embedding.shape}")  # (384,) - 384 boyutlu bir vektör

# Çoklu cümleler için embeddingler (batch işleme)
sentences = [
    "SentenceTransformers cümle embeddinglari için harikadır.",
    "Semantik arama için ideal bir çözümdür.",
    "BERT tabanlı modellerle çalışır."
]

# Embeddinglari hesaplama - varsayılan mean pooling kullanılır
embeddings = model.encode(sentences)
print(f"Batch embeddings boyutu: {embeddings.shape}")  # (3, 384) - 3 cümle, her biri 384 boyutlu
```

**Farklı SentenceTransformer Modelleri ve Kullanım Alanları:**

1. **Genel Amaçlı Modeller**:
   - `paraphrase-MiniLM-L6-v2`: Hızlı ve kompakt (384 boyut)
   - `all-mpnet-base-v2`: En iyi performans (768 boyut)
   - `all-MiniLM-L12-v2`: İyi performans/hız dengesi (384 boyut)

2. **Çoklu Dil Modelleri**:
   - `paraphrase-multilingual-MiniLM-L12-v2`: 50+ dil desteği
   - `distiluse-base-multilingual-cased-v1`: 15 dil için distilled model

3. **Özel Görev Modelleri**:
   - `msmarco-distilbert-base-v4`: Belge arama için optimize edilmiş
   - `multi-qa-MiniLM-L6-cos-v1`: Soru-cevap sistemleri için optimize edilmiş
   - `nli-distilroberta-base-v2`: Doğal dil çıkarımı için optimize edilmiş

**SentenceTransformers ile İleri Düzey Kullanım:**

```python
from sentence_transformers import SentenceTransformer, util
import torch

# Daha gelişmiş bir model yükleme
model = SentenceTransformer('all-mpnet-base-v2')  # Daha iyi performans için

# Özel encoding parametreleri
embeddings = model.encode(
    sentences,
    batch_size=32,           # Batch boyutu
    show_progress_bar=True,  # İlerleme çubuğu göster
    convert_to_tensor=True,  # PyTorch tensor olarak döndür
    normalize_embeddings=True  # L2 normalizasyon uygula (cosine similarity için)
)

# Embeddingler arasında cosine similarity hesaplama
cosine_scores = util.cos_sim(embeddings, embeddings)
print("Cosine similarity matrix:")
print(cosine_scores)

# En benzer cümle çiftlerini bulma
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j].item()})

# Benzerlik skoruna göre sırala
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

# En benzer çiftleri yazdır
for pair in pairs:
    i, j = pair['index']
    print(f"Benzerlik skoru: {pair['score']:.4f}")
    print(f"Cümle 1: {sentences[i]}")
    print(f"Cümle 2: {sentences[j]}\n")
```

**SentenceTransformers ile Özel Model Eğitimi:**

SentenceTransformers kütüphanesi, kendi özel veri setiniz üzerinde model eğitmeyi de destekler.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Eğitim örnekleri
train_examples = [
    InputExample(texts=['Bu bir elma.', 'Bu bir meyvedir.'], label=0.8),
    InputExample(texts=['Köpek havlıyor.', 'Kedi miyavlıyor.'], label=0.3),
    # Daha fazla örnek...
]

# Eğitim veri yükleyicisi (dataloader) oluşturma
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Eğitim için temel model seçme
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Loss fonksiyonu tanımlama (CosineSimilarityLoss - ile verilen etiketler 
# -1 ile 1 arasında olmalıdır, burada 0-1 arası kullanıldığından ölçeklendirme yaparız)
train_loss = losses.CosineSimilarityLoss(model)

# Modeli eğitme
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    evaluation_steps=1000,
    output_path='my-custom-model'
)
```

**SentenceTransformers'ın Avantajları:**

1. Semantik arama, benzerlik hesaplama ve bilgi çıkarımı için doğrudan kullanılabilir embeddingler
2. Standart BERT'e göre cümle temsilleri için çok daha iyi performans
3. Kullanımı kolay API
4. Çeşitli görevler ve diller için optimize edilmiş modeller
5. Hesaplama verimliliği - önceden hesaplanmış embeddingler yüksek hızlı benzerlik hesaplamasına olanak tanır

## 3. Metin Benzerlik Hesaplama

### 3.1. Benzerlik Metrikleri

İki metin parçasının semantik benzerliğini hesaplamak için çeşitli metrikler kullanılır. Bu metrikler, metin embeddinglari arasındaki ilişkiyi sayısal bir değere dönüştürür.

**Yaygın Benzerlik Metrikleri:**

1. **Cosine Similarity (Kosinüs Benzerliği)**:
   - İki vektör arasındaki açının kosinüsünü ölçer
   - -1 (tam zıt) ile 1 (tamamen aynı) arasında değer alır
   - Formül: $\text{cosine}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||}$
   - Avantajlar:
     - Vektör büyüklüğünden bağımsızdır (sadece yönü dikkate alır)
     - Yüksek boyutlu uzaylarda iyi çalışır
     - En yaygın kullanılan metriktir
   
   ```python
   def cosine_similarity(vec1, vec2):
       dot_product = np.dot(vec1, vec2)
       norm_vec1 = np.linalg.norm(vec1)
       norm_vec2 = np.linalg.norm(vec2)
       return dot_product / (norm_vec1 * norm_vec2)
   ```

2. **Euclidean Distance (Öklid Uzaklığı)**:
   - İki vektör arasındaki doğrudan mesafeyi ölçer
   - 0 (tamamen aynı) ile ∞ (sonsuz farklı) arasında değer alır
   - Formül: $\text{euclidean}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$
   - Avantajlar:
     - Sezgisel ve anlaşılması kolay
     - Fiziksel uzayda doğrudan anlamı var
   - Dezavantajlar:
     - Vektör büyüklüğüne duyarlıdır
     - Yüksek boyutlu uzaylarda "curse of dimensionality" sorunu

   ```python
   def euclidean_distance(vec1, vec2):
       return np.sqrt(np.sum((vec1 - vec2) ** 2))
   ```

3. **Manhattan Distance (Manhattan Uzaklığı)**:
   - İki vektör arasındaki "şehir blokları" mesafesini ölçer
   - Her boyuttaki farkların mutlak değerlerinin toplamı
   - Formül: $\text{manhattan}(A, B) = \sum_{i=1}^{n} |A_i - B_i|$
   - Avantajlar:
     - Gürültülü verilerde daha dayanıklı olabilir
     - Bazı uygulamalarda daha anlamlı sonuçlar verir
     - Hesaplaması daha hızlıdır

   ```python
   def manhattan_distance(vec1, vec2):
       return np.sum(np.abs(vec1 - vec2))
   ```

4. **Dot Product (Nokta Çarpımı)**:
   - İki vektör arasındaki çarpımların toplamı
   - Vektör büyüklüğüne duyarlıdır
   - Formül: $\text{dot}(A, B) = \sum_{i=1}^{n} A_i \cdot B_i$
   - Genellikle normalize edilmiş vektörlerle kullanılır (bu durumda cosine similarity'ye eşit olur)

   ```python
   def dot_product(vec1, vec2):
       return np.dot(vec1, vec2)
   ```

5. **Jaccard Similarity (Jaccard Benzerliği)**:
   - İki kümenin kesişiminin birleşime oranını ölçer
   - 0 (tamamen farklı) ile 1 (tamamen aynı) arasında değer alır
   - Formül: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
   - Genellikle sparse vektörler veya token setleri için kullanılır

   ```python
   def jaccard_similarity(set1, set2):
       intersection = len(set1.intersection(set2))
       union = len(set1.union(set2))
       return intersection / union if union != 0 else 0
   ```

**Metrik Seçimi Stratejileri:**

- **Cosine Similarity**: Genel amaçlı semantik benzerlik, özellikle normalize edilmiş vektörlerle kullanılır
- **Euclidean Distance**: Vektörlerin büyüklüğü de anlamlıysa
- **Manhattan Distance**: Gürültülü veri veya özellik vektörleriyle çalışırken
- **Dot Product**: Normalize edilmiş vektörlerle hızlı hesaplama için
- **Jaccard Similarity**: Token setleri, sparse vektörler veya binary özelliklerle çalışırken

**Kod Örneği - Farklı Benzerlik Metriklerinin Karşılaştırılması:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean, cityblock
import torch
import torch.nn.functional as F

# Model yükleme
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Test cümleleri
sentences = [
    "Yapay zeka günümüzde hızla gelişiyor.",
    "Makine öğrenmesi teknolojileri son yıllarda çok ilerledi.",
    "Kediler genellikle bağımsız hayvanlardır.",
    "Kedilerin bağımsız bir doğası vardır."
]

# Embeddinglari hesapla
embeddings = model.encode(sentences, convert_to_numpy=True)

# Farklı metrikleri karşılaştır
print("Cümle çiftleri ve benzerlik skorları:\n")

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        # Kosinüs benzerliği: 1 - cosine_distance (1'e yakın = benzer)
        cosine_sim = 1 - cosine(embeddings[i], embeddings[j])
        
        # Öklid mesafesi (0'a yakın = benzer)
        euc_distance = euclidean(embeddings[i], embeddings[j])
        
        # Manhattan mesafesi (0'a yakın = benzer)
        manhattan_dist = cityblock(embeddings[i], embeddings[j])
        
        # Normalizasyon için ortalama mesafe
        avg_length = (np.linalg.norm(embeddings[i]) + np.linalg.norm(embeddings[j])) / 2
        
        # Normalize edilmiş mesafeler (0 ile 1 arasında, 1 = benzer)
        norm_euc_sim = 1 / (1 + euc_distance / avg_length)
        norm_manhattan_sim = 1 / (1 + manhattan_dist / avg_length)
        
        print(f"Cümle 1: {sentences[i]}")
        print(f"Cümle 2: {sentences[j]}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"Normalized Euclidean Similarity: {norm_euc_sim:.4f}")
        print(f"Normalized Manhattan Similarity: {norm_manhattan_sim:.4f}")
        print("-" * 50)
```

### 3.2. Normalizasyon Teknikleri

Embedding vektörlerinin normalizasyonu, benzerlik hesaplamalarının doğruluğunu ve verimliliğini artırmak için kritik bir adımdır. Normalizasyon, vektörlerin farklı büyüklüklerinin etkisini azaltır ve daha tutarlı benzerlik skorları elde etmeyi sağlar.

**Normalizasyon Nedenleri:**

1. **Ölçek Etkilerini Azaltma**: Vektör büyüklüklerinden kaynaklanan farklılıkları ortadan kaldırır
2. **Cosine Similarity Hesaplamalarını Hızlandırma**: Normalize edilmiş vektörlerle cosine similarity sadece dot product hesabıyla bulunabilir
3. **Sayısal Stabilite**: Çok büyük veya çok küçük değerlerin oluşturduğu sayısal sorunları önler
4. **Model Performansını Artırma**: Normalize edilmiş vektörlerle genellikle daha iyi benzerlik skorları elde edilir

**Yaygın Normalizasyon Teknikleri:**

1. **L2 Normalizasyon (Birim Vektör Normalizasyonu)**:
   - Vektörü birim uzunluğa (1) ölçeklendirir
   - Formül: $v_{norm} = \frac{v}{||v||_2}$ , burada $||v||_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$
   - Bu, vektörün yönünü korur ancak büyüklüğünü 1'e eşitler
   - Özellikle cosine similarity için idealdir

   ```python
   def l2_normalize(vector):
       norm = np.linalg.norm(vector)
       return vector / norm if norm > 0 else vector
   ```

2. **L1 Normalizasyon**:
   - Vektör elemanlarının mutlak değerlerinin toplamına göre normalleştirme
   - Formül: $v_{norm} = \frac{v}{||v||_1}$ , burada $||v||_1 = \sum_{i=1}^{n} |v_i|$
   - Manhattan distance ile çalışırken yararlı olabilir

   ```python
   def l1_normalize(vector):
       norm = np.sum(np.abs(vector))
       return vector / norm if norm > 0 else vector
   ```

3. **Min-Max Normalizasyon**:
   - Vektörü belirli bir aralığa (genellikle [0,1]) ölçeklendirir
   - Formül: $v_{norm} = \frac{v - \min(v)}{\max(v) - \min(v)}$
   - Her özelliği aynı ölçeğe getirir
   - Outlier'lara karşı duyarlıdır

   ```python
   def min_max_normalize(vector):
       min_val = np.min(vector)
       max_val = np.max(vector)
       range_val = max_val - min_val
       return (vector - min_val) / range_val if range_val > 0 else vector
   ```

4. **Z-Score Normalizasyon**:
   - Vektörü ortalama 0, standart sapma 1 olacak şekilde dönüştürür
   - Formül: $v_{norm} = \frac{v - \mu}{\sigma}$
   - Normal dağılımlı verilerde iyi çalışır
   - Outlier'ları korur ancak etkilerini azaltır

   ```python
   def z_score_normalize(vector):
       mean = np.mean(vector)
       std = np.std(vector)
       return (vector - mean) / std if std > 0 else vector
   ```

**Farklı Kütüphanelerle Normalizasyon Teknikleri:**

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sk_normalize

# Örnek vektör
vector = np.array([0.2, 0.5, -0.3, 1.2, 0.8])

# NumPy ile L2 normalizasyon
np_l2_norm = vector / np.linalg.norm(vector)
print("NumPy L2 norm:", np_l2_norm)

# PyTorch ile L2 normalizasyon
pt_vector = torch.tensor(vector, dtype=torch.float32)
pt_l2_norm = F.normalize(pt_vector, p=2, dim=0)
print("PyTorch L2 norm:", pt_l2_norm.numpy())

# Scikit-learn ile normalizasyon
sk_l2_norm = sk_normalize(vector.reshape(1, -1), norm='l2')[0]
sk_l1_norm = sk_normalize(vector.reshape(1, -1), norm='l1')[0]
print("Scikit-learn L2 norm:", sk_l2_norm)
print("Scikit-learn L1 norm:", sk_l1_norm)

# SentenceTransformers ile normalizasyon
from sentence_transformers import util
st_vector = torch.tensor(vector, dtype=torch.float32).reshape(1, -1)
st_l2_norm = util.normalize_embeddings(st_vector)
print("SentenceTransformers norm:", st_l2_norm.numpy()[0])
```

**SentenceTransformers ile Embedding Normalizasyon:**

```python
from sentence_transformers import SentenceTransformer, util

# Model yükleme
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Cümleler
sentences = [
    "Bu bir örnek cümledir.",
    "Bu da başka bir örnek."
]

# Encoding sırasında otomatik normalizasyon
embeddings_normalized = model.encode(
    sentences, 
    normalize_embeddings=True  # L2 normalizasyon uygular
)

# Manuel normalizasyon
embeddings_raw = model.encode(sentences, normalize_embeddings=False)
embeddings_manual_norm = util.normalize_embeddings(torch.tensor(embeddings_raw))

# L2 normların kontrol edilmesi
l2_norms_auto = np.linalg.norm(embeddings_normalized, axis=1)
l2_norms_manual = np.linalg.norm(embeddings_manual_norm, axis=1)

print("Otomatik normalize edilen embeddinglarin L2 normları:", l2_norms_auto)
print("Manuel normalize edilen embeddinglarin L2 normları:", l2_norms_manual)
```

**Normalizasyon Seçim Kriterleri:**

- **Cosine Similarity için**: L2 normalizasyon (en yaygın kullanım)
- **Manhattan Distance için**: L1 normalizasyon
- **Farklı ölçeklerdeki veri için**: Min-Max veya Z-score normalizasyon
- **Sayısal stabilitede sorun yaşıyorsanız**: Min-Max normalizasyon
- **Performans öncelikli ise**: Cosine similarity + L2 normalizasyon kombinasyonu

### 3.3. Semantik Arama ve Retrieval

Semantik arama, metin embeddinglari kullanarak anlamsal olarak ilgili içeriği bulma işlemidir. Geleneksel anahtar kelime tabanlı aramanın aksine, semantik arama kelimelerin anlamsal benzerliğini dikkate alır ve böylece daha alakalı sonuçlar elde edilebilir.

**Semantik Aramanın Avantajları:**

1. **Eşanlamlıları (Synonyms) Anlama**: "Otomobil", "araba", "araç" gibi eşanlamlı kelimeleri anlayabilir
2. **Bağlamsal Anlama**: Kelimelerin bağlamını dikkate alır
3. **Kavram Bazlı Arama**: Tam kelime eşleşmesi yerine kavramsal eşleşmeye odaklanır
4. **Daha İyi Kullanıcı Deneyimi**: Kullanıcının sorgu niyetini daha iyi anlar

**Semantik Arama Süreci:**

1. **Ön İşlem (Offline)**: 
   - Corpus'taki tüm belgelerin embedding'lerini hesapla
   - Embedding'leri verimli arama için bir indekse kaydet

2. **Arama (Online)**:
   - Kullanıcı sorgusunun embedding'ini hesapla
   - Embedding'i belge embedding'leriyle karşılaştır (benzerlik hesapla)
   - En benzer belgeleri benzerlik skorlarına göre sırala ve döndür

**Basit Semantik Arama İmplementasyonu:**

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Model yükleme
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Corpus hazırlama - aranacak dokümanlar
corpus = [
    "Yapay zeka, insan zekasını taklit eden sistemlerdir.",
    "Derin öğrenme, yapay zeka alanında bir alt daldır.",
    "BERT, doğal dil işlemede devrim yaratan bir modeldir.",
    "Transformers mimarisi, dikkat mekanizmalarına dayanır.",
    "Python, yapay zeka uygulamaları için popüler bir programlama dilidir.",
    "Makine öğrenmesi, verilerden örüntüler öğrenir.",
    "NLP, bilgisayarların insan dilini anlamasını sağlar.",
    "GPT modelleri, metin üretimi için yaygın olarak kullanılır.",
    "Embedding vektörleri, kelimeleri veya cümleleri sayısal olarak temsil eder.",
    "Semantik arama, anlamsal benzerliğe dayalı bir arama yöntemidir."
]

# Corpus embeddingler hesaplama - ön işlem
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Kullanıcı sorguları
queries = [
    "Yapay zeka nedir?",
    "BERT ve transformer modelleri hakkında bilgi",
    "Vektör tabanlı metin temsilleri nasıl çalışır?",
    "Python ile makine öğrenmesi"
]

# Her sorgu için semantik arama
top_k = 3  # Her sorgu için kaç sonuç gösterilecek

for query in queries:
    # Sorgu embedding'i hesaplama
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Corpus ile cosine-similarity hesaplama
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # En yüksek benzerliğe sahip belgeleri alma
    top_results = torch.topk(cos_scores, k=top_k)
    
    print(f"\nSorgu: {query}")
    print(f"En alakalı {top_k} sonuç:")
    
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Skor: {score:.4f} - {corpus[idx]}")
```

**Büyük Ölçekli Semantik Arama için Optimizasyon:**

Büyük veri kümeleri için, brute-force benzerlik hesaplaması yerine verimli yaklaşık en yakın komşu (Approximate Nearest Neighbor - ANN) algoritmaları kullanılır.

```python
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Model yükleme
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Daha büyük bir corpus oluşturma (örnek olarak)
corpus = [
    "Yapay zeka, insan zekasını taklit eden sistemlerdir." + str(i) 
    for i in range(10000)  # 10,000 belge
]

# Corpus embeddingler hesaplama
corpus_embeddings = model.encode(corpus, show_progress_bar=True)
embedding_size = corpus_embeddings.shape[1]  # Embedding boyutu

# Embeddinglari normalize etme (opsiyonel, inner product araması için)
faiss.normalize_L2(corpus_embeddings)

# FAISS indeksi oluşturma
index = faiss.IndexFlatIP(embedding_size)  # IP = Inner Product (cosine sim için)
# Alternatif: İvmelendirme için
# index = faiss.IndexIVFFlat(quantizer, embedding_size, 100, faiss.METRIC_INNER_PRODUCT)
# index.train(corpus_embeddings)

# Vektörleri indekse ekleme
index.add(corpus_embeddings)

# Sorgu (arama) yapma
query = "Yapay zeka nedir?"
query_embedding = model.encode([query])[0]
faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize et

# En benzer k belgeyi bulma
k = 5
distances, indices = index.search(query_embedding.reshape(1, -1), k)

# Sonuçları gösterme
print(f"Sorgu: {query}")
for i in range(k):
    print(f"Skor: {distances[0][i]:.4f} - {corpus[indices[0][i]][:50]}...")
```

**Diğer Gelişmiş Semantik Arama Teknikleri:**

1. **Hibrit Arama (BM25 + Semantic Search)**:
   - Lexical arama ve semantik aramanın avantajlarını birleştirir
   - Daha yüksek doğruluk ve kapsama sağlar

   ```python
   from sentence_transformers import SentenceTransformer, util
   from rank_bm25 import BM25Okapi
   import numpy as np
   
   # Model yükleme
   model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
   
   # Corpus hazırlama
   corpus = [
       "Yapay zeka, insan zekasını taklit eden sistemlerdir.",
       "Derin öğrenme, yapay zeka alanında bir alt daldır.",
       # ...diğer belgeler
   ]
   
   # BM25 için tokenize corpus
   tokenized_corpus = [doc.lower().split() for doc in corpus]
   bm25 = BM25Okapi(tokenized_corpus)
   
   # Semantik arama için embeddingler
   corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
   
   # Hibrit arama fonksiyonu
   def hybrid_search(query, top_k=3, alpha=0.5):
       # BM25 skoru hesaplama
       tokenized_query = query.lower().split()
       bm25_scores = bm25.get_scores(tokenized_query)
       
       # Min-max normalizasyon
       bm25_scores = (bm25_scores - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-6)
       
       # Semantik skorlar hesaplama
       query_embedding = model.encode(query, convert_to_tensor=True)
       semantic_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
       
       # Hibrit skor hesaplama
       hybrid_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
       
       # En yüksek skorlu belgeleri alma
       top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
       
       return [(hybrid_scores[idx], corpus[idx]) for idx in top_indices]
   ```

2. **Dense Passage Retrieval (DPR)**:
   - Sorgu ve belge için ayrı encoder modelleri kullanır
   - Soru-cevap sistemleri için optimize edilmiştir

3. **Cross-Encoder Reranking**:
   - İlk aşamada Bi-encoder ile geniş bir aday seti bulunur
   - İkinci aşamada Cross-encoder ile adaylar yeniden sıralanır
   - Daha yüksek doğruluk ancak daha yavaş

   ```python
   from sentence_transformers import SentenceTransformer, CrossEncoder, util
   
   # Bi-encoder (ilk aşama retrieval için)
   bi_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
   
   # Cross-encoder (ikinci aşama reranking için)
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   
   # Corpus embeddingler
   corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True)
   
   # İki aşamalı arama fonksiyonu
   def two_stage_search(query, top_k_retrieval=100, top_k_rerank=10):
       # 1. Aşama: Bi-encoder ile geniş aday seti bulma
       query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
       hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k_retrieval)[0]
       
       # Adayları hazırlama
       candidates = [(hit['corpus_id'], corpus[hit['corpus_id']]) for hit in hits]
       
       # 2. Aşama: Cross-encoder ile yeniden sıralama
       cross_inp = [[query, cand[1]] for cand in candidates]
       cross_scores = cross_encoder.predict(cross_inp)
       
       # Sonuçları sıralama
       for idx in range(len(cross_scores)):
           candidates[idx] = (candidates[idx][0], candidates[idx][1], cross_scores[idx])
       
       candidates.sort(key=lambda x: x[2], reverse=True)
       return candidates[:top_k_rerank]
   ```

**Pratik Semantik Arama Uygulamaları:**

1. **Doküman Arşivi Arama**: Büyük belge koleksiyonlarında semantik arama
2. **Soru-Cevap Sistemleri**: Kullanıcı sorularına uygun cevapları bulma
3. **Öneri Sistemleri**: Kullanıcı ilgisine göre semantik olarak benzer içerik önerme
4. **Müşteri Desteği**: FAQ'lardan semantik olarak en ilgili cevapları bulma
5. **İçerik Filtreleme**: İstenmeyen içeriği semantik olarak tespit etme

