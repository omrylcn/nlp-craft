# NLI ve STS: Derin Teorik ve Pratik Analiz

## 1. Giriş: Doğal Dil Anlama Paradigmaları

Doğal dil işlemenin (NLP) özünde, makinelerin metinleri insanlar gibi "anlama" yeteneği bulunur. Bu "anlama" yeteneği, çok boyutlu ve katmanlı bir kavramdır. Bu noktada iki temel paradigma karşımıza çıkar: **çıkarımsal anlama** ve **benzerlik tabanlı anlama**. İşte NLI ve STS görevleri, bu iki temel anlama biçiminin somutlaştırılmış halleridir.

### 1.1 Anlama Paradigmalarının Bilişsel Temelleri

İnsan anlama süreçleri incelendiğinde iki temel mekanizma gözlenir:

1. **Çıkarımsal Mekanizmalar**: Okunan/duyulan metinden yeni bilgilerin çıkarılması, önermeler arası mantıksal ilişkilerin kurulması
2. **Benzerlik Mekanizmaları**: Kavramlar ve önermeler arasında anlamsal yakınlığın/uzaklığın algılanması

NLI ve STS, bu iki temel bilişsel mekanizmayı doğrudan modellemek için tasarlanmıştır. Bu görevler, basit görünseler de dil anlama sürecinin en temel bileşenlerini kapsamaktadır.

## 2. Teorik Temeller: Anlamın Matematiksel Çerçevesi

### 2.1 Anlamın Vektör Uzayı Hipotezi

NLI ve STS'nin teorik temellerinde "distributional semantics" (dağılımsal anlambilim) prensipleri yatar:

```
"Bir kelimenin anlamı, bulunduğu bağlamlar topluluğudur." - J.R. Firth, 1957
```

Bu prensip, modern vektör uzayı anlamsal modellerinin temelini oluşturur. Kelimeler ve cümleler, çok boyutlu bir semantik uzayda noktalar olarak temsil edilir. Bu uzayda:

- **Yakınlık**: Anlamsal benzerliği gösterir (STS'nin temeli)
- **Yönsel İlişkiler**: Anlam hiyerarşilerini ve çıkarımsal ilişkileri temsil eder (NLI'nin temeli)

### 2.2 Biçimsel Anlamsal Teoriler ve NLP Görevleri

NLI ve STS, biçimsel dilbilim teorilerinden de etkilenmiştir:

- **Mantıksal Form Teorisi**: NLI, önerme mantığı ve birinci derece mantık formülasyonlarıyla doğrudan ilişkilidir
- **Vektör Uzayı Anlambilimi**: STS, kelimelerin ve cümlelerin vektör reprezentasyonlarıyla doğrudan ilişkilidir
- **Olası Dünyalar Semantiği**: NLI, önermeler arası entailment (çıkarım) ilişkilerini modellemek için kullanılan bu teoriyi pratik bir görev formuna dönüştürür

## 3. Natural Language Inference (NLI): Derinlemesine Analiz

### 3.1 Teorik Çerçeve: Çıkarımsal Anlama

NLI, iki metin parçası arasındaki mantıksal ilişkiyi belirleyen bir görevdir. Tipik olarak bir "öncül" (premise) ve bir "hipotez" (hypothesis) verilir, ve model şu üç sınıftan birini tahmin etmelidir:

- **Entailment (Çıkarım)**: Öncül, hipotezi mantıksal olarak içerir/doğrular
- **Contradiction (Çelişki)**: Öncül, hipotezin yanlış olduğunu gösterir
- **Neutral (Nötr)**: Öncül, hipotez hakkında kesin bir çıkarım yapmaya yeterli değildir

NLI görevi, Montague semantiği ve biçimsel mantık teorilerinden esinlenen, ancak pratik NLP uygulamaları için operasyonel hale getirilmiş bir görevdir.

### 3.2 NLI'nin Modele Kazandırdıkları: Çıkarımsal Yetkinlikler

NLI eğitimi, bir dil modeline şu temel yetenekleri kazandırır:

1. **Mantıksal İlişkileri Anlama**: Metinler arası çıkarım yapabilme
2. **Anlam Belirsizliğini Çözme**: Bağlama dayalı anlam çözümleme
3. **Dünya Bilgisi Entegrasyonu**: İfadeler ve gerçek dünya arasındaki ilişkileri kavrama
4. **Sözdizimsel Dönüşümleri Anlama**: Aynı anlamı farklı sözdizimsel yapılarla ifade edildiğinde tanıma
5. **Dilbilimsel Önvarsayımları Tespit**: Bir ifadenin ima ettiği örtük bilgileri çıkarma

#### 3.2.1 Anlambilimsel Çerçeveler Açısından NLI'nin Önemi

```
"NLI, anlambilimsel temsillerin operasyonel bir tanımını sunar. Bir cümlenin anlambilimsel temsili, o cümlenin hangi diğer cümleleri doğruladığı, hangileriyle çeliştiği ve hangileriyle nötr olduğu bilgisidir." - Samuel R. Bowman
```

Bu perspektiften bakıldığında, NLI eğitimi bir modele gerçek anlamda "anlama" yeteneği kazandırmaya en yakın eğitim paradigmalarından biridir.

### 3.3 NLI Veri Kümeleri ve Zorlu Örnekler

Başlıca NLI veri kümeleri:

1. **SNLI**: Stanford Natural Language Inference, 570,000 insan tarafından etiketlenmiş örnek
2. **MultiNLI**: Multi-Genre NLI, birçok farklı metin türünden (konuşma dili, fikir yazıları, hükümet raporları) örnekler içerir
3. **XNLI**: Çok dilli NLI, 15 farklı dilde test setleri
4. **ANLI**: Adversarial NLI, modelleri yanıltacak şekilde tasarlanmış zorlayıcı örnekler

**Zorlu NLI Örnekleri ve Bilişsel Zorlukları:**

```
Öncül: "Anna suyu kaynatmak için ateşi açtı."
Hipotez: "Su kaynamaya başladı."
Etiket: Neutral (Nötr)
Zorluk: Zamansal çıkarım, nedensellik ve süreç anlama
```

```
Öncül: "Köpek, kedinin yanından koştu."
Hipotez: "Kedi, köpeğin yanından koştu."
Etiket: Contradiction (Çelişki)
Zorluk: Semantik rol değişimi, bakış açısı değişimi
```

Bu örnekler, NLI'nin basit bir metin karşılaştırma görevinden çok daha fazlası olduğunu ve insan düzeyinde anlama gerektirdiğini gösterir.

### 3.4 NLI Eğitimi: Teknik Detaylar ve Kod Örnekleri

#### 3.4.1 Transformers ile NLI Modeli Eğitimi

```python
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW

class NLIModel(nn.Module):
    def __init__(self, pretrained_model, num_labels=3):
        super(NLIModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        
        # Öncül ve hipotez embeddingler birleştirildiğinde oluşan boyut
        hidden_size = self.encoder.config.hidden_size
        combined_size = hidden_size * 3  # [u, v, |u-v|] birleştirmesi
        
        # Sınıflandırma katmanları
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids_premise, attention_mask_premise, 
                input_ids_hypothesis, attention_mask_hypothesis):
        # Öncül cümleyi kodla
        outputs_premise = self.encoder(
            input_ids=input_ids_premise,
            attention_mask=attention_mask_premise,
            return_dict=True
        )
        
        # Hipotez cümleyi kodla
        outputs_hypothesis = self.encoder(
            input_ids=input_ids_hypothesis,
            attention_mask=attention_mask_hypothesis,
            return_dict=True
        )
        
        # [CLS] token embeddingi al (alternatif olarak mean pooling kullanılabilir)
        premise_embedding = outputs_premise.last_hidden_state[:, 0, :]
        hypothesis_embedding = outputs_hypothesis.last_hidden_state[:, 0, :]
        
        # [u, v, |u-v|] birleştirmesi
        abs_diff = torch.abs(premise_embedding - hypothesis_embedding)
        combined = torch.cat([
            premise_embedding, 
            hypothesis_embedding, 
            abs_diff
        ], dim=1)
        
        # Dropout uygula
        combined = self.dropout(combined)
        
        # Sınıflandırma
        logits = self.classifier(combined)
        
        return logits
```

#### 3.4.2 NLI için Custom DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, idx):
        premise = str(self.premises[idx])
        hypothesis = str(self.hypotheses[idx])
        label = self.labels[idx]
        
        # Öncül tokenizasyonu
        premise_encoding = self.tokenizer(
            premise,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Hipotez tokenizasyonu
        hypothesis_encoding = self.tokenizer(
            hypothesis,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_premise': premise_encoding['input_ids'].squeeze(),
            'attention_mask_premise': premise_encoding['attention_mask'].squeeze(),
            'input_ids_hypothesis': hypothesis_encoding['input_ids'].squeeze(),
            'attention_mask_hypothesis': hypothesis_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

#### 3.4.3 Eğitim Döngüsü ve Loss Fonksiyonu

```python
def train_nli_model(model, train_dataloader, val_dataloader, device, num_epochs=3):
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Loss fonksiyonu (cross-entropy)
    criterion = nn.CrossEntropyLoss()
    
    # Eğitim döngüsü
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_dataloader:
            # Batch'i cihaza taşı
            input_ids_premise = batch['input_ids_premise'].to(device)
            attention_mask_premise = batch['attention_mask_premise'].to(device)
            input_ids_hypothesis = batch['input_ids_hypothesis'].to(device)
            attention_mask_hypothesis = batch['attention_mask_hypothesis'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(
                input_ids_premise, attention_mask_premise,
                input_ids_hypothesis, attention_mask_hypothesis
            )
            
            # Loss hesapla
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Ortalama eğitim kaybını hesapla
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validasyon
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Batch'i cihaza taşı
                input_ids_premise = batch['input_ids_premise'].to(device)
                attention_mask_premise = batch['attention_mask_premise'].to(device)
                input_ids_hypothesis = batch['input_ids_hypothesis'].to(device)
                attention_mask_hypothesis = batch['attention_mask_hypothesis'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(
                    input_ids_premise, attention_mask_premise,
                    input_ids_hypothesis, attention_mask_hypothesis
                )
                
                # Loss hesapla
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Doğruluk hesapla
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Ortalama validasyon kaybını ve doğruluğu hesapla
        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {accuracy:.4f}')
    
    return model
```

### 3.5 İleri Düzey NLI Kavramları ve Araştırma Yönelimleri

#### 3.5.1 Çıkarımsal Derinlik ve Zincir Çıkarımlar

Tek adımlı çıkarımlar yerine, çoklu öncül ve zincirleme çıkarım gerektiren kompleks NLI:

```
Öncül 1: "Tüm memeliler oksijen solur."
Öncül 2: "Tüm köpekler memelidir."
Hipotez: "Köpekler oksijen solur."
```

Bu tür "çok adımlı çıkarım" yeteneği, derin anlama için kritiktir.

#### 3.5.2 Monotonluk ve Doğal Dilde Çıkarım Kalıpları

Biçimsel mantıkta monotonluk, çıkarım kalıplarının geçerliliğini belirler:

```
Örnek (Yukarı Monotonluk):
"Kırmızı arabalar gördüm" → "Arabalar gördüm" (geçerli)

Örnek (Aşağı Monotonluk): 
"Hiç kırmızı araba görmedim" → "Hiç Ferrari görmedim" (geçersiz)
```

Bu tür dilbilimsel monotonluk kalıplarını öğrenmek, NLI modellerinin çıkarımsal yeteneklerini derinleştirir.

#### 3.5.3 Göreceli Entailment ve Dünya Bilgisi

NLI, giderek daha fazla dünya bilgisi gerektiren örneklere doğru evrilmektedir:

```
Öncül: "New York'tan Los Angeles'a uçtum."
Hipotez: "Dört saatten fazla yolculuk yaptım."
```

Bu çıkarım, dünya bilgisi (şehirlerarası mesafeler) olmadan yapılamaz.

## 4. Semantic Textual Similarity (STS): Derinlemesine Analiz

### 4.1 Teorik Çerçeve: Benzerlik Tabanlı Anlama

STS, iki metin arasındaki anlamsal benzerliği sürekli bir ölçekte değerlendiren bir görevdir. Tipik olarak:
- 0: Tamamen farklı anlamlar
- 5: Tamamen aynı anlam

STS, bilişsel psikolojideki "anlamsal uzay" teorileri ve vektör uzayı semantik modellerine dayanır. İki metin arasındaki benzerliği ölçmek, çıkarımsal ilişkileri belirlemekten daha temek bir görevi temsil eder.

### 4.2 STS'nin Modele Kazandırdıkları: Anlamsal Uzay Yapılandırması

STS eğitimi, bir dil modeline şu temel yetenekleri kazandırır:

1. **Semantik Uzayı Kalibre Etme**: Anlamsal yakınlık ve uzaklık ilişkilerini düzenleme
2. **Parafraz Anlama**: Aynı anlamı farklı sözcüklerle ifade etme yeteneği
3. **Derece Farkı Algılama**: İnce anlamsal farklılıkları ayırt etme
4. **Bağlamsal Benzerlik**: Kelime düzeyinden cümle düzeyine anlamsal benzerliği genişletme
5. **Çok Boyutlu Anlam Temsili**: Anlamın farklı boyutlarını (konu, üslup, duygu vs.) ayırt edebilme

#### 4.2.1 Vektör Uzayı Modelleri Açısından STS'nin Önemi

```
"STS eğitimi, vektör uzayında benzer anlamlı metinlerin yakın, farklı anlamlı metinlerin uzak konumlandığı bir 'anlamsal yerçekimi' oluşturur." - Eneko Agirre
```

Bu semantik yerçekimi, tüm diğer NLP görevleri için güçlü bir temel oluşturur.

### 4.3 STS Veri Kümeleri ve Değerlendirme Zorlukları

Başlıca STS veri kümeleri:

1. **STS Benchmark**: Çeşitli kaynaklardan derlenen 8,628 cümle çifti
2. **SICK**: Sentences Involving Compositional Knowledge, 10,000 cümle çifti
3. **SemEval STS**: SemEval 2012-2017 yarışmalarındaki veri setleri
4. **BIOSSES**: Biyomedikal alana özel STS veri seti

**Zorlu STS Örnekleri ve Nüansları:**

```
Cümle 1: "Çocuk sokakta oyun oynuyor."
Cümle 2: "Genç bir erkek dışarıda eğleniyor."
İnsan Skoru: 4.2/5.0
Zorluk: Kısmi sinonimleri anlamak (çocuk/genç erkek, sokak/dışarı, oyun/eğlence)
```

```
Cümle 1: "Film eleştirmenleri tarafından övüldü."
Cümle 2: "Film eleştirmenlerinin övgüsünü aldı."
İnsan Skoru: 5.0/5.0
Zorluk: Sözdizimsel farklılık, semantik eşdeğerlik
```

Bu örnekler, STS'nin nüanslı anlam eşleştirmelerinin değerlendirilmesini gerektirdiğini gösterir.

### 4.4 STS Eğitimi: Teknik Detaylar ve Kod Örnekleri

#### 4.4.1 STS için Sentence Transformer Eğitimi

```python
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader

# Sentence Transformer modeli oluştur
def create_sts_model(model_name):
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

# STS veri setini hazırla
def prepare_sts_dataset(sentences1, sentences2, scores, batch_size=16):
    # Skorları 0-1 aralığına normalize et (orijinal veriler genellikle 0-5 aralığındadır)
    normalized_scores = [score / 5.0 for score in scores]
    
    # InputExample nesneleri oluştur
    examples = []
    for sent1, sent2, score in zip(sentences1, sentences2, normalized_scores):
        examples.append(InputExample(texts=[sent1, sent2], label=score))
    
    # DataLoader oluştur
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    return dataloader

# STS modeli eğitme fonksiyonu
def train_sts_model(model, train_dataloader, evaluator=None, epochs=4):
    # CosineSimilarityLoss kullan
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Modeli eğit
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=1000,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    return model
```

#### 4.4.2 STS için Custom Loss Fonksiyonları

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class STSLoss(nn.Module):
    def __init__(self, loss_type="mse"):
        super(STSLoss, self).__init__()
        self.loss_type = loss_type
        
    def forward(self, embeddings1, embeddings2, labels):
        # Kosinüs benzerliği hesapla (-1 ile 1 arasında)
        cos_sim = F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        # Benzerliği 0-1 aralığına dönüştür
        cos_sim = (cos_sim + 1) / 2
        
        if self.loss_type == "mse":
            # Mean Squared Error
            return F.mse_loss(cos_sim, labels)
        
        elif self.loss_type == "contrastive":
            # Contrastive Loss
            # labels 1.0 = benzer, 0.0 = benzer değil olarak kabul edilir
            margin = 0.5
            similar_loss = labels * torch.pow(1.0 - cos_sim, 2)
            dissimilar_loss = (1.0 - labels) * torch.pow(torch.clamp(cos_sim - margin, min=0.0), 2)
            return torch.mean(similar_loss + dissimilar_loss)
        
        elif self.loss_type == "pearson":
            # Pearson korelasyonu optimize eden loss
            vx = cos_sim - torch.mean(cos_sim)
            vy = labels - torch.mean(labels)
            
            pearson_r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
            return 1.0 - pearson_r  # Maksimize etmek için minimize edilecek loss
```

#### 4.4.3 STS için Gelişmiş Değerlendirme Metrikleri

```python
from scipy.stats import pearsonr, spearmanr

def evaluate_sts_model(model, sentences1, sentences2, gold_scores):
    # Model tahminlerini al
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    
    # Kosinüs benzerliği hesapla
    cos_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
    predicted_scores = cos_sim.cpu().numpy().diagonal()
    
    # 0-1 aralığından 0-5 aralığına dönüştür
    predicted_scores = predicted_scores * 5.0
    
    # Korelasyon metrikleri hesapla
    pearson_correlation, _ = pearsonr(gold_scores, predicted_scores)
    spearman_correlation, _ = spearmanr(gold_scores, predicted_scores)
    
    # Mean Squared Error hesapla
    mse = ((gold_scores - predicted_scores) ** 2).mean()
    
    return {
        'pearson': pearson_correlation,
        'spearman': spearman_correlation,
        'mse': mse
    }
```

### 4.5 İleri Düzey STS Kavramları ve Araştırma Yönelimleri

#### 4.5.1 Anlamsal Benzerliğin Çok Boyutluluğu

Modern STS araştırmaları, benzerliğin tek boyutlu bir ölçek olmadığını, çeşitli boyutlarda değerlendirilebileceğini göstermiştir:

- **Tematik Benzerlik**: Aynı konudan bahsediyor mu?
- **Pragmatik Benzerlik**: Aynı amaca mı hizmet ediyor?
- **Üslupsal Benzerlik**: Benzer bir dil tonu ve stili mi kullanıyor?
- **Yapısal Benzerlik**: Benzer sözdizimsel yapıları mı kullanıyor?

#### 4.5.2 Asimetrik Benzerlik ve Yönlülük

Geleneksel STS, simetrik bir benzerlik varsayımı yapar, ancak gerçek dünyada benzerlik çoğu zaman asimetriktir:

```
A: "Kedi bir memeli hayvandır."
B: "Memeliler hayvanların bir alt sınıfıdır."

sim(A→B) ≠ sim(B→A)
```

Bu asimetriyi modelleyen STS yaklaşımları giderek önem kazanmaktadır.

#### 4.5.3 Çoklu Karar Benzerliği ve Konfigürasyonel STS

STS araştırmalarının yeni yönelimlerinden biri, benzerlikleri yapılandırmacı bir perspektiften değerlendirmektir:

```
"Benzerlik, tipik olarak iki nesnenin iç özelliklerinin bir fonksiyonu olarak değil, bir gözlemcinin belirli bir bağlamdaki konfigürasyonel değerlendirmesi olarak ortaya çıkar." - Tversky & Gati, 1978
```

Bu yönelim, bağlama duyarlı ve adaptif STS sistemlerine doğru ilerlemektedir.

## 5. NLI ve STS Arasındaki Simbiyotik İlişki

### 5.1 Tamamlayıcı Bilgi Yapıları

NLI ve STS görevleri, aslında anlamsal uzayın tamamlayıcı yapılarını modellerler:

- **NLI**: Anlamsal uzaydaki hiyerarşik ve mantıksal ilişkileri modeller
- **STS**: Anlamsal uzaydaki yakınlık ve uzaklık ilişkilerini modeller

Bu iki görev birlikte kullanıldığında, çok daha zengin bir anlamsal temsil oluşturulabilir.

### 5.2 İki Aşamalı Eğitim Paradigması

Son yıllardaki en başarılı yaklaşımlardan biri, modelleri önce NLI üzerinde, sonra STS üzerinde eğitmektir:

```python
# İki aşamalı eğitim örneği
def two_stage_training(base_model_name):
    # 1. Aşama: NLI eğitimi
    model = create_sts_model(base_model_name)
    nli_dataloader = prepare_nli_dataset(nli_premises, nli_hypotheses, nli_labels)
    
    # NLI için softmax loss kullan
    nli_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3  # entailment, contradiction, neutral
    )
    
    # NLI üzerinde eğit
    model.fit(
        train_objectives=[(nli_dataloader, nli_loss)],
        epochs=1,
        show_progress_bar=True
    )
    
    # 2. Aşama: STS fine-tuning
    sts_dataloader = prepare_sts_dataset(sts_sentences1, sts_sentences2, sts_scores)
    sts_loss = losses.CosineSimilarityLoss(model)
    
    # STS üzerinde fine-tuning
    model.fit(
        train_objectives=[(sts_dataloader, sts_loss)],
        epochs=4,
        show_progress_bar=True
    )
    
    return model
```

Bu yaklaşım, modelin önce anlamsal uzaydaki mantıksal ilişkileri (NLI), ardından metrik benzerlikleri (STS) öğrenmesini sağlar.

### 5.3 Çok Görevli Öğrenme ve Ortak Optimizasyon

İleri düzey bir yaklaşım, NLI ve STS görevlerini aynı anda optimize etmektir:

```python
def multitask_nli_sts_training(base_model_name):
    model = create_sts_model(base_model_name)
    
    # NLI ve STS veri yükleyicileri
    nli_dataloader = prepare_nli_dataset(nli_premises, nli_hypotheses, nli_labels)
    sts_dataloader = prepare_sts_dataset(sts_sentences1, sts_sentences2, sts_scores)
    
    # Loss fonksiyonları
    nli_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3
    )
    sts_loss = losses.CosineSimilarityLoss(model)
    
    # Çok görevli eğitim
    model.fit(
        train_objectives=[
            (nli_dataloader, nli_loss),
            (sts_dataloader, sts_loss)
        ],
        epochs=4,
        show_progress_bar=True
    )
    
    return model
```

Bu yaklaşım, iki görevin ortak optimizasyonunu sağlar ve anlamsal uzayın daha bütünsel bir şekilde yapılandırılmasına olanak tanır.

## 6. NLI ve STS Görevlerinin Bilişsel ve Nörolinguistik Temelleri

### 6.1 İnsan Bilişi ve Dil Modelleri Arasındaki Paralellikler

İnsan dilbilim süreçleri incelendiğinde, NLI ve STS görevlerinin aslında temel dil anlama mekanizmalarını yansıttığı görülür:

- **N400 ERP Bileşeni**: Beyin, anlamsal tutarsızlıklara 400ms civarında bir elektriksel tepki verir (NLI ile paralel)
- **Semantik Priming Etkisi**: İlişkili kelimelere tepki süresi daha hızlıdır (STS ile paralel)

```
"Dil modellerinin NLI ve STS görevlerindeki başarısı, insan dilbilim sisteminin temel mekanizmalarını ne ölçüde modelleyebildiklerinin bir göstergesidir." - Gary Marcus
```

### 6.2 Nedensellik ve İstatistiksel Korelasyon

Dilbilimsel felsefeye göre, NLI ve STS'nin temel farkı şöyle açıklanabilir:

- **NLI**: Nedensel ilişkileri ve çıkarımsal yapıları modellemek
- **STS**: İstatistiksel korelasyonları ve benzerlik ilişkilerini modellemek

Bu ayrım, sembolik ve alt-sembolik anlama sistemleri arasındaki klasik ayrımı yansıtır.

## 7. Pratik Uygulamalar: NLI ve STS'nin Etki Alanları

### 7.1 Endüstriyel Uygulamalar

NLI ve STS eğitimli modeller aşağıdaki alanlarda yaygın olarak kullanılır:

1. **Bilgi Erişimi ve Semantik Arama**:
   - Sorgu-doküman eşleştirme
   - Bilgi tabanı sorgulama
   - Anlamsal benzerliğe dayalı belge sıralama

2. **Soru-Cevap Sistemleri**:
   - Cevap doğrulama (NLI)
   - Cevap benzerliği ve devrikleme (STS)
   - Yan yana öğrenme (side-by-side learning)

3. **Otomatik Metin Değerlendirme**:
   - Kompozisyon değerlendirme
   - Makine çevirisi kalite değerlendirmesi
   - Metin özetleme kalite kontrolü

4. **Müşteri Deneyimi Analizi**:
   - Geri bildirim gruplandırma
   - Konu modelleme
   - Duygu analizi ve fikir madenciliği

### 7.2 Akademik ve Bilimsel Uygulamalar

1. **Biyomedikal Metin Madenciliği**:
   - Literatür taraması ve meta-analiz
   - İlaç-hastalık ilişkilerini keşfetme
   - Genetik ilişkileri çıkarma

2. **Hukuki Metin Analizi**:
   - Yasal içtihatları benzerlik temelinde gruplama
   - Argüman tespiti ve değerlendirmesi
   - Yasal akıl yürütme analizi

3. **Eğitim Teknolojileri**:
   - Otomatik ödev değerlendirme
   - Kişiselleştirilmiş eğitim içeriği eşleştirme
   - Öğrenme analitikleri ve kavram haritalama

```python
# Örnek Uygulama: Semantik Arama Motoru
def semantic_search_engine(query, documents, model):
    # Sorguyu encode et
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Dokümanları encode et
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    
    # Kosinüs benzerliği hesapla
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
    # En yüksek benzerliğe sahip dokümanları bul
    top_results = torch.topk(similarities, k=5)
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            'document': documents[idx],
            'score': score.item()
        })
    
    return results
```

## 8. Güncel Zorluklar ve Gelecek Araştırma Yönelimleri

### 8.1 NLI ve STS'nin Sınırlamaları

Her iki görevin de mevcut formülasyonlarındaki temel sınırlamalar:

1. **Dünya Bilgisi Entegrasyonu**: Her iki görev de, açık dünya bilgisi gerektiren durumlarda yetersiz kalabilir
2. **Bağlam Sınırlaması**: Genellikle iki cümleyle sınırlıdır, daha uzun metinlerdeki ilişkileri modellemekte zorlanabilir
3. **Kültürel ve Dilsel Yanlılık**: Veri setlerindeki kültürel ve dilsel yanlılıklar, modellerin genelleştirme yeteneğini sınırlar
4. **Anlamsal Derinlik**: Yüzeysel benzerlikler veya kalıplar üzerinden öğrenme eğilimi

### 8.2 İleri Araştırma Yönelimleri

1. **Neuro-Sembolik NLI ve STS**:
   - Sembolik mantık ve derin öğrenmenin entegrasyonu
   - Şeffaf ve açıklanabilir çıkarım mekanizmaları

2. **Çoklu-Modal NLI ve STS**:
   - Metin-görüntü, metin-ses NLI ve STS formülasyonları
   - Modaliteler arası semantik transfer

3. **Dillerarası ve Kültürlerarası NLI ve STS**:
   - Dil ve kültür bağımsız çıkarım ve benzerlik modelleri
   - Kültürel nüansları yakalayan STS metrikleri

4. **Derin Çıkarımsal Anlama**:
   - Çok adımlı çıkarım zincirleri
   - Hipotetik akıl yürütme ve karşı-olgusal çıkarım

```
"Gerçek anlamda insan düzeyinde anlama, yüzeysel metinsel ilişkileri modellemekten öte, derin anlamsal yapılara ve kavramsal bilgi grafiklerine erişim gerektirir." - Yoshua Bengio
```

## 9. Sonuç: NLI ve STS'nin Birleşik Teorisi

NLI ve STS, aslında anlamsal uzayın tamamlayıcı yapılarını modelleyen, dil anlama sürecinin temel bileşenleridir. Bu iki görev:

1. **Birlikte Çalışır**: NLI hiyerarşik ve mantıksal ilişkileri, STS yakınlık ve uzaklık ilişkilerini modelleyerek
2. **Birbirini Tamamlar**: NLI daha sınırlı ama kesin çıkarımları, STS daha yumuşak ama geniş anlamsal ilişkileri yakalar
3. **Anlama Sürecinin Temelini Oluşturur**: Bu iki yetenek, birlikte, daha yüksek seviyeli dil görevleri için temel oluşturur

Dilbilimci Ray Jackendoff'un deyişiyle:

```
"Anlamın iki temel boyutu vardır: referans (dünya ile ilişki) ve çıkarım (diğer anlamlarla ilişki)."
```

NLI ve STS, işte bu iki temel boyutu modelleyen görevlerdir ve doğal dil anlama sürecinin çekirdeğini oluşturur.

---

## Kaynaklar

1. Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large annotated corpus for learning natural language inference.
2. Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). SemEval-2017 Task 1: Semantic Textual Similarity.
3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
4. Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.
5. Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S. R., & Smith, N. A. (2018). Annotation Artifacts in Natural Language Inference Data.
6. Agirre, E., Banea, C., Cer, D., Diab, M., Gonzalez-Agirre, A., Mihalcea, R., ... & Wiebe, J. (2016). SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual and Cross-Lingual Evaluation.
7. Williams, A., Nangia, N., & Bowman, S. R. (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference.
8. Marelli, M., Menini, S., Baroni, M., Bentivogli, L., Bernardi, R., & Zamparelli, R. (2014). A SICK cure for the evaluation of compositional distributional semantic models.
9. Nie, Y., Williams, A., Dinan, E., Bansal, M., Weston, J., & Kiela, D. (2020). Adversarial NLI: A New Benchmark for Natural Language Understanding.
10. Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., & Van Durme, B. (2018). Hypothesis Only Baselines in Natural Language Inference.