# Transformer Tabanlı Semantik Modelleme: Teorik Temeller ve Uygulamalar


## 1. Semantik Modellemenin Teorik Temelleri

### 1.1 Transformer Mimarisinin Matematiksel Temeli

Transformer mimarisi, metin verilerinin anlamsal temsilini oluşturmak için dikkat mekanizmasını kullanan bir yapay sinir ağı modelidir. Bu mimarinin matematiksel temelini şöyle ifade edebiliriz:

$$\mathcal{T}: \mathcal{S} \rightarrow \mathcal{H}$$

Bu formül, transformer modelinin ($\mathcal{T}$) sözcükler uzayını ($\mathcal{S}$) gizli temsiller uzayına ($\mathcal{H}$) nasıl dönüştürdüğünü gösterir. Basitçe ifade etmek gerekirse, bu dönüşüm metni sayısal vektörlere çevirir.

Her bir katmanda, her token (kelime parçası) için yeni bir temsil oluşturulur:

$$\mathbf{h}_i^{(l)} = \text{TransformerLayer}_l(\mathbf{h}_i^{(l-1)}, \{\mathbf{h}_j^{(l-1)}\}_{j=1}^n)$$

Bu formül şunu anlatır:
- $\mathbf{h}_i^{(l)}$: $l$. katmandaki $i$. tokenin temsili
- $\mathbf{h}_i^{(l-1)}$: Bir önceki katmandaki aynı tokenin temsili
- $\{\mathbf{h}_j^{(l-1)}\}_{j=1}^n$: Önceki katmandaki tüm tokenlerin temsilleri (bağlam)

Transformer katmanı, her tokeni yalnızca kendisi açısından değil, tüm metin bağlamı açısından günceller. Bu, kelimelerin anlamını bağlama göre modelleyebilmemizi sağlar.

### 1.2 Dikkat Mekanizmasının Anlaşılır Açıklaması

Transformers mimarisinin kalbi, dikkat mekanizmasıdır. Dikkat mekanizması şu formülle hesaplanır:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Bu formülü adım adım açıklayalım:

1. **Sorgu, Anahtar, Değer Matrisleri:** 
   - $Q$ (Query/Sorgu): "Ne arıyorum?" sorusunu temsil eder
   - $K$ (Key/Anahtar): "Nerelerde arayabilirim?" bilgisini içerir
   - $V$ (Value/Değer): "Bulduğumda ne alacağım?" bilgisini içerir

2. **Benzerlik Hesaplama:** 
   - $QK^T$: Her sorgu vektörünün her anahtar vektörüyle nokta çarpımı, benzerliği ölçer
   - $\frac{QK^T}{\sqrt{d_k}}$: Ölçekleme faktörü ($\sqrt{d_k}$) ile bölme, gradyanların patlamasını önler

3. **Ağırlık Oluşturma:** 
   - $\text{softmax}(...)$: Sonuçları [0,1] aralığında olasılık dağılımına dönüştürür
   - Her token için diğer tüm tokenlere ne kadar "dikkat etmesi" gerektiğini belirler

4. **Ağırlıklı Toplamı Alma:**
   - $\text{softmax}(...) \times V$: Değer vektörlerinin ağırlıklı toplamını alır

Basit bir Türkçe örnek:
"Ahmet bankaya gitti" cümlesinde, "banka" kelimesi için model dikkat ağırlıklarını hesaplarken, kelimenin "para kurumu" mu yoksa "oturma bankı" mı anlamına geldiğini belirlemek için cümlenin diğer kısımlarına dikkat eder.

```python
def attention_mechanism(query, key, value):
    """
    Basit dikkat mekanizması hesaplaması
    
    Args:
        query: Sorgu tensörü [batch_size, seq_len, d_model]
        key: Anahtar tensörü [batch_size, seq_len, d_model]
        value: Değer tensörü [batch_size, seq_len, d_model]
        
    Returns:
        Dikkat uygulanmış değerler [batch_size, seq_len, d_model]
    """
    # Sorgu ile anahtarların benzerliğini hesapla
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Ölçekleme - gradyan patlamasını önler
    d_k = query.size()[-1]
    scaled_scores = scores / math.sqrt(d_k)
    
    # Softmax uygula - toplamı 1 olan ağırlıklar elde et
    weights = F.softmax(scaled_scores, dim=-1)
    
    # Değerlerin ağırlıklı toplamını al
    output = torch.matmul(weights, value)
    
    return output
```

### 1.3 Bağlamsal Temsiller ve Anlam Vektörleri

BERT gibi modeller, kelime gömmelerini bağlama duyarlı hale getirir. Örneğin, "banka" kelimesi farklı cümlelerde farklı anlamlara gelebilir:

- "Param için bankaya gittim." (Finansal kurum)
- "Parkta bankaya oturdum." (Oturak)

Bağlamsal gömmeler bu farkı yakalayabilir. Matematiksel olarak:

$$\phi(w|c) \neq \phi(w|c')$$

Bu notasyon şunu ifade eder:
- $\phi(w|c)$: $w$ kelimesinin $c$ bağlamındaki temsili
- $\phi(w|c')$: Aynı kelimenin farklı bir $c'$ bağlamındaki temsili

Bu bağlamsal temsiller, kelime anlamlarının bağlama göre değişebileceği gerçeğini modellemeye yardımcı olur.

## 2. BERT Modelinin İşleyişi ve Semantik Özellikleri

### 2.1 BERT'in Matematiksel Modeli (Basitleştirilmiş)

BERT, çift yönlü bir Transformer modelidir. İşleyişini şu şekilde formüle edebiliriz:

$$\mathbf{H} = \text{TransformerEncoder}(\mathbf{E} + \mathbf{P})$$

Bu formülün anlamı:
- $\mathbf{E}$: Token gömme matrisi (her kelime için bir vektör)
- $\mathbf{P}$: Pozisyon kodlama matrisi (kelimelerin sırasını kodlar)
- $\mathbf{H}$: Çıktı temsilleri (bağlamsal gömmeler)

Daha açık ifade etmek gerekirse:
1. Her token, bir gömme vektörüne dönüştürülür
2. Bu gömmelere, kelimelerin cümledeki konumunu belirten pozisyon kodlamaları eklenir
3. Bu toplam, Transformer kodlayıcı katmanlarından geçirilir
4. Çıktı, her token için bağlamsal bir gömme vektörüdür

BERT özellikle Maskelenmiş Dil Modelleme (MLM) göreviyle eğitilir:

$$P(w_i | w_1, ..., w_{i-1}, [MASK], w_{i+1}, ..., w_n)$$

Bu, cümledeki bazı kelimeleri "[MASK]" tokeniyle değiştirip modelin bu maskelenmiş kelimeleri tahmin etmesini sağlayarak, modelin iki yönlü bağlam anlayışı geliştirmesine yardımcı olur.

```python
class SimplifiedBERT(nn.Module):
    def __init__(self, vocab_size=30000, hidden_size=768, num_layers=12):
        super().__init__()
        
        # Token gömmeleri
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Pozisyon gömmeleri (maksimum 512 pozisyon için)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        
        # Transformer katmanları
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask):
        # Token gömmelerini al
        embeddings = self.token_embeddings(input_ids)
        
        # Pozisyon bilgisini ekle
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeddings = self.position_embeddings(positions)
        
        # Token ve pozisyon gömmelerini birleştir
        hidden_states = embeddings + position_embeddings
        
        # Transformer katmanlarından geçir
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        return hidden_states
```

### 2.2 Mean Pooling: Anlamsal Özet Çıkarma

BERT, her token için ayrı bir temsil üretir, ancak genellikle tüm cümlenin tek bir temsili gerekir. Mean pooling (ortalama havuzlama), token temsillerinin ağırlıklı ortalamasını alarak tüm cümlenin bir temsilini oluşturur:

$$\text{MeanPooling}(\mathbf{H}, \mathbf{M}) = \frac{\sum_{i=1}^{n} \mathbf{h}_i \cdot \mathbf{m}_i}{\sum_{i=1}^{n} \mathbf{m}_i}$$

Burada:
- $\mathbf{H}$ = BERT'in ürettiği tüm token temsilleri ($\mathbf{h}_i$ her bir token temsili)
- $\mathbf{M}$ = Dikkat maskesi ($\mathbf{m}_i$, token $i$ gerçek bir içerik ise 1, dolgu (padding) ise 0)

Mean pooling neden önemlidir?
1. **Değişken Uzunluk Sorunu**: Farklı uzunluktaki cümleler için sabit boyutlu bir temsil oluşturur
2. **Padding Tokenleri**: Dikkat maskesi sayesinde sadece gerçek içerik tokenlerini hesaba katar
3. **Tüm Bilgiyi Kullanma**: [CLS] token'a kıyasla cümlenin tüm bileşenlerindeki anlamsal bilgiyi kullanır

```python
def mean_pooling(token_embeddings, attention_mask):
    """
    Dikkat maskesi kullanarak token gömmelerinin ortalamasını alır
    
    Args:
        token_embeddings: Model çıktısı [batch_size, seq_len, hidden_size]
        attention_mask: Dikkat maskesi [batch_size, seq_len]
        
    Returns:
        Cümle gömmeleri [batch_size, hidden_size]
    """
    # Maskeyi gömme boyutuna genişlet
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Maskelenmiş token gömmelerinin toplamını al
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # Mask toplamını al (gerçek token sayısı)
    sum_mask = input_mask_expanded.sum(1)
    
    # Sıfıra bölünme hatasını önle
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    
    # Ortalama hesapla
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings
```

Mean pooling işlemi, neden daha ayrıntılı olarak adım adım açıklanmalı:

1. **Maskeyi Genişletme**: Dikkat maskesi, genellikle [batch_size, seq_len] boyutundadır ve her tokenin gerçek (1) mi yoksa padding (0) mı olduğunu gösterir. Bu maskeyi [batch_size, seq_len, hidden_size] boyutuna genişletiyoruz ki token gömmelerimizle aynı boyutta olsun.

2. **Maskeleme İşlemi**: Token gömmelerini maskeyle çarparak, padding tokenlerinin değerlerini sıfırlıyoruz. Böylece sadece gerçek içerik tokenlerinin değerleri kalır.

3. **Toplama**: Maskelenmiş gömmeler, sekans boyutu (1. boyut) üzerinde toplanarak her örnek için tek bir vektöre indirgenir.

4. **Normalizasyon**: Elde edilen toplamı, cümledeki gerçek token sayısına bölerek ortalama alırız. Bu, farklı uzunluktaki cümleler için karşılaştırılabilir temsiller oluşturur.

5. **Sıfıra Bölünme Koruması**: Çok nadir durumlarda, tamamen maskeli bir örnek olabilir. Bu durumda sıfıra bölünme hatasını önlemek için torch.clamp kullanılır.

### 2.3 Token Temsillerinin Anlamsal İçeriği

BERT ve benzeri modeller, farklı katmanlarında farklı tür bilgileri kodlar. Bu hiyerarşik yapı şu şekilde düşünülebilir:

- **Alt Katmanlar (1-4)**: Daha çok sözdizimsel (sentaktik) özellikleri ve temel dilbilgisel ilişkileri kodlar
- **Orta Katmanlar (5-8)**: Kelime anlamları ve yerel bağlam bilgisini işler
- **Üst Katmanlar (9-12)**: Daha karmaşık anlamsal ilişkileri ve bağlamsal anlamı kodlar

Bu bilgi hiyerarşisini matematiksel olarak ifade edersek:

$$I(H^{(l)}; \text{Syntax}) > I(H^{(l+1)}; \text{Syntax})$$
$$I(H^{(l)}; \text{Semantics}) < I(H^{(l+1)}; \text{Semantics})$$

Bu formüller şunu ifade eder:
- $I(X; Y)$: X ve Y arasındaki karşılıklı bilgi miktarı
- $H^{(l)}$: $l$. katmandaki gömmeler
- Alt katmanların sözdizimsel bilgiyle daha fazla, anlamsal bilgiyle daha az ilişkisi vardır
- Üst katmanlar ilerledikçe anlamsal bilgi artar, sözdizimsel bilgi azalır

Bu nedenle, farklı NLP görevleri için farklı katmanların temsilleri daha uygun olabilir:
- Sözdizimi analizinde alt katmanlar daha faydalı olabilir
- Anlamsal benzerlik görevlerinde üst katmanlar tercih edilebilir

## 3. Anlamsal Benzerlik İçin Etkili Cümle Temsilleri

### 3.1 Kosinüs Benzerliği ve Semantik Uzay

İki metin arasındaki anlamsal benzerliği ölçmek için en yaygın kullanılan metrik kosinüs benzerliğidir:

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

Bu formül, iki vektör arasındaki açının kosinüsünü hesaplar:
- $\mathbf{a} \cdot \mathbf{b}$: İki vektörün nokta çarpımı
- $||\mathbf{a}||$: a vektörünün L2 normu (uzunluğu)
- $||\mathbf{b}||$: b vektörünün L2 normu

Kosinüs benzerliği neden anlamsal benzerlik için uygundur?
1. **Yön Odaklı**: Vektörlerin büyüklüğünden ziyade yönlerine odaklanır
2. **Normalizasyon**: [-1, 1] aralığında değerler üretir, 1 tam benzerlik, -1 tam zıtlık, 0 ilişkisizlik anlamına gelir
3. **Boyut Bağımsızlığı**: Farklı uzunluktaki metinler için karşılaştırılabilir sonuçlar verir

```python
def compute_cosine_similarity(embeddings1, embeddings2):
    """
    İki gömme seti arasında kosinüs benzerliği hesaplar
    
    Args:
        embeddings1: İlk gömme seti [n_samples1, embedding_dim]
        embeddings2: İkinci gömme seti [n_samples2, embedding_dim]
        
    Returns:
        Benzerlik matrisi [n_samples1, n_samples2]
    """
    # Normalize et (opsiyonel, zaten normalize edilmişse)
    norm_emb1 = F.normalize(embeddings1, p=2, dim=1)
    norm_emb2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Kosinüs benzerliği = nokta çarpımı (normalize edilmiş vektörler için)
    return torch.matmul(norm_emb1, norm_emb2.transpose(0, 1))
```

Anlamsal uzayda, benzer anlamlı cümleler birbirine yakın vektörlerle temsil edilir. Bu, matematiksel olarak şu şekilde ifade edilebilir:

$$d(\phi(s_1), \phi(s_2)) \approx d_{semantic}(s_1, s_2)$$

Burada:
- $\phi(s)$: Cümle $s$'nin anlamsal gömme fonksiyonu
- $d$: Vektör uzayında mesafe fonksiyonu (genellikle kosinüs mesafesi)
- $d_{semantic}$: İki cümle arasındaki gerçek anlamsal mesafe

### 3.2 Normalizasyon ve Kalibre Etme

BERT gibi modellerin ürettiği gömmeler genellikle anizotropik'tir - yani vektörler uzayda düzgün dağılmaz, belirli yönlerde kümelenir. Bu durum kosinüs benzerliği hesaplamalarında sorunlara yol açabilir.

Bu sorunu çözmek için normalizasyon ve kalibrasyon teknikleri kullanılır:

1. **L2 Normalizasyon**: Her gömme vektörünü birim uzunluğa normalize etme:

$$\hat{\mathbf{h}} = \frac{\mathbf{h}}{||\mathbf{h}||}$$

2. **Whitening (Beyazlatma)**: Gömme dağılımını dönüştürerek anizotropiyi azaltma:

$$\mathbf{h}_{whitened} = \mathbf{\Sigma}^{-1/2}(\mathbf{h} - \boldsymbol{\mu})$$

Burada:
- $\boldsymbol{\mu}$: Gömmelerin ortalama vektörü
- $\mathbf{\Sigma}$: Kovaryans matrisi
- $\mathbf{\Sigma}^{-1/2}$: Kovaryans matrisinin karekök tersi

Whitening işlemi, gömmeler arasındaki korelasyonları kaldırır ve varyansı normalleştirir, böylece anlamsal uzayın daha iyi kalibre edilmesini sağlar.

```python
def normalize_embeddings(embeddings):
    """Gömme vektörlerini L2 normalizasyonu ile birim uzunluğa normalize eder"""
    return F.normalize(embeddings, p=2, dim=1)

def whiten_embeddings(embeddings):
    """
    Gömmelere whitening dönüşümü uygular
    """
    # Ortalamasını çıkar
    mean = embeddings.mean(dim=0, keepdim=True)
    centered_embeddings = embeddings - mean
    
    # Kovaryans matrisini hesapla
    n_samples = embeddings.size(0)
    cov = torch.mm(centered_embeddings.t(), centered_embeddings) / (n_samples - 1)
    
    # Özdeğer ayrışımı yap
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Whitening matrisi oluştur
    epsilon = 1e-10  # Sayısal stabilite için
    whitening_matrix = torch.mm(
        eigenvectors, 
        torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
    )
    whitening_matrix = torch.mm(whitening_matrix, eigenvectors.t())
    
    # Whitening uygula
    whitened_embeddings = torch.mm(centered_embeddings, whitening_matrix)
    
    return whitened_embeddings
```

### 3.3 Kontrastif Öğrenme ile Semantik Uzayı İyileştirme

Kontrastif öğrenme, anlamsal benzerlik görevleri için gömmeleri daha iyi hale getirmek için kullanılan güçlü bir tekniktir. Bu yaklaşım, benzer anlamları olan cümleleri vektör uzayında birbirine yakın, farklı anlamları olanları uzak konumlandırmayı amaçlar.

SimCSE gibi yaklaşımlar şu kayıp fonksiyonunu kullanır:

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}$$

Bu formülü daha açık şekilde anlamak için:
- $\mathbf{h}_i$: Çapa cümlenin gömmesi
- $\mathbf{h}_i^+$: Anlamsal olarak benzer cümlenin gömmesi (pozitif örnek)
- $\mathbf{h}_j$: Batch'teki diğer tüm cümlelerin gömmeleri (negatif örnekler)
- $\text{sim}$: Benzerlik fonksiyonu (genellikle kosinüs benzerliği)
- $\tau$: Sıcaklık parametresi (genellikle 0.05 - 0.1 arası)

Bu kayıp fonksiyonu, pozitif örneğe olan benzerliği maksimize ederken, negatif örneklerin benzerliğini minimize eder. Sıcaklık parametresi $\tau$, benzerlik skorlarının dağılımını kontrol eder.

SimCSE'de özellikle ilginç olan yaklaşım, aynı cümlenin iki farklı forward geçişini (farklı dropout desenleriyle) pozitif çift olarak kullanmasıdır. Bu sayede etiketli veri olmadan bile iyi sonuçlar elde edilir.

```python
def contrastive_loss(similarities, temperature=0.05):
    """
    Kontrastif kayıp (InfoNCE) hesaplar
    
    Args:
        similarities: Benzerlik matrisi [batch_size, batch_size]
        temperature: Sıcaklık parametresi
        
    Returns:
        Ortalama kontrastif kayıp
    """
    batch_size = similarities.size(0)
    
    # Etiketler: Kendi kendine benzerlik (diagonal)
    labels = torch.arange(batch_size, device=similarities.device)
    
    # Sıcaklık ile ölçeklendirme
    similarities = similarities / temperature
    
    # Çapraz entropi kaybı
    loss = F.cross_entropy(similarities, labels)
    
    return loss

def train_simcse_step(model, tokenizer, sentences, optimizer):
    """
    SimCSE eğitimi için tek bir adım
    """
    # Metinleri tokenize et
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Aynı batch'i iki kez forward et (farklı dropout desenleriyle)
    model.train()  # Dropout'u etkinleştir
    
    # İlk geçiş
    embeddings1 = model(**inputs)
    
    # İkinci geçiş (farklı dropout maskesi)
    embeddings2 = model(**inputs)
    
    # Normalize
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Tüm çiftler arasında benzerlik matrisini hesapla
    similarities = torch.matmul(embeddings1, embeddings2.t())
    
    # Kontrastif kaybı hesapla
    loss = contrastive_loss(similarities)
    
    # Geriye yayılım
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## 4. Semantik Model Eğitimi ve İnce Ayar

### 4.1 İki Aşamalı Eğitim: NLI ve STS

Anlamsal benzerlik modelleri genellikle iki aşamalı bir süreçle eğitilir:

1. **NLI (Natural Language Inference) Eğitimi**:
   - Doğal dil çıkarım veri setleri kullanılır (örn. SNLI, MNLI)
   - Bu veri setleri, "entailment" (içerme), "contradiction" (çelişki) ve "neutral" (nötr) ilişkilerini içerir
   - "Entailment" çiftleri pozitif örnekler, "contradiction" çiftleri negatif örnekler olarak kullanılır

2. **STS (Semantic Textual Similarity) İnce Ayarı**:
   - STS veri setleri, cümle çiftleri ve bunların benzerlik skorlarını içerir (genellikle 0-5 arasında)
   - Bu skorlar [0,1] aralığına normalize edilir
   - Model, tahmin edilen benzerlik ile gerçek benzerlik arasındaki farkı minimize etmek için eğitilir

Bu iki aşamalı yaklaşımın matematiği:

NLI Kaybı (kontrastif):
$$\mathcal{L}_{NLI} = -\log \frac{\exp(\text{sim}(\mathbf{h}_p, \mathbf{h}_h^+) / \tau)}{\exp(\text{sim}(\mathbf{h}_p, \mathbf{h}_h^+) / \tau) + \exp(\text{sim}(\mathbf{h}_p, \mathbf{h}_h^-) / \tau)}$$

Burada:
- $\mathbf{h}_p$: Öncül (premise) cümlenin gömmesi
- $\mathbf{h}_h^+$: İçerme (entailment) ilişkisindeki hipotez cümlenin gömmesi
- $\mathbf{h}_h^-$: Çelişki (contradiction) ilişkisindeki hipotez cümlenin gömmesi

STS Kaybı (regresyon):
$$\mathcal{L}_{STS} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \text{sim}(\mathbf{h}_{i1}, \mathbf{h}_{i2}))^2$$

Burada:
- $y_i$: i. çift için gerçek benzerlik skoru (0-1 arasında normalize edilmiş)
- $\mathbf{h}_{i1}, \mathbf{h}_{i2}$: i. çiftteki cümlelerin gömmeleri

```python
def train_nli_step(model, batch, optimizer):
    """
    NLI göreviyle eğitim adımı
    """
    # Batch'ten öğeleri çıkar
    premises, hypotheses_pos, hypotheses_neg = batch
    
    # Gömmeleri hesapla
    premise_embeddings = model.encode(premises)
    pos_hypothesis_embeddings = model.encode(hypotheses_pos)
    neg_hypothesis_embeddings = model.encode(hypotheses_neg)
    
    # Normalize
    premise_embeddings = F.normalize(premise_embeddings, p=2, dim=1)
    pos_hypothesis_embeddings = F.normalize(pos_hypothesis_embeddings, p=2, dim=1)
    neg_hypothesis_embeddings = F.normalize(neg_hypothesis_embeddings, p=2, dim=1)
    
    # Pozitif ve negatif çiftler arasındaki benzerlikler
    pos_similarities = torch.sum(premise_embeddings * pos_hypothesis_embeddings, dim=1)
    neg_similarities = torch.sum(premise_embeddings * neg_hypothesis_embeddings, dim=1)
    
    # Kayıp hesapla (InfoNCE formülasyonunun basitleştirilmiş versiyonu)
    logits = torch.stack([pos_similarities, neg_similarities], dim=1) / 0.05
    labels = torch.zeros(len(premises), dtype=torch.long, device=model.device)
    loss = F.cross_entropy(logits, labels)
    
    # Geriye yayılım
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_sts_step(model, batch, optimizer):
    """
    STS göreviyle ince ayar adımı
    """
    # Batch'ten öğeleri çıkar
    sentences1, sentences2, similarity_scores = batch
    
    # Gömmeleri hesapla
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)
    
    # Normalize
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Kosinüs benzerliği
    pred_similarities = torch.sum(embeddings1 * embeddings2, dim=1)
    
    # MSE kaybı
    loss = F.mse_loss(pred_similarities, similarity_scores)
    
    # Geriye yayılım
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 4.2 Çok Dilli Semantik Modelleme

Çok dilli semantik modeller, farklı dillerdeki benzer anlamları aynı vektör uzayında temsil etmeyi amaçlar. Bu modeller genellikle iki yaklaşımla eğitilir:

1. **Birleşik Çok Dilli Ön Eğitim**:
   - mBERT veya XLM-R gibi modeller, çok dilli veri ile birlikte eğitilir
   - Aynı anlamı taşıyan farklı dillerdeki cümleler, benzer vektör temsillerine sahip olur

2. **Çapraz Dil Transfer Öğrenimi**:
   - Paralel korpuslar kullanılarak, farklı dillerdeki eşdeğer cümleler birbirine yakın konumlandırılır
   - Paralellik kaybı şöyle ifade edilir:

$$\mathcal{L}_{parallel} = \frac{1}{|P|} \sum_{(s_i^{L_1}, s_i^{L_2}) \in P} (1 - \text{sim}(E(s_i^{L_1}), E(s_i^{L_2})))$$

Burada:
- $P$: Paralel cümle çiftleri kümesi
- $s_i^{L_1}$: i. cümlenin birinci dildeki versiyonu
- $s_i^{L_2}$: i. cümlenin ikinci dildeki versiyonu
- $E$: Cümle kodlayıcı model
- $\text{sim}$: Benzerlik fonksiyonu

Bu yaklaşım, çok dilli vektör uzayının oluşturulmasına yardımcı olur, böylece farklı dillerdeki anlamsal benzer ifadeler birbirine yakın konumlandırılır.

```python
def train_multilingual_step(model, batch, optimizer):
    """
    Çok dilli semantik model eğitimi adımı
    """
    # Batch'ten paralel cümleleri al
    sentences_lang1, sentences_lang2 = batch
    
    # Her iki dildeki cümleleri kodla
    embeddings_lang1 = model.encode(sentences_lang1)
    embeddings_lang2 = model.encode(sentences_lang2)
    
    # Normalize
    embeddings_lang1 = F.normalize(embeddings_lang1, p=2, dim=1)
    embeddings_lang2 = F.normalize(embeddings_lang2, p=2, dim=1)
    
    # Paralel benzerlik kaybı
    # Aynı endeksteki cümleler paralel (birbirinin çevirisi) olmalı
    # Kosinüs benzerliği 1'e yakın olmalı, bu yüzden (1 - sim) ifadesini minimize ediyoruz
    similarities = torch.sum(embeddings_lang1 * embeddings_lang2, dim=1)
    loss = torch.mean(1 - similarities)
    
    # Ek olarak, kontrastif kayıp da eklenebilir
    # Burada paralel olmayan çiftlerin benzerliği azaltılır
    
    # Geriye yayılım
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## 5. Semantik Modellerin Değerlendirilmesi ve Analizi

### 5.1 Anlamsal Benzerlik Değerlendirme Metrikleri

Semantik modellerin başarısı, genellikle STS (Semantic Textual Similarity) benchmark'ları üzerindeki performanslarıyla ölçülür. Ana değerlendirme metrikleri şunlardır:

1. **Pearson Korelasyon Katsayısı**:
   
$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

Burada:
- $x_i$: Model tarafından tahmin edilen benzerlik skoru
- $y_i$: İnsan tarafından verilen benzerlik skoru
- $\bar{x}, \bar{y}$: x ve y'nin ortalamaları

2. **Spearman Sıra Korelasyon Katsayısı**:

$$\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

Burada $d_i$, i. örnek için x ve y'nin sıralamalarındaki farktır.

Bu korelasyon metrikleri, modelin tahminlerinin insan değerlendirmeleriyle ne kadar uyuştuğunu ölçer. 1'e yakın değerler mükemmel korelasyonu, 0'a yakın değerler zayıf korelasyonu gösterir.

```python
def evaluate_semantic_model(model, eval_pairs, human_scores):
    """
    Semantik modeli değerlendir
    
    Args:
        model: Değerlendirilecek model
        eval_pairs: (sentence1, sentence2) şeklinde değerlendirme çiftleri
        human_scores: İnsan değerlendirme skorları
        
    Returns:
        dict: Değerlendirme metrikleri
    """
    # Modelin benzerlik skorlarını hesapla
    predicted_scores = []
    
    for sent1, sent2 in eval_pairs:
        # Cümleleri kodla
        emb1 = model.encode([sent1])[0]
        emb2 = model.encode([sent2])[0]
        
        # Kosinüs benzerliği
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        predicted_scores.append(similarity)
    
    # Pearson korelasyonu
    pearson_corr, pearson_p = pearsonr(human_scores, predicted_scores)
    
    # Spearman korelasyonu
    spearman_corr, spearman_p = spearmanr(human_scores, predicted_scores)
    
    return {
        "pearson": pearson_corr,
        "pearson_p": pearson_p,
        "spearman": spearman_corr,
        "spearman_p": spearman_p
    }
```

### 5.2 Gömme Uzayının Görselleştirilmesi ve Analizi

Semantik gömme uzayını analiz etmek ve anlamak için çeşitli boyut indirgeme ve görselleştirme teknikleri kullanılır:

1. **PCA (Principal Component Analysis)**:
   - Veri varyansını en iyi açıklayan bileşenleri bulur
   - Matematiksel olarak, kovaryans matrisinin özdeğer ayrışımını kullanır:
   
   $$\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$
   
   Burada $\mathbf{\Sigma}$ kovaryans matrisi, $\mathbf{V}$ özvektör matrisi, $\mathbf{\Lambda}$ özdeğer matrisidir.

2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
   - Yüksek boyutlu uzaydaki benzerlik ilişkilerini koruyarak düşük boyutlu görselleştirme yapar
   - Olasılık dağılımlarını KL-yakınsamasıyla minimize eder:
   
   $$KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
   
   Burada $p_{ij}$ orijinal uzaydaki benzerlik, $q_{ij}$ düşük boyutlu uzaydaki benzerliktir.

3. **UMAP (Uniform Manifold Approximation and Projection)**:
   - Topolojik yapıyı koruyarak boyut indirme yapar
   - t-SNE'den daha hızlı ve ölçeklenebilirdir

Bu teknikler, semantik uzayın yapısını anlamak, kümelenmeleri görmek ve modelin ne tür ilişkileri yakaladığını görselleştirmek için kullanılır.

```python
def visualize_embeddings(embeddings, labels=None, method='tsne'):
    """
    Gömme uzayını görselleştir
    
    Args:
        embeddings: Görselleştirilecek gömmeler
        labels: Nokta etiketleri (opsiyonel)
        method: Görselleştirme metodu ('pca', 'tsne', 'umap')
    """
    # Boyut indirgeme
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
        raise ValueError(f"Bilinmeyen görselleştirme metodu: {method}")
    
    # Boyut indirgeme uygula
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Renkli gösterim
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=labels, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Etiket')
    else:
        # Tek renkli gösterim
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(f'Gömmelerin {method.upper()} Görselleştirmesi')
    plt.xlabel('Boyut 1')
    plt.ylabel('Boyut 2')
    plt.grid(alpha=0.3)
    plt.show()
```

## 6. Sonuç ve Pratik Uygulama Önerileri

Transformer tabanlı semantik modelleme, anlamsal benzerlik, metin sınıflandırma, bilgi erişimi ve daha birçok NLP uygulaması için güçlü bir temel sağlar. Bu modellerin etkin kullanımı için birkaç pratik öneri:

1. **Doğru Pooling Stratejisi Seçimi**:
   - Genel anlamsal benzerlik için mean pooling genellikle en iyi performansı gösterir
   - Sınıflandırma görevleri için [CLS] token veya max pooling faydalı olabilir
   - Farklı stratejileri karşılaştırıp görevinize en uygun olanı seçin

2. **Normalizasyon ve Ön İşleme**:
   - Gömmeleri her zaman normalize edin (L2 normalizasyonu)
   - Gerekirse whitening gibi ek normalizasyon teknikleri kullanın
   - Türkçe metinler için uygun ön işleme adımlarını uygulayın (normalize, lowercase vb.)

3. **Model Seçimi ve İnce Ayar**:
   - Türkçe için özel olarak eğitilmiş modeller seçin (BERTurk, XLM-R vb.)
   - Görevinize uygun şekilde ince ayar yapın (NLI, STS veya domain-spesifik verilerle)
   - İki aşamalı eğitim yaklaşımını kullanın (NLI + STS)

4. **Değerlendirme ve Analiz**:
   - Modelin performansını standart STS benchmark'larıyla karşılaştırın
   - Gömme uzayını görselleştirerek modelin davranışını anlayın
   - Hata analizi yaparak modelin iyileştirilebileceği alanları belirleyin

