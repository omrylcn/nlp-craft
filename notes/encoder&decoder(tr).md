# Encoder-Only ve Decoder-Only Modeller: Kapsamlı Teknik Analiz ve Karşılaştırma

## Özet

Bu makale, modern doğal dil işleme (NLP) alanında iki baskın mimari paradigmayı—encoder-only ve decoder-only modellerini—kapsamlı şekilde analiz etmektedir. Transformer mimarisinin temel prensiplerinden başlayarak, her iki yaklaşımın matematiksel temelleri, mimari tasarım tercihleri, eğitim stratejileri ve pratik uygulamaları açısından derinlemesine incelemesini sunmaktadır. Makale, performans ölçütleri, hesaplama verimliliği ve kullanım senaryosu optimizasyonları perspektiflerinden sistematik karşılaştırma içermekte, gelecekteki araştırma yönleri için içgörüler sağlamaktadır.

**Anahtar Kelimeler:** Transformer, Encoder-Only, Decoder-Only, BERT, GPT, Attention Mechanism, Language Modeling

---

## 1. Giriş: Transformer Mimarisinin Anatomisi

### 1.1 Tarihsel Bağlam ve Gelişim

Transformer mimarisi, Vaswani ve arkadaşları (2017) tarafından "Attention is All You Need" makalesi ile tanıtıldığında, sıralı işlemden paralel işlemeye geçişi temsil eden bir paradigma değişimi yaratmıştı. Orijinal Transformer, encoder-decoder mimarisi kullanıyordu, ancak sonraki gelişmeler iki farklı yönde uzmanlaşmaya başladı:

1. **Encoder-only modeller**: BERT (2018) ile başlayıp ModernBERT'e kadar uzanan soy
2. **Decoder-only modeller**: GPT (2018) ile başlayıp GPT-4, Claude, LLaMA'ya uzanan evrim hattı

### 1.2 Temel Mimari Yapı Taşları

Her iki yaklaşım da temel attention mekanizmasını paylaşır, ancak bilgi akış kalıpları ve eğitim hedefleri açısından temel farklılıklar gösterir.

**Ölçeklendirilmiş Nokta Çarpım Dikkat (Scaled Dot-Product Attention) Matematiksel Temeli:**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Çok Başlı Dikkat (Multi-Head Attention) Formülasyonu:**
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
burada head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Bu matematiksel temel yapılar her iki mimaride kullanılsa da, uygulama detayları ve kullanım kalıpları dramatik şekilde farklılaşır.

---

## 2. Encoder-Only Mimarisi: Çift Yönlü Anlama Paradigması

### 2.1 Mimari Derinlemesine İnceleme

Encoder-only modeller, girdi dizisinin tam çift yönlü bağlamını eş zamanlı olarak işler. Bu yaklaşım, "anlama" görevleri için optimize edilmiş tasarım felsefesini temsil eder.

#### 2.1.1 Encoder'larda Self-Attention Mekanizması

Encoder'larda her token, dizi içerisindeki tüm diğer token'larla etkileşime girebilir. Bu çift yönlü görünürlük, zengin bağlamsal temsiller oluşturur:

```python
class EncoderSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Her token tüm token'larla etkileşime girebilir
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Çift yönlü dikkat hesaplaması
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Başları birleştir
        context = context.view(batch_size, seq_len, d_model)
        return self.W_o(context)
```

#### 2.1.2 Encoder Katman Yığını Mimarisi

Tipik encoder yığını, N özdeş katmandan oluşur, her biri şunları içerir:
- Çok başlı self-attention alt katmanı
- Pozisyon bazlı ileri beslemeli ağ
- Artık bağlantılar ve katman normalizasyonu

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = EncoderSelfAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Artık bağlantı ile self-attention
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Artık bağlantı ile ileri beslemeli ağ
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 2.2 Eğitim Hedefleri: Maskeli Dil Modelleme Paradigması

#### 2.2.1 Maskeli Dil Modelleme (MLM)

MLM, encoder-only modellerin birincil eğitim hedefidir. Bu kendi kendini denetleyen yaklaşım, çift yönlü bağlamdan maskelenmiş token'ları tahmin etmeyi öğrenir:

```python
def create_mlm_batch(texts, tokenizer, mask_prob=0.15):
    """BERT tarzı MLM veri hazırlama"""
    input_ids = tokenizer(texts, padding=True, return_tensors='pt')['input_ids']
    labels = input_ids.clone()
    
    # Maskeleme stratejisi: token'ların %15'i
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    special_tokens_mask = get_special_tokens_mask(input_ids, tokenizer)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Sadece maskelenmiş token'larda kayıp hesapla
    
    # %80 [MASK], %10 rastgele, %10 orijinal
    mask_token_indices = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[mask_token_indices] = tokenizer.mask_token_id
    
    random_token_indices = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~mask_token_indices
    random_words = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[random_token_indices] = random_words[random_token_indices]
    
    return input_ids, labels

class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
```

#### 2.2.2 Alternatif Eğitim Hedefleri

**Sonraki Cümle Tahmini (NSP):** BERT'de kullanılan ikincil hedef, cümle çifti ilişkilerini öğrenir.

**Cümle Sırası Tahmini (SOP):** ALBERT'de NSP yerine kullanılan geliştirilmiş hedef.

**Değiştirilmiş Token Tespiti (RTD):** ELECTRA'da kullanılan verimli alternatif, ayırt edici model ile üreteciden gelen değiştirilmiş token'ları tespit eder.

### 2.3 Encoder-Only Model Gelişimi

#### 2.3.1 BERT Ailesi Gelişimi

**BERT (2018):** Çift yönlü encoder paradigmasının öncüsü
- 110M/340M parametreler (base/large)
- 512 maksimum dizi uzunluğu
- WordPiece tokenization
- NSP + MLM eğitim hedefleri

**RoBERTa (2019):** Eğitim prosedürü optimizasyonları
- NSP'nin kaldırılması
- Dinamik maskeleme
- Daha büyük batch boyutları ve daha uzun eğitim
- BPE tokenization

**ALBERT (2019):** Parametre verimliliği odaklı
- Katmanlar arası parametre paylaşımı
- Çarpanlarına ayrılmış gömme parametrizasyonu
- NSP yerine SOP

**DeBERTa (2020):** Geliştirilmiş dikkat mekanizması
- Ayrıştırılmış dikkat (içerik vs pozisyon)
- Geliştirilmiş maske kod çözücü
- Göreli pozisyonel kodlama

**ModernBERT (2024):** Çağdaş mimari
- 8192 dizi uzunluğu
- RoPE pozisyonel gömmeler
- GeGLU aktivasyonları
- Alternatif yerel-küresel dikkat

#### 2.3.2 Özelleşmiş Encoder Çeşitleri

**DistilBERT:** Bilgi damıtma ile sıkıştırma
**ELECTRA:** Üretici-ayırt edici eğitim paradigması
**BigBird:** Uzun diziler için seyrek dikkat
**Longformer:** Kayan pencere + küresel dikkat

---

## 3. Decoder-Only Mimarisi: Otoregresif Üretim Paradigması

### 3.1 Mimari Derinlemesine İnceleme

Decoder-only modeller, nedensel dil modelleme paradigmasını uygular. Bilgi akışı, sıkı şekilde soldan sağa yönde kısıtlıdır, bu da otoregresif metin üretimini mümkün kılar.

#### 3.1.1 Decoder'larda Maskeli Self-Attention

Decoder'larda dikkat mekanizması, gelecekteki token'lara erişimi engelleyen nedensel maske kullanır:

```python
class DecoderSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def create_causal_mask(self, seq_len, device):
        """Gelecekteki token'lara erişimi engelleyen üçgen maske"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Nedensel dikkat hesaplaması
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Nedensel maske uygula - gelecek pozisyonları -inf olarak ayarla
        causal_mask = self.create_causal_mask(seq_len, x.device)
        scores.masked_fill_(~causal_mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.view(batch_size, seq_len, d_model)
        return self.W_o(context)
```

#### 3.1.2 Decoder Katman Mimarisi

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = DecoderSelfAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Modern decoder'lar genellikle GELU kullanır
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Nedensel self-attention
        attn_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # İleri beslemeli ağ
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 3.2 Eğitim Hedefleri: Nedensel Dil Modelleme

#### 3.2.1 Sonraki Token Tahmini

Decoder-only modellerin temel eğitim hedefi, önceki token'lardan sonraki token'ı tahmin etmektir:

```python
def causal_language_modeling_loss(logits, targets):
    """
    Nedensel LM kayıp hesaplaması
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    """
    # Hedefleri kaydır: sonraki token'ı tahmin et
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    
    # Kayıp hesaplaması için düzleştir
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Çapraz entropi kaybı
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss

class CausalLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, hidden_states):
        return self.dense(hidden_states)
```

#### 3.2.2 Gelişmiş Eğitim Stratejileri

**Öğretmen Zorlama vs Serbest Çalıştırma:**
- Eğitim sırasında öğretmen zorlama (gerçek girdi)
- Çıkarım sırasında otoregresif üretim

**Örnekleme Stratejileri:**
- Açgözlü kod çözme
- Işın araması
- Top-k örnekleme
- Top-p (çekirdek) örnekleme
- Sıcaklık ölçeklendirme

```python
def generate_with_sampling(model, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
    """Gelişmiş örnekleme stratejisi uygulaması"""
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Top-k filtreleme
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Top-p filtreleme
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sonraki token'ı örnekle
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Bitiş token'ını kontrol et
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return generated
```

### 3.3 Decoder-Only Model Gelişimi

#### 3.3.1 GPT Ailesi Gelişimi

**GPT-1 (2018):** Dil anlama için Transformer decoder
- 117M parametreler
- Denetimsiz ön eğitim + denetimli ince ayar
- 512 bağlam uzunluğu

**GPT-2 (2019):** Ölçek ve yetenek genişlemesi
- 124M ile 1.5B parametreler arası
- "Sıfır atış görev transferi" gösterimi
- Potansiyel kötüye kullanım nedeniyle tartışmalı gecikmeli yayın

**GPT-3 (2020):** Az atışlı öğrenmenin ortaya çıkışı
- 175B parametreler
- Bağlam içi öğrenme yetenekleri
- Minimal göreve özgü ince ayar gereksinimi

**GPT-4 (2023):** Çok modlu ve gelişmiş akıl yürütme
- Bilinmeyen mimari detayları (tahmini 1.7T parametreler)
- Görsel yetenekler
- Sofistike akıl yürütme ve hizalama

#### 3.3.2 Alternatif Decoder-Only Mimarileri

**LLaMA Serisi:** Meta'nın verimli büyük dil modelleri
- Parametre-verimli tasarım
- Daha küçük ölçeklerde güçlü performans
- Açık araştırma topluluğu etkisi

**PaLM:** Google'ın yol dil modeli
- 540B parametreler
- Düşünce zinciri akıl yürütme yetenekleri

**Claude Serisi:** Anthropic'in anayasal AI yaklaşımı
- Güvenlik odaklı eğitim
- Uzun bağlam yetenekleri

**Mistral/Mixtral:** Uzmanların karışımı mimarileri
- Uzman yönlendirmesi ile verimli hesaplama
- Güçlü performans-verimlilik dengesi

---

## 4. Mimari Karşılaştırma: Temel Farklar Analizi

### 4.1 Dikkat Kalıbı Analizi

#### 4.1.1 Bilgi Akış Kalıpları

**Encoder-Only: Çift Yönlü Bilgi Akışı**
```
Token_i şunlara dikkat edebilir: [Token_1, Token_2, ..., Token_N]
Bilgi Yoğunluğu: Tam O(N²) etkileşimler
Bağlam Kullanımı: Maksimum çift yönlü bağlam
```

**Decoder-Only: Nedensel Bilgi Akışı**
```
Token_i şunlara dikkat edebilir: [Token_1, Token_2, ..., Token_i]
Bilgi Yoğunluğu: Üçgen O(N²/2) etkileşimler
Bağlam Kullanımı: Sadece önceki bağlam
```

#### 4.1.2 Dikkat Görselleştirme Karşılaştırması

```python
def visualize_attention_patterns():
    """Encoder vs Decoder dikkat kalıbı görselleştirmesi"""
    seq_len = 8
    
    # Encoder dikkat kalıbı (çift yönlü)
    encoder_attention = torch.ones(seq_len, seq_len)
    
    # Decoder dikkat kalıbı (nedensel)
    decoder_attention = torch.tril(torch.ones(seq_len, seq_len))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(encoder_attention, cmap='Blues')
    ax1.set_title('Encoder: Çift Yönlü Dikkat')
    ax1.set_xlabel('Anahtar Pozisyonları')
    ax1.set_ylabel('Sorgu Pozisyonları')
    
    ax2.imshow(decoder_attention, cmap='Reds')
    ax2.set_title('Decoder: Nedensel Dikkat')
    ax2.set_xlabel('Anahtar Pozisyonları')
    ax2.set_ylabel('Sorgu Pozisyonları')
    
    plt.tight_layout()
    plt.show()
```

### 4.2 Hesaplama Karmaşıklığı Analizi

#### 4.2.1 Eğitim Karmaşıklığı

**Encoder-Only Modeller:**
- **Zaman Karmaşıklığı:** Katman başına O(N² × d_model)
- **Alan Karmaşıklığı:** O(N² + N × d_model)
- **Paralelleştirme:** Eğitim sırasında tam dizi paralelliği
- **Bellek Kalıbı:** Statik bellek kullanımı

**Decoder-Only Modeller:**
- **Zaman Karmaşıklığı:** Katman başına O(N² × d_model) (eğitim)
- **Alan Karmaşıklığı:** O(N² + N × d_model)
- **Paralelleştirme:** Eğitim sırasında tam dizi paralelliği
- **Bellek Kalıbı:** Çıkarım sırasında KV önbellek optimizasyonu

#### 4.2.2 Çıkarım Karmaşıklığı

```python
class InferenceComplexityAnalysis:
    def __init__(self, seq_len, d_model, num_layers):
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_layers = num_layers
    
    def encoder_inference_cost(self):
        """Encoder: Tüm dizi için tek ileri geçiş"""
        attention_cost = self.seq_len ** 2 * self.d_model * self.num_layers
        feed_forward_cost = self.seq_len * self.d_model * 4 * self.d_model * self.num_layers
        return attention_cost + feed_forward_cost
    
    def decoder_inference_cost(self, generated_length):
        """Decoder: Aşamalı üretim maliyeti"""
        total_cost = 0
        for pos in range(1, generated_length + 1):
            # Her pozisyonda, önceki pozisyonlara dikkat
            attention_cost = pos * self.d_model * self.num_layers
            feed_forward_cost = self.d_model * 4 * self.d_model * self.num_layers
            total_cost += attention_cost + feed_forward_cost
        return total_cost
    
    def compare_costs(self, decoder_gen_length):
        encoder_cost = self.encoder_inference_cost()
        decoder_cost = self.decoder_inference_cost(decoder_gen_length)
        
        return {
            'encoder_flops': encoder_cost,
            'decoder_flops': decoder_cost,
            'oran': decoder_cost / encoder_cost
        }
```

### 4.3 Bellek Kullanım Kalıpları

#### 4.3.1 KV Önbellek Optimizasyonu (Decoder-Only)

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, head_dim):
        self.cache_k = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.cache_v = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.seq_len = 0
    
    def update(self, key, value):
        """Otoregresif üretim için aşamalı KV önbellek güncellemesi"""
        batch_size, num_heads, _, head_dim = key.shape
        
        self.cache_k[:batch_size, :, self.seq_len] = key.squeeze(2)
        self.cache_v[:batch_size, :, self.seq_len] = value.squeeze(2)
        self.seq_len += 1
        
        return self.cache_k[:batch_size, :, :self.seq_len], self.cache_v[:batch_size, :, :self.seq_len]
    
    def reset(self):
        self.seq_len = 0
```

#### 4.3.2 Bellek Verimliliği Karşılaştırması

**Encoder-Only Bellek Profili:**
- **Statik Tahsis:** Girdi dizi uzunluğuna dayalı sabit bellek
- **Tepe Kullanımı:** Dikkat hesaplaması sırasında O(N²)
- **Optimizasyon:** Gradyan kontrol noktası, karışık hassasiyet

**Decoder-Only Bellek Profili:**
- **Dinamik Tahsis:** Üretim uzunluğu ile büyüyen bellek
- **Tepe Kullanımı:** Uzun diziler için KV önbellek baskın
- **Optimizasyon:** KV önbellek sıkıştırma, sayfalama stratejileri

---

## 5. Eğitim Stratejileri ve Metodolojileri

### 5.1 Ön Eğitim Yaklaşımları

#### 5.1.1 Encoder-Only Ön Eğitimi

**Veri Hazırlama Stratejileri:**
```python
class EncoderPretrainingDataset:
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_training_batch(self, batch_texts):
        """MLM + NSP batch oluşturma"""
        mlm_inputs, mlm_labels = self.create_mlm_examples(batch_texts)
        nsp_inputs, nsp_labels = self.create_nsp_examples(batch_texts)
        
        return {
            'input_ids': mlm_inputs,
            'mlm_labels': mlm_labels,
            'nsp_labels': nsp_labels,
            'attention_mask': (mlm_inputs != self.tokenizer.pad_token_id)
        }
    
    def create_mlm_examples(self, texts):
        # MLM veri oluşturma uygulaması
        pass
    
    def create_nsp_examples(self, texts):
        # NSP veri oluşturma uygulaması
        pass
```

**Eğitim Döngüsü Optimizasyonu:**
```python
def train_encoder_model(model, dataloader, optimizer, scheduler, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['mlm_labels']
            )
            
            mlm_loss = outputs.loss
            
            # NSP kaybı (eğer uygulanabilirse)
            if 'nsp_labels' in batch:
                nsp_outputs = model.classifier(outputs.pooler_output)
                nsp_loss = F.cross_entropy(nsp_outputs, batch['nsp_labels'])
                total_loss = mlm_loss + nsp_loss
            else:
                total_loss = mlm_loss
                
            total_loss.backward()
            optimizer.step()
            scheduler.step()
```

#### 5.1.2 Decoder-Only Ön Eğitimi

**Veri İşleme Hattı:**
```python
class DecoderPretrainingDataset:
    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_training_batch(self, batch_texts):
        """Nedensel LM için veri hazırlama"""
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        labels = input_ids.clone()
        
        # Ped token'larını -100 ile maskele (kayıp hesaplanmayacak)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encoded['attention_mask']
        }
```

### 5.2 İnce Ayar Stratejileri

#### 5.2.1 Encoder-Only İnce Ayar Yaklaşımları

**Görev Spesifik Başlıklar:**
```python
class TaskSpecificHeads:
    """Encoder modeller için farklı görev başlıkları"""
    
    @staticmethod
    def classification_head(hidden_size, num_labels):
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    @staticmethod
    def token_classification_head(hidden_size, num_labels):
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    @staticmethod
    def question_answering_head(hidden_size):
        return nn.Linear(hidden_size, 2)  # Başlangıç ve bitiş pozisyonları
```

**Adaptif İnce Ayar Teknikleri:**
- **Discriminative Fine-tuning:** Farklı katmanlar için farklı öğrenme hızları
- **Gradual Unfreezing:** Katmanların aşamalı olarak çözülmesi
- **Task-specific Layer Normalization:** Görev başına ayrı normalizasyon

#### 5.2.2 Decoder-Only İnce Ayar Yaklaşımları

**Talimat İnce Ayarı (Instruction Tuning):**
```python
def format_instruction_data(instruction, input_text, output_text):
    """Talimat takibi için veri formatı"""
    if input_text:
        prompt = f"### Talimat:\n{instruction}\n\n### Girdi:\n{input_text}\n\n### Yanıt:\n"
    else:
        prompt = f"### Talimat:\n{instruction}\n\n### Yanıt:\n"
    
    return prompt + output_text

class InstructionTuningDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        item = self.data[idx]
        formatted_text = format_instruction_data(
            item['instruction'],
            item.get('input', ''),
            item['output']
        )
        
        encoded = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=2048,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }
```

**Parametre Verimli İnce Ayar (PEFT):**
```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation katmanı"""
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Düşük sıralı matrisler
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Başlatma
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, original_weight):
        # Orijinal ağırlık + LoRA adaptasyonu
        lora_output = x @ self.lora_A @ self.lora_B * self.scaling
        original_output = x @ original_weight
        return original_output + lora_output
```

### 5.3 Optimizasyon Teknikleri

#### 5.3.1 Gradyan Birikimi ve Karışık Hassasiyet

```python
class OptimizationStrategies:
    @staticmethod
    def setup_mixed_precision_training(model, optimizer):
        """Karışık hassasiyet eğitimi kurulumu"""
        from torch.cuda.amp import GradScaler, autocast
        
        scaler = GradScaler()
        
        def training_step(batch):
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            return loss.item()
        
        return training_step
    
    @staticmethod
    def gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4):
        """Gradyan birikimi ile etkili batch boyutu artırma"""
        model.train()
        accumulated_loss = 0
        
        for i, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        return accumulated_loss
```

#### 5.3.2 Öğrenme Hızı Programlama

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Isınma ile kosinüs öğrenme hızı programı"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## 6. Performans Karşılaştırması ve Kullanım Senaryoları

### 6.1 Görev Bazlı Performans Analizi

#### 6.1.1 Anlama Görevleri

**Encoder-Only Modellerin Üstün Olduğu Alanlar:**

1. **Metin Sınıflandırma:**
   - Duygu analizi
   - Konu sınıflandırma
   - Spam tespiti
   - Niyet tanıma

2. **Token Seviyesi Görevler:**
   - Adlandırılmış varlık tanıma (NER)
   - Parça etiketleme (POS tagging)
   - Sözcük anlamı belirsizliği giderme

3. **Cümle Çifti Görevleri:**
   - Doğal dil çıkarımı (NLI)
   - Anlamsal benzerlik
   - Soru-cevap eşleştirme

**Performans Metrikleri:**
```python
class EncoderPerformanceMetrics:
    @staticmethod
    def evaluate_classification(model, dataloader, num_labels):
        """Sınıflandırma görevleri için değerlendirme"""
        model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        
        # Metrik hesaplama
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
```

#### 6.1.2 Üretim Görevleri

**Decoder-Only Modellerin Üstün Olduğu Alanlar:**

1. **Metin Üretimi:**
   - Yaratıcı yazım
   - Kod üretimi
   - Metin tamamlama
   - Diyalog sistemleri

2. **Bağlam İçi Öğrenme:**
   - Few-shot öğrenme
   - Zero-shot görev transferi
   - Talimat takibi

3. **Akıl Yürütme Görevleri:**
   - Matematiksel problem çözme
   - Mantıksal çıkarım
   - Çok adımlı akıl yürütme

**Üretim Kalitesi Metrikleri:**
```python
class GenerationMetrics:
    @staticmethod
    def calculate_perplexity(model, dataloader):
        """Perplexity hesaplama"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                
                # Maskelenmemiş token sayısını hesapla
                num_tokens = (batch['labels'] != -100).sum()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity
    
    @staticmethod
    def calculate_bleu_score(references, hypotheses):
        """BLEU skoru hesaplama"""
        from nltk.translate.bleu_score import corpus_bleu
        return corpus_bleu(references, hypotheses)
```

### 6.2 Hesaplama Verimliliği Karşılaştırması

#### 6.2.1 Eğitim Verimliliği

```python
class EfficiencyBenchmark:
    def __init__(self, model_type, model_size, batch_size, seq_length):
        self.model_type = model_type
        self.model_size = model_size
        self.batch_size = batch_size
        self.seq_length = seq_length
    
    def measure_training_throughput(self, model, dataloader, num_steps=100):
        """Eğitim verimi ölçümü (token/saniye)"""
        model.train()
        start_time = time.time()
        total_tokens = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
                
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # İşlenen token sayısı
            total_tokens += batch['input_ids'].numel()
        
        elapsed_time = time.time() - start_time
        throughput = total_tokens / elapsed_time
        
        return {
            'tokens_per_second': throughput,
            'batches_per_second': num_steps / elapsed_time,
            'avg_batch_time': elapsed_time / num_steps
        }
```

#### 6.2.2 Çıkarım Verimliliği

```python
def compare_inference_efficiency():
    """Encoder vs Decoder çıkarım verimliliği karşılaştırması"""
    
    # Encoder modeli - tek geçişte tüm çıktılar
    def encoder_inference(model, input_ids):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state
        inference_time = time.time() - start_time
        return embeddings, inference_time
    
    # Decoder modeli - otoregresif üretim
    def decoder_inference(model, input_ids, max_new_tokens=50):
        start_time = time.time()
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        inference_time = time.time() - start_time
        return generated, inference_time
    
    return {
        'encoder_constant_time': True,
        'decoder_linear_time': True,
        'encoder_parallelizable': True,
        'decoder_sequential': True
    }
```

### 6.3 Pratik Kullanım Kılavuzları

#### 6.3.1 Model Seçim Kriterleri

**Encoder-Only Modelleri Tercih Etme Durumları:**

1. **Sabit uzunlukta çıktı gerektiren görevler**
2. **Çift yönlü bağlam kritik olan durumlar**
3. **Gerçek zamanlı sınıflandırma sistemleri**
4. **Kaynak kısıtlı ortamlar (çıkarım için)**

**Decoder-Only Modelleri Tercih Etme Durumları:**

1. **Değişken uzunlukta çıktı gerektiren görevler**
2. **Yaratıcı içerik üretimi**
3. **İnteraktif sohbet sistemleri**
4. **Genel amaçlı dil modelleme**

#### 6.3.2 Hibrit Yaklaşımlar

```python
class HybridArchitecture(nn.Module):
    """Encoder ve Decoder güçlerini birleştiren hibrit yaklaşım"""
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = EncoderModel(encoder_config)
        self.decoder = DecoderModel(decoder_config)
        self.cross_attention = CrossAttentionLayer(
            encoder_config.hidden_size,
            decoder_config.hidden_size
        )
    
    def forward(self, input_ids, decoder_input_ids=None):
        # Encoder çıktılarını al
        encoder_outputs = self.encoder(input_ids)
        
        if decoder_input_ids is not None:
            # Decoder'ı encoder çıktılarıyla besle
            decoder_outputs = self.decoder(
                decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state
            )
            return decoder_outputs
        
        return encoder_outputs
```

---

## 7. Gelecek Perspektifleri ve Araştırma Yönleri

### 7.1 Mimari İnovasyonlar

#### 7.1.1 Verimlilik Odaklı Gelişmeler

**Seyrek Dikkat Mekanizmaları:**
- Flash Attention: Bellek verimli dikkat hesaplama
- Sliding Window Attention: Yerel bağlam penceresi
- Dilated Attention: Aralıklı dikkat kalıpları

**Dinamik Hesaplama:**
- Mixture of Experts (MoE): Koşullu hesaplama
- Early Exit: Dinamik derinlik
- Adaptive Computation Time: Değişken hesaplama süresi

#### 7.1.2 Uzun Bağlam Yetenekleri

```python
class LongContextInnovations:
    """Uzun bağlam işleme için yenilikçi yaklaşımlar"""
    
    @staticmethod
    def rotary_position_embedding(seq_len, dim):
        """RoPE: Göreli pozisyon kodlaması"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    @staticmethod
    def alibi_attention_bias(seq_len, num_heads):
        """ALiBi: Pozisyon bilgisini dikkat içinde kodlama"""
        slopes = torch.tensor([2**(-8/num_heads * i) for i in range(num_heads)])
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * positions.unsqueeze(0)
        return alibi
```

### 7.2 Eğitim Paradigması Değişimleri

#### 7.2.1 Kendi Kendini Denetleyen Öğrenme Gelişmeleri

**Gelişmiş Maskeleme Stratejileri:**
- Span maskeleme
- Geometrik maskeleme
- Anlamsal maskeleme

**Kontrastif Öğrenme:**
```python
class ContrastiveLearning:
    @staticmethod
    def simclr_loss(embeddings1, embeddings2, temperature=0.1):
        """SimCLR tarzı kontrastif kayıp"""
        batch_size = embeddings1.shape[0]
        
        # Normalize
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Benzerlik matrisi hesapla
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / temperature
        
        # Pozitif çiftler köşegen üzerinde
        labels = torch.arange(batch_size).to(embeddings1.device)
        
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
```

#### 7.2.2 Çok Görevli ve Çok Modlu Öğrenme

**Unified Model Yaklaşımları:**
- T5 tarzı metin-metin formatı
- Görüntü-metin birleşik modelleme
- Konuşma-metin entegrasyonu

### 7.3 Pratik Uygulamalar ve Endüstri Trendleri

#### 7.3.1 Uygulama Alanları

**Encoder-Only Uygulamaları:**
- Gerçek zamanlı içerik moderasyonu
- Otomatik etiketleme sistemleri
- Anlamsal arama motorları
- Müşteri niyet analizi

**Decoder-Only Uygulamaları:**
- Kod asistanları
- İçerik üretim platformları
- Eğitim ve öğretim asistanları
- Yaratıcı yazım araçları

#### 7.3.2 Deployment Stratejileri

```python
class DeploymentOptimization:
    """Model dağıtımı için optimizasyon teknikleri"""
    
    @staticmethod
    def quantize_model(model, quantization_config):
        """Model kuantizasyonu"""
        import torch.quantization as quantization
        
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        prepared_model = quantization.prepare(model, inplace=False)
        quantized_model = quantization.convert(prepared_model, inplace=False)
        
        return quantized_model
    
    @staticmethod
    def optimize_for_inference(model):
        """Çıkarım için optimizasyon"""
        model.eval()
        
        # JIT derleme
        example_input = torch.randint(0, 1000, (1, 128))
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimizasyonları uygula
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        return traced_model
```

---

## 8. Sonuç

Encoder-only ve decoder-only mimariler, modern NLP'nin iki temel direğini oluşturmaktadır. Her birinin kendine özgü güçlü yönleri ve optimal kullanım senaryoları bulunmaktadır:

**Encoder-Only Modeller:**
- Anlama görevlerinde üstün performans
- Verimli çıkarım için sabit zaman karmaşıklığı
- Çift yönlü bağlam kullanımı
- Sınıflandırma ve etiketleme görevleri için ideal

**Decoder-Only Modeller:**
- Üretim görevlerinde esneklik
- Bağlam içi öğrenme yetenekleri
- Ölçeklenebilir mimari
- Genel amaçlı dil modelleme için optimal

Gelecekte, bu iki paradigmanın güçlü yönlerini birleştiren hibrit yaklaşımların ve görev-spesifik optimizasyonların önem kazanması beklenmektedir. Araştırmacılar ve uygulayıcılar, spesifik kullanım durumlarına göre en uygun mimariyi seçmeli ve sürekli gelişen bu alandaki yenilikleri takip etmelidirler.

### Kaynaklar ve İleri Okuma

1. Vaswani, A., et al. (2017). "Attention is All You Need"
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training"
4. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
5. Brown, T., et al. (2020). "Language Models are Few-Shot Learners"
6. Raffel, C., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
