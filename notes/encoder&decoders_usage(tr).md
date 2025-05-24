# Encoder-Only vs Decoder-Only: Neden Ayrı Kullanılıyor?

## 1. Temel Fark: Dikkat Mekanizmasının Yönü

### 1.1 Encoder: Çift Yönlü Dikkat (Bidirectional Attention)
```
Cümle: "Bugün hava çok güzel"

Encoder'da "hava" kelimesi:
- "Bugün" kelimesine bakabilir (GERİ)
- "çok" kelimesine bakabilir (İLERİ)  
- "güzel" kelimesine bakabilir (İLERİ)

Her kelime, cümledeki TÜM kelimeleri görebilir!
```

### 1.2 Decoder: Tek Yönlü Dikkat (Causal/Unidirectional Attention)
```
Cümle: "Bugün hava çok güzel"

Decoder'da "hava" kelimesi:
- "Bugün" kelimesine bakabilir (GERİ)
- "çok" kelimesine BAKAMAZ (İLERİ)
- "güzel" kelimesine BAKAMAZ (İLERİ)

Her kelime, sadece KENDİNDEN ÖNCEKİ kelimeleri görebilir!
```

## 2. Neden Bu Fark Kritik?

### 2.1 Encoder'ın Çift Yönlü Olma Nedeni: Anlama Odaklı

**Örnek: Cümle Anlamını Belirleme**
```
"Banka yanında duran adam" cümlesinde "banka" kelimesinin anlamı:

Encoder:
- İleriye bakıp "yanında duran adam" görür
- Geriye bakıp başka bağlam arar
- SONUÇ: "Oturma yeri" anlamını çıkarır (nehir kenarı değil)

Decoder:
- Sadece geriye bakabilir
- "yanında duran adam" bilgisine erişemez
- SONUÇ: Anlamı tam belirleyemez
```

### 2.2 Decoder'ın Tek Yönlü Olma Nedeni: Üretim Odaklı

**Örnek: Metin Üretimi**
```python
# Decoder metin üretimi
girdi = "Yarın hava"
# Decoder tahmin eder: "güzel"
# Sonra tahmin eder: "olacak"

# Eğer decoder ileriye bakabilseydi:
# "güzel" kelimesini tahmin ederken zaten "olacak"ı görürdü
# Bu hile yapmak olurdu! Gelecekteki cevabı görmüş olurdu.
```

## 3. Matematiksel Fark: Dikkat Maskeleri

### 3.1 Encoder Dikkat Maskesi
```python
# Encoder: Tam dikkat (herkes herkesi görür)
encoder_mask = [
    [1, 1, 1, 1],  # 1. kelime herkesi görür
    [1, 1, 1, 1],  # 2. kelime herkesi görür
    [1, 1, 1, 1],  # 3. kelime herkesi görür
    [1, 1, 1, 1]   # 4. kelime herkesi görür
]
```

### 3.2 Decoder Dikkat Maskesi
```python
# Decoder: Üçgen maske (sadece geçmişi görür)
decoder_mask = [
    [1, 0, 0, 0],  # 1. kelime sadece kendini görür
    [1, 1, 0, 0],  # 2. kelime 1. ve kendini görür
    [1, 1, 1, 0],  # 3. kelime 1., 2. ve kendini görür
    [1, 1, 1, 1]   # 4. kelime herkesi görür
]
```

## 4. Eğitim Farklılıkları

### 4.1 Encoder Eğitimi: Maskeli Dil Modelleme (MLM)
```python
# BERT tarzı eğitim
orijinal = "Bugün hava çok güzel"
maskelenmiş = "Bugün [MASK] çok güzel"

# Model görevi: [MASK] yerine "hava" tahmin et
# Avantaj: Hem öncesini hem sonrasını kullanabilir
```

### 4.2 Decoder Eğitimi: Sonraki Kelime Tahmini
```python
# GPT tarzı eğitim
girdi = "Bugün hava"
hedef = "çok"

# Model görevi: "Bugün hava" dan sonra "çok" geldiğini öğren
# Kısıtlama: Sadece önceki kelimeleri kullanabilir
```

## 5. Kullanım Alanı Farklılıkları

### 5.1 Encoder İçin İdeal Görevler
```
1. Sınıflandırma: "Bu e-posta spam mı?"
   - Tüm metni görmek gerekir
   - Baştan sona analiz

2. Duygu Analizi: "Bu yorum olumlu mu olumsuz mu?"
   - Cümlenin tamamını anlamak kritik
   - "İyi... ama kötü" gibi yapıları yakalar

3. Soru Cevaplama: "Metinde X nerede geçiyor?"
   - İleri geri bakarak bağlamı anlar
```

### 5.2 Decoder İçin İdeal Görevler
```
1. Metin Üretimi: "Bir hikaye yaz"
   - Kelime kelime üretim
   - Doğal akış

2. Kod Tamamlama: "def calculate_"
   - Sonraki token'ı tahmin
   - Otoregresif üretim

3. Sohbet: "Merhaba, nasılsın?"
   - Ardışık yanıt üretimi
   - Bağlama dayalı devam
```

## 6. Performans ve Verimlilik Karşılaştırması

### 6.1 Çıkarım (Inference) Hızı
```python
# Encoder: Sabit zaman
def encoder_inference(text):
    # Tüm metni bir kerede işle
    embeddings = model(text)  # Tek geçiş
    return embeddings

# Decoder: Lineer zaman
def decoder_inference(prompt, max_length=50):
    output = prompt
    for i in range(max_length):
        # Her kelime için yeni bir geçiş
        next_token = model(output)
        output += next_token
    return output
```

### 6.2 Bellek Kullanımı
```
Encoder:
- Girdi uzunluğu kadar bellek
- Sabit bellek kullanımı
- Tahmin edilebilir

Decoder:
- Üretilen her token için artan bellek
- KV-cache biriktirme
- Dinamik bellek yönetimi
```

## 7. Neden Her İkisini Birden Kullanmıyoruz?

### 7.1 Orijinal Transformer: Encoder + Decoder
```
Çeviri görevi için:
Encoder: Kaynak dili anla (Fransızca)
Decoder: Hedef dili üret (İngilizce)

Ancak bu yaklaşım:
- Daha karmaşık
- Daha fazla parametre
- Özel görevler için gereksiz
```

### 7.2 Uzmanlaşmanın Avantajları
```
Encoder-Only (BERT):
✓ Anlama görevlerinde mükemmel
✓ Daha basit mimari
✓ Verimli fine-tuning

Decoder-Only (GPT):
✓ Üretim görevlerinde mükemmel
✓ Few-shot learning yeteneği
✓ Genel amaçlı kullanım
```

## 8. Somut Örneklerle Farklar

### 8.1 Aynı Cümleyi İşleme
```python
cümle = "Kedi masanın üstünde uyuyor"

# Encoder perspektifi
encoder_temsil = {
    "Kedi": [kedi + masa + üst + uyuyor],     # Tüm bağlam
    "masanın": [kedi + masa + üst + uyuyor],  # Tüm bağlam
    "üstünde": [kedi + masa + üst + uyuyor],  # Tüm bağlam
    "uyuyor": [kedi + masa + üst + uyuyor]    # Tüm bağlam
}

# Decoder perspektifi
decoder_temsil = {
    "Kedi": [kedi],                            # Sadece kendisi
    "masanın": [kedi + masa],                  # Öncekiler
    "üstünde": [kedi + masa + üst],           # Öncekiler
    "uyuyor": [kedi + masa + üst + uyuyor]    # Tüm öncekiler
}
```

### 8.2 Görev Performansı Örneği
```python
# Görev: "Bu cümlede hangi hayvan var?"

# Encoder yaklaşımı:
def encoder_approach(sentence):
    # Tüm cümleyi bir kerede analiz et
    full_context = encode_bidirectional(sentence)
    # "Kedi" kelimesini bulmak için tüm bağlamı kullan
    return extract_animal(full_context)

# Decoder yaklaşımı:
def decoder_approach(sentence):
    # Kelime kelime oku
    for word in sentence:
        # Sadece önceki kelimeleri kullanarak tahmin yap
        if is_animal(word, previous_context):
            return word
```

## 9. Hibrit Kullanım Senaryoları

### 9.1 Ne Zaman Her İkisi de Gerekli?
```
1. Makine Çevirisi:
   - Encoder: Kaynak dili anla
   - Decoder: Hedef dili üret

2. Soru Cevaplama + Açıklama:
   - Encoder: Soruyu ve metni anla
   - Decoder: Detaylı açıklama üret

3. Özetleme:
   - Encoder: Uzun metni anla
   - Decoder: Kısa özet üret
```

## 10. Pratik Seçim Rehberi

### Encoder-Only Seç Eğer:
- ✓ Sabit boyutlu çıktı istiyorsan
- ✓ Sınıflandırma yapacaksan
- ✓ Anlama odaklı görevlerin varsa
- ✓ Hızlı inference gerekiyorsa

### Decoder-Only Seç Eğer:
- ✓ Değişken uzunlukta çıktı istiyorsan
- ✓ Metin üreteceksen
- ✓ Few-shot learning yapacaksan
- ✓ Genel amaçlı model istiyorsan

## Özet

**Encoder ve Decoder ayrımının temel nedeni görev optimizasyonudur:**

1. **Encoder**: "Anlamaya" optimize - Çift yönlü dikkat sayesinde zengin bağlamsal temsiller
2. **Decoder**: "Üretmeye" optimize - Tek yönlü dikkat sayesinde doğal metin akışı

Her ikisi de Transformer mimarisini kullanır ama farklı amaçlar için farklı kısıtlamalar uygular. Bu uzmanlaşma, her bir model tipinin kendi alanında daha başarılı olmasını sağlar.
