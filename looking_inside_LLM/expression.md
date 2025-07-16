
# Looking Inside LLM with Expressions

## 🎯 ANA KAVRAMLAR (5 Pillar)

### 1. **Next Token Prediction** ✅ (LLM'nin Özü)

*"LLM'ler sadece bir sonraki kelimeyi tahmin eden sistemlerdir"*

- **PDF Bağlantısı:** Sayfa 2-8 (Generation süreci)
- **Sunum Süresi:** 8-10 dakika
- **Demo:** Canlı ChatGPT kelime kelime yazması

---

### 2. **Autoregressive Model** ✅ (LLM'nin Çalışma Şekli)

*"Kendi çıktısını tekrar kendine input olarak verir"*

- **PDF Bağlantısı:** Sayfa 3-4 (Token ekleme döngüsü)
- **Sunum Süresi:** 8-10 dakika  
- **Demo:** "Hikaye anlatma oyunu" canlı demo

---

### 3. **Attention is All You Need** (Transformer'ın Kalbi)

*"Dikkat mekanizması her şeyin anahtarı - hangi kelimelerin önemli olduğunu anlar"*

- **Analoji:** Kokteyl partisinde belirli konuşmalara odaklanmak
- **PDF Bağlantısı:** Sayfa 14-22 (Attention mechanism detayları)
- **Sunum değeri:** Transformer'ın temel prensibi
- **Demo:** "The cat sat on the mat" attention visualization

---

### 4. **In-Context Learning** (Prompting'in Sihri)

*"Parametrelerini değiştirmeden, sadece örnek vererek öğretebilirsiniz"*

- **Günlük analoji:** Arkadaşınıza örnekle anlatmak gibi
- **PDF Bağlantısı:** Sayfa 25-28 (Prompting techniques)
- **Sunum değeri:** Few-shot learning'in gücü
- **Demo:** Zero-shot vs Few-shot comparison

---

### 5. **Foundation Models** (Yeni Bölüm - Versatility)

*"Tek model, binlerce farklı görev - çok amaçlı AI temeli"*

- **Analoji:** İsviçre çakısı gibi
- **Sunum değeri:** Versatility vurgusu
- **Örnekler:**
  - Aynı GPT-4: Email yazer, kod yazer, çevirir, analiz yapar
  - Specialized model'ler vs Foundation model karşılaştırması
- **Demo:** Bir model, farklı görevler showcase

---

## 🔤 TOKENIZER BÖLÜMÜ

### 6. **Text-to-Numbers, Numbers-to-Text Magic**

*"LLM'ler sayılarla çalışır, ama biz kelimelerle konuşuruz - tokenizer köprü görevi görür"*

**Ana İfade:**

- **Encode:** "Merhaba" → [45, 123, 789]
- **Decode:** [156, 278, 934] → "dünya"

**Analojiler:**

- **Çevirmen:** İnsan dili ↔ Bilgisayar dili
- **Morse Kodu:** Kelimeler ↔ Nokta/tire
- **Şarkı Notaları:** Müzik ↔ Nota yazısı

**PDF Bağlantısı:** Sayfa 30-34
**Kritik Nokta:** "Tokenization kalitesi = Model performansı"
**Demo:** "Türkçe vs İngilizce tokenization farklılıkları"
