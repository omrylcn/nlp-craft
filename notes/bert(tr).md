# BERT: Kapsamlı Teorik ve Pratik İnceleme

## 1. BERT'in Teorik Temelleri

### 1.1 BERT Nedir ve Neden Önemlidir?

BERT (Bidirectional Encoder Representations from Transformers), 2018 yılında Google tarafından geliştirilen ve NLP alanında devrim yaratan bir dil modelidir. Önceki modellerin aksine, BERT metindeki kelimeleri hem soldan sağa hem de sağdan sola (çift yönlü) analiz eder, bu da bağlamı daha iyi anlamasını sağlar.

BERT'in en önemli katkısı, kelimelerin bağlama bağlı temsillerini öğrenebilmesidir. Örneğin, "banka" kelimesinin "nehir bankası" ve "para bankası" ifadelerindeki farklı anlamlarını ayırt edebilir.

### 1.2 Mimari Yapısı

BERT, Transformer mimarisinin kodlayıcı (encoder) kısmını temel alır. Temel BERT modelleri iki boyutta gelir:

- **BERT-Base**: 12 transformer katmanı, 12 dikkat başlığı, 768 gizli boyut (toplam 110M parametre)
- **BERT-Large**: 24 transformer katmanı, 16 dikkat başlığı, 1024 gizli boyut (toplam 340M parametre)

Her BERT katmanı şunlardan oluşur:

- Multi-head self-attention mekanizması: Her başlık, girişteki farklı ilişki türlerine odaklanabilir
- Feed-forward sinir ağları: Doğrusal dönüşümler ve aktivasyon fonksiyonları içerir
- Bağlantı kalıntıları (residual connections): Gradyan akışını kolaylaştırır
- Katman normalizasyonu: Eğitimi stabilize eder

BERT varyasyonlarının karşılaştırması:

| Model Versiyonu | Kodlayıcı Katman Sayısı | Gizli Katman Boyutu | Dikkat Başlığı Sayısı | Parametre Sayısı |
|-----------------|-------------------------|---------------------|----------------------|-----------------|
| BERT-Tiny       | 2                       | 128                 | 2                    | ~4M             |
| BERT-Mini       | 4                       | 256                 | 4                    | ~11M            |
| BERT-Small      | 4                       | 512                 | 8                    | ~29M            |
| BERT-Base       | 12                      | 768                 | 12                   | ~110M           |
| BERT-Large      | 24                      | 1024                | 16                   | ~340M           |

### 1.3 Ön-Eğitim Hedefleri

BERT iki görevle ön-eğitime tabi tutulur:

1. **Masked Language Modeling (MLM)**: Giriş metnindeki kelimelerin %15'i rastgele maskelenir ve model bu maskelenmiş kelimeleri tahmin etmeye çalışır. Bu, modelin metni çift yönlü olarak anlamasını sağlar.

2. **Next Sentence Prediction (NSP)**: Modele iki cümle verilir ve bu cümlelerin metinde ardışık olup olmadığını tahmin etmesi beklenir. Bu, cümleler arası ilişkileri anlama yeteneğini geliştirir.

### 1.4 Giriş Temsilleri

BERT'e girdi olarak verilen her token üç temsil katmanının toplamından oluşur:

- **Token Embeddings**: Kelimenin anlamsal temsili
- **Segment Embeddings**: Cümlenin hangi parçasına ait olduğunu belirten temsil (0 veya 1)
- **Position Embeddings**: Kelimenin cümle içindeki konumunu belirten temsil

Bu üç embedding vektörü aşağıdaki şekilde birleştirilir:

```
Final Embedding = Token Embedding + Segment Embedding + Position Embedding
```

## 2. Masked Language Modeling'in Derinlemesine İncelenmesi

Masked Language Modeling (MLM), doğal dil işlemede dil modellerini eğitmek için kullanılan güçlü bir tekniktir. Bu yaklaşım, BERT gibi Transformer tabanlı modellerin özünü oluşturur.

### 2.1 MLM vs. Word2Vec

**Word2vec'in Yaklaşımı**:

- **CBOW (Continuous Bag of Words)**: Bağlam kelimelerini kullanarak merkez kelimeyi tahmin eder.
- **Skip-gram**: Merkez kelimeyi kullanarak bağlam kelimelerini tahmin eder.
- Her iki durumda da **sabit boyutlu bir pencere** kullanır (örn. 5 kelimelik pencere).
- Tek yönlü bağlam öğrenimi (soldan sağa veya sağdan sola)
- Her kelime için tek bir statik vektör temsili üretir

**MLM'in Farklı Yaklaşımı**:

- Metin içerisindeki kelimelerin belirli bir yüzdesi (genellikle %15) rastgele maskelenir.
- Maskelenen kelimeler `[MASK]` gibi özel bir token ile değiştirilir.
- Model, tüm metni bir bütün olarak görerek maskelenen kelimeleri tahmin etmeye çalışır.
- Word2vec'ten farklı olarak, MLM sabit bir pencere kullanmak yerine, tüm cümleyi veya paragrafı bağlam olarak kullanır.
- Çift yönlü bağlam öğrenimi (hem önceki hem sonraki kelimeleri dikkate alır)
- Kelimenin bağlama göre değişebilen, bağlama duyarlı temsiller üretir

### 2.2 MLM'in Nasıl Çalıştığı (BERT Örneği)

BERT'teki MLM uygulamasını adım adım inceleyelim:

1. **Maskeleme Süreci:**
   - Girdi metninin tokenlere ayrılması
   - Tokenlerin %15'inin seçilmesi
   - Bu seçilen tokenlerin:
     - %80'i `[MASK]` tokeni ile değiştirilir: "Bugün hava çok güzel" → "Bugün [MASK] çok güzel"
     - %10'u rastgele başka bir kelimeyle değiştirilir: "Bugün hava çok güzel" → "Bugün kitap çok güzel"
     - %10'u aynı kalır (bu, modelin kelimeleri değiştirmemeyi de öğrenmesini sağlar)

2. **Tahmin Süreci:**
   - Model, tüm konteksti kullanarak maskelenen veya değiştirilen kelimeleri tahmin etmeye çalışır
   - Bu süreçte, bir kelimenin öncesindeki ve sonrasındaki tüm kontekst kullanılır
   - Kayıp (loss) fonksiyonu, yalnızca değiştirilen tokenler üzerinden hesaplanır

### 2.3 MLM'in Avantajları

MLM'in geniş bağlam anlayışı, daha zengin ve bağlama duyarlı kelime temsillerinin öğrenilmesini sağlar. Bu, Word2vec'in genellikle yakalayamadığı çok anlamlı kelimelerin (homonyms) farklı anlamlarını ayırt etmeyi mümkün kılar.

Örneğin, "banka" kelimesi "nehir bankası" ve "para bankası" ifadelerinde farklı anlamlara sahiptir. MLM, tüm cümleyi görebildiği için bu farkı öğrenebilir, oysa Word2vec sabit pencere boyutu nedeniyle bu ayrımı yapmakta zorlanabilir.

## 3. Next Sentence Prediction ve Özel Tokenler

### 3.1 Next Sentence Prediction (NSP) Nedir?

NSP, BERT'in ön eğitim sürecindeki ikinci temel hedeftir. Bu görev, modelin dil anlayışını cümle seviyesine yükseltmeyi amaçlar.

**NSP'nin Çalışma Prensibi:**

1. **Veri Hazırlama**:
   - Eğitim verisindeki her örnek için, iki cümle seçilir
   - %50 olasılıkla, gerçekten ardışık olan iki cümle seçilir (positive example)
   - %50 olasılıkla, aynı korpustan rasgele seçilen, birbirleriyle ilişkisiz iki cümle birleştirilir (negative example)

2. **Öğrenme Hedefi**:
   - Model, verilen iki cümlenin gerçekten ardışık olup olmadığını tahmin etmeye çalışır
   - Bu binary sınıflandırma görevi, modelin cümleler arası ilişkileri anlamasını sağlar

Bu eğitim, BERT'e şu yetenekleri kazandırır:

- Cümleler arasındaki tematik tutarlılığı değerlendirebilme
- Metin içindeki konu geçişlerini algılayabilme
- Çıkarımsal ilişkileri anlayabilme

### 3.2 BERT'in Özel Tokenleri: [CLS] ve [SEP]

BERT, girdi metnini işlemek için bazı özel tokenler kullanır:

**[CLS] Tokeni (Classification Token)**

- Her BERT girdisinin **en başına** yerleştirilir
- Başlangıçta anlamsal bir değere sahip değildir
- Transformer'ın öz-dikkat mekanizması sayesinde, tüm cümlenin bütünsel bir temsilini oluşturur
- Son katmanda, [CLS] tokeninin çıktısı (H₀), metin sınıflandırma görevleri için kullanılır
- [CLS] vektörü şu uygulamalarda kullanılır:
  - Siamese BERT: İki cümlenin kosinüs benzerliği hesaplanması
  - Özellik çıkarımı: Cümle temsil vektörü olarak kullanılması
  - Kümeleme: Benzer anlamlı cümlelerin gruplandırılması

**[SEP] Tokeni (Separator Token)**

- BERT'e birden fazla cümle verildiğinde bunları ayırmak için kullanılır
- Bir cümlenin nerede bitip diğerinin nerede başladığını modele gösterir
- İki cümleli görevlerde (NSP, doğal dil çıkarımı, anlam benzerliği vb.) kullanımı zorunludur:

  ```
  [CLS] Bugün hava çok güzel. [SEP] Dışarı çıkıp yürüyüş yapacağım. [SEP]
  ```

- Segment gömmeleri ile kombinasyon halinde kullanılır:
  - İlk cümledeki tüm tokenler (ilk [SEP] dahil): Segment A gömmesi (0)
  - İkinci cümledeki tüm tokenler (ikinci [SEP] dahil): Segment B gömmesi (1)

### 3.3 Özel Tokenlerin Pratik Uygulamaları

**Cümle Çifti Sınıflandırma**

```
[CLS] Kadın bir kitap okuyor. [SEP] Kadın okuma eylemini gerçekleştiriyor. [SEP]
```

BERT, [CLS] tokeninin son temsilini kullanarak "çıkarım", "çelişki" veya "nötr" sınıflandırması yapabilir.

**Anlamsal Benzerlik Ölçümü**

```
Model 1: [CLS] Bu film harikaydı. [SEP]
Model 2: [CLS] Filmi çok beğendim. [SEP]
```

Her iki cümlenin [CLS] token temsillerinin kosinüs benzerliği hesaplanabilir.

**Soru Cevaplama**

```
[CLS] Soru metni? [SEP] Cevabın bulunduğu bağlam metni [SEP]
```

BERT, bağlam metnindeki her tokenin, cevabın başlangıç ve bitiş tokeni olma olasılığını hesaplar.

## 4. BERT'in İki Aşamalı Eğitim Süreci

BERT'in eğitimi iki ayrı aşamadan oluşur: ön eğitim (pre-training) ve ince ayar (fine-tuning).

### 4.1 Ön Eğitim ve İnce Ayar Arasındaki Fark

| Özellik | Ön Eğitim (Pre-training) | İnce Ayar (Fine-tuning) |
|---------|--------------------------|--------------------------|
| Amaç | Genel dil anlayışı | Spesifik görev çözümü |
| Veri | Büyük, etiketlenmemiş metinler | Küçük, etiketlenmiş veri kümeleri |
| Eğitim hedefleri | MLM, NSP | Görev-spesifik hedefler (sınıflandırma, vs.) |
| Süre | Günler/Haftalar | Saatler/Günler |
| Çıktı | Genel dil modeli | Görev-spesifik model |
| Kaynak ihtiyacı | Yüksek (çoklu GPU) | Daha düşük (tek GPU yeterli olabilir) |

### 4.2 İnce Ayar Süreci

İnce ayar sürecinin adımları şöyledir:

1. **Ön Eğitimli Modeli Yükleme**: Önceden eğitilmiş BERT modelinin ağırlıkları yüklenir.

2. **Model Mimarisini Uyarlama**: BERT'in çıkış katmanı, hedef göreve göre modifiye edilir:
   - **Sınıflandırma görevleri için**: [CLS] token çıktısının üzerine bir sınıflandırma katmanı eklenir
   - **Token-seviyesi görevler için**: Her token çıktısına bir tahmin katmanı eklenir
   - **Soru cevaplama için**: Başlangıç ve bitiş pozisyonlarını tahmin eden katmanlar eklenir

3. **Veri Hazırlama**: Hedef görevin etiketlenmiş verileri, BERT'in beklediği formata dönüştürülür:
   - Gerekli özel tokenler ([CLS], [SEP]) eklenir
   - Metin tokenize edilir ve BERT'in maksimum giriş uzunluğuna (genellikle 512 token) göre kesilir
   - Padding ve attention maskeleri oluşturulur

4. **İnce Ayar Eğitimi**: Model, görev-spesifik verilerle eğitilir:
   - Genellikle daha düşük öğrenme hızları kullanılır (2e-5 ila 5e-5 arası)
   - Daha az sayıda epoch (2-4 arası) kullanılır
   - Tüm model parametreleri güncellenir (tam ince ayar) veya sadece eklenen katmanlar güncellenir (katmanlı ince ayar)
   - Görev-spesifik kayıp fonksiyonu kullanılır (örn. sınıflandırma için cross-entropy loss)

5. **Değerlendirme ve Optimize Etme**: Model, görevin test veri kümesinde değerlendirilir ve gerekirse hiperparametreler ayarlanır.

### 4.3 İnce Ayar Örnekleri

**Duygu Analizi (Sınıflandırma Görevi)**

```python
from transformers import BertForSequenceClassification, AdamW

# Ön eğitimli modeli yükleme ve sınıflandırma için uyarlama
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Olumlu/Olumsuz

# İnce ayar için optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Eğitim döngüsü
for epoch in range(3):
    for batch in train_dataloader:
        # Forward geçişi
        outputs = model(input_ids=batch['input_ids'], 
                       attention_mask=batch['attention_mask'], 
                       labels=batch['labels'])
        
        loss = outputs.loss
        
        # Geriye yayılım ve optimizasyon
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Adlandırılmış Varlık Tanıma (Token Sınıflandırma)**

```python
from transformers import BertForTokenClassification

# Ön eğitimli modeli yükleme ve token sınıflandırma için uyarlama
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)  # Varlık etiketleri
```

## 5. Bağlamsal Gömmeler (Contextual Embeddings)

### 5.1 Geleneksel Gömmeler vs. Bağlamsal Gömmeler

**Geleneksel Gömmeler** (Word2Vec, GloVe gibi):

- Her kelime için tek ve sabit bir vektör temsili üretir
- Bir kelimenin farklı anlamlarını tek bir vektörde birleştirir
- Bağlamdan bağımsızdır
- Genellikle basit bir sinir ağı (tek gizli katmanlı) kullanır

**Bağlamsal Gömmeler** (BERT gibi):

- Aynı kelime için, içinde bulunduğu bağlama göre farklı vektör temsilleri üretir
- Çok anlamlı kelimelerin farklı anlamlarını ayırt edebilir
- Bağlama duyarlıdır
- Genellikle Transformer kodlayıcısı gibi daha karmaşık mimariler kullanır

### 5.2 Dinamik Bağlamsal Temsiller

Statik gömmelerden dinamik, bağlamsal gömmeler aşağıdaki temel farkları gösterir:

1. **Kelime Anlamı Ayrımı**: Bağlamsal gömmeler, kelimelerin farklı anlamlarını bağlama göre ayırt edebilir.

2. **Sözdizimsel Roller**: Aynı kelimenin cümle içindeki farklı sözdizimsel rolleri (özne, nesne vb.) farklı temsillere sahip olabilir.

3. **Eş Referanslar**: Aynı varlığa atıfta bulunan farklı kelimeler (örn. "o", "kendisi", "başkan") benzer temsillere sahip olabilir.

4. **Semantik İlişkiler**: Özel bağlamlara özgü semantik ilişkileri yakalayabilir.

### 5.3 Örnek Üzerinden Açıklama

1. **"cold-hearted killer"** cümlesinde "cold" kelimesi:
   - Burada "cold" duygusal bir soğukluğu, merhametsizliği ifade eder
   - BERT, bu bağlamda "cold" kelimesini, duygusal özelliklerle ilişkilendiren bir vektör temsili üretir
   - Muhtemelen "cruel", "merciless", "heartless" gibi kelimelerle vektör uzayında yakın olacaktır

2. **"cold weather"** cümlesinde "cold" kelimesi:
   - Burada "cold" fiziksel bir sıcaklık durumunu ifade eder
   - BERT, bu bağlamda "cold" kelimesini, sıcaklık özellikleriyle ilişkilendiren bir vektör temsili üretir
   - Muhtemelen "freezing", "chilly", "winter" gibi kelimelerle vektör uzayında yakın olacaktır

BERT, dikkat mekanizması sayesinde her kelimenin kendisine ve cümledeki diğer kelimelere olan ilişkisini hesaplayarak bu bağlamsal temsilleri oluşturur.

### 5.4 BERT'in Semantic Embedding Üretim Süreci

BERT, cümle gömmelerini üretirken şu şekilde çalışır:

1. **Tokenizasyon**: Metin önce alt-kelime (subword) tokenlarına ayrılır.
2. **Özel Tokenler Ekleme**: Her girdinin başına `[CLS]` ve sonuna `[SEP]` tokenleri eklenir.
3. **Forward Pass**: Tokenize edilmiş giriş, BERT modelinden geçirilir.
4. **Kontekst Anlama**: BERT, öz-dikkat mekanizması sayesinde bir cümledeki her kelimenin diğer tüm kelimelerle olan ilişkisini öğrenir.
5. **Gömme Çıkarımı**: Cümleyi temsil eden gömme vektörü, genellikle `[CLS]` token çıktısından alınır.
6. **Vektör Uzayı Eşlemesi**: Eğitim sırasında, model benzer anlamlı cümleleri vektör uzayında yakın noktalara, farklı anlamlı cümleleri ise uzak noktalara yerleştirmeyi öğrenir.

## 6. Tokenizasyon ve BERT'in İç Mekanizmaları

### 6.1 Tokenizasyon Neden Önemlidir?

Tokenizasyon, ham metni modelin işleyebileceği sayısal temsillere dönüştürmenin ilk adımıdır. Tokenizasyon şekli şunları etkiler:

1. Modelin işlemesi gereken kelime dağarcığı boyutu
2. Modelin görülmemiş kelimeleri genelleştirme yeteneği
3. Modelin anlamlı dilbilimsel birimleri yakalama etkinliği
4. Modelin morfolojik açıdan zengin dilleri işleme yeteneği

### 6.2 WordPiece Tokenizasyon

WordPiece, BERT'in kullandığı tokenizasyon yöntemidir. Bu yöntem, kelimeleri anlamlı alt parçalara böler:

**WordPiece'in Nasıl Çalıştığı: Algoritma**

1. **Başlangıç**: Tek karakterlerden oluşan minimal bir kelime dağarcığı ile başlayın
2. **Eğitim korpusu hazırlama**: Hedef dili temsil eden büyük bir metin korpusu alın
3. **Yinelemeli birleştirme**:
   - Her olası karakter çifti birleşimi için olasılık artışını hesaplayın
   - Eğitim verisinin olasılığını en çok artıran birleşimi seçin
   - Bu yeni alt kelimeyi kelime dağarcığına ekleyin
   - İstenen kelime dağarcığı boyutuna ulaşılana veya olasılık iyileştirmeleri minimal hale gelene kadar tekrarlayın

**Örnek**:

```
"unhappiness" → ["un", "##happi", "##ness"]
```

"##" öneki, alt kelimenin bir kelime başlangıcı olmadığını, önceki tokenin devamı olduğunu gösterir.

### 6.3 Karşılaştırma: WordPiece vs. BPE vs. SentencePiece

**Byte Pair Encoding (BPE)**:

- GPT modellerinin kullandığı tokenizasyon yöntemi
- Yinelemeli olarak çalışır:
  1. Karakterlerden/baytlardan oluşan bir kelime dağarcığı ile başlayın
  2. Eğitim korpusundaki bitişik karakter çiftlerinin frekansını sayın
  3. En sık görülen çifti birleştirin ve kelime dağarcığına ekleyin
  4. Korpustaki tüm çift oluşumlarını yeni birleştirilmiş sembolle değiştirin
  5. İstenen kelime dağarcığı boyutuna ulaşana kadar 2-4 adımlarını tekrarlayın
- Sadece frekansa dayalıdır, olabilirlik iyileştirmesine değil
- Genellikle tokenları soldan sağa doğru açgözlü bir şekilde uygular

**SentencePiece**:

- Google tarafından geliştirilen gerçekten "uçtan uca" bir tokenizasyon yöntemi
- Girdiye ham unicode dizisi olarak davranır, dile özgü ön işleme gerektirmez
- Boşlukları normal semboller olarak ele alır, tersine çevrilebilirlik sağlar
- Alt kelime düzenlileştirme: Sağlamlık için aynı metnin birden fazla tokenizasyonunu üretebilir
- Kelime dağarcığı kontrolü: Kelime dağarcığı boyutunun tam olarak belirtilmesine izin verir
- Unigram dil modeli varyantı, olasılıksal bir yaklaşım kullanır

### 6.4 BERT Tokenizer'in İş Akışı

BERT tokenizer'ın metni işleme adımları:

1. **Metin Normalizasyonu**:

   ```
   "Hello, world!" → "hello, world!"
   ```

   - Metni küçük harfe çevirme (BERT-uncased için)
   - Unicode normalizasyonu
   - Whitespace düzenlemesi

2. **Temel Tokenizasyon (Basic Tokenizer)**:

   ```
   "hello, world!" → ["hello", ",", "world", "!"]
   ```

   - Noktalama işaretlerini ayırma
   - Boşluklara göre kelimelere ayırma

3. **WordPiece Tokenizasyonu**:

   ```
   ["hello", ",", "world", "!"] → 
   ["hello", ",", "world", "!"]  // Bu örnekte kelimelerin bölünmesi gerekmedi
   ```

   - Her kelime önce sözlükte tam olarak var mı diye kontrol edilir
   - Sözlükte bulunmayan kelimeler, sözlükte var olan en uzun alt-kelimelere bölünür

4. **Özel Token Ekleme**:

   ```
   ["hello", ",", "world", "!"] → 
   ["[CLS]", "hello", ",", "world", "!", "[SEP]"]
   ```

   - [CLS] ve [SEP] gibi özel tokenler eklenir

5. **Token ID'lerine Dönüştürme**:

   ```
   ["[CLS]", "hello", ",", "world", "!", "[SEP]"] → 
   [101, 7592, 1010, 2088, 999, 102]
   ```

   - Her token için sözlükte tanımlı benzersiz bir ID atanır

6. **Attention Mask Oluşturma**:

   ```
   [101, 7592, 1010, 2088, 999, 102] → 
   [1, 1, 1, 1, 1, 1]
   ```

   - Her token için 1 değeri, modelin bu tokene dikkat etmesi gerektiğini gösterir
   - Dolgu tokenları için 0 değeri kullanılır

7. **Token Tipi ID'leri Oluşturma**:

   ```
   Tek cümle: [101, 7592, 1010, 2088, 999, 102] → [0, 0, 0, 0, 0, 0]
   İki cümle: [101, 7592, 1010, 102, 2571, 2024, 102] → [0, 0, 0, 0, 1, 1, 1]
   ```

   - İlk cümledeki tüm tokenlar için 0, ikinci cümledeki tokenlar için 1 değeri atanır

## 7. Pozisyonel Kodlama

### 7.1 Transformerlar Neden Pozisyonel Kodlamaya İhtiyaç Duyar?

Transformer mimarisinin temelini oluşturan öz-dikkat (self-attention) mekanizması, doğası gereği kelime sırasına duyarsızdır. Yani, "köpek adamı ısırdı" ve "adam köpeği ısırdı" cümlelerini birbirinden ayırt edemez. Ek bilgi olmadan, model için tüm girdiler bir "kelime torbası" gibidir.

Recurrent neural network (RNN) mimarilerinde, kelime sırası kelimelerin birer birer işlenmesiyle zımnen yakalanır. Ancak transformerlar tüm kelimeleri eş zamanlı olarak işler, bu da hesaplama verimliliği sağlar fakat sıra bilgisinin kaybına neden olur.

### 7.2 Pozisyonel Kodlamanın Matematiksel Formülasyonu

Pozisyonel kodlama, her kelime gömme vektörüne, kelimenin cümledeki konumunu belirten ek bilgi ekler. Orijinal Transformer mimarisinde, sinüs ve kosinüs fonksiyonlarına dayalı bir formül kullanılır:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Burada:

- `pos`, kelimenin dizideki konumu (0, 1, 2, ...)
- `i`, gömme vektöründeki boyut indeksi (0 ≤ i < d_model/2)
- `d_model`, gömme vektörünün boyutu (BERT-base için 768)

Bu formül, şu özelliklere sahip pozisyonel kodlamalar üretir:

1. **Deterministik**: Aynı konum her zaman aynı kodlamayı alır
2. **Sınırlı değerler**: -1 ile 1 arasında değerler
3. **Göreceli konumların hesaplanabilirliği**: Model, konumlar arasındaki mesafeyi kolayca hesaplayabilir
4. **Uzun dizileri destekleme**: Teorik olarak, formül keyfi uzunluktaki dizileri destekler

### 7.3 BERT'te Pozisyonel Kodlama

BERT, orijinal Transformer'ın sabit sinüzoidal pozisyonel kodlaması yerine, öğrenilebilir pozisyonel gömmeleri kullanır. Bu yaklaşımda:

1. Pozisyon gömmeleri, modelin diğer parametreleriyle birlikte eğitim sırasında öğrenilir
2. Her pozisyon (0'dan maksimum dizge uzunluğuna kadar, tipik olarak 512) için ayrı bir gömme vektörü vardır
3. Bu, modelin dil yapısına göre optimize edilmiş pozisyon bilgisini öğrenmesine olanak tanır
4. Çok yönlü pozisyon ilişkilerini yakalayabilir: örneğin, cümle sınırlarını, sözdizimsel yapıları vb.

BERT'teki pozisyon gömmeleri token ve segment gömmelerine aşağıdaki şekilde eklenir:

```
Final Embedding = Token Embedding + Segment Embedding + Position Embedding
```

Bu yaklaşım, modelin hem kelimenin ne olduğunu (token gömme), hangi cümleye ait olduğunu (segment gömme) hem de cümle içindeki nerede olduğunu (pozisyon gömme) anlamasını sağlar.

## 8. BERT İle Sınıflandırma Yaklaşımları

### 8.1 [CLS] Token Sınıflandırması (Standart Yaklaşım)

```
[CLS] token embedding → Dense Layer → Softmax → Classification
```

Bu, en yaygın yaklaşımdır, orijinal BERT makalesinde önerilmiştir:

- Son katmanın [CLS] token gömme vektörünü alın
- Bunu tek bir yoğun (tam bağlantılı) katmandan geçirin
- Sınıf olasılıklarını elde etmek için softmax aktivasyonunu uygulayın
- Yoğun katman boyutu, BERT'in gizli boyutundan (BERT-base için 768) sınıf sayısına geçiş yapar

### 8.2 Ortalama Token Gömme Yaklaşımı

```
All token embeddings → Average Pooling → Dense Layer → Softmax → Classification
```

Bu yaklaşımda:

- Son katmandan tüm token gömmelerini çıkarın
- Ortalamasını alın (ortalama havuzlama)
- Sonuçta ortaya çıkan vektör tüm diziyi temsil eder
- Bunu bir sınıflandırma katmanından geçirin

Bu yaklaşım, özellikle ince ayar sınırlı olduğunda veya önemli bilgiler tokenlar arasında dağıtıldığında, bazen yalnızca [CLS] tokeninden daha fazla bilgi yakalayabilir.

### 8.3 BERT Üzerinde LSTM

```
All token embeddings → LSTM → Final hidden state → Dense Layer → Softmax → Classification
```

Bu yaklaşım, BERT üzerine sıralı işleme ekler:

- BERT'in son katmanından tüm token gömmelerini LSTM'e besleyin
- LSTM diziyi işler ve son gizli durumu üretir
- Bu gizli durum daha sonra sınıflandırma için kullanılır

LSTM, BERT'in öz-dikkat mekanizmasında tam olarak temsil edilemeyen bağımlılıkları yakalayabilen ek bir sıralı modelleme katmanı ekler.

### 8.4 BERT Üzerinde CNN

```
All token embeddings → Convolutional Layers → Pooling → Dense Layer → Softmax → Classification
```

CNN yaklaşımı:

- BERT'in son katmanından tüm token gömmelerini alır
- Özellikler çıkarmak için evrişimli filtreler uygular
- Boyutsallığı azaltmak için havuzlama kullanır
- Sonuçta ortaya çıkan özellikleri bir sınıflandırıcıya besler

CNN'ler, BERT gömmelerinden yerel kalıpları ve n-gram benzeri özellikleri etkili bir şekilde yakalayabilir, bu da yerel metin kalıplarının önemli olduğu görevler için özellikle yararlı olabilir.

### 8.5 Softmax vs. Sigmoid Aktivasyonu

**Softmax Aktivasyonu (Tek-etiketli Sınıflandırma)**

- Her girdinin tam olarak bir sınıfa ait olduğu durumlarda kullanılır
- Ham skorları 1'e toplanan olasılıklara dönüştürür
- En yüksek olasılık, tahmin edilen sınıfı gösterir
- Matematiksel fonksiyon: softmax(z_i) = e^(z_i) / Σ(e^(z_j))

**Sigmoid Aktivasyonu (Çok-etiketli Sınıflandırma)**

- Her girdinin aynı anda birden fazla sınıfa ait olabileceği durumlarda kullanılır
- Her çıktı düğümünü bağımsız bir ikili sınıflandırıcı olarak ele alır
- Her çıktı 0 ile 1 arasında bir olasılık verir
- Genellikle 0.5 eşiği, etiketin atanıp atanmadığını belirler
- Matematiksel fonksiyon: sigmoid(z) = 1 / (1 + e^(-z))

## 9. Pratik Uygulamalar: Türkçe Metinlerle BERT

### 9.1 Cümle Benzerliği için BERTurk Modeli

Türkçe metinler arasındaki semantik benzerliği ölçmek için BERT modelini kullanma örneği:

```python
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# BERTurk modelini yükle
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")

# Cümle embedding'lerini oluşturan fonksiyon
def get_embeddings(sentences):
    # Tokenizasyon
    encoded_input = tokenizer(sentences, padding=True, truncation=True, 
                              max_length=128, return_tensors='pt')
    
    # Model çıktıları
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # [CLS] tokeninin temsilini al
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    
    # Normalize et
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

# İki cümle arasındaki benzerliği hesaplayan fonksiyon
def semantic_similarity(sentence1, sentence2):
    embeddings = get_embeddings([sentence1, sentence2])
    
    return F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

# Test cümleleri
test_pairs = [
    ("Bugün hava çok güzel.", "Hava bugün oldukça iyi."),
    ("Türkiye'nin başkenti Ankara'dır.", "İstanbul Türkiye'nin en kalabalık şehridir."),
    ("Bu film harika.", "Filmi çok beğendim.")
]

# Benzerlik skorlarını hesapla
for sentence1, sentence2 in test_pairs:
    similarity = semantic_similarity(sentence1, sentence2)
    print(f"Cümle 1: {sentence1}")
    print(f"Cümle 2: {sentence2}")
    print(f"Benzerlik skoru: {similarity:.4f}")
    print("-" * 50)
```

### 9.2 Duygu Analizi için BERTurk Fine-tuning

```python
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import torch

# Model ve tokenizer yükleme
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Pozitif, nötr, negatif

# Örnek veri seti sınıfı
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Eğitim döngüsü örneği
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # GPU'ya taşı
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward ve optimize et
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## 10. BERT Varyantları ve Türkçe Modeller

### 10.1 BERT Varyantları

BERT'in başarısından sonra birçok türevi geliştirilmiştir:

**RoBERTa (Robustly Optimized BERT)**

- Facebook AI tarafından geliştirilmiştir
- NSP görevini kaldırır ve sadece MLM kullanır
- Daha büyük batch boyutları kullanır
- Daha uzun süre ve daha fazla veri üzerinde eğitilmiştir
- Dinamik maskeleme: Her örnek için yeni bir maskeleme kalıbı kullanır
- BERT'ten daha iyi performans göstermiştir

**DistilBERT**

- Hugging Face tarafından geliştirilen damıtılmış BERT versiyonu
- Bilgi damıtma (knowledge distillation) tekniği kullanılarak küçültülmüştür
- Orijinal BERT'in %40 daha küçük, %60 daha hızlı versiyonu
- BERT'in performansının %97'sini korur
- Mobil uygulamalar ve düşük kaynaklı ortamlar için idealdir

**ALBERT (A Lite BERT)**

- Google tarafından geliştirilmiştir
- Parametre paylaşımı yoluyla model boyutunu küçültür:
  - Katmanlar arası ağırlık paylaşımı
  - Kelime gömme parametrelerini faktörize etme
- NSP yerine Sentence Order Prediction (SOP) kullanır
- Daha küçük bellek ayak izi ile daha büyük modeller oluşturulabilir

**ELECTRA**

- "Replaced Token Detection" (RTD) adı verilen alternatif bir ön eğitim görevi kullanır
- İki bileşenli bir sistemdir:
  - Generator: MLM gibi maskelenmiş token'ları tahmin eder
  - Discriminator: Token'ların orijinal mi yoksa generator tarafından değiştirilmiş mi olduğunu tespit eder
- Daha verimli eğitim: Tüm token'lar üzerinde eğitim yapar, sadece maskelenmiş olanlar değil
- Aynı hesaplama kaynakları ile daha iyi performans sağlar

**DeBERTa-v3 (2021-2023)**

- Microsoft tarafından geliştirilen, BERT'in en güçlü varyantlarından biri
- Disentangled Attention: İçerik ve pozisyon bilgisini ayrı ayrı işler
- Enhanced Mask Decoder: MLM tahminini iyileştirir
- DeBERTa-v3 (2023): ELECTRA tarzı eğitim ile daha da geliştirilmiş
- SuperGLUE benchmark'ta insan seviyesini aşan ilk modellerden biri

```python
from transformers import AutoModel, AutoTokenizer

# DeBERTa-v3 kullanımı
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
```

**ModernBERT (Aralık 2024)**

- Answer.AI ve LightOn tarafından geliştirilen yeni nesil encoder modeli
- Modern mimari optimizasyonları:
  - Rotary Position Embeddings (RoPE)
  - GeGLU aktivasyon fonksiyonu
  - Flash Attention 2 entegrasyonu
  - Alternating local (128 token) ve global attention
- 8192 token context window (BERT'in 512'sine karşı)
- 2 trilyon token üzerinde eğitilmiş
- Aynı boyutta BERT/RoBERTa'dan önemli ölçüde daha iyi performans

```python
from transformers import AutoModel, AutoTokenizer

# ModernBERT kullanımı
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
```

| Model | Yıl | Parametre | Context | GLUE Score | Özellikler |
|-------|-----|-----------|---------|------------|------------|
| BERT-base | 2018 | 110M | 512 | 79.6 | Orijinal |
| RoBERTa-base | 2019 | 125M | 512 | 87.6 | Daha iyi eğitim |
| DeBERTa-v3-base | 2023 | 86M | 512 | 88.1 | Disentangled attention |
| ModernBERT-base | 2024 | 149M | 8192 | 88.0+ | Modern optimizasyonlar |

### 10.2 Türkçe BERT Modelleri

Türkçe için geliştirilen ve kullanılabilen BERT modelleri:

**BERTurk**

- Türkçe metinler üzerinde eğitilmiş BERT modeli
- OSCAR ve Common Crawl veri kümeleri üzerinde eğitilmiştir
- Hem cased (büyük/küçük harf duyarlı) hem de uncased versiyonları vardır
- Hugging Face'te "dbmdz/bert-base-turkish-cased" ve "dbmdz/bert-base-turkish-uncased" isimleriyle erişilebilir

**mBERT (Multilingual BERT)**

- Google'ın çok dilli BERT modeli
- Türkçe dahil 104 dil destekler
- Wikipedia verileri üzerinde eğitilmiştir
- Dil-spesifik modellere göre daha az performans gösterebilir, ancak çok dilli uygulamalar için faydalıdır

**XLM-RoBERTa**

- Facebook'un geliştirdiği çok dilli RoBERTa modeli
- 100 dil destekler (Türkçe dahil)
- 2.5TB temizlenmiş Common Crawl verisi üzerinde eğitilmiştir
- mBERT'ten daha iyi çapraz dil transferi yeteneklerine sahiptir
- Düşük kaynaklı diller için avantajlıdır

### 10.3 Model Seçim Kriterleri

Hangi BERT versiyonunu seçeceğiniz aşağıdaki faktörlere bağlıdır:

1. **Görev Karmaşıklığı**: Daha karmaşık görevler için daha büyük modeller (BERT-Large gibi) daha iyi performans gösterebilir.

2. **Veri Miktarı**: Daha küçük veri kümeleri için daha küçük modeller (BERT-Small, BERT-Mini gibi) aşırı uyumu (overfitting) önleyebilir.

3. **Hesaplama Kaynakları**: Sınırlı GPU/CPU kaynakları varsa, daha küçük modeller (DistilBERT gibi) tercih edilebilir.

4. **Çıkarım Hızı Gereksinimleri**: Gerçek zamanlı uygulamalar için daha küçük ve hızlı modeller (BERT-Tiny, DistilBERT gibi) daha uygundur.

5. **Doğruluk/Hız Dengesi**: Uygulamanızın doğruluk ve hız arasındaki öncelikleri hangi modeli seçeceğinizi etkiler.

6. **Dil Özellikleri**: Türkçe gibi sondan eklemeli diller için genellikle daha büyük kelime dağarcığına sahip modeller (30k-50k token) daha faydalı olabilir.

## 11. Sonuç ve İleri Seviye Uygulamalar

BERT, NLP alanında çığır açmış ve birçok uygulamada başarıyla kullanılmıştır. Çift yönlü bağlam anlayışı, zengin önceden eğitilmiş temsiller ve kolay ince ayarlanabilirliği sayesinde, metin sınıflandırma, adlandırılmış varlık tanıma, soru cevaplama ve cümle benzerliği gibi çeşitli görevlerde güçlü performans gösterir.

### 11.1 BERT'in Gelişmiş Uygulamaları

BERT'in gelişmiş uygulamaları şunları içerir:

- **Semantik Arama Motorları**: Kullanıcı sorgularını semantik olarak benzer belgelere eşleştirme
- **Özetleme Sistemleri**: Benzer cümleleri gruplandırarak metin özetleme
- **Duygu Analizi**: Metinlerin duygusal tonu hakkında bilgi çıkarma
- **Soru-Cevap Sistemleri**: Soruları semantik olarak alakalı yanıtlarla eşleştirme
- **Doküman Sınıflandırma**: Belgeleri içerik benzerliğine göre kategorilere ayırma
- **Makine Çevirisi**: Diller arası anlam aktarımı
- **Metin Düzeltme ve Gramer Kontrolü**: Dil hatalarını tespit etme ve düzeltme
- **Metin Oluşturma**: Fine-tuning sonrası belirli tarzda metin üretme

### 11.2 Türkçe NLP için BERT Kullanımında İpuçları

Türkçe dili için BERT kullanırken dikkat edilmesi gereken hususlar:

1. **Morfolojik Zenginlik**: Türkçe sondan eklemeli bir dildir. Kelimeler birçok ek alabilir, bu nedenle tokenizasyon önemlidir.

2. **Tokenizasyon Stratejisi**: Türkçe için BPE veya SentencePiece genellikle WordPiece'ten daha iyi performans gösterir.

3. **Kelime Dağarcığı**: Türkçe modeller için daha büyük kelime dağarcığı boyutları (30k-50k token) faydalı olabilir.

4. **Ön İşleme**: Türkçe metinlerde normalizasyon (örn. büyük/küçük harf, aksanlı karakterler) önemlidir.

5. **Veri Artırma**: Türkçe için veri kısıtlı olabilir, bu nedenle veri artırma teknikleri kullanmak faydalı olabilir.

### 11.3 Gelecek Gelişmeler

BERT'in başarısı yeni modellerin geliştirilmesini teşvik etmiştir. Gelecekte şu gelişmeleri bekleyebiliriz:

1. **Daha Verimli Transformer Modelleri**: Hesaplama kaynakları daha az gerektiren, ancak benzer performans gösteren modeller

2. **Çok Dilli ve Çok Modaliteli Modeller**: Metin, görüntü, ses gibi farklı veri türlerini birleştiren modeller

3. **Alan-Spesifik BERT Modelleri**: Belirli alanlara (tıp, hukuk, bilim) özel olarak eğitilmiş BERT modelleri

4. **Daha Uzun Bağlamı İşleyebilen Modeller**: 512 tokenden daha fazlasını işleyebilen BERT varyantları

5. **Düşük Kaynaklı Diller için Daha İyi Destek**: Türkçe gibi diller için daha fazla kaynak ve daha iyi modeller

BERT ile çalışmaya başlamak için en iyi yol, küçük bir veri kümesiyle basit bir görevde ince ayar yapmaktır. Transformers gibi kullanıcı dostu kütüphaneler, BERT'i projenize entegre etmeyi kolaylaştırır.

BERT'in güçlü yönlerini ve sınırlamalarını anlayarak, kendi NLP uygulamalarınızda en iyi sonuçları elde etmek için doğru stratejileri belirleyebilirsiniz.
