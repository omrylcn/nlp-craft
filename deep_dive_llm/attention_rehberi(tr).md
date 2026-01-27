# Tek Başına Bir Hiçsin: Çevren Seni Sen Yapar

> *Attention Mekanizması: Kapsamlı Öğrenme Rehberi*

---

Bu rehber, Transformer mimarisinin temelini oluşturan Attention mekanizmasını sezgisel ve matematiksel perspektiflerden ele almaktadır.

---

## Bölüm 1: Attention Mekanizmasının Varoluş Amacı

### 1.1 Barkod mu, Parmak İzi mi?

Geleneksel bilgisayar sistemleri için kelimeler birer barkod gibidir. "Gelmek" kelimesi her yerde aynı barkoda sahiptir — sabit, değişmez, cansız.

Ama insan zihni böyle çalışmıyor.

Şu iki cümleyi düşün:
- "Eve geldim." 
- "Gaza geldim."

İkisinde de kelime aynı: "Geldim". Sözlükte tek bir madde altında yazılır. Ama bu iki cümleyi okurken zihninde tamamen farklı şeyler canlanıyor:
- Birinde kapıdan giren bir insan
- Diğerinde heyecanlanan bir ruh hali

Aynı kelime, aynı harfler, aynı ses — ama tamamen farklı deneyimler.

#### Eski Yaklaşım: Her Kelimeye Tek Bir Barkod

Word2Vec, GloVe gibi geleneksel yöntemlerde her kelime için tek bir sabit vektör vardı. "Gelmek" kelimesinin vektörü:

```
[0.4, 0.3, 0.3, ...]
→ Biraz fiziksel hareket, biraz mecazi anlam, biraz deyimsel kullanım
→ Hepsinin ortalaması, hiçbirinin kendisi değil
```

Model, "Eve geldim" ile "Gaza geldim" cümlelerini gördüğünde "geldim" kısmını matematiksel olarak birebir aynı sanıyordu.

#### Daha Derin Bir Problem: Aynı Anlam, Farklı Rol

Ama mesele sadece "farklı anlamlar" değil. Daha ince bir şey var.

"Ahmet" kelimesini düşün — üç cümlede de aynı kişi, aynı anlam:
- "Ahmet topu attı" → Ahmet burada **etken** — eylemi yapan, güç kaynağı
- "Topu Ahmet'e verdiler" → Ahmet burada **alıcı** — eylemin hedefi
- "Ahmet'in topu kayboldu" → Ahmet burada **sahip** — bir ilişkinin tarafı

Sözlük anlamı aynı. Ama cümledeki rolü, diğer kelimelerle ilişkisi ve temsil ettiği "şey" tamamen farklı.

İşte eski modellerin asıl sorunu burada: **Kelimeyi "ne olduğu"na göre kodluyorlardı, "ne yaptığı"na göre değil.**

### 1.2 Attention'ın Görevi: Barkoddan Parmak İzine

Dil canlıdır. Bir kelime her kullanıldığında, yanındaki kelimelerden renk alır ve o ana özel, eşsiz bir anlama bürünür. Tıpkı parmak izi gibi — bir daha asla tıpatıp aynısı üretilemez.

Attention mekanizmasının görevi, bu parmak izini oluşturmaktır.

#### Dönüşüm Nasıl Oluyor?

**Girdi (X):** Kelimenin "ham" hali — sözlükteki donuk, ruhsuz tanımı.

**İşlem:** Kelime, cümledeki diğer tüm kelimelerle etkileşime girer:
- "Eve" kelimesi "geldim"e fiziksel bir boyut katar
- "Gaza" kelimesi "geldim"e mecazi bir boyut katar
- "Attı" fiili "Ahmet"e etken rolü yükler
- "Verdiler" fiili "Ahmet"e alıcı rolü yükler

**Çıktı (Z):** Artık elindeki vektör sadece genel bir "gelmek" ya da "Ahmet" değil. O cümlenin bağlamıyla boyanmış, o ana özel bir temsil:
- "Gaza geldim"deki "geldim" → heyecan, coşku, mecaz boyutları öne çıkmış
- "Ahmet topu attı"daki "Ahmet" → güç, irade, başlatıcı boyutları öne çıkmış

#### Tek Cümleyle Özet

Attention, kelimeyi alır ve ona sorar: 

> *"Şu an, bu cümlede, bu arkadaşlarının yanında — tam olarak kimsin ve ne yapıyorsun?"*

Bu sorunun cevabı, kelimenin o cümleye özel parmak izidir.

### 1.3 Somut Örnek: Dönüşümü Adım Adım Görelim

**Cümle:** "Denizde yüz"

1. **Attention öncesi:** "Yüz" kelimesi belirsiz — surat mı, sayı mı, eylem mi?
2. **Attention sırasında:** "Yüz" kelimesi çevresini tarıyor, "Denizde" ile etkileşime giriyor
3. **Skor hesaplanıyor:** "Denizde" ile yüksek uyum çıkıyor (0.95)
4. **Bilgi aktarılıyor:** "Denizde"nin içeriği "Yüz"e katılıyor
5. **Attention sonrası:** "Yüz" artık net olarak "yüzmek eylemi" — o cümleye özel parmak izi oluştu

Boyut değişmiyor (hala aynı uzunlukta vektör), ama **içerik tamamen değişiyor**.

### 1.4 Asosyal'den Sosyal'e: Girdi-Çıktı Dönüşümü

#### Girdi Matrisi: "Asosyal" Kelimeler

Elimizdeki ilk `X ∈ ℝⁿˣᵈ` matrisi, kelimelerin sözlükteki yalnız halleridir.

- Her kelime kendi kutusundadır
- Yanında kimin oturduğundan haberi yoktur
- "Run" kelimesi yanında "marathon" olsa bile, hala nötr ve belirsizdir

#### Çıktı Matrisi: "Sosyalleşmiş" Kelimeler

Attention'dan geçtikten sonra `Z ∈ ℝⁿˣᵈ` matrisi:

- Satır sayısı değişmez, ama içindeki değerler tamamen değişir
- Her kelime, cümledeki diğer kelimelerin bilgisini emmiş haldedir
- "Run" kelimesi artık net olarak "koşmak eylemi" vektörüne dönüşmüştür

**Analoji:** Hamur malzemeleri (un, su, tuz) ayrı ayrı duruyordu → Yoğrulmuş hamur oldu. Aynı maddeler, ama artık birbirinin içine geçmiş halde.

---

## Bölüm 2: Çözüm Arayışı — Bağlamsal Anlam Nasıl Kazandırılır?

Bölüm 1'de problemi tanımladık: Statik embedding'ler bağlamı görmüyor, bu yüzden "Denizde yüz" ile "Yüzü güldü" aynı muameleyi görüyor.

Şimdi kritik soru şu: **Bir token'a bağlamsal anlam katmak için ne yapmalıyız?**

Bu soruyu adım adım, sezgisel olarak çözelim. Sonra bu adımları matematiksel formüle dökeceğiz.

### 2.1 Sezgisel Çözüm: Adım Adım Düşünelim

#### Adım 1: Token'lar Birbirleriyle "Konuşmalı"

Eğer "Yüz" kelimesi bağlamını anlayacaksa, yanındaki kelimelere bakması gerekir.

- "Yüz" tek başına → Belirsiz (surat? sayı? yüzmek?)
- "Yüz" + "Denizde" → Aha! Yüzmek eylemi olmalı

**İhtiyaç:** Her token, diğer tüm token'lara erişebilmeli.

#### Adım 2: Her Token "Kime Ne Kadar Dikkat Edeceğini" Belirlemeli

Ama her kelimeye eşit dikkat etmek mantıklı değil.

"Denizde yüz metre koştum" cümlesinde:
- "Yüz" kelimesi için "Denizde" önemli mi? Hayır, burada "yüz" sayı
- "Yüz" kelimesi için "metre" önemli mi? Evet! Çünkü "yüz metre" bir ölçü birimi

**İhtiyaç:** Token'lar arası "uyum skoru" hesaplanmalı. Kim kimle alakalı?

#### Adım 3: Uyumlu Token'lardan Bilgi Çekilmeli

Uyum skorunu bulduk diyelim. Şimdi ne yapacağız?

Yüksek skorlu token'lardan bilgi alacağız, düşük skorlulardan almayacağız.

- "Yüz" ← "metre" (yüksek skor) → "metre"nin bilgisini al
- "Yüz" ← "Denizde" (düşük skor) → "Denizde"nin bilgisini alma

**İhtiyaç:** Skorlara göre ağırlıklı bilgi toplama mekanizması.

### 2.2 Bu Adımları Nasıl Formüle Ederiz?

Şimdi sezgisel adımlarımızı matematiksel bileşenlere dönüştürelim.

#### Problem 1: Bir Token'ın İki Farklı Rolü Var

Dikkat et: Bir cümlede her token **aynı anda iki farklı şapka** takmak zorunda:

- **Sorgulayan şapkası:** "Ben kime bakmalıyım? Kimden bilgi almalıyım?"
- **Sorgulanan şapkası:** "Bana kim bakmalı? Kim benden bilgi almalı?"

**Örnek:** "Kedi fare kovaladı" cümlesinde "kovaladı" kelimesi:
- Sorgulayan olarak: "Kovalayan kim? Kovalanan ne?" diye sorar → "kedi" ve "fare"ye bakar
- Sorgulanan olarak: "Kedi" kelimesi "ne yaptı?" diye sorduğunda cevap verir

**Kritik içgörü:** Aynı token, aynı anda hem soru soruyor hem soruya cevap oluyor!

Tek bir vektörle bu iki rolü ayırt edemeyiz. Çünkü:

```
uyum(i, j) = xᵢᵀ xⱼ = xⱼᵀ xᵢ = uyum(j, i)   ← Simetrik!
```

Ama roller simetrik değil:
- "Kedi"nin "kovaladı"ya bakması ≠ "Kovaladı"nın "kedi"ye bakması
- Biri özne arıyor, diğeri eylem arıyor — farklı niyetler!

**Peki neden doğrudan X·Xᵀ kullanmıyoruz?**

İlk akla gelen fikir: "Ham embedding'leri direkt çarpsak olmaz mı?"

```
A = softmax(X · Xᵀ)   ← Basit ama çalışmaz!
```

Bu yaklaşımın 3 kritik sorunu var:

1. **Simetri:** `xᵢᵀ xⱼ = xⱼᵀ xᵢ` — "Kedi kovaladı'ya ne kadar bakıyor?" ile "Kovaladı kedi'ye ne kadar bakıyor?" aynı skor çıkar. Ama dilde bu ilişkiler asimetrik!

2. **Tek perspektif:** Her token sadece tek bir vektörle temsil ediliyor. "Sorgularken nasıl görünmeliyim?" ile "Sorgulanırken nasıl görünmeliyim?" ayrımı yok.

3. **Öğrenme kapasitesi yok:** X sabit (embedding tablosundan geliyor). Wq ve Wk olmadan, model "neye dikkat etmeli?" sorusunu öğrenemez.

Bu yüzden **öğrenilebilir projeksiyon matrisleri (Wq, Wk)** şart.

#### Çözüm: Her Token İki Kimliğe Bürünsün — Query ve Key

Her token için iki farklı vektör üretelim:

```
Query (Q): "Ben sorgulayan modundayken böyle görünürüm"
Key (K):   "Ben sorgulanan modundayken böyle görünürüm"
```

Bu vektörleri ham embedding'den (X) üretiyoruz, ama **farklı dönüşüm matrisleriyle:**

```
Q = X × Wq   (Sorgulayan kimliği)
K = X × Wk   (Sorgulanan kimliği)
```

Artık uyum skoru:
```
uyum(i, j) = Qᵢ · Kⱼ = (Wq xᵢ)ᵀ (Wk xⱼ)
```

`Wq ≠ Wk` olduğu için, `uyum(i,j) ≠ uyum(j,i)` — **asimetri sağlandı!**

**Ve işte "Self" Attention'ın sırrı burada:**

Hem sorgulayan hem sorgulanan aynı cümledeki token'lar! Dışarıdan başka bir şeye ihtiyaç yok — herkes birbirine bakıyor, herkes birbirine cevap veriyor. Bu yüzden "**Self**-Attention" :)

#### Problem 2: Ham Embedding'i Aktarmak Neden Kötü?

Uyum skorlarını bulduk. Şimdi yüksek skorlu token'lardan bilgi çekeceğiz.

Ama hangi bilgiyi çekeceğiz? Ham embedding'i (X) olduğu gibi mi aktaralım?

**Hayır!** Üç ciddi sorun var:

**Sorun 1: Gürültü Problemi**

Ham embedding her şeyi içerir. "Elma" kelimesinin vektörü:
```
[Meyve, Kırmızı, Yuvarlak, Newton, Apple Şirketi, Vitamin, ...]
```

"Steve Jobs yeni elma tanıttı" cümlesinde "elma"dan sadece "Apple Şirketi" bilgisi lazım. Ama ham embedding'i aktarırsan, "Kırmızı" ve "Vitamin" gürültüsü de gelir.

**Sorun 2: Çoklu Anlam Problemi**

Aynı kelime farklı bağlamlarda farklı bilgi aktarmalı:
- "Elmanın içinden kurt çıktı" → Meyve bilgisi aktarılmalı
- "Elma hisseleri düştü" → Şirket bilgisi aktarılmalı

Tek bir ham vektör bu ayrımı yapamaz.

**Sorun 3: Boyut Esnekliği**

Ham embedding 1024 boyutlu olabilir, ama belki sadece 64 boyutluk özet bilgi yeterli.

#### Çözüm: Üçüncü Projeksiyon — Value (Wv)

```
V = X × Wv   (Aktarılacak içerik)
```

**Wv'nin görevi:** Ham embedding'i alıp, "Bu bağlamda aktarılması gereken bilgi nedir?" sorusuna cevap vermek.

Wv bir **filtre** gibi çalışır:
- Girdi: Karman çorman ham vektör (her şeyi içerir)
- Çıktı: Süzülmüş, göreve özel vektör (sadece gerekeni içerir)

**Analoji:** 
- Ham embedding = Tüm kütüphane
- Wv ile çarpım = O kütüphaneden sadece lazım olan sayfaları fotokopi çekip götürmek

**Peki farklı bağlamlarda farklı bilgi aktarma problemi?**

Bu soruyu şimdilik askıda bırakalım. Tek bir Wv matrisi bu problemi tam çözemez — bunun için **Multi-Head Attention** gerekecek. Bölüm 5'te bu konuya döneceğiz.

#### Neden Üç Ayrı Projeksiyon?

| Bileşen | Kimlik | Görevi | Neden Ayrı? |
|---------|--------|--------|-------------|
| **Query** | Sorgulayan ben | "Kime bakmalıyım?" | Aktif arama yönünü belirler |
| **Key** | Sorgulanan ben | "Kim bana bakmalı?" | Pasif görünürlüğü belirler |
| **Value** | Bilgi taşıyan ben | "Ne bilgi vereceğim?" | Gürültüsüz, filtrelenmiş içerik |

Bu üçlü yapı, **görevlerin ayrılması (separation of concerns)** prensibini uygular:
- **Eşleşme görevi:** Q ve K (kim kime bakacak?)
- **Aktarım görevi:** V (ne aktarılacak?)

### 2.3 Metafor: Postacı, Kapı Numarası ve Ev Sakini

Bu üçlü yapıyı somutlaştıralım:

- **Query** = Postacı (mektup götüren, adres arayan)
- **Key** = Evin kapı numarası (adres, eşleşme kriteri)
- **Value** = Evde yaşayan kişi (asıl içerik, alınacak bilgi)

**Süreç:**
1. Postacı (Query) elindeki adresle sokağa çıkar
2. Tüm kapı numaralarına (Key) bakar → "Bu adres benimkiyle uyuşuyor mu?"
3. Uyum skorlarını hesaplar → Bazı evler çok uyumlu, bazıları hiç değil
4. Yüksek uyumlu evlere girer, sakinlerinden (Value) bilgi alır
5. Aldığı bilgileri birleştirir → Yeni, bağlamsal temsil oluşur

**Kritik nokta:** Mektup kapıya değil, **ev sakinlerine** verilir!
- `Q × K` → Adres eşleşmesi (kimle konuşulacak?)
- `A × V` → İçerik aktarımı (ne alınacak?)

### 2.4 Üçlü Yapının Özeti

```
Girdi: X (ham embedding'ler, statik)
         ↓
    ┌────┴────┐────────┐
    ↓         ↓        ↓
Q = X·Wq   K = X·Wk  V = X·Wv
    ↓         ↓        ↓
 Sorgula   Eşleş    İçerik
    └────┬────┘        ↓
         ↓             ↓
   Uyum Skoru (A)      ↓
         └──────┬──────┘
                ↓
         Ağırlıklı Toplam
                ↓
Çıktı: Z (bağlamsal embedding'ler, dinamik)
```

Bu yapı sayesinde:
- Her token diğerlerine farklı şekillerde bakabilir (asimetri)
- Karşılaştırma ve içerik aktarımı ayrı tutulur (görev ayrımı)
- Bağlama göre dinamik bilgi entegrasyonu sağlanır

---

> **⚠️ Önemli Not: Attention Sıradan Habersizdir**
> 
> Attention mekanizması kelimelerin **"ne"** olduğunu bilir, ama **"nerede"** durduğunu bilmez. 
> 
> Yani "Ahmet Ali'yi sevdi" ile "Ali Ahmet'i sevdi" cümleleri, saf Attention için aynı kelime torbasıdır — özne ve nesneyi sıradan çıkaramaz.
> 
> Bu yüzden Transformer modellerinde girdiye ayrıca **Positional Encoding** (sıra bilgisi) eklenir. Ayrıca GPT gibi modellerde geleceği görmeyi engellemek için **Masking** işlemi de uygulanır. Bu konular yazının kapsamı dışında, ama Attention'ın tek başına "sihirli değnek" olmadığını bilmek önemli.

---

## Bölüm 3: Matematiksel Mekanizma — Formülün Detayları

Bölüm 2'de sezgisel olarak Q, K, V yapısına ulaştık. Şimdi bu yapının matematiksel detaylarını inceleyelim.

### 3.1 Temel Formül

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

Bu formülü Bölüm 2'deki adımlarla eşleştirelim:

| Sezgisel Adım | Matematiksel Karşılık |
|---------------|----------------------|
| "Kime ne kadar dikkat edeyim?" | `QKᵀ` (uyum skorları) |
| "Skorları normalize et" | `softmax(... / √d)` |
| "Bilgiyi topla" | `× V` (ağırlıklı toplam) |

### 3.2 Adım 1: Uyum Skorlarının Hesaplanması (QKᵀ)

#### Ne Yapıyoruz?

Her token'ın Query'sini, diğer tüm token'ların Key'leriyle karşılaştırıyoruz.

```
QKᵀ ∈ ℝⁿˣⁿ   (n token varsa, n×n'lik bir skor matrisi)
```

#### Bu Matris Ne Anlama Geliyor?

```
         Key₁    Key₂    Key₃
        ┌─────┬───────┬───────┐
Query₁  │ 2.1 │  0.3  │  0.8  │  ← Token 1, herkese ne kadar dikkat etmeli?
Query₂  │ 0.5 │  1.9  │  0.2  │  ← Token 2, herkese ne kadar dikkat etmeli?
Query₃  │ 0.7 │  0.4  │  1.5  │  ← Token 3, herkese ne kadar dikkat etmeli?
        └─────┴───────┴───────┘
```

- **(i, j) hücresi:** i'inci token'ın (Query olarak), j'inci token'a (Key olarak) verdiği ham uyum skoru
- **Satır:** Bir token'ın "bakış profili" — kime ne kadar bakıyor
- **Sütun:** Bir token'ın "görünürlük profili" — kim ona ne kadar bakıyor

#### Matematiksel Detay

```
(QKᵀ)ᵢⱼ = qᵢᵀ kⱼ = Σₖ qᵢₖ × kⱼₖ
```

İki vektörün dot product'ı — ne kadar "aynı yöne bakıyorlarsa" o kadar yüksek skor.

#### Neden Q Solda, K Sağda?

Matris çarpımında satır × sütun mantığı var:
- Q'nun her **satırı** = bir token'ın sorgulayan kimliği
- Kᵀ'nin her **sütunu** = bir token'ın sorgulanan kimliği

Bu düzen sayesinde:
- i. satır = "i. token kime bakıyor?"
- j. sütun = "j. token'a kim bakıyor?"

Dilin doğal akışı "özne → fiil → nesne" şeklinde. Formül de bu yönü yansıtıyor.

#### Neden Dot Product? Başka Yöntemler Yok mu?

İki vektör arasındaki "uyumu" ölçmenin farklı yolları var. Neden dot product seçildi?

**Alternatif 1: Cosine Similarity**
```
cos(q, k) = (q · k) / (||q|| × ||k||)
```
- Vektörlerin büyüklüğünü normalize eder, sadece yöne bakar
- Sorun: Bazen büyüklük de önemli bilgi taşır. "Çok önemli özne" vs "az önemli özne" ayrımı kaybolur.

**Alternatif 2: Additive Attention (Bahdanau)**
```
score = Wᵥ × tanh(Wq·q + Wk·k)
```
- Daha esnek, non-linear
- Sorun: Daha yavaş (ekstra matris çarpımları), paralelize etmesi zor

**Dot Product'ın Avantajları:**

1. **Hız:** Sadece matris çarpımı — GPU'lar bunu çok hızlı yapar
2. **Basitlik:** Ekstra parametre yok (Wᵥ gibi)
3. **Yeterli ifade gücü:** Wq ve Wk zaten vektörleri dönüştürüyor, dot product bu dönüştürülmüş vektörlerdeki uyumu yakalamak için yeterli

Transformer'ın seçimi: **Hız + Basitlik + Yeterli güç = Dot Product**

### 3.3 Adım 2: Normalizasyon (Softmax ve √d)

#### √d ile Bölme Neden Gerekli?

Dot product'lar, boyut (`d`) arttıkça büyür. Eğer `d = 512` ise, skorlar çok büyük olabilir.

Büyük skorlar softmax'ı "sertleştirir":
- Softmax([10, 1, 1]) ≈ [0.99, 0.005, 0.005] — neredeyse one-hot
- Softmax([2, 1, 1]) ≈ [0.58, 0.21, 0.21] — daha yumuşak

**Neden sert softmax kötü?**

İki kritik sorun var:

1. **Genelleme kaybı:** Dikkat tek bir token'a odaklanırsa (one-hot gibi), model diğer bağlamları görmezden gelir.

2. **Gradient sorunu:** Softmax'in türevi, çıktı 0 veya 1'e yaklaştıkça sıfıra yaklaşır. Yani sert softmax = küçük gradient = yavaş/durmuş öğrenme (vanishing gradient).

`√d` ile bölme her iki sorunu da çözer:
- Skorları makul aralığa çeker → yumuşak dağılım
- Gradient'ler sağlıklı akar → stabil öğrenme

#### Softmax Ne Yapıyor?

```
A = softmax(QKᵀ / √d) ∈ ℝⁿˣⁿ
```

Softmax, her satırı olasılık dağılımına çevirir:
- Her satırın toplamı = 1
- Tüm değerler [0, 1] arasında

`Aᵢⱼ` = i'inci token'ın, j'inci token'a vereceği dikkat ağırlığı.

#### Softmax'in Rekabetçi Doğası

Softmax'in toplamı 1'e eşitlemesi, token'ları birbirleriyle **rekabete** sokar.

Dikkat sınırlı bir kaynaktır: Eğer "Elma" kelimesi dikkatinin %90'ını "Borsa"ya verirse, "Meyve"ye verecek dikkati kalmaz.

Bu kısıt (constraint), modeli **en önemli olana odaklanmaya** zorlar. Herkes eşit dikkat alamaz — biri kazanırsa, diğeri kaybeder.

### 3.4 Adım 3: Bilgi Toplama (A × V)

Son adımda, hesaplanan ağırlıklarla Value'ları topluyoruz:

```
Z = A × V ∈ ℝⁿˣᵈᵛ
```

Her `zᵢ` (i'inci token'ın yeni temsili):

```
zᵢ = Σⱼ Aᵢⱼ × vⱼ
```

- `Aᵢⱼ` = i'inci token, j'inci token'a verdiği önem ağırlığı
- `vⱼ` = j'inci token'ın Value vektörü (taşıdığı içerik)

**Sonuç:** Her token, tüm token'ların Value'larının ağırlıklı ortalamasını alır. Ağırlıklar ise Query-Key uyumundan gelir.

### 3.5 Neden V Skor Üretmiyor? Neden QV veya KV Yok?

Akla takılan bir soru: Q, K, V hepsi aynı boyutta (`n × d`). Neden V de skora katılmıyor? Neden `QV` veya `KV` yapmıyoruz?

#### Soru 1: "V neden skor üretmiyor?"

Kısa cevap: **Karşılaştırma görevi ile içerik taşıma görevi birbirine karışmamalı.**

Detaylı cevap:

**Q ve K'nın tek işi:** Birbirleriyle çarpılıp skor üretmek. `QKᵀ` bir **geçici skor matrisidir** — sadece "kim kime ne kadar dikkat etsin?" sorusuna cevap verir. Sonra bu skorlar kullanılır ve "unutulur".

**V'nin tek işi:** Bilgi taşımak. V hiçbir şeyle "karşılaştırılmaz" — sadece son adımda `A × V` ile ağırlıklı toplanır.

Eğer V de skora katılsaydı (`QVᵀ` gibi), şu sorun çıkardı:

> "Ahmet" kelimesinin Value'su "özne" bilgisini taşıyor olsun. Bu bilgi, **içerik olarak** değerli. Ama aynı bilgiyi **eşleşme kriteri** olarak kullanırsan, "özne" bilgisi fiil aramayı zorlaştırabilir.

Yani: **İçerik bilgisi, eşleşme kararını bozabilir.** Transformer bunu bilerek ayırıyor.

#### Soru 2: "Ama boyutları aynı, neden karışmıyor?"

Bu çok iyi bir soru. Cevap: **Lineer cebirde boyut değil, kullanım bağlamı önemlidir.**

Aynı `[0.8, -0.2, 0.5]` vektörü:
- **Q olarak kullanılırsa:** "Ben fiil arıyorum"
- **K olarak kullanılırsa:** "Ben özneyim, fiil beni bulsun"  
- **V olarak kullanılırsa:** "Ben bir insanım, eylemi yapan"

Aynı sayılar, **formüldeki pozisyonlarına göre** farklı anlamlar taşıyor.

Bu, programlamadaki tip sistemine benzer: Aynı bitler, `int` mi `float` mı diye kullanıma göre yorumlanır. Burada da aynı boyutlu vektörler, formüldeki yerlerine göre farklı roller üstleniyor.

#### Soru 3: "QV veya KV neden yapılmıyor?"

**QV yapmak anlamsız:**
- Query "kime bakacağımı" söyler, "ne alacağımı" değil
- Sorgu ile içeriği doğrudan çarpmak, "adres aramadan eve girmek" gibi

**KV yapmak gereksiz:**
- Key zaten "eşleşme kriteri" için var
- İçerik aktarımı Value'ın işi

#### Mimari Prensip: Görev Ayrımı

| Görev | Sorumlu | Diğerleri Karışmaz |
|-------|---------|-------------------|
| Karşılaştırma / Skor üretme | Q, K | V karışmaz |
| İçerik taşıma | V | Q, K karışmaz |

Bu ayrımın iki faydası var:

1. **Gradyan temizliği:** Her ağırlık matrisi tek bir görevi optimize eder. Wv sadece "daha iyi içerik üret" baskısı alır, Wq/Wk sadece "daha iyi eşleş" baskısı alır.

2. **Esneklik:** Aynı kelime, farklı bağlamlarda farklı Key (nasıl bulunmalı?) ve farklı Value (ne bilgi vermeli?) üretebilir.

### 3.6 Somut Örnek: 3 Token'lık Cümle

**Cümle:** "Kedi uyudu" (3 token)

```
X = [x_kedi; x_uyudu; x_nokta] ∈ ℝ³ˣ⁴  (3 token, 4 boyutlu embedding)
```

**Adım 1: Projeksiyonlar**
```
Q = X Wq   →  3×4 matris
K = X Wk   →  3×4 matris  
V = X Wv   →  3×4 matris
```

**Adım 2: Uyum Skorları**
```
QKᵀ = [q_kedi·k_kedi   q_kedi·k_uyudu   q_kedi·k_nokta ]
      [q_uyudu·k_kedi  q_uyudu·k_uyudu  q_uyudu·k_nokta]
      [q_nokta·k_kedi  q_nokta·k_uyudu  q_nokta·k_nokta]
```

**Adım 3: Softmax**
```
A = softmax(QKᵀ / √4)

Örnek sonuç:
A = [0.7  0.2  0.1]   ← Kedi: kendine %70, uyudu'ya %20, noktaya %10
    [0.3  0.6  0.1]   ← Uyudu: kediye %30, kendine %60, noktaya %10
    [0.2  0.3  0.5]   ← Nokta: kediye %20, uyuduya %30, kendine %50
```

**Adım 4: Bilgi Toplama**
```
z_kedi = 0.7×v_kedi + 0.2×v_uyudu + 0.1×v_nokta
z_uyudu = 0.3×v_kedi + 0.6×v_uyudu + 0.1×v_nokta
z_nokta = 0.2×v_kedi + 0.3×v_uyudu + 0.5×v_nokta
```

**Sonuç:** "Uyudu" kelimesi artık "kedi" bilgisini de içeriyor — bağlamsal anlam kazandı!

### 3.7 Özet Tablo: Formül Bileşenleri

| Bileşen | Formül | Boyut | İşlevi |
|---------|--------|-------|--------|
| Query | Q = X Wq | n × dₖ | Sorgulama vektörleri |
| Key | K = X Wk | n × dₖ | Eşleşme vektörleri |
| Value | V = X Wv | n × dᵥ | İçerik vektörleri |
| Ham Skorlar | QKᵀ | n × n | Token çiftleri arası uyum |
| Dikkat Ağırlıkları | A = softmax(QKᵀ/√d) | n × n | Normalize edilmiş ağırlıklar |
| Çıktı | Z = AV | n × dᵥ | Bağlamsal temsiller |

---

## Bölüm 4: Wv Dönüşümünün Gerekliliği — Neden Ham Girdiyi Kullanmıyoruz?

### 4.1 Temel Soru

Attention'da şöyle bir ara adım var:
- `QKᵀ` ile "hangi bilgiyi çekeyim, ne kadar çekeyim" kararını alıyoruz
- Sonra `V = X Wv` ile bilgiyi dönüştürüyoruz

Soru şu: **Elimizde zaten `X` (ham girdi) var. Neden onu olduğu gibi kullanmıyoruz da `Wv` ile çarpıp şeklini değiştiriyoruz?**

Bu sorunun 3 temel cevabı var.

### 4.2 Sebep 1: Gürültüden Arınma (Ham Madde vs. İşlenmiş Ürün)

Girdi vektörü `xᵢ`, o kelimeye dair **her şeyi** karman çorman barındırır.

**Örnek:** "Elma" kelimesinin girdi vektörü:
```
[Meyve, Kırmızı, Yuvarlak, Newton, Teknoloji Şirketi, Vitamin, ...]
```

**Cümle:** "Steve Jobs yeni elma modelini tanıttı."

Burada modelin "Elma"dan alması gereken bilgi sadece **"Teknoloji Şirketi"** özelliğidir. "Kırmızı" veya "Vitamin" bilgisi burada **gürültüdür (noise)**.

Eğer `Wv` matrisi olmasaydı ve direkt `X`'i kullansaydık:
- Model cümleyi işlerken gereksiz yere vitaminlerden ve renklerden bahseden sinyalleri de taşırdı
- Bilgi kirliliği oluşurdu

**Wv'nin görevi:** Girdiyi alır ve "Bu katmanda/görevde vitaminler önemli değil, sadece şirketsel özellikleri öne çıkar" diyerek vektörü filtreler.

**Analoji:**
- Girdiyi (`X`) olduğu gibi kullanmak = Tüm kütüphaneyi sırtlayıp götürmek
- `Wv` ile çarpmak = O kütüphaneden sadece lazım olan sayfayı fotokopi çekip götürmek

### 4.3 Sebep 2: Çoklu Bakış Açısı (Multi-Head Attention Hazırlığı)

Bu en güçlü sebeptir.

Multi-Head yapısında model, kelimeye **aynı anda 8-12 farklı gözle** bakar.

Eğer `Wv` dönüşümü olmasaydı:
- Kafa 1, "Elma"nın `X`'ini alırdı
- Kafa 2 de "Elma"nın aynı `X`'ini alırdı
- Hepsi aynı şeyi taşırdı — hiçbir fark olmazdı

Ama `Wv` matrisleri her kafa için farklıdır (`Wv¹, Wv², Wv³, ...`):

- **Kafa 1 (Wv¹):** `X`'i alır, içinden sadece **Dilbilgisi** (İsim/Özne) bilgisini süzüp taşır
- **Kafa 2 (Wv²):** `X`'i alır, içinden sadece **Anlam** (Şirket) bilgisini süzüp taşır
- **Kafa 3 (Wv³):** `X`'i alır, içinden sadece **Renk** özelliklerini süzüp taşır

**Sonuç:** `Wv` dönüşümü, aynı girdiden **farklı lezzetler** üretmemizi sağlar.

### 4.4 Sebep 3: Boyut Bağımsızlığı (Matematiksel Esneklik)

Girdi vektörümüz `X` çok büyük olabilir (örneğin 1024 boyutlu).

Ama belki attention mekanizmasının o an sadece 64 boyutlu, sıkıştırılmış bir bilgiye ihtiyacı vardır.

`Wv` matrisi bize bu esnekliği verir:

```
Wv ∈ ℝᵈˣᵈ'  →  V = X Wv ∈ ℝⁿˣᵈ'
```

Böylece gereksiz yükten kurtulup, ağın içinde akan veriyi optimize ederiz.

### 4.5 Metafor: Vana ve Su

Attention mekanizmasında:
- **QKᵀ** = "Vana" (Ne kadar açılacak?)
- **V** = "Su" (Borudan ne geçecek?)

`Wv` dönüşümü, suyun (ham bilginin) boruya girmeden önce filtrelenmesini sağlar. Böylece boruda sadece o an işe yarayan bilgi akar.

---

## Bölüm 5: Multi-Head Attention — Matematiksel Bir Zorunluluk

**Multi-Head Attention Nedir?**

Aynı girdi üzerinde **paralel çalışan birden fazla attention katmanı**. Her "head" (kafa) kendi Wq, Wk, Wv matrislerine sahiptir ve girdiyi farklı bir perspektiften işler. Sonunda tüm head'lerin çıktıları birleştirilir.

```
Multi-Head = [Head₁ ; Head₂ ; ... ; Headₕ] × Wₒ

Her Head = Attention(X Wqⁱ, X Wkⁱ, X Wvⁱ)
```

Peki neden tek bir attention yetmiyor? Neden birden fazla "kafa" gerekiyor?

### 5.1 Problem: Tek Kafa Neden Yetersiz?

Şu iki cümleyi düşün:
- "Steve Jobs yeni elma modelini tanıttı" → Elma = Şirket
- "Elmanın içinden kurt çıktı" → Elma = Meyve

Eğer sadece tek bir kafa (Single Head) ve tek bir `Wv` olsaydı, model eğitim sırasında "Elma" kelimesi için ortada kalırdı:
- Veri setinin yarısında "Elma = Meyve"
- Diğer yarısında "Elma = Şirket"

Tek bir matris (`Wv`) bu iki zıt anlamı aynı anda öğrenmeye çalışsaydı ne yapardı?

**Ortalamasını alırdı.**

**Sonuç:** Ne meyveye benzeyen ne şirkete benzeyen, "Meyvemsi Şirket" gibi bulanık, işe yaramaz (garbage) bir vektör ortaya çıkardı.

Bu yüzden Multi-Head bir "lüks" değil, **matematiksel bir zorunluluktur**.

### 5.2 Çözüm: "Hepsini Üret, Sonra Yok Et" (Süperpozisyon)

Çözüm, modelin "Hangi anlama dönüştüreceğim?" diye **seçim yapmamasıdır**.

Bunun yerine **tüm ihtimalleri aynı anda üretmesidir**.

Bunun için birden fazla kafa var: `Wv¹, Wv², Wv³, ...`

Eğitim bittiğinde (milyonlarca iterasyondan sonra), bu matrisler şöyle uzmanlaşmış olurlar:

| Kafa | Matris | Uzmanlık | Elma'yı Görünce |
|------|--------|----------|-----------------|
| Kafa 1 | Wv¹ | Botanik/Gıda | **Daima** "Meyve Vektörü" üretir |
| Kafa 2 | Wv² | Teknoloji/Finans | **Daima** "Şirket Vektörü" üretir |
| Kafa 3 | Wv³ | Renk özellikleri | **Daima** "Kırmızı/Yeşil" üretir |

**Şu an masada ne var?**

Masada aynı anda hem Meyve, hem Şirket, hem Renk vektörleri duruyor. (Süperpozisyon hali)

Burada şans yok. Her kafa kendi ezberlediği (eğitildiği) sabit dönüşümü yaptı.

### 5.3 Kritik Müdahale: Bağlamın (Context) Seçiciliği

Şimdi kritik soru: "Model hangi kafanın çıktısını kullanacağını nereden biliyor?"

**Cevap:** `Wv` bilmez. Ama **Attention Score (A)** bilir.

**Örnek Cümle:** "Borsa düştü, Elma değer kaybetti."

**Kafa 1 (Meyve Kanalı):**
```
Q(Borsa) · K(Meyve_Elması) → Skor: 0.001 (Hiç alaka yok)
İşlem: 0.001 × V_meyve ≈ 0 (Meyve anlamı silindi)
```

**Kafa 2 (Şirket Kanalı):**
```
Q(Borsa) · K(Şirket_Elması) → Skor: 0.99 (Tam isabet)
İşlem: 0.99 × V_şirket ≈ V_şirket (Şirket anlamı parladı)
```

### 5.4 Sistemin Zekası Nerede?

Kritik bir içgörü:

> **Sistemin zekası, Wv'nin "doğru dönüşümü yapmasında" değildir.**
> **Sistemin zekası, "Yanlış dönüşümlerin sesini kısmasındadır" (Noise Suppression).**

- **Wv'ler bağlama kördür:** Her head kendi Wv'sini eğitimle özelleştirir, ama çalışma anında bağlama bakmaz. Girdisi ne olursa olsun, öğrendiği sabit dönüşümü uygular. (Meyve de üretir, Şirket de üretir — hangisinin kullanılacağına karar vermez)
- **Attention Score zekidir:** Bağlama bakar ve "Şu an şirket olanı duymak istiyorum, meyve olanın sesini (katsayısını) sıfıra indir" der

Yani her head, Wv'yi kendi uzmanlık alanına göre özelleştirir. Ama **hangi head'in çıktısının ön plana çıkacağına** attention skoru karar verir.

### 5.5 Pratikte Kaç Kafa Kullanılır?

En küçük GPT-2 modelinde bile **12 Kafa** vardır.

Neden? Çünkü bir kelimenin ortalama 12 farklı nüansı/bağlamı olabilir:
- Dilbilgisel rol (özne, nesne, yüklem)
- Anlamsal kategori (somut, soyut)
- Zamansal ilişki (geçmiş, şimdi, gelecek)
- Duygusal ton (pozitif, negatif, nötr)
- ...

Her bir nüansı yakalamak için ayrı bir `Wv` filtresine (uzmanına) ihtiyaç vardır.

---

## Bölüm 6: Eğitim Dinamikleri — Roller Nasıl Ortaya Çıkıyor?

### 6.1 Başlangıç: Rastgele Matrisler

Eğitimin başında `Wq, Wk, Wv` matrisleri tamamen rastgele. Hiçbir "rol" atanmamış.

İlk loss sinyali geldiğinde her şey değişmeye başlar.

### 6.2 Gradient Akışı ve Rol Ataması

Loss fonksiyonu tahmin edilen kelimenin yanlış olduğunu söyler. Zincir kuralıyla (backpropagation) hata geri yayılır.

**Kritik Gözlem:** Dikkat mekanizmasında `V` doğrudan çıktıya katkı sağlar, bu yüzden **en güçlü gradyanı alır**.

```
∂L/∂Wv = ∂L/∂y · ∂y/∂Attention · ∂Attention/∂V · ∂V/∂Wv
```

Burada `∂Attention/∂V = A` (attention ağırlıkları) — doğrudan ve basit.

Ama:
```
∂L/∂Wq = ∂L/∂y · ∂y/∂Attention · ∂Attention/∂Q · ∂Q/∂Wq
```

Burada `∂Attention/∂Q = A · softmax-türevi · Kᵀ` — çok daha karmaşık ve dolaylı.

### 6.3 Doğal Rol Ataması

Bu gradient yapısı, doğal olarak şu rol atamasını yaratır:

| Matris | Gradient Tipi | Öğrendiği Şey |
|--------|---------------|---------------|
| **Wv** | Doğrudan, güçlü | "Doğru bilgiyi kodla" → İçerik kalitesi |
| **Wq** | Dolaylı, stratejik | "Doğru yere bak" → Bağlamsal sorgulama |
| **Wk** | Dolaylı, stratejik | "Doğru şekilde görün" → Keşfedilebilirlik |

**Sonuç:**
- **Value daha hızlı ve net öğrenir** (içerik odaklı)
- **Query/Key daha yavaş ama stratejik öğrenir** ("kimle konuşulacağı" stratejisi)

Model, "sen cevap ol" ya da "sen soru sor" demez. **Loss'un yapısı ve zincir kuralı, her ağırlığa farklı görev yükler.**

### 6.4 Formüldeki Yer = Öğrenilen Rol

Peki bu roller neden karışmıyor? Neden Wq gidip "Key gibi davranmayı" öğrenmiyor?

**Çünkü formüldeki konum, öğrenilebilecek şeyi kısıtlıyor.**

Bunu şöyle düşün: Üç kişiyi üç farklı odaya koyduk.

| Oda | Formüldeki Yer | Hatayı Düzeltmek İçin Yapabileceği Tek Şey |
|-----|----------------|-------------------------------------------|
| **Sol oda (Wq)** | `Q` × Kᵀ'nin sol tarafı | "Daha iyi aramayı" öğrenmek |
| **Sağ oda (Wk)** | Q × `Kᵀ`'nin sağ tarafı | "Daha görünür olmayı" öğrenmek |
| **Arka oda (Wv)** | A × `V`'nin sağ tarafı | "Daha iyi paketlemeyi" öğrenmek |

Loss fonksiyonu odaya girdiğinde sadece "İşler yürümedi!" diye bağırıyor. Ama:

- **Sol odadaki** (Wq), işi düzeltmek için **daha iyi bakmayı** öğreniyor — çünkü formülde "arayan" pozisyonunda
- **Sağ odadaki** (Wk), işi düzeltmek için **daha belirgin olmayı** öğreniyor — çünkü formülde "aranan" pozisyonunda  
- **Arka odadaki** (Wv), işi düzeltmek için **daha kaliteli içerik üretmeyi** öğreniyor — çünkü formülde "aktarılan" pozisyonunda

**Kimse "sen Query ol" demedi.** Formüldeki konum + Loss'un baskısı + Verinin yapısı birleşince, roller kendiliğinden ortaya çıktı.

### 6.5 Örnek: "Ahmet" → Etken mi, Alıcı mı?

Bu **contextualization** sürecidir — ve burada kritik bir içgörü var:

**Eski modeller kelimeyi "ne olduğu"na göre kodlardı.** "Ahmet" bir isim, bir insan, erkek — hep aynı vektör.

**Transformer ise kelimeyi "ne yaptığı"na göre kodluyor.** Aynı "Ahmet", farklı cümlelerde farklı şeyler "yapıyor":

**Cümle 1:** "Ahmet topu attı"
- Ahmet burada **etken** (agent) — eylemi başlatan, güç kaynağı
- Wv, "Ahmet" vektöründe şu özellikleri öne çıkarır: `[irade, güç, başlatıcı, aktif...]`

**Cümle 2:** "Topu Ahmet'e verdiler"
- Ahmet burada **alıcı** (recipient) — eylemin hedefi
- Wv, "Ahmet" vektöründe şu özellikleri öne çıkarır: `[hedef, pasif, etkilenen, sonuç...]`

**Cümle 3:** "Ahmet'in topu kayboldu"
- Ahmet burada **sahip** (possessor) — bir ilişkinin tarafı
- Wv, "Ahmet" vektöründe şu özellikleri öne çıkarır: `[sahiplik, ilişki, bağlam...]`

Üç cümlede de "Ahmet" aynı kişi. Sözlük anlamı aynı. Ama **cümledeki rolü** — yani "ne yaptığı" — tamamen farklı.

İşte Attention'ın asıl gücü burada: Polysemy'yi (çok anlamlılığı) çözmek sadece başlangıç. Asıl devrim, **aynı anlamdaki kelimenin bile bağlamsal rolünü kodlayabilmesi**.

**Sonuç:** Wv kelimenin yüzey formunu değil, o cümledeki **işlevini** öğrenir. Bu yüzden Bölüm 1'de söylediğimiz gibi — her kelime, her cümlede kendine özel bir "parmak izi" kazanır.

---

## Bölüm 7: Sonuç — Transformer Bir "Rol Oyunu Makinesidir"

### 7.1 Üçlü Uyum

Transformer mimarisi, dilin doğasını yansıtan bir üçlü uyum kurar:

| Bileşen | Soru | Rol | Analoji |
|---------|------|-----|---------|
| **Query** | "Ben kiminle ilgilenmeliyim?" | Aktif arayış | Postacı |
| **Key** | "Ben kimin dikkatini çekebilirim?" | Pasif sunum | Kapı numarası |
| **Value** | "İşte size vermem gereken şey." | Saf içerik | Ev sakini |

### 7.2 Rollerin Kaynağı

Bu roller nereden geliyor? Üç kaynaktan:

1. **Matris çarpımının yönü:** Sol operand (Q) satır bazında çalışır → sorgulayan
2. **Gradient akışı:** V doğrudan loss'a bağlı → içerik; Q/K dolaylı → strateji
3. **Loss fonksiyonunun yapısı:** Doğru tahmini ödüllendir → her bileşen kendi rolünü öğrenir

### 7.3 Nihai İçgörü

**"Neden üç ayrı projeksiyon (Q, K, V)?"** sorusunun cevabı:

Çünkü bir token'ın cümle içinde **üç farklı görevi** var:
1. **Sorgulayan olarak:** Diğerlerine bakıp bilgi toplamak (Query)
2. **Sorgulanan olarak:** Başkalarının sorularına cevap olmak (Key)  
3. **Bilgi kaynağı olarak:** Toplandığında aktarılacak içeriği taşımak (Value)

Tek bir vektörle bu üç görevi aynı anda temsil edemezsin. Her görev için ayrı bir "kimlik" gerekiyor.

**Ve "Self" Attention'ın güzelliği:**

Tüm bu roller aynı cümle içinde, aynı token'lar tarafından oynanıyor. Dışarıdan hiçbir şeye ihtiyaç yok — her token hem soruyor, hem cevaplıyor, hem bilgi veriyor. Sistem kendi kendine yeterek bağlamı öğreniyor.

### 7.4 Neden Devrim Yarattı?

Transformer'dan önce NLP'de iki ana yaklaşım vardı:

**RNN (Recurrent Neural Networks):**
- Token'ları sırayla işler: t₁ → t₂ → t₃ → ...
- Her adım öncekine bağımlı → **paralelleştirilemez**
- Uzun cümlelerde iki problem:
  - **Forward:** Hidden state sürekli çarpılınca eski bilgi "sönüyor" (long-term dependency)
  - **Backward:** Gradient'ler küçülüyor, öğrenme zorlaşıyor (vanishing gradient)

**CNN (Convolutional Neural Networks):**
- Sabit pencere boyutu (örn: 5 token)
- Uzak token'lar birbirini göremez
- Birden fazla katman lazım uzun mesafe için

**Transformer'ın farkı:**

```
RNN:   t₁ → t₂ → t₃ → t₄  (sıralı, yavaş)
Trans: t₁ ↔ t₂ ↔ t₃ ↔ t₄  (paralel, hızlı)
       ↕    ↕    ↕    ↕
      hepsi herkesi görüyor
```

- **Paralel işlem:** Tüm token'lar aynı anda işlenir → GPU'lar tam kapasite çalışır
- **Global görüş:** Her token, cümlenin tamamını tek seferde "görür"
- **Uzun mesafe:** İlk ve son kelime arasında bilgi kaybı yok

Bu üç özellik, Transformer'ı hem **hızlı** hem **güçlü** yaparak NLP'de devrim yarattı.

---

## Ek A: Hızlı Referans Formülleri

### Temel Attention

```
Q = X Wq
K = X Wk
V = X Wv

A = softmax(QKᵀ / √d)
Z = A × V
```

### Multi-Head Attention

```
head_i = Attention(X Wqⁱ, X Wkⁱ, X Wvⁱ)
MultiHead = Concat(head_1, ..., head_h) × Wo
```

### Boyutlar

| Tensor | Boyut |
|--------|-------|
| X (girdi) | n × d |
| Wq, Wk, Wv | d × dₖ |
| Q, K, V | n × dₖ |
| A (attention weights) | n × n |
| Z (çıktı) | n × dₖ |

---

## Ek B: Anahtar Terimler Sözlüğü

| Terim | Tanım |
|-------|-------|
| **Static Embedding** | Bağlamdan bağımsız, sabit kelime vektörü |
| **Contextual Embedding** | Bağlama göre değişen kelime vektörü |
| **Attention Score** | Bir token'ın diğerine verdiği önem ağırlığı |
| **Softmax** | Skorları olasılık dağılımına çeviren fonksiyon |
| **Multi-Head** | Birden fazla paralel attention katmanı |
| **Süperpozisyon** | Tüm olası anlamların aynı anda var olması |
| **Contextualization** | Kelimenin bağlama göre anlam kazanması |