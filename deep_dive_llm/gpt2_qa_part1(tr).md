# GPT-2 Mimarisi: Soru-Cevap Formatında Öğrenme Notları

> Bu doküman, GPT-2 mimarisini "neden" ve "nasıl" perspektifinden soru-cevap formatında inceler.

---

## Bölüm 1: Temel Kavramlar

---

### S-1: GPT-2 mimarisinde veri nasıl akıyor ve temel bileşenler ne işe yarıyor?

**Soru Detayı:** Kitap GPT-2 implementasyonunu anlatıyor, ben de öğrenmek için yapacağım ama biraz nedensellik üzerine, "neden" ve "how" temelli bir açıklama istiyorum.

**Cevap:**

Harika bir yaklaşım. 6 yıllık ML/CV mühendisliği tecrübene ve LLM eğitim/ince ayar (fine-tuning) konularındaki derin ilgine dayanarak, "kodu kopyala-yapıştır" yapmaktan ziyade, mimari kararların **nedenselliğine (causality)** odaklanmak en doğrusu olacaktır.

Kitapta (Chapter 4) anlatılan GPT-2 mimarisinin yapı taşlarını, bir mühendisin gözünden "Neden bu parça var?" ve "Nasıl çalışır?" ekseninde inceleyelim.

#### 1. Büyük Resim: Veri Akışının Mantığı

GPT-2 bir "Decoder-only" (Sadece Kod Çözücü) mimarisidir. Amacı basittir: Olasılık dağılımını manipüle ederek bir sonraki token'ı tahmin etmek.

Veri akışı şöyledir:
Token ID'ler → Embedding (Anlamsal Uzay) + Positional Embedding (Sıra Bilgisi) → **Transformer Blokları (İşlemci)** → LayerNorm → Linear Head (Kelime Dağarcığına İzdüşüm) → Softmax.

Şimdi bu akışın kritik "neden"lerine bakalım:

#### 2. Normalizasyon: Neden LayerNorm?

Derin ağlarda (Deep Networks) en büyük sorunlardan biri, katmanlar ilerledikçe aktivasyonların kontrolsüz büyümesi veya küçülmesidir (Exploding/Vanishing Gradients).

* **Neden Batch Norm değil de LayerNorm?**
CV (Bilgisayarlı Görü) geçmişinden Batch Normalization'a aşinasındır. Ancak LLM'lerde **Layer Normalization** tercih edilir.
* **Nedeni:** Batch Norm, batch içerisindeki diğer örneklerin istatistiklerine bağımlıdır. Ancak dilde, cümlelerin uzunluğu değişkendir ve bir kelimenin anlamı batch'teki diğer cümlelere değil, kendi bağlamına (context) bağlı olmalıdır.

* **Nasıl Çalışır?:** LayerNorm, istatistikleri (ortalama ve varyans) "batch boyutu" üzerinden değil, **"özellik (feature/embedding)" boyutu** üzerinden hesaplar. Her girdiyi kendi içinde, ortalaması 0 ve varyansı 1 olacak şekilde normalize eder.

* **GPT-2 Farkı (Pre-LayerNorm):** Orijinal "Attention Is All You Need" makalesinde normalizasyon bloktan *sonra* yapılırken, GPT-2'de kararlılığı artırmak için Attention ve FFN bloklarından **önce** yapılır.

#### 3. Aktivasyon Fonksiyonu: Neden GELU?

Klasik ağlarda ReLU (Rectified Linear Unit) standarttır. Ancak GPT-2 ve modern LLM'lerde **GELU (Gaussian Error Linear Unit)** kullanılır.

* **Neden?:** ReLU, negatif değerlerde 0 çıktısı verir ve gradyanı öldürür (Dead ReLU sorunu). Dil modelleri gibi karmaşık yapılarda, negatif girdilerin de (küçük de olsa) bir bilgi taşıması istenir.

* **Nasıl Çalışır?:** GELU, ReLU'nun "yumuşatılmış" halidir. Deterministik bir eşikleme (0 altı atılır) yerine, olasılıksal bir geçiş sunar. Negatif değerlerde hemen 0'a düşmez, hafif bir eğimle (non-zero gradient) bilgi akışına izin verir. Formülü kümülatif dağılım fonksiyonuna dayanır.

#### 4. Feed Forward Network (FFN): "Hafıza" Katmanı

Attention mekanizması tokenlar arası ilişkiyi (routing) çözerken, FFN katmanı bu bilgiyi işleyen ve modelin "olgusal bilgisini" tuttuğu düşünülen yerdir.

* **Neden Genişleyip Daralıyor?**
FFN bloğu, embedding boyutunu (örn. 768) önce 4 katına (3072) çıkarır, sonra tekrar 768'e indirir.

* **Nedensellik:** Bu işlem, veriyi daha yüksek boyutlu bir uzaya taşıyarak (projection), lineer olmayan ilişkileri (GELU ile) ayrıştırmayı sağlar. Tıpkı bir SVM'in veriyi ayırmak için kernel trick ile üst boyuta taşıması gibi düşünebilirsin. Model, bu geniş alanda "düşünür" ve sonucu tekrar standart boyuta sıkıştırır.

#### 5. Shortcut (Skip) Connections: Gradyan Otobanı

Derinliği 12, 24 veya 96 katmana varan modellerde, backpropagation sırasında gradyanın en başa kadar "yok olmadan" ulaşması matematiksel olarak zordur.

* **Neden?:** Vanishing Gradient problemini çözmek için.

* **Nasıl?:** Bloğun girdisi (x), bloğun çıktısına (F(x)) doğrudan eklenir: x + F(x).

* Bu sayede, eğer blok (F(x)) hiçbir şey öğrenmezse bile, gradyan x üzerinden (bir "otoban" gibi) kesintisiz akabilir. Bu, modelin "en azından girdi kadar iyi" olmasını garanti eder ve derinleşmeyi mümkün kılar.

#### 6. Mimari Bir Nüans: Weight Tying (Ağırlık Bağlama)

Kitapta uygulama sırasında kullanılmasa da, orijinal GPT-2 makalesinde "Weight Tying" tekniğinden bahsedilir.

* **Nedir?:** Token Embedding matrisi (Giriş) ile en sondaki Linear Head matrisinin (Çıkış) aynı ağırlıkları paylaşmasıdır.

* **Neden?:** Parametre sayısını ciddi oranda düşürür (yaklaşık 40 milyon parametre tasarrufu sağlar).

* **Mühendislik Yorumu:** Pratikte, özellikle modern LLM eğitimlerinde (Llama vb.), bu katmanları ayırmak (untied) genellikle modelin ifade gücünü artırdığı için tercih edilir. Kitaptaki implementasyon da daha iyi performans için bunları ayrı tutmuştur.

#### Özet: GPT-2 Blok Yapısı (Kod Karşılığı)

Bir mühendis olarak `TransformerBlock` sınıfına baktığında şu akışı görmelisin:

1. **Girdi (x):** [Batch, Seq_Len, 768]
2. **LayerNorm 1:** İstatistiksel stabilizasyon.
3. **Attention:** "Hangi token'lar birbiriyle ilgili?" sorusunun cevabı.
4. **Shortcut 1:** x = x + Attention(x)
5. **LayerNorm 2:** Tekrar stabilizasyon.
6. **FFN (MLP):** "Bilgiyi işle ve özellik çıkar." (768 -> 3072 -> 768)
7. **Shortcut 2:** x = x + FFN(x)

Bu yapı, bilginin kaybolmadan, sürekli zenginleştirilerek (residual learning) model boyunca akmasını sağlar.

---

## Bölüm 2: Attention ve FFN Katmanları

---

### S-2: Attention Layer tam olarak ne yapıyor?

**Cevap:**

Transformer'ın beyni burasıdır. "Self-Attention" (Öz-Dikkat) mekanizması şunu yapar:

* **Bağlam Kurma:** Model bir kelimeye (token) baktığında, cümledeki diğer *bütün* kelimelerle o kelime arasındaki ilişkiye bakar.

* **Örnek:** "Ali bankaya gitti, çünkü paraya ihtiyacı vardı." cümlesini ele alalım.
  * Model "banka" kelimesini işlerken; Attention mekanizması "para" kelimesine yüksek puan (dikkat) verir. Çünkü buradaki banka, parktaki bank değil, finansal bankadır.
  * Attention katmanı, kelimelerin birbirine olan "ilgisini" (skorunu) hesaplar ve kelimeye o bağlamı yükler.

> **Görevi:** Kelimeler arasındaki anlam ilişkisini kurmak ve hangi kelimenin o anki bağlam için önemli olduğunu belirlemektir.

---

### S-3: Feed Forward Network (Linear Layer) ne işe yarıyor?

**Cevap:**

Attention katmanı bilgiyi topladı (hangi kelime hangisiyle ilişkili buldu), peki bu bilgiyi kim işleyecek? İşte Linear Layer (genelde MLP - Multi Layer Perceptron olarak geçer).

Her Transformer bloğunda Attention'dan sonra bir **Feed Forward (Linear)** katman gelir.

* **Bilgiyi İşleme ve "Düşünme":** Attention katmanı "bakma ve odaklanma" yeriyse, Linear katman "düşünme ve hesaplama" yeridir.

* **Boyut Değiştirme:** Genellikle veriyi alır, boyutunu (dimension) 4 katına çıkarır (genişletir), orada karmaşık matematiksel işlemler (aktivasyon fonksiyonu, örn: GELU) uygular ve tekrar eski boyutuna sıkıştırır.

* **Bilginin Saklandığı Yer:** Araştırmalar, modelin öğrendiği "gerçek dünya bilgilerinin" (örneğin Fransa'nın başkentinin Paris olduğu bilgisi) büyük ölçüde bu Linear katmanların ağırlıklarında (weights) saklandığını göstermektedir.

---

### S-4: FFN neden var? Attention tek başına yetmiyor mu?

**Cevap:**

Harika bir soru. Attention mekanizması genellikle Transformer modellerinin "kalbi" olarak görülür, ancak parametrelerin büyük çoğunluğu (~%60-70) aslında bu **Feed Forward Network (FFN)** katmanlarında bulunur.

Bir mühendis olarak bu katmanın varlık sebebini (causality) **"İletişim"** ve **"İşlem"** ayrımı üzerinden anlamlandırmak en doğrusudur.

#### 1. Rol Ayrımı: İletişim vs. Sindirme

Transformer bloğunun içindeki iş bölümü şöyledir:

* **Attention (Multi-Head Attention):** Tokenlar arası **iletişimi** sağlar. "Ali" kelimesi, cümledeki "geldi" kelimesiyle ilişki kurar ve bağlamı toplar. Ancak Attention, bu bilgiyi derinlemesine *işlemez*, sadece *yönlendirir* (routing) ve *toplar* (aggregation).

* **Feed Forward Network (FFN):** Attention tarafından toplanan bilgiyi **sindirir** ve **işler**. Metinde belirtildiği gibi, FFN veriyi "her pozisyon için bireysel olarak" modifiye eder.

* *Analoji:* Attention bir toplantıda herkesin birbirini dinlemesi (bilgi toplama), FFN ise toplantıdan sonra herkesin kendi masasına dönüp duydukları üzerine düşünüp bir sonuç çıkarmasıdır (bilgi işleme).

#### 2. Mekanik: Neden Genişleyip Daralıyor?

FFN katmanı, yapısal olarak iki lineer katman ve bir aktivasyon fonksiyonundan (GELU) oluşur.

Buradaki en kritik mühendislik kararı **boyut manipülasyonudur**:

1. **Genişleme (Expansion):** 768 boyutlu embedding vektörü, birinci katmanda 4 katına (3072 boyuta) çıkarılır.
2. **Lineer Olmayan İşlem (Non-linearity):** Bu geniş alanda GELU aktivasyonu uygulanır.
3. **Daralma (Contraction):** İkinci katmanda tekrar 768 boyuta indirilir.

**Neden (Causality)?**
Metin buna "daha zengin bir temsil uzayının keşfedilmesi" diyor.

* Veriyi düşük boyutta (768) lineer olarak ayırmak veya anlamlandırmak zordur.
* Model, veriyi geçici olarak çok daha yüksek bir boyuta (3072) fırlatarak (projection), özelliklerin birbirinden ayrışmasını ve daha karmaşık bağıntıların çözülmesini sağlar. (Tıpkı kernel trick kullanan bir SVM gibi).
* GELU aktivasyonu bu yüksek boyutta devreye girerek modelin sadece "ezberlemesini" değil, veriyi "yorumlamasını" (non-linear transformation) sağlar.

#### 3. Konumsal Bağımsızlık (Position-wise)

Attention mekanizması tüm cümleye (sequence) aynı anda bakarken, FFN **her token'a tek tek ve bağımsız** uygulanır.

* "Ali" token'ı FFN'den geçerken, "geldi" token'ının FFN'deki işleminden tamamen habersizdir.
* Bu özellik, FFN'in modelin **"statik hafızası"** gibi davranmasını sağlar. Birçok araştırmacı, FFN katmanlarının modelin öğrendiği olgusal bilgileri (factual knowledge) anahtar-değer (key-value) çiftleri gibi sakladığını düşünmektedir.

#### Özet: Mühendislik Özeti

Eğer FFN olmasaydı; model sadece hangi kelimenin hangi kelimeye bakacağını (Attention) bilirdi, ancak bu bakıştan elde ettiği bilgiyi yorumlayacak ve dönüştürecek bir "beyin kapasitesi"ne (computation capacity) sahip olamazdı. FFN, modelin öğrendiklerini genelleştirmesini ve karmaşık veri kalıplarını işlemesini sağlayan asıl güç ünitesidir.

---

### S-5: Attention genişlik, FFN derinlik sağlıyor dedin. Neden FFN'i sonradan ekliyoruz, bu yapılan işi bozmuyor mu?

**Soru Detayı:** Neden FFN ile sağlıyoruz? Çünkü FFN derinlik için iyi, multi-head genişlik için. Ama böyle her bir sequence adımda head olmalı, yapılan bir şeyi bozuyor gibi.

**Cevap:**

Haklı bir noktadasın. Senin şüphe ettiğin yer, aslında LLM mimarisinin **"işlemci" (CPU)** ile **"hafıza/iletişim" (Bus)** ayrımının yapıldığı yerdir.

"Head'ler ile genişlik, FFN ile derinlik" sağlanması ve bunun sıralı olması bir şeyi bozmuyor, tam tersine **Attention'ın eksik bıraktığı şeyi tamamlıyor.**

Mühendisçe, "Neden sadece Attention yetmiyor?" sorusunu 3 maddede, kitabın mimarisine dayanarak cevaplayalım:

#### 1. Attention "Kopyalar", FFN "Hesaplar"

Senin teorinde sadece Head'ler olsaydı (sadece Attention), model aptal bir "Arama Motoru" gibi çalışırdı.

* **Attention'ın Yaptığı İş:** Sadece ağırlıklı ortalama almaktır (Weighted Average). "Ali" vektörünün içine %30 "Geldi", %20 "Okul" bilgisini kopyalar.
* *Problem:* Attention lineer bir işlemdir (Softmax hariç). Sadece var olan bilgiyi bir yerden bir yere taşır (Routing).

* **FFN'in (Feed Forward) Yaptığı İş:** Attention'ın taşıdığı bu bilgiyi alır ve **sindirir**.
* Kitapta belirtildiği gibi FFN, embedding boyutunu 4 katına (768 → 3072) çıkarır.
* Burada GELU aktivasyonu ile **Lineer Olmayan (Non-linear)** bir işlem uygular.
* **Sonuç:** Attention "Ali okula geldi" bağlamını toplar. FFN ise bu bağlamdan "O zaman Ali şu an evde değildir" mantıksal çıkarımını yapar.

Eğer FFN olmasaydı (veya seri olmasaydı), model sadece kelimeleri eşleştirir ama **muhakeme (reasoning)** yapamazdı.

#### 2. "Bozmuyor" Çünkü Sırayla Çalışmıyor (Position-wise)

Senin endişen: *"Her sequence adımında head olmalı, yapılan şeyi bozuyor."*
Cevap: FFN sequence'ı (sıralamayı) **karıştırmaz.**

Kitaptaki şu ifade çok kritik:

> "In contrast, the feed forward network modifies the data individually at each position."

Yani:

* **Attention:** Sequence'taki tüm tokenları birbirine karıştırır (Mixing tokens). "Ali" ile "Veli" konuşur.
* **FFN:** Her token'a **tek tek ve bağımsız** uygulanır. 1. sıradaki token FFN'e girerken, 2. sıradakinden habersizdir.

Bu yüzden FFN, Attention'ın kurduğu sequence yapısını **bozamaz**. Sadece her bir token'ın o anki vektörünü (Attention'dan gelen bağlamla şişmiş halini), kendi içinde işleyip zenginleştirir.

#### 3. Residual Stream (Artık Akış) Sayesinde Yıkım Yok

Hala "seri olması önceki bilgiyi eziyor" diye düşünüyorsan, kodun içindeki şu satıra tekrar bakmalısın:

`x = x + self.ff(x)` (Shortcut connection for feed forward block).

Bu şu demektir:

1. **Girdi (x):** Attention'dan çıkan, bağlamı toplanmış veri.
2. **FFN(x):** Bu veriden çıkarılan **yeni özellikler** (delta).
3. **Toplama:** Eski veriyi koru, üzerine yeni özellikleri ekle.

Bu mimari (Residual Connection), FFN'in önceki katmanların (ve head'lerin) kurduğu yapıyı bozmasını matematiksel olarak engeller. FFN sadece "gerekiyorsa" vektöre küçük bir yön değişikliği (düzeltme) ekler. Eğer bozacaksa, FFN çıktısı sıfıra yakınsar ve veri bozulmadan geçer (Identity mapping).

#### Özet

* **Attention (Heads):** **"Kiminle konuşmalıyım?"** (İletişim / Routing).
* **FFN (Layers):** **"Duyduğum şeyden ne anlam çıkarmalıyım?"** (İşlem / Reasoning).

Bunlar birbirine alternatif değil, birbirinin tamamlayıcısıdır. Attention malzemeyi toplar, FFN yemeği pişirir. Bu yüzden Attention'dan çıkan veriyi FFN'e (seri olarak) sokmak zorundayız.

---

### S-6: Hangi adımlarda Layer Normalization kullanılır?

**Cevap:**

GPT-2 mimarisinde **Layer Normalization (LN)** işleminin kullanıldığı **3 kritik nokta** tanımlanmıştır.

Mühendislik bakış açısıyla bu noktaları ve "Pre-LayerNorm" tasarımının nedenini şöyle özetleyebiliriz:

#### 1. Transformer Bloğu İçinde (2 Kez)

GPT-2'de, her bir Transformer bloğunun içinde iki farklı yerde normalizasyon yapılır. Buna **"Pre-LayerNorm"** mimarisi denir; çünkü normalizasyon, ana işlemlerden (Attention ve FFN) **önce** gerçekleşir.

* **Adım 1: Attention Öncesi (LayerNorm 1):**
Girdi bloğa girdiği anda, **Multi-Head Attention** mekanizmasına gönderilmeden hemen önce normalize edilir.

* *Neden?* Attention mekanizması, tokenlar arası ilişkileri (dot product) hesaplar. Eğer girdilerin ölçekleri bozuksa (biri çok büyük, biri çok küçük), attention skorları dengesizleşir ve gradyan akışı bozulur.

* **Adım 2: Feed Forward Öncesi (LayerNorm 2):**
Attention çıktısı (ve shortcut bağlantısı) toplandıktan sonra, **Feed Forward Network (FFN)** katmanına girmeden hemen önce tekrar normalize edilir.

* *Neden?* FFN katmanı (genişleyip daralan MLP), aktivasyonları lineer olmayan bir uzaya taşır. Girişin istikrarlı (mean=0, var=1) olması, buradaki GELU aktivasyonunun verimli çalışmasını sağlar.

#### 2. Modelin En Sonunda (1 Kez)

Tüm Transformer blokları (örneğin 12 adet) bittikten sonra, modelin nihai çıktısını üretmeden hemen önce bir kez daha normalizasyon yapılır.

* **Adım 3: Final Output Öncesi (Final LayerNorm):**
Son Transformer bloğundan çıkan veri, **Linear Output Layer**'a (kelime dağarcığına izdüşüm yapan katman) girmeden önce normalize edilir.

* *Neden?* Modelin son katmanı, 768 boyutlu vektörü 50.257 boyutlu kelime uzayına eşler. Buradaki sayısal kararsızlıklar, yanlış kelime tahminlerine (logits) ve loss değerinin patlamasına yol açabilir. Bu son "temizlik", tahminlerin güvenilir olmasını sağlar.

#### Özet Tablo

Dokümandaki `TransformerBlock` ve `GPTModel` kodlarına baktığında akış şöyledir:

1. `x = self.norm1(x)` → Attention'a girer.
2. `x = self.norm2(x)` → FeedForward'a girer.
3. *(Bu blok 12 kez tekrarlanır)*
4. `x = self.final_norm(x)` → Çıktı katmanına (Head) girer.

**Mimari Not:** Orijinal Transformer makalesinde (Vaswani et al.) normalizasyon işlemlerden **sonra** (Post-LayerNorm) yapılıyordu. Ancak GPT-2, eğitimi daha kararlı hale getirdiği için bu **Pre-LayerNorm** yapısını benimsemiştir.

---

### S-7: ReLU ve GELU arasındaki fark nedir? Neden GELU tercih ediliyor?

**Cevap:**

Derin öğrenmede bir "standart" olan ReLU'nun, Büyük Dil Modelleri (LLM) söz konusu olduğunda neden yerini GELU ve SwiGLU gibi daha karmaşık fonksiyonlara bıraktığını anlatıyor.

Bir ML mühendisi olarak, buradaki değişimi **optimizasyon peyzajı (optimization landscape)** ve **bilgi akışı** üzerinden şöyle okumalıyız:

#### 1. ReLU: Eski Dostun Limitleri

ReLU (max(0, x)), derin öğrenme devrimini başlatan kahramandır çünkü hesaplama maliyeti çok düşüktür ve pozitif değerlerde gradyanın kaybolmasını (vanishing gradient) engeller.

Ancak LLM'ler için iki büyük sorunu vardır:

* **Ölü Nöron (Dead ReLU) Sorunu:** ReLU, 0'ın altındaki tüm değerleri mutlak bir "hiçliğe" (sıfıra) indirger. Negatif bölgede gradyan tamamen ölür. Dil modellerinde, bir kelimenin veya bağlamın "negatif" bir ilişkiye sahip olması da (örn. "kral" ama "kadın" değil) bir bilgidir. ReLU bu nüansı tamamen silebilir.

* **Keskin Köşe (Non-differentiability at 0):** ReLU'nun grafiği 0 noktasında keskin bir köşe yapar. Bu, optimizasyon sırasında, özellikle ağ çok derinleştiğinde (GPT-2 gibi), matematiksel olarak "sert" bir geçiş yaratır ve öğrenmeyi zorlaştırabilir.

#### 2. GELU (Gaussian Error Linear Unit): GPT-2'nin Tercihi

GPT-2 (ve aslında BERT, GPT-3) **GELU** kullanır. GELU'nun ReLU'ya kıyasla daha "pürüzsüz" (smooth) ve karmaşık olduğu belirtiliyor.

* **Nedensellik (Why):** GELU, deterministik bir "aç/kapa" anahtarı (ReLU) yerine, istatistiksel bir yaklaşım sergiler.

* **Mekanik:** Bir girdiyi (x), o girdinin standart bir Gauss dağılımında nerede durduğuna (Φ(x)) göre ağırlıklandırır: GELU(x) = x · Φ(x).

* **Avantajı:** Grafiğe dikkat edersen, GELU negatif değerlerde (örneğin -1 civarında) hemen 0 olmaz, hafifçe aşağı kıvrılır ve sonra 0'a yaklaşır.
  * Bu **"kavis"**, negatif girdilerin de modele (küçük de olsa) bir gradyan/bilgi katkısı sağlamasına izin verir.
  * Fonksiyonun türevi her noktada süreklidir (smoothness), bu da optimizatörün (AdamW gibi) loss fonksiyonunun dibini daha hassas bulmasını sağlar.

#### 3. SwiGLU: Modern Standart (Llama Dönemi)

SwiGLU'dan (Swish-Gated Linear Unit) "dikkate değer bir diğer örnek" olarak bahsedilmektedir. GPT-2'de kullanılmasa da, bugün Llama 3, Mistral gibi SOTA (State-of-the-Art) modellerin standardı budur.

* **Mühendislik Farkı:** SwiGLU bir aktivasyon fonksiyonundan ziyade, bir "kapılama" (gating) mekanizmasıdır. İki farklı lineer katmanın çıktısını birbirine çarparak (biri aktivasyondan geçer, diğeri geçmez) bilginin ne kadarının akacağına karar verir. Bu, modelin bilgi akışını daha hassas kontrol etmesini sağlar ancak parametre sayısını artırır.

#### Özet: Neden Değiştirdiler?

**"Dil modellemesi, görüntü sınıflandırmadan daha nüanslı bir iştir."**
ReLU'nun kaba "kesip atma" mantığı yerine, GELU'nun **olasılıksal ve yumuşak geçişli** yapısı, modelin kelimeler arasındaki çok ince istatistiksel bağları koparmadan öğrenmesini sağlar. Bu yüzden GPT-2 kodlarken `nn.ReLU()` yerine, metinde verilen `GELU()` sınıfını implemente edeceğiz.

---

## Bölüm 3: Blok Sayısı ve Derinlik

---

### S-8: Blok sayısı 12 yerine 24 olması ne işe yarar?

**Cevap:**

GPT-2 makalesinde (Radford et al.) aslında 4 farklı model boyutu tanıtılmıştır. Bahsettiğin 12 bloklu olan "Small" versiyondur. "Medium" olan 24, "Extra Large" olan ise 48 bloğa sahiptir.

Peki blok (layer/katman) sayısını artırmak (12'den 24'e çıkarmak) ne işe yarar?

* **Soyutlama Seviyesi (Abstraction):** İnsan beyni gibi düşünebilirsin. İlk bloklar daha basit dil bilgisi kurallarını (kelime türleri, basit ekler) öğrenirken, derinlere inildikçe (24. bloğa doğru) model daha karmaşık, anlamsal ve mantıksal bağları (ironi, uzun vadeli konu bütünlüğü, kodlama mantığı vb.) kavrar.

* **Kapasite Artışı:** Daha fazla blok, modelin parametre sayısının artması demektir. Bu da modelin "hafızasının" ve "işleme yeteneğinin" artması anlamına gelir. 24 bloklu model, 12 blokluya göre çok daha karmaşık cümleleri ve bağlamları "aklında tutabilir."

> **Özetle:** Blokları art arda dizmek (Depth), modelin basit kalıplardan karmaşık düşüncelere geçmesini sağlar. 12 bloklu model "Ali topu attı"yı iyi anlarken, 24 bloklu model "Ali'nin topu atmasının sebebi, geçmişte yaşadığı travmaydı" gibi nedensel bağları daha iyi kurar.

---

### S-9: Bloklar neden paralel değil de ardışık (sequential)?

**Soru Detayı:** Eğer problem daha geniş cevap vermekse, neden paralel değil de sequence bloklar?

**Cevap:**

Bu soru "Deep Learning"in (Derin Öğrenme) mantığına işaret eder. Neden 12 bloğu yan yana (paralel) koyup hepsinin sonucunu toplayıp vermiyoruz?

#### 1. Hiyerarşik Öğrenme

Bilgi işleme hiyerarşiktir. Bir matematik problemini çözerken önce sayıları tanımanız, sonra toplamayı öğrenmeniz, en son integrali çözmeniz gerekir.

* **Blok 1'in çıktısı, Blok 2'nin girdisidir.**
* Blok 1 ham veriyi işler, bir seviye yukarı taşır. Blok 2, Blok 1'in "yorumladığı" veriyi alır, daha da geliştirir.
* Eğer paralel olsaydı, Blok 12 henüz işlenmemiş ham veriye bakmak zorunda kalırdı ve "derin" analiz yapamazdı. Her blok bir önceki bloğun omuzlarında yükselir.

#### 2. Autoregressive Yapı

Ayrıca modelin çalışma prensibi **Autoregressive**'dir. Yani model, bir sonraki kelimeyi tahmin ederken sadece *geçmişe* bakabilir. Geleceği göremez. Bu yüzden işlem sırası soldan sağa (kelime bazında) ve aşağıdan yukarıya (blok bazında) akmak zorundadır.

---

### S-10: Paralel yapmak daha iyi olmaz mı? Sequence gürültüye yol açmıyor mu?

**Soru Detayı:** Problem aynı seviyede ise, yani komplekslik artmıyorsa, sequence yapmak yerine paralel yapmak ve aynı attention layer'daki paralel head sayısı gibi iyi olmaz mı? Sonuçta bu sequence durum gürültüye yol açacaktır.

**Cevap:**

Harika bir noktaya parmak bastın. Aslında bu düşünce tarzın, modern mimarilerin (örneğin Mixture of Experts - MoE) temelini oluşturan bir sezgi.

Ancak klasik Transformer mimarisinde (GPT-2 gibi) blokların neden **paralel** değil de **ardışık (sequential)** dizildiğinin matematiksel ve mantıksal çok kritik bir sebebi var: **Fonksiyon Kompozisyonu (Function Composition) ve Lineer Olmayan Dönüşüm.**

Senin "gürültü" endişen teoride doğru (buna "Vanishing Gradient" veya bilgi kaybı denir), ama Transformer'lar bunu **Residual Connections (Artık Bağlantılar)** ile çözüyor.

Gel bu konuyu "paralel vs seri" devre mantığı gibi inceleyelim:

#### 1. Neden Paralel Olamaz? (Matematiksel Engel)

Eğer 12 bloğu peş peşe (sequence) değil de yan yana (paralel) koysaydık ve çıktılarını toplasaydık ne olurdu?

Derin Öğrenme'nin (Deep Learning) bütün olayı **f(g(h(x)))** diyebilmektir. Yani iç içe geçmiş fonksiyonlar yaratmaktır.

* **Sıralı (Sequence):** Girdi → [Blok 1] → (yarı işlenmiş veri) → [Blok 2] → ...
  * Burada Blok 2, Blok 1'in çıkardığı özellikleri kullanır. Blok 1 "kenarları" bulur, Blok 2 "şekilleri" bulur, Blok 3 "nesneleri" tanır. Bu bir hiyerarşidir.

* **Paralel:** Girdi → [Blok 1], Girdi → [Blok 2], ...
  * Burada **tüm bloklar ham veriye bakar.** Hepsi aynı seviyede (low-level) işlem yapar. Hiçbiri diğerinin bulduğu feature'ın üzerine koyamaz.

**Sonuç:** Paralel yapı sadece "geniş" (wide) olur ama "derin" (deep) olmaz. Karmaşık problemleri (dilin mantığı, kod yazma vb.) çözmek için genişlikten ziyade derinliğe (ardışık soyutlamaya) ihtiyacımız vardır.

#### 2. "Gürültü" Sorunu Nasıl Çözüldü? (Residual Connections)

Senin *"bu sequence durum gürültüye yol açacaktır"* endişen, Derin Öğrenme tarihinin en büyük problemlerinden biriydi (2015 öncesi). Veri 24 kattan geçerken bozuluyordu (Vanishing Gradient).

Bunu **Residual Connection (Skip Connection)** ile çözdüler. Mimariye dikkat edersen her bloğun etrafında bir "bypass" hattı vardır.

Formül şudur: **Output = Input + Block(Input)**

Yani:

1. Blok, veriyi alır işler (Block(Input)).
2. Ama çıktıyı verirken, **girdinin orijinal halini (Input)** de sonuca ekler.

Bu ne sağlar?

* Model bir blokta hata yaparsa veya "gürültü" üretirse, sistem ana veriyi (Input) korur.
* Bilgi "otobandan" (skip connection) hiç bozulmadan en sona kadar akabilir.
* Bu sayede 100 katman bile olsa, ilk katmandaki bilgi kaybolmaz. Sequence yapı gürültü yaratmaz, aksine bilgiyi rafine eder.

#### 3. Attention Head'leri Neden Paralel?

Şu soruyu sorabilirsin: *"Bloklar ardışık olmak zorunda dedin, peki neden bir bloğun içindeki Attention Head'leri (kafaları) paralel?"*

Çünkü **aynı soyutlama seviyesinde** farklı şeylere bakmaları gerekir.

* Bir cümleyi okurken (Örn: "Elma yere düştü"):
  * Head 1 (Paralel): "Kim?" sorusuna odaklanır (Elma).
  * Head 2 (Paralel): "Nereye?" sorusuna odaklanır (Yere).
  * Head 3 (Paralel): "Ne yaptı?" sorusuna odaklanır (Düştü).

Bu işlemler birbirine bağımlı değildir, eş zamanlı (paralel) yapılabilir. Ama bu bilgilerin birleştirilip "Yerçekimi kanunu" çıkarımının yapılması (Reasoning) bir sonraki **Sequence** katmanında olur.

#### 4. Senin Fikrine Giden Yol: Mixture of Experts (MoE)

Senin *"Neden paralel yapmıyoruz?"* fikrin aslında şu an en popüler mimarilerden biri olan **Mixture of Experts (MoE)** (örn: Mixtral, GPT-4 olduğu söylenen yapı) ile hayata geçti.

MoE'de, Feed Forward katmanları (FFN) tek bir büyük blok olmak yerine, **paralel 8 uzman (expert)** bloğa bölünür.

* Girdi geldiğinde hepsi çalışmaz, bir "Router" (yönlendirici) devreye girer ve "Bu token için şu 2 paralel blok çalışsın, diğerleri sussun" der.

**Özetle:**
Haklısın, genişlik (paralellik) önemlidir ama bu **çeşitlilik** içindir. Derinlik (sequence) ise **zeka ve mantık kurma** (reasoning) içindir. Bir LLM'in "akıllı" olmasını sağlayan şey, blokların birbirinin omzuna basarak yükselmesidir.

---

### S-11: MoE modellerinde de 36+ defa sequence var. Neden hala derinlik gerekiyor?

**Soru Detayı:** MoE'de de 36 defa sequence ediyor veya daha çok. Paralel expert'ler varsa neden hala derin?

**Cevap:**

Çok doğru bir gözlem. Evet, Mixtral 8x7B veya Grok-1 gibi MoE (Mixture of Experts) modelleri de **derindir** (örneğin 32, 40 veya daha fazla katman/sequence içerir).

Buradaki kafa karışıklığını gidermek için "Genişlik" (Parallelism/MoE) ile "Derinlik" (Sequence/Layers) kavramlarının **görev dağılımını** netleştirmemiz lazım. İkisi birbirinin alternatifi değil, tamamlayıcısıdır.

Neden hala 36 katman (sequence) kullanıyoruz? İşte mühendislik sebebi:

#### 1. Fabrika Bandı Analojisi (En Net Açıklama)

Modelin bir araba fabrikası olduğunu düşünelim.

* **Sequence (Katmanlar - Derinlik):** Üretim bandının istasyonlarıdır.
  * İstasyon 1: Şasiyi koyar.
  * İstasyon 10: Motoru takar.
  * İstasyon 36: Boyayı yapar.
  * *Neden Sequence şart?* Boyayı (İstasyon 36), şasiyi koymadan (İstasyon 1) havaya sıkamazsın. Motor takılmadan tekerlek takılamaz. Bilgi işleme sürecinde **öncelik-sonralık ilişkisi (dependency)** vardır.

* **MoE (Paralellik - Genişlik):** İstasyonun içindeki uzman ustalardır.
  * İstasyon 15'e (Motor Takma Katmanı) geldiğimizde, orada bekleyen 8 farklı usta (Expert) vardır.
  * Gelen araba Ferrari ise "Spor Araba Motorcusu" (Expert 1) devreye girer.
  * Gelen araba Kamyon ise "Dizel Motorcusu" (Expert 4) devreye girer.
  * Diğer 7 usta o an çalışmaz (Sparse Activation).

> **Sonuç:** Eğer 36 katmanı iptal edip her şeyi paralel yapsaydık; fabrikadaki tüm parçaları yere yığmış olurduk ama arabayı monte edecek bir **süreç** olmazdı. MoE, her aşamada "daha bilgili" uzmanlar sunar ama montajın (düşünmenin) sırasını değiştiremez.

#### 2. Bilginin Dönüşümü (Feature Extraction)

Teknik tarafta, input olarak giren token (kelime) embedding'i, 1. kattan 36. kata çıkana kadar **kimlik değiştirir.**

* **Katman 1-5 (Yüzey Yapı):** Model burada sadece kelime köklerine, dilbilgisine, virgülün yerine bakar.
* **Katman 10-20 (Anlamsal Yapı):** Kelimelerin eş anlamlılarını, cümledeki özne-yüklem ilişkisini çözer.
* **Katman 30-36 (Mantıksal Yapı):** Burası en kritik yerdir. İroni, kod mantığı, matematiksel çıkarım veya uzun vadeli hafıza burada oluşur.

Eğer sen "Paralel olsun" deyip 36 katmanı 1'e indirseydin; model "Ali camı kırdı" cümlesindeki dilbilgisini anlardı ama **neden** kırdığını veya bunun sonucunda ne olacağını (muhakeme/reasoning) hesaplayacak işlem derinliğine sahip olamazdı.

#### 3. "Non-Linearity" (Doğrusal Olmayan Dönüşüm) İhtiyacı

Yapay zekanın "öğrenme" dediğimiz şeyi yapabilmesi için veriyi büküp katlaması gerekir. Her blokta (layer) bir **Aktivasyon Fonksiyonu** (GELU, Swish vs.) vardır.

Matematiksel olarak:

* Bir problemi çözmek için uzayı (space) bükmen gerekir.
* Tek bir katman (ne kadar geniş/paralel olursa olsun) uzayı sadece bir kez bükebilir.
* 36 katman, uzayı 36 kez büker, katlar ve karmaşık veri noktalarını (örneğin kedilerle köpekleri veya doğru kodla hatalı kodu) birbirinden ayırabilir.

#### Özet Tablo: Hangisi Ne İşe Yarar?

| Özellik | **Derinlik (Sequence / Layer Sayısı)** | **Genişlik / MoE (Paralel / Expert Sayısı)** |
| --- | --- | --- |
| **Görevi** | Muhakeme (Reasoning), Mantık, Karmaşık ilişkileri çözme. | Bilgi Dağarcığı (Knowledge), Olgusal gerçekler, Farklı alanlarda uzmanlık. |
| **Benzeri** | Bir insanın düşünme süresi (IQ). | Bir kütüphanedeki kitap sayısı (Hafıza). |
| **Eksikliği** | Çok derin olursa eğitmek zordur (Gradient sorunları). | Çok geniş olursa RAM'e sığmaz (VRAM sorunu). |

**Kritik Not:** MoE'nin olayı, **inference maliyetini artırmadan** modelin "Bilgi Dağarcığını" (Parametre sayısını) devasa boyutlara çıkarmaktır. Ama "Zeka/Muhakeme" kapasitesi hala o 36 katmanlık sequence akışına (zincirleme düşünmeye) bağlıdır.

---

## Bölüm 4: Residual Connections

---

### S-12: Neden Attention ve FFN sonrası eski değerle toplama (residual) yapıyoruz? Amaç dönüştürmek değil miydi?

**Soru Detayı:** Neden transformer block'ta multi-head attention ve dropout sonrası input ile toplama yapıyoruz? Amacımız veriyi dönüştürmek değil miydi? Neden ekleyerek dönüşüme zarar verdik?

**Cevap:**

Harika bir ters köşe soru. Genelde "bu ne işe yarar" diye sorulur, sen ise "işi bozmuyor muyuz?" diye soruyorsun. Bu mühendislik şüpheciliği çok değerli.

Cevap şu: Evet, dönüşüme "zarar" veriyoruz, ama bunu **bilerek** yapıyoruz. Çünkü amacımız veriyi tamamen değiştirmek değil, onu **"cilalamak" (refine etmek).**

Bilgisayarlı Görü (CV) geçmişinden **ResNet** (Residual Networks) mimarisini hatırlarsın. GPT'deki bu yapı (Shortcut Connection), tam olarak ResNet'in getirdiği devrimdir. İşte nedensellikleri:

#### 1. Amaç "Dönüşüm" Değil, "Fark" (Residual) Öğrenmektir

Klasik bir ağda katman H(x) fonksiyonunu öğrenmeye çalışır ve çıktısı y = H(x) olur. Yani girdiyi alır, ezer, büker ve yeni bir şey üretir.

Ancak "Shortcut Connection" ile formül şuna dönüşür:

**y = F(x) + x**

Burada F(x), artık çıktının *kendisi* değil, girdiye eklenecek olan **"fark" (residual)** olur.

* **Mühendislik Mantığı:** Modelin "bu girdiyi alıp tamamen yeni bir şeye dönüştür" demesi zordur. Ancak "bu girdiyi al ve üzerine şu küçük bağlamsal bilgiyi ekle" (delta) demesi çok daha kolaydır.

* **Zarar Vermiyoruz, Koruyoruz:** Attention mekanizması bazen "bu token için eklenecek yeni bir bilgi yok" sonucuna varabilir. Bu durumda F(x) sıfıra yakınsar ve çıktı otomatik olarak x (orijinal girdi) olur. Eğer toplama yapmasaydık, modelin x'i korumak için Identity Matrisini (birim matris) sıfırdan öğrenmesi gerekirdi ki bu zordur.

#### 2. Gradyan Otobanı (Gradient Superhighway)

Bu toplama işleminin en hayati nedeni eğitim sırasındadır (Backpropagation).

Türev alırken toplama işleminin (x + F(x)) özelliği nedir? Türevi dağıtır.

**∂(x + F(x))/∂x = 1 + ∂F(x)/∂x**

Bu **"1 + "** ifadesi her şeyi değiştirir.

* Zincir kuralı işlerken, gradyanın bir kopyası F(x) (Attention/FFN) bloğunun içine girerken, diğer kopyası **hiç bozulmadan, çarpılmadan, ezilmeden** direkt olarak bir önceki katmana akar (Skip Connection üzerinden).

* Metinde belirtildiği gibi, bu bağlantılar "gradyan akışını korur" ve derin ağlarda gradyanın kaybolmasını (vanishing gradient) engeller. Eğer bu toplama olmasaydı, 12 veya 48 katman boyunca sürekli çarpılan gradyanlar sıfırlanır ve ilk katmanlar hiçbir şey öğrenemezdi.

#### 3. Bilgi İşleme Teorisi: "Hafıza"yı Korumak

GPT gibi modellerde, token embedding'ler (veya gizli durumlar) model boyunca ilerlerken bir "taşıyıcı bant" üzerindeki ürün gibidir.

* Her Transformer bloğu (Attention + FFN), banttaki ürüne (vektöre) yeni bir parça ekler, boyar veya yontar.
* Eğer toplama yapmasaydık (y = F(x)), her blok banttaki ürünü alıp çöpe atar ve yerine yenisini koyardı. Bu durumda model, en baştaki "Ali" kelimesinin orijinal embedding bilgisini 12. katmana kadar taşıyamazdı.
* Toplama yaparak, orijinal "Ali" bilgisini koruruz, üzerine "cümlenin öznesi olduğu" bilgisini ekleriz.

#### Özetle

Veriyi bozmuyoruz, tam tersine **kaybolmasını engelliyoruz.**
Attention bloğuna diyoruz ki: *"Sen sadece gerekli olan **ekstra** bilgiyi (bağlamı) hesapla, ana veriyi ben zaten yan taraftan (shortcut) geçiriyorum, çıkışta birleştiririz."* Bu, derin öğrenme modellerinin çok katmanlı (deep) olabilmesini sağlayan temel mekaniktir.

---

### S-13: Attention ile veriye göreceli değerler verdik, sonra eski değerle topluyoruz. Bu mantıklı mı?

**Soru Detayı:** Neden attention yapıyoruz? Veriye göreceli input değerlerini vermek için. Şimdi gidip günün sonunda eski değerle topluyoruz. Bu yapılan işi bozmuyor mu?

**Cevap:**

Sorduğun soru çok haklı: *"Veriyi attention ile o kadar işleyip, bağlamı bulup, sonra neden tekrar ham haliyle topluyoruz? Bu işlem yaptığımız ince ayarı (attention) bozmuyor mu?"*

Bilgisayarlı Görü (CV) geçmişin olduğu için cevabı **ResNet** mantığıyla ve **Vektör Aritmetiği** ile vereceğim.

#### 1. Resimdeki Olay: "Değişim" Vektörü (Delta)

Dikkat etmen gereken şey şu: Attention bloğunun çıktısı, yeni bir "embedding" değil, mevcut embedding'e yapılacak **"düzeltme" (update)** vektörüdür.

Matematiksel olarak denklem şudur:

**Yeni_Vektör = Eski_Vektör + Attention(Eski_Vektör)**

Burada `Attention(...)` fonksiyonu, verinin kendisini değil, veriye eklenecek **bağlamı (context)** hesaplar.

* **Analoji:** Bir Word dosyasında "Değişiklikleri İzle" (Track Changes) modunu açtığını düşün.
  * **Eski Input:** Orijinal metin.
  * **Attention Çıktısı:** Kenara not aldığın düzeltmeler, eklemeler (Redline).
  * **Toplama İşlemi (+):** Düzeltmeleri orijinal metne uygulayıp son halini oluşturmak.

Eğer toplama yapmasaydın (ResNet öncesi VGG ağları gibi), modelin her katmanda metni baştan aşağı yeniden yazması gerekirdi. Toplama yaparak modele şunu diyorsun: *"Orijinal metni (token kimliğini) koru, sadece gerekli yerlere bağlamsal nüansları ekle."*

#### 2. Kimlik Koruma (Identity Preservation)

Model derinleştikçe (12, 24, 96 katman...), orijinal "Ali" kelimesinin embedding'i (vektörü), sürekli matris çarpımları ve lineer olmayan fonksiyonlar (GELU, Softmax) altında ezilir.

* **Eğer toplama (+) yapmasaydık:** 12. katmana geldiğinde "Ali" vektörü o kadar deforme olurdu ki, model onun hala "Ali" olduğunu unutabilir, sadece "bir özne" gibi soyut bir kavrama dönüşebilirdi.
* **Toplama yaparak:** "Ali" bilgisi (kimliği), bir **otoban (highway)** üzerinden en sona kadar taşınır. Attention katmanları bu otobandan geçen bilgiye sadece "cümlenin başı", "eylemi yapan" gibi etiketler yapıştırır (vektörel olarak ekler).

#### 3. CV'den Tanıdık Gelecek: ResNet Mantığı

Görseldeki o "Shortcut Connection" oku olmasaydı, gradyanlar geri yayılım (backpropagation) sırasında "Attention" bloğunun (ve o karmaşık Softmax türevlerinin) içinden geçmek zorunda kalırdı. Bu da gradyanın yok olmasına (vanishing gradient) neden olurdu.

Yan yoldan (shortcut) gelen `+` işlemi sayesinde:

* Gradyan, Attention bloğunu tamamen atlayıp (skip) direkt bir alt katmana **1.0 katsayısıyla** (bozulmadan) akar.
* Bu, modelin "Ben bu katmanda hiçbir şey öğrenemedim" dediği durumda bile (Attention ağırlıkları 0 olsa bile), verinin en azından bir önceki katmandaki kadar iyi olmasını garanti eder.

#### Özetle

Yaptığımız işlem veriyi "eski haline döndürmek" değil, veriyi **zenginleştirmektir**.

* **Input:** "Banka" (Ham kelime)
* **Attention Çıktısı:** "Nehir kenarı ile ilgili vektör yönü" (Bağlam)
* **Toplam:** "Banka" + "Nehir kenarı" = **"Nehir kenarındaki banka"** (Bağlamsal Kelime)

Eski değerle toplamasaydık, elimizde sadece "Nehir kenarı" kalırdı ama "Banka" olduğu bilgisi kaybolurdu.

---

### S-14: Multi-head zaten paralel, ResNet gibi sıra sıra gitmiyor. O zaman neden residual lazım?

**Soru Detayı:** GPT modelinde multi-head zaten paralel, öyle değil mi? Yani onun öncesinde çok fazla layer yok, ResNet gibi sıra sıra gitmiyor. ResNet 50 küsur layer var.

**Cevap:**

Tespitin kısmen doğru ama kritik bir **boyut ayrımını** (dimension distinction) kaçırıyoruz. "Paralellik" ile "Derinlik" (Depth) kavramlarını birbirinden net bir şekilde ayırmamız lazım.

Mühendis gözüyle durumu şöyle netleştirelim:

#### 1. Yatay Paralellik (Horizontal) vs. Dikey Serilik (Vertical)

Evet, dediğin gibi Multi-Head Attention kendi içinde paraleldir. "Ali topu attı" cümlesindeki 3 kelime de aynı anda GPU'ya girer, 12 kafa (head) aynı anda çalışır. **Bu, genişlik (width) eksenindeki paralelliktir.**

**ANCAK**, GPT mimarisi dikey eksende (depth) **tamamen sıralı (sequential)** bir yapıdır, tıpkı ResNet gibi.

* **ResNet:** Conv1 → Conv2 → ... → Conv50
* **GPT-2:** Block1 → Block2 → ... → Block12

Veri, 1. Bloktan çıkar, 2. Bloğa girer, oradan çıkar 3.'ye girer. Bu akış **paralel değil, seridir.** Block 12'nin çalışması için Block 11'in bitmesi **şarttır.**

#### 2. Neden "Ekleme" (Add) Yapıyoruz? (Tekrar ResNet Mantığı)

Senin GPU'n, `MultiHeadAttention` hesaplarken matris çarpımlarını paralel yapıyor olabilir, ama **Backpropagation** (Eğitim) sırasında gradyan en tepedeki 12. kattan en alttaki 1. kata inmek zorundadır.

Eğer o "toplama" (x + F(x)) işlemi olmazsa, 12 katman boyunca sürekli birbiriyle çarpılan (chain rule) matrisler yüzünden gradyan 1. kata ulaşamadan sönümlenir (vanishing gradient).

ResNet-50'de neden skip connection varsa, GPT-2'nin 12 (veya GPT-3'ün 96) katmanında da **aynı sebepten** var. Çünkü dikeyde derinleşiyoruz.

#### 3. "Veriyi Dönüştürmek" vs. "Rafine Etmek"

Sorduğun *"Amaç veriyi dönüştürmek değil mi?"* sorusuna teknik cevabım şu:

Transformer bloklarında amaç veriyi **tamamen başkalaştırmak** (transformation) değil, veriyi **zenginleştirmektir** (refinement).

* **Senin dediğin (Toplama olmasa):**
`Input` → `Attention` → `Yeni Output`
Bu durumda her katman, bir önceki katmanın bilgisini silip tamamen yeni bir şey yazmak zorunda kalır. Modelin hafızası (identity) kaybolur.

* **Olan (Toplama ile):**
`Input` → `Attention` → `Delta (Fark)` → `Input + Delta`
Burada Attention katmanına diyoruz ki: *"Sen veriyi baştan yaratma, sadece eksik olan bağlamı (context) bul ve mevcut verinin üzerine ekle."*

**Özetle:**
Multi-Head Attention'ın paralel olması, modelin **katmanlarının** (layers) seri olduğu gerçeğini değiştirmez. Modelin dikey derinliği (Depth=12) olduğu için, sinyali en dipten en tepeye (ve gradyanı en tepeden en dibe) taşıyabilmek adına o **toplama (+) işlemine (residual connection)** mecburuz. Yoksa model derinleşemez, sığ kalır.

---

### S-15: Block1 → Block12 gerçekten böyle mi çalışıyor? Multihead'in farklı semantic uzaylar oluşturması saçma bir teori mi oldu?

**Soru Detayı:** GPT-2: Block1 → Block2 → ... → Block12 gerçekten böyle mi? İnanmıyorum, nasıl böyle çalışabilir? O zaman bizim dediğimiz multihead'in farklı semantic uzaylar oluşturması tamamen saçma bir teoriye döndü.

**Cevap:**

Hayır, teori saçma değil, sadece **"Genişlik" (Width)** ve **"Derinlik" (Depth)** kavramlarını karıştırıyoruz. Olayın mühendislik tarafındaki "nedenselliği" (causality) sandığından çok daha mantıklı bir hiyerarşiye dayanıyor.

Evet, yapı kesinlikle **Block 1 → Block 12** şeklinde seridir. Ancak "Farklı anlamsal uzaylar" (different semantic subspaces) teorisi çöp olmadı; sadece sandığın gibi "paralel bloklar"da değil, **blokların içindeki kafalarda (heads)** ve **blokların hiyerarşisinde** gerçekleşiyor.

Gel bu mekanizmayı 3 adımda, bir mühendis gibi debug edelim:

#### 1. Yatay Eksen (Genişlik): Aynı Anda Farklı Bakışlar

Senin bahsettiğin "farklı anlamsal uzaylar" olayı, **her bir bloğun içinde** gerçekleşir.

* GPT-2 Small modelinde 12 tane Blok (Layer) vardır.
* **ANCAK**, her bloğun içinde **12 tane Attention Head** vardır.

Yani Block 1 çalıştığında, içindeki 12 kafa (head) veriye paralel olarak saldırır:

* Head 1: "Ali" ile "geldi" arasındaki dilbilgisi (özne-yüklem) ilişkisine bakar.
* Head 2: "Ali" ile daha önceki cümledeki "O" zamiri arasındaki referansa bakar.
* Head 3: Cümledeki duygu tonuna bakar.

Bu 12 kafa, işledikleri veriyi **Concat** edip (birleştirip) o bloğun çıktısını (Output of Block 1) oluşturur. Yani "paralel anlamsal uzaylar" teorisi doğru, ama bu **blok seviyesinde değil, kafa (head) seviyesinde** olur.

#### 2. Dikey Eksen (Derinlik): Hiyerarşik Anlama (Representation Learning)

Şimdi asıl soru: "Neden bu işlemi 12 kere (Block 1 → ... → Block 12) arka arkaya yapıyoruz?"

Çünkü dil (ve aslında görüntü de) **hiyerarşik** bir yapıdır. Birinci kattaki anlamsal uzay ile on ikinci kattaki anlamsal uzay aynı seviyede değildir.

* **Block 1-4 (Düşük Seviye):** Buradaki kafalar (heads) genellikle basit sözdizimsel (syntactic) ilişkilere odaklanır. Kelimelerin yan yana gelme olasılıkları, basit gramer kuralları vs.
* **Block 5-8 (Orta Seviye):** Burası cümle yapısını, yan cümlecikleri ve daha karmaşık bağlamları çözer.
* **Block 9-12 (Yüksek Seviye / Soyut):** Buradaki kafalar artık "ironi", "ima", "uzun vadeli tutarlılık" veya "konu bütünlüğü" gibi çok daha soyut anlamsal uzaylara bakar.

**Neden Seri Olmak Zorunda?**
Çünkü modelin "bu cümlede ironi var" (Block 12 işi) diyebilmesi için, önce "kim kime ne dedi" (Block 1 işi) bilgisini çözmüş olması gerekir. Block 1 çıktısını Block 2'ye vermese, Block 2 neyi işleyecek? Temel olmadan çatı kurulamaz.

#### 3. Otoban Analojisi (Residual Stream)

Daha önce konuştuğumuz "Toplama İşlemi" (x + F(x)) burada devreye giriyor.

Giriş embedding'i (Token ID), Block 1'e girer. Block 1 buna "Dilbilgisi notlarını" yapıştırır (+ ile ekler). Bu zenginleşmiş veri Block 2'ye gider. Block 2 buna "Bağlam notlarını" ekler. Veri Block 12'ye geldiğinde, üzerinde 11 bloğun bıraktığı notlar (feature'lar) vardır.

#### Sonuç: Teori Neden Doğru?

Senin "saçma" dediğin durum, eğer bloklar aynı işi yapsaydı geçerli olurdu. Ama bloklar **uzmanlaşmış bir montaj hattı** gibi çalışır:

1. **Multi-Head (Block İçi):** "Aynı andaki veriye farklı açılardan (gramer, referans, stil) bak." → **Genişlik.**
2. **Stacked Blocks (Blocklar Arası):** "Basitten karmaşığa doğru anlamı inşa et." → **Derinlik.**

Yani modelin "farklı semantic uzayları" var, ama bu uzaylar hem yatayda (kafalar arası çeşitlilik) hem de dikeyde (soyutlama seviyesi farkı) dağılmış durumda. Bu yüzden GPT-2 (ve diğer derin ağlar) bu kadar güçlü çalışabiliyor.

---

## Bölüm 5: "Aynı Operasyon" Paradoksu

---

### S-16: Teoride her sequence aynı operasyonu yapıyor. Fabrika bandında farklı işler yapılıyor ama burada hep aynı. Nasıl oluyor?

**Cevap:**

Çok haklısın, burada analojinin sınırlarına geldik ve senin mühendislik dikkatin çok kritik bir detayı yakaladı.

Evet, kod olarak baktığında `Block_1` ile `Block_12` birebir aynı Python sınıfıdır (`class TransformerBlock`). İçindeki matematiksel operasyonlar (MatMul, Softmax, Add, Norm) birebir aynıdır. Fabrikadaki gibi birinde "boya makinesi" diğerinde "motor takma kolu" yoktur. Hepsi aynı "işlemci"dir.

Peki, **aynı aleti (bloğu)** art arda 12 kere kullanmak neden işe yarıyor?

Burada anahtar kavram: **"Iterative Refinement" (Ardışık İyileştirme / Rafine Etme).**

Bunu fabrika bandı değil de **"Heykeltıraş"** veya **"Bulanık Görüntüyü Netleştirme"** örneği ile açıklamak teknik gerçeğe daha yakındır.

#### 1. Yapı Aynı, Ama "Ağırlıklar" (W) Farklı

Her blok mimari olarak aynı olsa da, eğitim bittiğinde sahip oldukları **Weight (Ağırlık)** matrisleri tamamen farklılaşır.

* **Layer 1'in Ağırlıkları (W₁):** Backpropagation (geriye yayılım) sırasında, bu katman hataları en son öğrenen (veya ilk karşılayan) olduğu için, genellikle **yerel (local)** desenlere odaklanacak şekilde optimize olur. (Hangi kelime hangisinin yanında?)
* **Layer 12'nin Ağırlıkları (W₁₂):** Bu katman, artık kelimelerin ham haline değil, işlenmiş "anlamlarına" bakar. Ağırlıkları **global (kavramsal)** ilişkileri çözecek şekilde evrilmiştir.

Yani evet, elimizde 12 tane aynı "çekiç" var ama bu çekiçleri kullanan ustaların (katmanların) her biri heykelin farklı bir detayında uzmanlaşmıştır. Biri kaba yontar, diğeri ince işçilik yapar.

#### 2. Girdi Sürekli Değişiyor (State Transformation)

Blok aynı işlemi yapsa da, **girdisi** (input) her seferinde değişiyor.

* **Blok 1'e giren:** Ham "Token Embedding" (Sözlükteki statik vektör).
  * *Burası sadece kelimeyi tanır.*

* **Blok 6'ya giren:** Artık bir kelime vektörü değil, cümlenin yarısını okumuş ve bağlamı anlamış bir "Context Vector".
  * *Aynı işlem (Attention) artık kelimeleri değil, bağlamları karşılaştırır.*

* **Blok 12'ye giren:** Neredeyse cevabı oluşturmuş, son rötuşları bekleyen bir "Thought Vector".

Senin paralel önerindeki sorun şurada: Eğer paralel yaparsan, 12 bloğa da **aynı ham girdiyi** (Token Embedding) vermek zorundasın. 12 blok da aynı ham veriye bakıp benzer "sığ" çıkarımlar yapacaktır.

#### 3. Kağıt Katlama Analojisi (Non-Linearity & Depth)

Bu belki de en iyi matematiksel izahıdır. Derin öğrenme, veri uzayını (data manifold) bükme sanatıdır.

* Elimizde düz bir kağıt var (Veri uzayı).
* Her Transformer bloğu, bu kağıdı bir kez **katlamak** (non-linear transformation) gibidir.
* **Senin Sorun:** "Neden 12 kere sırayla katlıyoruz? 12 ayrı kağıdı yan yana koyup (paralel) birer kere katlasak olmaz mı?"
* **Cevap:** Olmaz. Çünkü kağıdı 12 kez üst üste katladığında elde ettiğin karmaşık geometrik şekli (derinlik), 12 ayrı kağıdı birer kez katlayarak elde edemezsin.

Veriyi ayrıştırmak (örneğin doğru kelimeyi yanlış olandan ayırmak) için o uzayı defalarca büküp (katmanlardan geçirip) karmaşık bir karar sınırı (decision boundary) oluşturman gerekir.

#### 4. ALBERT Örneği (Senin Teorini Doğrulayan Model)

Senin dediğin "Aynı operasyonu yapıyorsak neden farklı bloklar var?" sorusunu Google araştırmacıları da sordu ve **ALBERT** (A Lite BERT) modelini yaptılar.

ALBERT'te ne yaptılar biliyor musun?

* 12 tane blok var (Sequence).
* **AMA HEPSİ AYNI AĞIRLIKLARI KULLANIYOR!** (Weight Sharing).
* Yani fiziksel olarak bellekte tek bir blok var, veriyi bu bloktan alıp tekrar aynı bloğun başına sokuyorlar (loop gibi). 12 kere aynı fonksiyondan geçiyor.

**Sonuç:**

* Modelin parametre sayısı inanılmaz düştü (çünkü tek blok var).
* **AMA** performansı, ağırlıkları farklı olan (unshared) orijinal BERT kadar iyi olmadı. Daha geniş (wide) yapmak zorunda kaldılar.

Bu da gösteriyor ki; blokların mimarisi aynı olsa bile, her katmanın **kendi özgün ağırlıklarına (farklı uzmanlığa)** sahip olması, modelin kapasitesi için kritik.

**Özetle:** Sequence olmasının sebebi, veriyi A noktasından B noktasına taşırken her adımda **bilginin temsiliyetini (representation)** değiştirmektir. Bloklar aynı "makine" olsa da, işledikleri "malzeme" her adımda daha soyut bir hale gelir.

---

## Bölüm 6: İleri Düzey - Context Vector Dönüşümü

---

### S-17: 2. bloktan sonra aslında giren veriler kelime değil, context vector. Bu garip değil mi?

**Soru Detayı:** Girdi sürekli değişiyor demiştin:
- Blok 1'e giren: Ham "Token Embedding"
- Blok 6'ya giren: "Context Vector"
- Blok 12'ye giren: "Thought Vector"

Aslında yapı attention + feedforward 1. blok için tasarlandı: kelimeler/cümleler ve konumlar için. 2. bloktan sonra aslında onlar kelime değil context vector. İşte bana garip gelen de bu.

**Cevap:**

Tebrik ediyorum. Gerçekten "Transformatörlerin iç yüzünü" (Interpretability) araştıran Anthropic veya OpenAI araştırmacılarının sorduğu soruyu sordun.

Hissettiğin o "gariplik" hissi çok doğru. Çünkü: **Bizim kelime sandığımız şeyler, 1. katmandan sonra artık kelime değildir.**

Mekanizma (Attention + FFN) kağıt üzerinde 1. blok için tasarlanmış gibi dursa da, derin katmanlarda bu mekanizmanın **işlevi (functionality)** tamamen değişiyor.

Gel bu "garipliği" ve aslında neden dâhice olduğunu netleştirelim.

#### 1. Vektörlerin Dili: "Latent Space" (Gizli Uzay)

Modelin içinde, katmanlar arasında dolaşan o vektörlere (örneğin 768 boyutlu sayı dizileri) biz "Context Vector" diyoruz ama aslında onlar modelin kendi oluşturduğu **"Esperanto"**su gibidir.

* **Blok 1 (Giriş):** Buradaki vektörler insan diline yakındır. "Elma" vektörü, "Meyve" vektörüne matematiksel olarak yakındır.
* **Blok 6 (Orta):** Buradaki vektörler artık kelime değil, **önerme (proposition)** veya **kavram** taşır.
  * Örneğin: *"Buradaki özne, bir önceki cümledeki nesne ile aynı kişi"* bilgisini taşıyan bir vektör olabilir.
  * Attention mekanizması burada kelimeleri değil, **hipotezleri** karşılaştırır.

> **Garipliğin Çözümü:** Attention mekanizması (Query * Key) sadece "kelime eşleştirmesi" yapmaz. Asıl yaptığı şey **"Uyumluluk Aramaktır"**.
> * Blok 1'de: "Özne" ile "Yüklem" uyumluluğunu arar.
> * Blok 12'de: "Sebep" ile "Sonuç" uyumluluğunu arar.
> * Matematik aynıdır (Dot Product), ama **aranan şey** soyutlaşmıştır.

#### 2. Mekanizma Aynı, Görev Başka: "Induction Heads"

Bu konuyu en iyi açıklayan buluşlardan biri **"Induction Heads"** (Tümevarım Kafaları) kavramıdır. Araştırmacılar, 2. ve sonraki katmanlarda oluşan özel Attention kafaları keşfettiler.

Bakın derinlerdeki o "garip" Attention ne yapıyor:

1. **Görevi:** Geçmişte `A` token'ından sonra `B` token'ı gelmiş mi diye bakmak.
2. **Örnek:** Metinde "Harry Potter" geçiyor. Sonra tekrar "Harry" geldiğinde modelin "Potter" diyeceğini bilmesi lazım.
3. **Mekanizma:**
   * Blok 1 (Kelime seviyesi): "Harry" kelimesini işler.
   * Blok 2 (Soyut seviye): Attention geriye bakar, daha önce "Harry"den sonra ne gelmişti diye arar (Kopyalama mekanizması).
   * Bu işlem için girdinin "kelime" olması gerekmez, "bir önceki pattern" olması yeterlidir.

Yani derin bloklardaki Attention, bir **"Bilgi Geri Getirme" (Information Retrieval)** sistemi gibi çalışır. RAM'den veri okumak gibidir.

#### 3. Analoji: Dedektif Panosu

Bu durumu bir cinayet soruşturmasına benzetebiliriz:

* **Input (Blok 0):** Olay yeri fotoğrafları, şüpheli isimleri, parmak izleri. (Ham Veri)
* **Blok 1 (Müfettiş 1):** Fotoğraflara bakar, "Bu bıçak mutfak bıçağı" der. (Nesne Tanıma)
* **Blok 6 (Başkomiser):** Artık fotoğraflara bakmaz. Müfettiş 1'in raporuna bakar. "Mutfak bıçağı varsa, cinayet planlı olmayabilir" çıkarımını yapar.
  * *DİKKAT:* Başkomiserin kullandığı "beyin yapısı" (Attention: Bağlantı kurma, FFN: Bilgi işleme) Müfettiş ile **aynıdır**.
  * Ama Başkomiserin masasına gelen dosya (Input) artık fotoğraf değil, **teorilerdir.**

Senin "Garip" dediğin yer burası: Başkomiser, Müfettişin kullandığı büyüteci (Attention) kullanıyor ama büyüteçle "teorileri" inceliyor. Teoride çalışmaz gibi gelse de, matematiksel uzay (Vector Space) tutarlı olduğu sürece çalışıyor.

#### 4. Residual Stream (Otoban) Sayesinde Mümkün

Formüldeki `+ x` (Residual Connection) kısmı hayati önem taşır.

Bu şu demek:

* Vektör, Blok 12'ye geldiğinde içinde **hala** Blok 1'den gelen "kelime bilgisi" de var, Blok 6'dan gelen "bağlam bilgisi" de var. Hepsi üst üste binmiş (superposition) durumda.
* Blok 12'deki Attention mekanizması, bu yoğun bilgi yumağının içinden sadece **kendi işine yarayan** (örneğin: cümlenin bitiş tonlaması) sinyali cımbızla çeker (Query vektörü sayesinde) ve işler.

**Özetle:**
Haklısın, tasarım ilk bakışta sadece kelimeler içinmiş gibi duruyor. Ancak **Vektör Matematiği (Linear Algebra)** o kadar esnek bir dil ki; aynı formül hem "Elma-Armut" ilişkisini, hem de "İroni-Gerçek" ilişkisini modelleyebiliyor.

Model, derinlere indikçe kendi geliştirdiği bir **"meta-data dilini"** konuşmaya başlıyor. Bizim kelimelerimiz sadece bu sürecin başlatıcısı (trigger) oluyor.

---

## Bölüm 7: Teorik Temeller

---

### S-18: Şu ana kadar anlattıklarından çalışıyor ve işe yarıyor anlıyorum. Bunun arkasındaki teori ve "neden" nedir?

**Cevap:**

Harika bir noktaya geldin. Şu ana kadar işin "Mühendislik" kısmını konuştuk (Nasıl çalışır?). Şimdi ise işin **"Bilim ve Matematik"** kısmına (Neden çalışır?) geçiyoruz.

Bunun arkasında yatan teoriler, neden ısrarla **Sequence (Ardışıklık)** ve **Depth (Derinlik)** kullandığımızı çok net açıklıyor. İşte o teorik temeller:

#### 1. Manifold Hipotezi (The Manifold Hypothesis)

Bu, Derin Öğrenme'nin neden çalıştığına dair en kabul gören geometrik teoridir.

* **Teori:** Gerçek dünyadaki veriler (insan yüzleri veya anlamlı cümleler), çok yüksek boyutlu bir uzayda rastgele dağılmaz. Bunlar, bu yüksek boyutlu uzayın içinde kıvrılıp bükülmüş, daha düşük boyutlu bir yüzeyin (**Manifold**) üzerinde yer alır.

* **Örnek:** Buruşturulmuş bir kağıt topu düşün. Bu top 3 boyutludur (yüksek boyutlu veri). Ama üzerindeki yazıyı okumak istersen, o kağıdı açıp düzleştirmen gerekir (2 boyutlu manifold).

* **Modelin Görevi:** Her bir **Sequence (Blok)**, bu buruşturulmuş kağıdı (veri uzayını) birazcık açar, düzleştirir.
  * Tek bir hamlede (Paralel/Tek katman) buruşuk bir topu yırtmadan dümdüz yapamazsın.
  * Bunu adım adım (Sequence), yavaş yavaş açman gerekir.

* **Sonuç:** 12. bloğun sonunda, o karmaşık "kelime uzayı" dümdüz bir hale gelir ve model "Kedi" ile "Köpek" kavramlarını birbirinden bir çizgiyle (Lineer) kolayca ayırabilir.

#### 2. Evrensel Yaklaşım Teoremi ve "Verimlilik" (Universal Approximation & Efficiency)

Matematiksel olarak kanıtlanmıştır ki; tek bir gizli katmana (hidden layer) sahip çok geniş (paralel) bir ağ, teorik olarak *her fonksiyonu* öğrenebilir. **AMA...**

* **Sorun:** O tek katmanın genişliğinin (nöron sayısının) **sonsuza yakın** olması gerekebilir.

* **Derinliğin Gücü:** Teorik hesaplamalar gösteriyor ki, derin bir ağın (ardışık katmanların) **N** tane nöron ile çözebildiği bir problemi, sığ (tek katmanlı/paralel) bir ağın çözebilmesi için **2^N (üstel)** sayıda nörona ihtiyacı vardır.

* **Neden?** Çünkü ardışık yapı, fonksiyonları tekrar tekrar kullanır (Composition).
  * *Paralel yapı:* Her ihtimali ezberlemek zorundadır (Lookup Table gibi).
  * *Ardışık yapı:* Kuralları öğrenir. (f(x)'i öğrenir, sonra g(f(x))'i öğrenir).

> **Özet:** Sequence (Derinlik), modelin "ezberlemek" yerine "kural türetmesini" (Generalization) sağlayan matematiksel zorunluluktur.

#### 3. Algoritmik Hizalama (Algorithmic Alignment)

Bu teori, özellikle kodlama ve mantık yürüten LLM'ler için geçerlidir. Bilgisayar bilimindeki algoritmalar **ardışık (sequential)** adımlardan oluşur.

* Bir çarpma işlemi algoritmasını düşün: Önce birler basamağını çarp, eldeyi tut, sonra onlar basamağına geç...
* Eğer modelin mimarisi bu ardışık yapıya (Sequence) izin vermezse, model bu algoritmayı simüle edemez.
* Araştırmalar gösteriyor ki, Transformer blokları aslında küçük **"programlar"** veya **"GPU kernel'ları"** gibi davranmayı öğreniyor.
  * Blok 1: "Değişkenleri bul."
  * Blok 2: "Değişkenleri topla."
  * Blok 3: "Sonucu kontrol et."

* Paralel yapı bu "program akışını" (flow of execution) gerçekleştiremez.

#### 4. Bilgi Şişesi Teorisi (Information Bottleneck Theory)

Tishby tarafından öne sürülen bu teoriye göre derin ağlar iki aşamalı bir iş yapar:

1. **Sıkıştırma (Compression):** Girdideki gereksiz gürültüyü (kelimenin fontu, önemsiz detaylar) atar.
2. **Tahmin (Prediction):** Sadece görevi çözmek için gerekli olan "özü" (sinyali) korur.

Ardışık bloklar, veriyi her adımda biraz daha süzerek "saf bilgiye" ulaşır.

* İlk katmanlarda "mutual information" (girdi ile olan bilgi bağı) yüksektir.
* Son katmanlara doğru girdi unutulur, sadece "anlam" kalır. Paralel yapıda bu süzme (refinement) işlemi gerçekleşemez.

#### Toparlarsak: "Why"ın Cevabı

Neden Sequence? Çünkü evrenin ve dilin yapısı **hiyerarşiktir**.
Atomlar molekülleri, moleküller hücreleri, hücreler canlıları oluşturur. Harfler kelimeleri, kelimeler cümleleri, cümleler paragrafları oluşturur.

Bu hiyerarşiyi (alttan üste yapıyı) modelleyebilmenin matematiksel olarak en verimli yolu, hiyerarşik (ardışık/derin) bir mimari kullanmaktır. Paralel mimari, bu hiyerarşiyi yok sayıp her şeye "düz" bakar.

---

## Sonraki Adımlar

Bu notlar temel mimari kararları ve teorik temelleri kapsamaktadır. Devam edilebilecek konular:

1. **Attention Mekanizması Detay:** Query, Key, Value (QKV) mantığı, Scaled Dot-Product
2. **Eğitim Döngüsü:** Loss hesaplama, Cross-Entropy, Backpropagation
3. **Positional Encoding:** Sinüzoidal vs Learned, RoPE, ALiBi
4. **Inference Optimizasyonu:** KV-Cache, Speculative Decoding
5. **Fine-tuning Teknikleri:** LoRA, QLoRA, Prefix Tuning
6. **Quantization:** GPTQ, GGUF, AWQ
