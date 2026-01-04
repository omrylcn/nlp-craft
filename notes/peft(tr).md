# Parametre-Verimli İnce Ayar (PEFT): LLM'ler için Teorik Temeller ve İleri Teknikler

## 1. Teorik Temeller ve Matematiksel Formülasyon

### 1.1 PEFT Temel Prensibi

Parametre-Verimli İnce Ayar (Parameter-Efficient Fine-Tuning, PEFT), milyarlarca parametreye sahip büyük dil modellerini (LLM) tüm parametrelerini güncellemek yerine, küçük bir parametre alt kümesini optimize ederek adapte etme yöntemidir. PEFT'in temel prensibi şu matematiksel formülasyonla ifade edilebilir:

Tam ince ayar (full fine-tuning) problemi:
$$\min_{\theta} \mathcal{L}(\theta; \mathcal{D})$$

Burada $\theta \in \mathbb{R}^d$ modelin tüm parametreleridir ve $d >> 10^9$ olabilir (örn. GPT-3 için 175 milyar).

PEFT alternatifi:
$$\min_{\Delta\theta} \mathcal{L}(f(\theta_0, \Delta\theta); \mathcal{D})$$

Burada:
- $\theta_0$: Ön-eğitimli model parametreleri (dondurulmuş)
- $\Delta\theta \in \mathbb{R}^k$: Optimize edilecek küçük parametre seti ($k << d$)
- $f$: Orijinal parametreleri ve ince ayar parametrelerini birleştiren fonksiyon

Parametre verimliliği oranı:
$$\text{Efficiency ratio} = \frac{d}{k}$$

### 1.2 Teorik Avantajlar

PEFT'in teorik avantajları şunları içerir:

1. **Aşırı-parametre düzenlemesi**: Daha az parametre güncellenerek aşırı uyumun önüne geçilir. Matematik olarak, PEFT bir düzenleme terimi olarak görülebilir:

$$\mathcal{L}_{PEFT}(\theta) = \mathcal{L}(\theta) + \lambda R(\theta - \theta_0)$$

Burada $R$ bir düzenleme fonksiyonu ve $\lambda$ bu düzenlemenin belirli boyutlarda nasıl uygulanacağını belirleyen bir maskedir.

2. **Catastrophic forgetting azaltma**: Önceki bilgilerin korunmasını sağlar. Bilgi-teorik bir perspektiften:

$$I(X_{pretrain}; \theta_{PEFT}) \geq I(X_{pretrain}; \theta_{full})$$

Burada $I(X;\theta)$ önceki veriler ($X_{pretrain}$) ile parametreler arasındaki karşılıklı bilgiyi gösterir.

3. **Transfer öğrenme ve genelleme**: PEFT genellikle görevler arası transferde daha iyi performans gösterir:

$$\mathbb{E}_{x \sim \mathcal{D}_{target}}[L(x, \theta_{PEFT})] \leq \mathbb{E}_{x \sim \mathcal{D}_{target}}[L(x, \theta_{full})]$$

## 2. PEFT Yöntemleri Taksonomisi

### 2.1 Adaptör-Tabanlı Yöntemler

Adaptörler, Transformer katmanlarına eklenen küçük, öğrenilebilir modüllerdir:

$$h' = h + f(h)$$

Burada $h$ orijinal aktivasyonlar, $f$ adaptör fonksiyonu ve $h'$ modifiye edilmiş aktivasyonlardır. Adaptör fonksiyonu tipik olarak aşağıdaki formda tanımlanır:

$$f(h) = W_{up} \cdot \sigma(W_{down} \cdot h + b_{down}) + b_{up}$$

Burada:
- $W_{down} \in \mathbb{R}^{d \times r}$, $W_{up} \in \mathbb{R}^{r \times d}$ ($r << d$)
- $\sigma$: Aktivasyon fonksiyonu (genellikle ReLU veya GeLU)

Adaptörlerin teorik karmaşıklığı:
- Parametre sayısı: $2rd + 2r$ ($r$ boyutlu ara temsil için)
- Bellek karmaşıklığı: $O(rd)$
- Hesaplama karmaşıklığı: $O(nrd)$ ($n$ token sayısı)

#### Paralel vs. Seri Adaptörler

Seri (Houlsby) Adaptörler:
$$h' = h + W_{up} \cdot \sigma(W_{down} \cdot \text{LayerNorm}(h) + b_{down}) + b_{up}$$

Paralel (Pfeiffer) Adaptörler:
$$h' = \text{LayerNorm}(h + W_{up} \cdot \sigma(W_{down} \cdot h + b_{down}) + b_{up})$$

### 2.2 Düşük-Rankli Adaptasyon (LoRA)

LoRA, ağırlık matrislerindeki değişimleri düşük rankla yaklaştırarak modelin davranışını değiştirir:

$$W = W_0 + \Delta W = W_0 + BA$$

Burada:
- $W_0 \in \mathbb{R}^{d \times k}$: Ön-eğitimli ağırlık matrisi
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Düşük rankla güncelleme matrisleri
- $r$: İnce ayar rankı ($r << \min(d, k)$)

LoRA, eğitim sırasında aşağıdaki parametrizasyonu kullanır:

$$W = W_0 + \frac{\alpha}{r}BA$$

Burada $\alpha$ ölçeklendirme faktörüdür.

LoRA'nın teorik avantajları:
- Parametre sayısı: $r(d+k)$ (Tam güncelleme: $d \times k$)
- Verimlilik oranı: $\frac{dk}{r(d+k)} \approx \frac{\min(d,k)}{r}$
- İleri geçiş maliyeti asimptotik olarak tam ince ayarla aynıdır: $O(ndk)$

#### QLoRA ve Diğer Optimizasyonlar

QLoRA, dört-bit niceleme ile LoRA'yı birleştirir:

1. Ön-eğitimli modeli 4-bit'e niceler: $Q(W_0, 4-bit)$
2. LoRA parametrelerini tam hassasiyette tutar: $A, B \text{ in FP16}$
3. Hesaplama esnasında 4-bit matrisleri 16-bit'e dequantize eder

QLoRA'daki anahtar yenilikler:
- NormalFloat (NF4) nicelemesi: $W_{NF4} = \text{round}(W \cdot s_{NF4})$
- Sayfalamalı Optimizasyon (paged optimizers): GPU belleğini daha etkin kullanır
- İleri/geri geçişte dequantize işlemi: $W_{dequant} = W_{quant} / s_{NF4}$

Matematiksel avantaj:
- QLoRA ile bellek kullanımı: $O(d+k)+O(r(d+k))$
- Tam ince ayara göre tasarruf: ~65x

### 2.3 Prefix ve Prompt Tuning

#### Prefix Tuning

Prefix Tuning, her bir Transformer katmanında anahtar-değer önbelleğine öğrenilebilir prefixler ekler:

$$\text{Attention}(Q, [P_K; K], [P_V; V])$$

Burada:
- $P_K, P_V \in \mathbb{R}^{l \times d}$: Öğrenilebilir prefixler
- $l$: Prefix uzunluğu (genellikle 20-100 token arası)

Parametre verimliliği:
- Toplam parametre sayısı: $2 \times l \times d \times L$ ($L$ katman sayısı)
- Verimlilik oranı: $\frac{N_{total}}{2ldL}$ ($N_{total}$ toplam model parametresi)

Li ve Liang (2021), reparametrizasyon trickini kullanarak eğitim stabilitesini artırırlar:

$$P_K = \text{MLP}_{\theta_K}(R), \quad P_V = \text{MLP}_{\theta_V}(R)$$

Burada $R \in \mathbb{R}^{l \times d_r}$ rastgele başlatılan bir matristir ve $d_r < d$.

#### Prompt Tuning

Prompt Tuning daha basit bir yaklaşım kullanır:

$$E_{\text{input}} = [E_{\text{prompt}}; E_{\text{tokens}}]$$

Burada:
- $E_{\text{prompt}} \in \mathbb{R}^{p \times d}$: Öğrenilebilir prompt embeddinglari
- $E_{\text{tokens}}$: Gerçek token embeddinglari
- $p$: Prompt uzunluğu (genellikle 5-100 token)

Parametre verimliliği:
- Toplam parametre sayısı: $p \times d$
- Verimlilik oranı: $\frac{N_{total}}{pd}$

### 2.4 Diğer Parametre-Verimli Yaklaşımlar

#### BitFit

BitFit, yalnızca bias terimlerini günceller:

$$\min_{\{b_i\}} \mathcal{L}(f(\theta_0, \{b_i\}); \mathcal{D})$$

Teorik verimlilik:
- Parametre verimliliği: ~1000x (bias parametreleri genellikle toplam parametrelerin ~%0.1'i)
- Toplam güncellenen parametre sayısı: $O(d)$ (vs $O(d^2)$ tam ince ayarda)

#### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

IA³, aktivasyonları eleman bazlı ölçeklendirerek çalışır:

$$\tilde{h} = h \odot s$$

Burada:
- $h$: Orijinal aktivasyonlar
- $s \in \mathbb{R}^d$: Öğrenilebilir ölçeklendirme vektörü
- $\odot$: Hadamard (eleman bazlı) çarpım

IA³ genellikle şu bileşenlerde uygulanır:
- Anahtar ve değer projeksiyon çıktıları
- Feed-forward ara aktivasyonlar

Verimlilik:
- Parametre sayısı: $2d \times L$ (gizli boyut başına 2 parametre, $L$ katman)
- Verimlilik oranı: $\frac{N_{total}}{2dL}$, genellikle 10,000x seviyelerinde

## 3. PEFT Yöntemlerinin Teorik Karşılaştırması

### 3.1 Ekspresif Güç ve Optimizasyon Uzayı

PEFT yöntemlerinin ekspresif gücü, optimize edilen alt uzayın boyutsallığı ve bu alt uzayın tam parametre uzayıyla etkileşimiyle ilgilidir.

LoRA'nın ekspresif gücü şöyle hesaplanabilir:
- Tam parametre uzayı: $\mathbb{R}^{d \times k}$
- LoRA ile erişilebilir alt uzay: $W_0 + \text{span}(B) \times \text{span}(A^T)$
- Maksimum rank: $\min(r, d, k)$

Adaptör yöntemleri, ara katmanlar ekleyerek farklı bir optimizasyon uzayı kullanır:
- Adaptör uzayı: $\mathcal{H} + f_{\text{adapter}}(\mathcal{H})$
- Bu, doğrusal olmayan dönüşümleri de yakalayabilir

Prefix/Prompt tuning, cross-attention veya input uzayında modifikasyon yapar:
- Erişilebilir uzay: Orijinal input uzayını genişleterek yeni bir manifold oluşturur

### 3.2 Matematiksel Karmaşıklık ve Analiz

PEFT yöntemlerinin teorik karmaşıklığını şu denklemlerle karşılaştırabiliriz:

| Yöntem | Parametre Sayısı | Verimlilik Oranı | İleri Geçiş Karmaşıklığı |
|--------|-------------------|-----------------|---------------------------|
| Tam İnce Ayar | $N_{total}$ | 1x | $O(ND)$ |
| Adaptörler | $2rD + 2r$ | $\frac{N_{total}}{2rD + 2r}$ | $O(ND + NrD)$ |
| LoRA | $r(D + K)$ | $\frac{N_{total}}{r(D+K)}$ | $O(ND)$ |
| QLoRA | $r(D + K)$ | $\frac{N_{total}}{r(D+K)}$ | $O(ND)$ |
| Prefix Tuning | $2lDL$ | $\frac{N_{total}}{2lDL}$ | $O(ND + NlD)$ |
| Prompt Tuning | $pD$ | $\frac{N_{total}}{pD}$ | $O(ND + NpD)$ |
| BitFit | $~0.001N_{total}$ | ~1000x | $O(ND)$ |
| IA³ | $2DL$ | $\frac{N_{total}}{2DL}$ | $O(ND)$ |

Burada:
- $N_{total}$: Toplam parametre sayısı
- $D$: Gizli boyut
- $r$: Rank/adaptör boyutu
- $l$: Prefix uzunluğu
- $p$: Prompt uzunluğu
- $L$: Transformer katman sayısı
- $K$: Projeksiyon boyutu

### 3.3 Gradyan Flow Analizi

PEFT yöntemlerinin eğitim dinamiklerini anlamak için gradyan akışı analizi önemlidir:

LoRA için gradyan akışı:
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial W} \cdot B^T \cdot \frac{\alpha}{r}$$
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial W} \cdot A^T$$

Adaptörler için gradyan akışı:
$$\frac{\partial \mathcal{L}}{\partial W_{up}} = \frac{\partial \mathcal{L}}{\partial h'} \cdot \sigma(W_{down} \cdot h + b_{down})^T$$
$$\frac{\partial \mathcal{L}}{\partial W_{down}} = W_{up}^T \cdot \frac{\partial \mathcal{L}}{\partial h'} \cdot \sigma'(W_{down} \cdot h + b_{down}) \cdot h^T$$

Farklı katmanlardaki gradyan büyüklükleri şunları gösterir:
- LoRA en doğrudan yaklaşımı sunar (orijinal parametrelerle aynı gradyan yolu)
- Adaptörler gradyanları küçük boyutlu bir katmandan geçirir, bu da potansiyel olarak stabilite sağlar
- Prefix/Prompt tuning, sınırlı sayıda parametreden çok katmana geriye doğru gradyan akışına dayanır

## 4. İleri PEFT Yöntemleri ve Teknikler

### 4.1 MultiModal ve Cross-Modal PEFT

Çoklu-modalite PEFT şu yaklaşımları içerir:

1. **Modalite-spesifik uyarlamalar**:
   $$f_{MM-PEFT}(x) = f_{pre-trained}(x; \theta_0) + \sum_{m \in M} \mathbb{1}_{m}(x) \cdot \Delta f_m(x; \phi_m)$$

   Burada $M$ modaliteler kümesi, $\mathbb{1}_{m}(x)$ girdinin m modalitesine ait olup olmadığını gösteren bir indikatör fonksiyonu, ve $\phi_m$ modalite-spesifik ince ayar parametreleridir.

2. **Cross-Modal Projeksiyonlar**:
   $$h'_m = W_{m \rightarrow common} \cdot h_m$$

   Burada $W_{m \rightarrow common}$ m modalitesinden ortak temsil uzayına bir projeksiyon matrisidir.

### 4.2 PEFT için Compositional ve Multi-Task Yaklaşımlar

Görev-spesifik parametre setleri nasıl etkili bir şekilde kombine edilebilir:

1. **Modüler PEFT Kompozisyonu**:
   $$f_{compositional}(x) = f_{base}(x; \theta_0) + \sum_{i=1}^{T} \alpha_i \cdot \Delta f_i(x; \phi_i)$$

   Burada $\phi_i$ görev $i$ için PEFT parametreleri ve $\alpha_i$ ağırlık faktörüdür.

2. **Mixture-of-Adapters (MoA)**:
   $$h' = h + \sum_{i=1}^{N} g_i(h) \cdot f_i(h)$$

   Burada $g_i(h)$ adaptör $i$ için routing ağırlığıdır.

### 4.3 Knowledge Distillation ve PEFT

PEFT'te knowledge distillation, küçük modelleri veya daha verimli PEFT konfigürasyonlarını eğitmek için kullanılır:

$$\mathcal{L}_{KD}(\phi) = \alpha\mathcal{L}_{CE}(f(x; \theta_0, \phi), y) + (1-\alpha)\mathcal{L}_{KL}(f(x; \theta_0, \phi), f_{teacher}(x))$$

Burada:
- $\mathcal{L}_{CE}$: Çapraz entropi kaybı
- $\mathcal{L}_{KL}$: Kullback-Leibler yakınsama kaybı
- $f_{teacher}$: Öğretmen model (genellikle tam ince ayarlanmış model)
- $\alpha$: İki kayıp terimi arasındaki dengeyi kontrol eden parametre

### 4.4 Neural Architecture Search (NAS) ile Otomatik PEFT Optimizasyonu

NAS, optimal PEFT konfigürasyonlarını otomatik olarak keşfetmek için kullanılabilir:

$$\phi^* = \arg\min_{\phi \in \Phi} \mathcal{L}_{val}(f(x; \theta_0, \phi))$$
$$\text{subject to } |\phi| \leq C$$

Burada:
- $\Phi$: Tüm olası PEFT konfigürasyonlarının kümesi
- $C$: Parametre bütçesi kısıtlaması

Bu optimizasyon şu boyutlarda yapılabilir:
- En uygun rank $r$ seçimi (LoRA için)
- En uygun katmanları seçme (tüm katmanlar üzerinde PEFT gerekmeyebilir)
- Farklı PEFT yöntemlerinin kombinasyonu (hibrit yaklaşımlar)
- Sparsity seviyeleri veya bitfit paternleri

## 5. Gelecek Yönelimler ve Açık Araştırma Soruları

### 5.1 Uzun Bağlamlarda PEFT

LoRA gibi PEFT yöntemleri uzun bağlamlara genellemeyi iyileştirmede etkili olabilir:

$$P(y_{n+1:n+m}|x_{1:n}) \approx P_{\phi}(y_{n+1:n+m}|x_{1:n})$$

Burada PEFT parametreleri $\phi$, pozisyon kodlama ve bağlam penceresi faktörlerini optimize edebilir.

### 5.2 Continuous Learning için PEFT

PEFT, sürekli öğrenme için ideal bir çerçeve sunar:

$$\phi_{t+1} = \text{Update}(\phi_t, \mathcal{D}_{t+1})$$

Burada $\phi_t$ t zamanındaki PEFT parametreleri ve $\mathcal{D}_{t+1}$ yeni veri noktalarıdır.

Teorik avantaj, taban modelin sabit kalması ve yalnızca kompakt PEFT parametrelerinin güncellenmesi ve saklanması gerekliliğidir.

### 5.3 Quantization-Aware PEFT

Quantization ve PEFT'in birleşimi, ek bellek verimliliği sağlayabilir:

$$\tilde{\phi} = Q(\phi, b)$$

Burada $Q$ bir quantization işlevi ve $b$ bit genişliğidir.

Quantization-aware PEFT eğitimi:

$$\min_{\phi} \mathcal{L}(f(\theta_0, \text{STE}(Q(\phi, b))); \mathcal{D})$$

Burada STE (Straight-Through Estimator) geri yayılım sırasında quantization gradyanlarının düzgün akmasını sağlar.

## 6. Teorik Kısıtlamalar ve Dikkat Edilmesi Gerekenler

### 6.1 Model Kapasitesi ve Ekspresif Güç

PEFT yöntemleri, tam ince ayara göre daha düşük ekspresif güç sunar:

$$\mathcal{H}_{full} \supseteq \mathcal{H}_{PEFT}$$

Burada $\mathcal{H}$ hipotez uzayıdır.

Ekspresif gücün potansiyel kısıtlamaları:
- Çok farklı domainlere transfer olma kabiliyeti sınırlı olabilir
- Belirli öğrenme görevleri için yetersiz kapasiteye sahip olabilir

### 6.2 Katman Seçimi ve Optimal PEFT Yerleşimi

Tüm katmanların eşit önemde olmadığı gösterilmiştir. Optimal PEFT yerleşiminde katman seçimine dikkat etmek gerekir:

$$\phi_{optimal} = \arg\min_{\phi} \mathcal{L}(f(\theta_0, \phi); \mathcal{D}) \text{ subject to } |\text{Layers}(\phi)| = k$$

Araştırmalar gösteriyor ki:
- Transformer mimarilerinde üst katmanlar genellikle görev-spesifik adaptasyonlar için daha kritiktir
- Alt katmanlar genellikle dil modelleme gibi temel özellikleri kodlar

### 6.3 Hiperparametre Seçimi

PEFT yöntemlerinin performansı hiperparametre seçimine kritik şekilde bağlıdır:

**LoRA için önemli hiperparametreler:**
- $r$: Rank boyutu
- $\alpha$: Ölçeklendirme faktörü
- Target modüllerin seçimi (genellikle query ve value projeksiyonları)

**Adaptörler için önemli hiperparametreler:**
- Adaptör boyutu (MLP'nin ara representasyon boyutu)
- Dropout değeri
- Paralel vs seri entegrasyon

İyi bir ampirik kural:
- LoRA için: $r$ genellikle 4-64 arası, $\alpha$ genellikle 16-32 arası
- Adaptörler için: Boyut genellikle orijinal boyutun %1-10'u arası

## 7. Sonuç

PEFT teknikleri, LLM'lerin ekonomik ve verimli adaptasyonu için kritik öneme sahiptir. Teorik avantajları ve pratik uygulamaları, bu tekniklerin hem akademik araştırmalarda hem de endüstriyel uygulamalarda giderek daha yaygın hale gelmesini sağlamıştır.

Optimal PEFT seçimi modelinize, görevinize ve kaynak kısıtlamalarınıza bağlıdır:
- Bellek kritik ise: QLoRA veya BitFit
- Hesaplama kritik ise: IA³ veya Adaptörler
- İnce ayarlanan model paylaşılacaksa: LoRA veya Adaptörler
- Çok görevli transfer gerekiyorsa: Adaptörler veya Prompt-tuning

PEFT yöntemleri, tam ince ayara çok yakın veya bazen daha iyi performans sunar, ancak parametre sayısında %0.1-1 gibi dramatik bir azalma ile. Bu, LLM'lerin demokratikleştirilmesi ve daha geniş uygulamalar için erişilebilir hale getirilmesi açısından büyük önem taşır.