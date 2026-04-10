# Pre-training İşini Yapmış Mı?

*Fine-tune'dan önce kimsenin sormadığı soru*

---

Bir dil modelinin kabaca iki temel aşamada eğitildiğini söyleyebiliriz.

- **İlk aşama, *pre-training* (ön eğitim):** Model, trilyonlarca veri (teknik tabirle *token*) üzerinden bir sonraki kelimeyi tahmin etmeye çalışarak (*next token prediction*) dili, dünyayı ve kavramları öğrenir. Modeli başlangıçta boş bir levha (*tabula rasa*) gibi düşünebilirsin. Sezgisel olarak baktığımızda bile, bir kelimenin öncesinde ve sonrasında gelen kelimeler aslında o kelimenin anlamını şekillendirir. Diyelim ki dilimizde hiç bilmediğin, tamamen uydurma bir kelime var: "**kasdJ**". Telaffuz bile edemiyoruz ama şimdi şu cümlelere bir bak:

  > *"kasdJ"yi soyarak yedim.*
  > *Elma yerine "kasdJ" al.*
  > *"kasdJ"nin tadı fena değil, biraz armuda benziyor.*
  > *Kabukları kolayca soyulan "kasdJ", sanki mandalinayı andırıyor.*

  Bütün bu cümleleri okuduğunda, sana net bir "kasdJ" tanımı yapmasam bile bu kelime zihninde bir yere oturdu (muhtemelen bir meyve). "Zihninde bir yere oturması" çok güzel ve isabetli bir tabir; çünkü matematiksel olarak da tam olarak bunu yapıyoruz. Kelimeleri çok boyutlu bir vektör uzayına (*embedding space*) indirgeyip, alakalı oldukları kavramlarla benzer yerlere konumlandırıyoruz. İşte model, *pre-training* sırasında "kasdJ" kelimesinin anlamını, hangi kavramlara benzediğini, nelerde ayrıştığını ve ne ile alakalı olduğunu bu şekilde öğreniyor.

  *[Buraya Görsel Önerisi: "Elma", "armut", "mandalina" kelimelerinin bir vektör uzayında (örneğin 3 boyutlu bir koordinat sisteminde) birbirine yakın kümelendiği ve "kasdJ" kelimesinin de onların tam arasına yerleştiğini gösteren basit bir grafik (PCA/t-SNE tarzı bir illüstrasyon) harika oturur.]*

- **İkinci aşama, *fine-tune* (ince ayar):** Bu aşamada, *pre-train* edilmiş modele belirli bir davranış kazandırılır; soru-cevap formatına uyum sağlama, talimatları izleme (*instruction following*) veya spesifik bir alana (*domain*) adapte olma gibi. Şöyle özetleyebiliriz: *Pre-training* modelin **"ne bildiğini"** belirler, *fine-tune* ise bunu **"nasıl sunacağını"**. Modeli biraz terbiye etmek, ona adap kazandırmak gibidir. Model zaten "kasdJ"nin ne olduğunu biliyor; *fine-tune* aşaması ona "kasdJ nedir?" diye sorulduğunda nasıl cevap vereceğini, hangi formatta konuşacağını ve soruları nasıl karşılayacağını öğretiyor. Yani bilgi zaten orada, *fine-tune* sadece o bilginin sunum şeklini şekillendiriyor. Tıpkı bir çocuğun dünyayı tanıması (içerik/bilgi) ile nerede nasıl konuşacağını öğrenmesi (davranış/terbiye) arasındaki fark gibi.

Buraya kadar temel kavramları biraz oturtmuş ya da hafızamızı tazelemiş olduğumuzu umuyorum. Yazının devamı için aynı dili konuşmamız, aynı frekansta olmamız önemli :)

---

## Motivasyon

Diyelim ki Türkçe çalışacak bir finans chatbot'u yapman gerekiyor. Girdin [Hugging Face](https://huggingface.co/)'e, Türkçe yetkinliği olan bir model arıyorsun. Bir tane buluyorsun; model kartında "Turkish support" yazıyor, hatta *instruction-tuned* versiyonu bile var. İşin biraz daha kolaylaşsın diye hazır *instruct* (yani halihazırda *fine-tune* edilmiş) versiyonunu alıyorsun.

Metriklerine bakıyorsun; PPL (Perplexity) fena değil. *(Ufak bir not: Perplexity, modelin yeni bir metin gördüğünde ne kadar "şaşırdığını" veya kafasının karıştığını ölçer. Puan ne kadar düşükse, model o dile o kadar hakimdir ve bir sonraki kelimeyi kendinden emin tahmin ediyordur).* MMLU skorları da idare eder.

Oturup *fine-tune* ediyorsun. Elinde kendi Finans QA (Soru-Cevap) veri setin var, birkaç bin örnekten oluşuyor. Eğitim sorunsuz bitiyor, heyecanla *deploy* ediyorsun.

Sonra garip şeyler olmaya başlıyor.

Model "faiz artırımı kararı" hakkında gramer olarak kusursuz, akıcı cümleler kuruyor — ama faiz artınca dövizin düşme eğilimine gireceğini bilmiyor. Ya da "döviz" kelimesini hiç tanımıyor; yabancı para birimine ne dendiğini bilmemesi demek, o alanın (*domain*) daha ABC'sini bilmemesi demek. Bazen terimler tamam, ilişkiler tamam ama bir bakıyorsun "evlerden" yerine "evlerdan" yazıyor, Türkçenin sondan eklemeli yapısında çuvallıyor.

Peki ne oldu?

Sen model kartındaki o rakamlara ve *benchmark* skorlarına güvendin, ama modelin **gerçekte neyi bilip neyi bilmediğine**, o temel eğitimde zihnine ne kazındığına bakmadın.

---

## Peki Neye Bakmalıydık?

"Bu model ne biliyor?" aslında çok geniş ve muğlak bir soru. Böyle sorunca cevabı da muğlak geliyor — "işte genel olarak iyi" ya da "fena değil" gibi. Halbuki modeli gerçekten tanımak istiyorsak bu soruyu parçalara ayırmamız lazım.

Ben modelleri değerlendirirken kafamda hep 3 ayrı katman oluyor. Hemen söyleyeyim, bu benim uydurduğum keyfi bir ayrım değil; NLP *probing* literatüründe (yani modelin iç dünyasını kurcalayan çalışmalarda) buna benzer birçok ölçüm metriği zaten mevcut. Ama biz belki de kendi problemimiz gereği işte bu üçüne bakıyoruz — çünkü *pre-training*'den beklediğimiz şey aslında bu üç katmanın kazanılmış olması.

*Ufak bir not: Bu yazıyı hazırlarken faydalandığım makaleleri ve çalışmaları yazının sonunda paylaşacağım, detaylı bakmak isteyenler için.*

- **Dili bilmek** — modelin Türkçenin kurallarını ne kadar içselleştirdiği. Linzen ve ark. bunu özne-yüklem uyumu (*subject-verb agreement*) üzerinden ölçmüş.
- **Dünyayı bilmek** — modelin kafasında ne gibi olgusal bilgilerin (*factual knowledge*) kazındığı. Petroni ve ark.'nın LAMA veri seti tam olarak bunu test ediyor: "Türkiye'nin başkenti nedir?" tarzı sorularla.
- **Kavramlar arası ilişkiyi bilmek** — modelin "A olunca B olur" tarzı çıkarımları ne kadar tutarlı yapabildiği. Elazar ve ark. bunu farklı sorma biçimleri altında modelin aynı cevabı verip vermediğine bakarak ölçmüş.

Bu üç katmanın birbirinden bağımsız olduğunu anlamak çok önemli. Çünkü üçünün de eğitim sırasında oluşma şekli farklı, ölçüm imzaları farklı ve eksik olduklarında uygulayacağın çözüm tamamen farklı. Mesela dili bilmiyorsa *fine-tune* bunu kurtarmaz, modeli değiştirmen gerekir. Ama dünyayı bilmiyorsa belki RAG ile bu açığı kapatabilirsin. Bunlara ileride tek tek değineceğiz.

İşte tam da bu yüzden PPL veya MMLU gibi *aggregate* metrikler (yani tek bir skorda her şeyi toplayan ölçümler) tek başına yetmiyor. Birbirinden bağımsız üç farklı problemi tek skora sıkıştırınca, hangi katmanın zayıf hangi katmanın güçlü olduğunu göremiyorsun. Model "genel olarak 7/10" geliyor ama bu 7'nin içinde dilbilgisi 10'dan, kavram ilişkisi 2'den gelmiş olabilir. Sen *fine-tune*'a başladığında da bunu fark etmiyorsun, ta ki model sana "evlerdan" yazana kadar.

Şimdi bu üç katmana sırayla bakalım.

---

### Birincisi: Dili Bilmek

En temel şey. Model Türkçeyi biliyor mu? Sonuçta bizim problemimiz Türkçe, soruları da değerlendirmeyi de Türkçe yapacağız.

"Evlerden" mi, "evlerdan" mı? "Çocuklar oyun oynuyor" mu, "çocuklar oyun oynuyorum" mu? "Ali okula gitti" mi, "gitti Ali okula" mı?

Bir insan için "Türkçe biliyor" demek ne anlama geliyorsa, model için de aynı şeyi kastediyorum. Yani sadece kuralları ezberlemiş olmak değil — kuralları bilmek ve onları akıcı bir şekilde kullanabilmek. Bu ikisi aslında ayrılmaz. Ünlü uyumunu (hani o "ev-den" ama "kapı-dan" dediğimiz kuralı) teorik olarak bilen ama yazarken "evlerdan" diyen bir insan için "Türkçe biliyor" demezsin. Model için de aynı şey geçerli. Dili bilmek demek, bu kuralların modele oturmuş ve doğal olarak kullanılabiliyor olması demek.

Peki bu olmazsa ne olur? Mesela model "çocuklar" dedikten sonra yanına 1. tekil şahıs eki getiriyorsa, özne-yüklem uyumu (*agreement*) kuralı kafasında oturmamış demektir. Bu ilk bakışta küçük bir detay gibi görünür ama değildir, büyük resmin temelidir. Morfoloji ve uyum yoksa model yazarken sürekli tökezler, ürettiği metin rahatsız edici olur. Ve en kötüsü: bu hatalar *fine-tune* sonrasında da yakanı bırakmaz. Çünkü *fine-tune* sana dili öğretmez — *fine-tune* modele "sorulduğunda nasıl cevap vereceğini" öğretir, dilin kendisini değil. Dili zaten *pre-training*'de öğrenmiş olması gerekiyordu.

Bu katmanın bir de ilginç bir yanı var: Dünya hakkında hiçbir iddia içermiyor. "Çocuklar oynuyor" ile "çocuklar koşuyor" cümleleri model için aynı derecede makul olmalı. Burada test ettiğimiz şey çocukların ne yaptığı değil, sadece dilin kendi iç kuralları. "Çocuklar" dedikten sonra 1. tekil ek gelemez — o kadar. Bu yüzden bu katman, modelde eğer oturmuşsa, genellikle *pre-training*'in çok erken aşamalarında oturuyor. Dünyayı henüz tanımadan bile dilin ritmini yakalayabiliyor. Dikkat edersen test ederken finans terimleri ya da karmaşık kavramlar kullanmıyoruz — "evler", "çocuklar", "bahçe" gibi en temel kelimelerle çalışıyoruz. Çünkü burada amacımız modelin alan, terim, kavram bilgisini değil, saf dil kabiliyetini anlamak.

Pratikte bunu ölçmek aslında çok basit. Modele iki alternatif cümle veriyorsun, hangisine daha yüksek olasılık atadığına bakıyorsun.

```python
compare("Evler", "den", "dan")                           # ünlü uyumu
compare("Çocuklar bahçede", " oynuyor.", " oynuyorum.")  # agreement
compare("Dün akşam eve geç", " geldim.", " gelir.")      # tense (zaman) uyumu
```

---

### İkincisi: Olgusal Bilgiyi Bilmek

Direkt bir soru ile başlayalım: "Model Türkçeyi bilse bile, şeylerin **ne olduğunu** biliyor mu?" (Burada "bilmek" ile kastettiğimiz şey, o bilginin modelin milyarlarca parametresi arasına kalıcı bir istatistiksel ağırlık olarak kazınmış olması).

```python
"Türkiye'nin başkenti ___"          → "Ankara"
"Yabancı ülke paralarına ___ denir" → "döviz"
"Hisse senetlerinin alınıp satıldığı piyasaya ___ denir" → "borsa"
```

Dikkat edersen bunlar bir ilişki kurma ya da mantık yürütme değil. Düpedüz ansiklopedik bilgi — yani modelin bir şeyin ne olduğunu, nasıl adlandırıldığını doğrudan "ezberlemiş" olması. "Türkiye'nin başkenti Ankara" bir fact (olgu). "Yabancı paraya döviz denir" bir terminoloji bilgisi. Burada mantık yürütmeye gerek yok, sadece bilip bilmemek var. Model bunu *pre-training* sırasında yeterince görmüşse biliyor, görmemişse bilmiyor. O kadar.

Bu katmanın işleyişi, biraz önce konuştuğumuz dil katmanından oldukça farklı. Chang ve ark.'nın (NeurIPS 2024) güzel bir çalışması var: LLM'lerin(Large Language Model) *pre-training* sırasında olguları nasıl öğrendiğini incelemişler ve ilginç bir şey bulmuşlar. Model bir fact'i her gördüğünde, o fact'i doğru tahmin etme olasılığı biraz artıyor. Ama sonraki eğitim adımlarında bu artış bir miktar sönümleniyor, yani model bir kısmını unutuyor (*forgetting*). Olgusal bilgi tek seferde kazınmıyor, tekrarla pekişiyor — biraz ders çalışmaya benziyor aslında. Bir kere okuduğun bilgi unutulur, tekrar ettiğin kalır.

Bunun pratik sonucu şu: *pre-training* corpus'unda sık geçen bilgileri model iyi biliyor, nadir geçenleri bilmiyor. Literatürde buna *long-tail knowledge* problemi deniyor. "Türkiye'nin başkenti" internette milyonlarca yerde geçer, model bunu rahat bilir. Ama "FAVÖK" gibi daha niş bir finans terimi? Orada iş değişir. Model bu terimi corpus'ta ne sıklıkla görmüş, ne kadar tekrarlı görmüş — performansı doğrudan buna bağlı.

İşte bu yüzden Türkçe bir finans chatbot'u kurarken domain terimlerinin modelin kafasında oturup oturmadığını baştan test etmek lazım. Akıcı cümle kurması seni yanıltmasın.

```python
complete("Türkiye'nin başkenti")           # "Ankara" gelmeli
complete("Yabancı ülke paralarına")        # "döviz" gelmeli

compare("Şirketin FAVÖK değeri geçen yıla göre",
        " yükseldi.", " pişti.")
```

---

### Üçüncüsü: Kavramlar Arası İlişkiyi Bilmek

Diyelim ki model Türkçeyi de biliyor, terimleri de biliyor. Peki aralarındaki **ilişkiyi** de biliyor mu?

```python
"Sobayı yaktık, oda ___"
```

Model buraya "ısındı" mı yazıyor, "soğudu" mu? "Isındı" demesi lazım, değil mi? Bu karmaşık bir mantık yürütme (*reasoning*) değil — sadece *"soba yanınca oda ısınır"* gibi en temel bir çağrışım (*association*). Üstelik "soğudu" alternatifi fizik kanununa aykırı, yani modelin buradaki tercih farkı çok net olmalı. Diyelim %95'e %5 gibi. Eğer %52'ye %48 gibi bir sonuç çıkıyorsa, model bu ilişkiyi aslında bilmiyor demektir — sadece rastgele tutturmuş.

Birkaç örnek daha:

```python
"Hava karardı, çocuklar ___ gitti"    → "eve" mi "parka" mı?
"Kış gelince ağaçlar yapraklarını ___" → "döker" mi "açar" mı?
```

Şimdi ikinci katmandan (olgusal bilgi) ne farkı var diye sorabilirsin. Orada test ettiğimiz şey tek bir bağdı — "Türkiye ↔ Ankara," "yabancı para ↔ döviz." Bir kelime ile bir başka kelime arasında tek bir doğrudan ilişki. Burada ise birden fazla kavram arasında bir *örüntü* var: "A durumu olunca B sonucu olur." Bu artık tek bir fact değil, bir **şema** (*schema*) — kavramlar arası beklentinin zihne oturması. Model "döviz" kelimesini tanıyor olabilir ama "döviz kuru yükselince ne olur" sorusunun cevabını bilmeyebilir. Hatta tersi de olabilir — ilişkiyi sezer ama terimi tanımaz.

Bu ayrım pratikte çok kritik. Neden? Çünkü ikinci katmandaki eksikleri RAG ile büyük ölçüde kapatabilirsin — modelin bilmediği terimi *context*'e koyarsın, model okur ve kullanır. Ama şema öyle çalışmıyor. Şema çalışma anında (*runtime*) inşa edilmez, *pre-training*'de modelin parametrelerine yazılmış olması gerekir. Sen *context*'e "faiz artarsa döviz düşer" diye bir cümle koyabilirsin, ama model bu cümleyi alıp "o zaman bugünkü faiz kararı sonrası USD/TL düşer" çıkarımına bağlayamıyorsa, o metnin orada olması bir işe yaramıyor. Şema *weights*'te eksikse, *context*'e koyduğun metin tek başına bu açığı kapatmıyor.

Finans domain'inde bunu şöyle test edebiliriz:

```python
compare("Faiz oranları düştüğünde tahvil fiyatları",
        " yükselir.", " düşer.")

compare("Petrol fiyatları yükselince üretim maliyetleri arttı, bu da",
        " enflasyonu yükseltti.", " enflasyonu düşürdü.")
```

Burada sadece hangi cevabın geldiğine değil, **ne kadar emin geldiğine** de bakmak lazım. %90'a %10 bir sonuç → model emin, ilişkiyi biliyor. %52'ye %48 → aslında bilmiyor, sadece iki seçenekten birini rastgele söylüyor.

---

## Neden Bunları Fine-tune'dan Önce Ölçmeliyiz?

"Tamam ama fine-tune yaparsam bu eksikleri kapatamaz mıyım?" diye sorabilirsin. Haklı bir soru. Cevap için *fine-tune*'un (daha teknik tabirle *SFT* veya *instruction tuning*) ne yaptığına, daha doğrusu ne yapmak için tasarlandığına bakmak lazım. Gelin üç açıdan bakalım:

**Birincisi, veri formatına bak.** *Fine-tune*'un girdisi nedir? **(soru, cevap)** çiftleri. Mesela:

> *"Soru: Enflasyon nedir? → Cevap: Enflasyon, fiyatların genel seviyesinin artmasıdır."*

Şimdi bir dakika düşün — bu veri çifti modele "enflasyon" **kavramını** mı öğretiyor, yoksa "nedir sorusu gelince tanım formatında cevap ver" **davranışını** mı? Kesinlikle ikincisini. Model aslında "enflasyon"un ne olduğunu zaten biliyor (ya da bilmiyor, o ayrı mesele); *fine-tune* ona sadece bu bilgiyi nasıl servis edeceğini öğretiyor. Wei ve ark. (2022) *instruction tuning*'i ilk tanımladıklarında tam da bunu söylüyordu: Görevleri (*tasks*) bir talimat (*instruction*) formatına çevirip, modeli bu formatta cevap vermeye alıştırmak.

**İkincisi, eğitim hedefine (*training objective*) bak.** Model eğitilirken neyi minimize ediyor? Cevap kısmının *loss*'unu (hata/kayıp oranını). Yani "doğru formatta cevap üretmeyi" öğreniyor — "doğru bilgiyi" öğrenmeyi değil. Bu ikisi çok farklı şeyler. Birincisi davranış, ikincisi içeriktir.

**Üçüncüsü, veri hazırlama sürecine bak.** Bir *fine-tune* veri seti hazırlarken kafanda ne var? Çeşitli sorular ve kaliteli cevaplar toplamak. Veri kalitesinden söz ederken bile aslında "Cevaplar ne kadar iyi formatlanmış, ne kadar tutarlı?" diye bakıyorsun — "Cevaplar modele yeni bilgi katıyor mu?" diye değil. Veri hazırlayan kişi de modele bilgi enjekte etmeye değil, davranışı tanımlamaya odaklanır.

İşin tasarımı, amacı, veri formatı, eğitim hedefi ve veri hazırlığı üçü de aynı hikayeyi anlatıyor. Hiçbiri modele bilgi kazandırmak için tasarlanmamış. Dilbilgisi kazandırmak için hiç tasarlanmamış. Kavramlar arası ilişki kurmak için de hiç tasarlanmamış. Peki bu yetenekler nereden gelecek? Tek bir yerden: *Pre-training*'den, öylede oluyor.

Zaten bu yeni bir fikir de değil. LIMA çalışması (Zhou ve ark., NeurIPS 2023) bunu "Yüzeysel Hizalama Hipotezi" (*Superficial Alignment Hypothesis*) olarak formülize etmişti: *"Bir modelin bilgisi ve yetenekleri neredeyse tamamen pre-training sırasında öğrenilir; hizalama (alignment), modele kullanıcılarla etkileşime girerken sadece hangi format alt-dağılımını kullanacağını öğretir."* Bu hipotez o zamandan beri tartışılıyor; sonraki çalışmalar bazı yönlerini nitelendirdi, eleştirdi, inceltti. Mesela yakın tarihli bir çalışma (Vergara-Browne ve ark., 2026), aynı iddiayı bilgi kuramsal (*information-theoretic*) bir zemine oturttu ve şunu gösterdi: *Pre-training*, bir görevde iyi performansa ulaşmanın maliyetini dramatik olarak düşürüyor. Yani *fine-tune* sıfırdan bir yetenek yaratmıyor — zaten orada olan yeteneklere kolay bir erişim kapısı açıyor.

Yani mesele aslında *"Fine-tune bunu yapabilir mi?"* değil. Mesele şu: ***Fine-tune* bunun için tasarlanmamış.** Dilbilgisi öğretmek *fine-tune*'un işi değil. "Döviz nedir" öğretmek *fine-tune*'un işi değil. "Faiz artınca döviz düşer" öğretmek *fine-tune*'un işi değil.

Bu yetenekler *pre-training*'den geliyor. O yüzden *fine-tune*'a başlamadan önce bu yeteneklerin zaten orada olup olmadığına bakmak lazım. Yoksa olmayan bir şey üzerine davranış inşa etmeye çalışıyorsun ve bu işe yaramıyor.

> **Küçük ama acı bir not:** Bu sadece teorik bir endişe değil, ampirik olarak da kanıtlandı. Gekhman ve ark. (EMNLP 2024) şunu ortaya koydu: *Fine-tune* ile modele yeni olgular (*facts*) enjekte etmeye çalıştığınızda, modelin halüsinasyon (uydurma) eğilimi doğrusal olarak artıyor. Yani *fine-tune* bu yetenekleri kazandıramıyor olmakla kalmıyor, buna zorlandığında aktif olarak modele zarar veriyor.

Peki bu üç katmandan biri ya da birkaçı eksikse ne yapacağız? Kısaca söyleyeyim: Dil eksikse fine-tune kurtarmaz, başka bir model arıyorsun ya da Türkçe CPT (continual pre-training) yapıyorsun. Olgusal bilgi eksikse, en azından dar bir terminoloji için RAG genelde yeterli oluyor; geniş domain bilgisi gerekiyorsa yine CPT. Kavramlar arası ilişki eksikse işte en zor durum bu RAG yetmiyor, domain CPT şart. Ve bir tehlikeli kombinasyon var: Fact'ler tamam ama ilişki eksik. Model doğru terimleri kullanıp yanlış çıkarımlar yapıyor. Finans gibi ciddi domain'lerde en kötü failure mode'u bu yüzeysel akıcılık, altında boş ilişki zemini.

---

## Mevcut Ölçümler Bunu Gösteriyor Mu?

Peki şimdi haklı olarak soracaksın: "Madem durum böyle, niye herkes PPL ve *benchmark* skorlarına bakıp model seçiyor? Bunlar yeterli değil mi?" Açık söyleyeyim: Değiller.

PPL, *benchmark* skorları, BPC (Bits Per Character) — bunların hepsi birer *aggregate* (bütünleşik) metrik. Yani modelin "Genel olarak iyi mi?" sorusuna bir cevap veriyorlar. Ama biz zaten yukarıda gördük ki "genel" diye bir şey yok; birbirinden tamamen bağımsız üç farklı katman var. Bu metrikler ise bu üç katmanı alıp tek bir skorda harmanlayarak önüne koyuyor. Sen de o tek skora bakıp "Tamam, bu iyi bir modelmiş" diyorsun.

Şöyle düşün: PPL skoru 15 olan iki farklı model hayal et. Birincisi dilbilgisinde mükemmel ama olgusal (*fact*) bilgisi neredeyse sıfır. İkincisinin ise terminolojisi harika ama morfolojisi sürekli dökülüyor. Bu iki modelin PPL skoru tamamen aynı çıkabilir ama biri yapacağın finans chatbot'u için harika çalışırken, diğeri hiçbir işe yaramaz. Sadece o tek skora bakan biri bu hayati farkı asla göremez.

Ve inanın, bu sadece teorik bir endişe de değil. Ankner ve ark. (ICLR 2025), "Perplexed by Perplexity" adlı çalışmalarında bunu çok somut bir şekilde gösterdi: Eğitimi sırasında *perplexity-based pruning* (veriyi perplexity skoruna göre ayıklama) yapılan modeller, *pre-training* test verisinde daha kötü PPL skorları sergiliyorlar; ancak buna rağmen *downstream task* (gerçek dünyadaki alt görevler) doğruluklarında 2 puana kadar iyileşme gösteriyorlar! Yani PPL'in iyileşmesi ile gerçek performansın iyileşmesi sadece birbiriyle alakasız (korelasyonsuz) değil, bazen tam tersi yönde hareket ediyor. PPL kötüleşirken görev başarısı artabiliyor ya da tam tersi olabiliyor.

Ben bu durumu şöyle özetliyorum: PPL skoru sana *"Hastanın ateşi var mı?"* sorusunun cevabını verir. Faydalı bir bilgidir, evet, ama tek başına yetersizdir. Çünkü ateşin birçok sebebi olabilir — sorun ciğerde mi, böbrekte mi, yoksa kalpte mi? İşte bu yazıda başından beri konuştuğumuz katman testleri, tam olarak şu sorunun cevabını arıyor: **Hangi organ?**

---

## Özet: Ne Neyi Ölçüyor?

Yazı boyunca konuştuğumuz testleri ve her birinin hangi katmanı ölçtüğünü kısa bir tabloda toparlayalım:

| Test | Ne Soruyor | Hangi Katman |
|--------|-----------|---------------|
| Ünlü uyumu, özne-yüklem uyumu | "Türkçenin temel kurallarını biliyor mu?" | Dili bilmek |
| Söz dizimi (word order) testleri | "Doğal cümle düzenini tanıyor mu?" | Dili bilmek |
| *Fact completion* ("X'in başkenti ___") | "Ansiklopedik bilgi kafasında var mı?" | Dünyayı bilmek |
| Terim tanıma ("FAVÖK", "döviz" vb.) | "Domain terimlerini tanıyor mu?" | Dünyayı bilmek |
| *Contrastive pairs* ("A olunca B olur") | "Kavramlar arası ilişkiyi kurabiliyor mu?" | Kavramlar arası ilişki |
| *Cloze completion* (bağlamlı boşluk doldurma) | "Bağlama uygun ilişki seçebiliyor mu?" | Kavramlar arası ilişki |

Ve bir de hatırlatma: PPL, MMLU gibi *aggregate* metrikler de elbette kullanılır, ama bunlar tek başına yeterli değil. Üç katmanı tek skorda sıkıştırdıkları için hangisinin zayıf hangisinin güçlü olduğunu söylemiyorlar. Onları üstüne **ek olarak** bu testleri yapıyoruz, yerine değil.

---

*Bu yazı, aklımdaki üçlemenin ilki oldu. İlk olarak pre-training'den ne beklediğimizi ve bu beklentileri nasıl teşhis edeceğimizi konuştuk — yani "hastanın hangi organı sağlam?" sorusunu nasıl soracağımızı.*

*Bir sonraki yazıda tablodaki her bir testin detayına ineceğiz: Hangi probe geleneğinden geliyor, nasıl implemente edilir, hangi eşik değerleri anlamlı, modelin cevabını nasıl yorumlamak lazım. Bir nevi "Türkçe LLM değerlendirme el kitabı" gibi düşünebilirsin.*

*Üçüncü yazıda ise teoriden çıkıp uygulamaya geçiyoruz: Kumru-2B (base) ile Kara-Kumru'yu (fine-tuned) karşılaştıracağız. 150+ test yapıp bu üç katmanın fine-tune sonrası gerçekten değişip değişmediğine bakacağız.*

---

## Referanslar

[1] Wei, J., et al. (2022). Finetuned Language Models Are Zero-Shot Learners. *ICLR 2022.*

[2] Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. *NeurIPS 2023.*

[3] Vergara-Browne, T., Patil, D., Titov, I., Reddy, S., Pimentel, T., Mosbach, M. (2026). Operationalising the Superficial Alignment Hypothesis via Task Complexity. *arXiv:2602.15829.*

[4] Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., Seo, M. (2024). How Do Large Language Models Acquire Factual Knowledge During Pretraining? *NeurIPS 2024.*

[5] Gekhman, Z., et al. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? *EMNLP 2024.*

[6] Ankner, Z., Blakeney, C., Sreenivasan, K., Marion, M., Leavitt, M. L., Paul, M. (2025). Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models. *ICLR 2025.*

[7] Linzen, T., Dupoux, E., Goldberg, Y. (2016). Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies. *TACL.*

[8] Petroni, F., et al. (2019). Language Models as Knowledge Bases? *EMNLP.*

[9] Elazar, Y., et al. (2021). Measuring and Improving Consistency in Pretrained Language Models. *TACL.*

---
