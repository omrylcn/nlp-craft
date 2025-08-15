# 🧠 LangGraph Durum Yönetimi: Yeni Başlayanlar İçin Adım Adım Rehber

> **Bu rehber, LangGraph’te "durum (state)" nasıl doğru yönetilir" sorusuna cevap verir. Kod yazmadan önce mantığını anlatır, kafa karışıklığını ortadan kaldırır.**

---

## 🤔 Başlamadan Önce: Hangi Terimler Ne Demek?

| Terim | Açıklama |
|------|---------|
| **Durum (State)** | İş akışının hafızası. Tüm düğünlerin erişebileceği ortak veri deposu. |
| **Immutable (Değişmez)** | Bir veriyi doğrudan değiştirmek yerine, **yeni bir kopya** oluşturmak. |
| **Reducer** | Eski veriyle yeni veriyi birleştiren kurallar. Mesela: "listeye ekle", "sayıyı artır". |
| **TypedDict** | Python’da tip güvenli sözlük tanımlamak için kullanılan yapı. |
| **Annotated** | Bir veriye "ek bilgi" ekler. Burada: "Bu alan nasıl birleştirilsin?" |

> 🔍 **İpucu:** `Annotated[list, operator.add]` → "Bu liste, yeni gelenle toplansın (birleştirilsin)" demek.

---

## 🌐 LangGraph'te Durum Nedir?

**LangGraph State**, bir iş akışında (workflow) tüm düğünlerin (nodes) ortak kullandığı **paylaşımlı bir hafıza**dır.

### 🔄 Geleneksel Zincir vs LangGraph Durumu

#### ❌ Geleneksel Yaklaşım (Zayıf)
```python
girdi → düğüm1 → çıktı1 → düğüm2 → çıktı2 → düğüm3 → sonuç
```
- Her düğüm sadece öncekinden gelen veriyi görür.
- Bağlam kaybolabilir.  
- Hata ayıklamak zor.

#### ✅ LangGraph Yaklaşımı (Güçlü)
```python
                    📦 ORTAK HAFIZA (State)
                         ↕   ↕   ↕
girdi → düğüm1 → state → düğüm2 → state → düğüm3 → sonuç
```
- Her düğüm, **tüm geçmişi ve veriyi** görebilir.
- Veri kaybolmaz.
- Her değişiklik izlenebilir.

> 💡 **Benzetme:**  
> Geleneksel zincir → bir posta kutusundan diğerine taşınan zarf.  
> LangGraph → bir Google Docs belgesi. Herkes okur, herkes ekler, tüm değişiklikler kayıtlı.

---

## 🧱 Durum Nasıl Tanımlanır?

```python
from typing import TypedDict, Annotated
import operator

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]    # Mesajları ekle
    counter: Annotated[int, operator.add]     # Sayacı artır
    context: Annotated[dict, merge_dicts]     # Sözlüğü birleştir
```

### 📌 Her Satır Ne Anlama Geliyor?

| Satır | Açıklama |
|------|---------|
| `messages: Annotated[list, add_messages]` | Bu listeye yeni mesaj geldiğinde, `add_messages` fonksiyonu eskiyle birleştirir. |
| `counter: Annotated[int, operator.add]` | Yeni bir sayı geldiğinde, eskiyle toplanır. |
| `context: Annotated[dict, merge_dicts]` | Yeni veri geldiğinde, eski sözlükle birleştirilir. |

> ⚠️ **Dikkat:** `Annotated` olmadan yazarsanız, yeni veri eskiyi **siler**!  
> ❌ `messages: list` → yeni mesaj gelirse eski kaybolur!

---

## 🔁 Durumun Yaşam Döngüsü

### 1. Düğüm, Durumu Okur (Salt Okunur!)
```python
def mesaj_isleme_dugumu(state: WorkflowState):
    # state["messages"] okunabilir ama değiştirilemez!
    mesaj_sayisi = len(state["messages"])
```

### 2. İşlem Yapılır
```python
    yeni_mesaj = {"role": "assistant", "content": "Merhaba!"}
```

### 3. Sadece **Değişiklik** Döndürülür
```python
    return {
        "counter": 1,           # 1 artır (reducer toplar)
        "messages": [yeni_mesaj] # Yeni mesaj ekle
    }
```

### 4. LangGraph Otomatik Birleştirir
```python
# LangGraph bunu yapar:
# yeni_counter = eski_counter + 1
# yeni_messages = eski_messages + [yeni_mesaj]
```

> ✅ **Doğru:** Sadece değişikliği döndür.  
> ❌ **Yanlış:** `state["counter"] += 1` yapmaya çalışma!

---

## 🔧 Reducer Nedir? (Birleştirme Kuralları)

**Reducer**, iki veriyi birleştirmek için kullanılan bir fonksiyondur.

```python
def reducer(eski, yeni):
    return birlestirilmis
```

### 🛠️ Sık Kullanılan Reducer’lar

#### `operator.add` → Ekle, Topla, Birleştir
```python
counter: Annotated[int, operator.add]     # 5 + 3 = 8
messages: Annotated[list, operator.add]   # [1,2] + [3] = [1,2,3]
log: Annotated[str, operator.add]         # "a" + "b" = "ab"
```

#### `merge_dicts` → Sözlükleri Birleştir
```python
def merge_dicts(eski: dict, yeni: dict) -> dict:
    return {**eski, **yeni}

user_data: Annotated[dict, merge_dicts]
# {"isim": "Ali"} + {"yas": 30} → {"isim": "Ali", "yas": 30}
```

#### `unique_append` → Tekrar Etmeyen Liste
```python
def unique_append(eski: list, yeni: list) -> list:
    for item in yeni:
        if item not in eski:
            eski.append(item)
    return eski

tags: Annotated[list, unique_append]
# ["a", "b"] + ["b", "c"] → ["a", "b", "c"]
```

---

## ❌ Yanlış: Değişebilir (Mutable) Yaklaşım

```python
def kotu_dugum(state):
    state["counter"] += 1                  # ❌ DOĞRUDAN DEĞİŞTİRİYOR!
    state["messages"].append(yeni_mesaj)   # ❌ ORJİNAL VERİYE DOKUNUYOR!
    return state
```

### 🚨 Neden Kötü?
- 🔥 **Veri kaybı:** Başka bir düğün aynı anda çalışırsa çakışır.
- 🐞 **Hata ayıklama zor:** Hangi düğüm neyi değiştirdi? Bilinmez.
- 📉 **Üretimde riskli:** Çoklu kullanıcıda sistem çöker.

---

## ✅ Doğru: Değişmez (Immutable) Yaklaşım

```python
def iyi_dugum(state):
    return {
        "counter": 1,                     # LangGraph: eski + 1
        "messages": [yeni_mesaj]          # LangGraph: eski + [yeni]
    }
```

### ✅ Neden İyi?
- ✅ **Güvenli:** Aynı anda çalışan düğünler çakışmaz.
- ✅ **İzlenebilir:** Her değişiklik kayıtlı.
- ✅ **Basit:** Sadece farkı döndür, gerisini LangGraph halletsin.

> 💡 **Kural:**  
> "Düğüm fonksiyonu, sadece **ne değiştiğini** döner. Nasıl birleşeceğini **reducer** bilir."

---

## 🛠️ Gerçek Dünya Örnekleri

### 1. Sohbet Uygulaması
```python
def add_message(eski: list, yeni: list) -> list:
    """Yeni mesajlara otomatik tarih ekle"""
    for msg in yeni:
        if "tarih" not in msg:
            msg["tarih"] = datetime.now().isoformat()
    return eski + yeni

class ChatState(TypedDict):
    messages: Annotated[list, add_message]
    user_id: str
    session_ Annotated[dict, lambda e, y: {**e, **y}]
```

### 2. Hata Takibi
```python
def collect_errors(eski: list, yeni: list) -> list:
    """Hata mesajlarını zenginleştir"""
    for err in yeni:
        if isinstance(err, str):
            err = {"mesaj": err, "tarih": datetime.now().isoformat()}
    return eski + yeni

class ErrorState(TypedDict):
    errors: Annotated[list, collect_errors]
    error_count: Annotated[int, operator.add]
```

---

## 🔍 Hata Ayıklama: Ne Zaman Ne Değişti?

### Reducer’a Log Ekle
```python
def debug_reducer(alan_adi, orijinal_reducer):
    def sarmal(eski, yeni):
        print(f"🔧 {alan_adi}: {eski} + {yeni}")
        sonuc = orijinal_reducer(eski, yeni)
        print(f"   → {sonuc}")
        return sonuc
    return sarmal

# Kullanımı
messages: Annotated[list, debug_reducer("mesajlar", operator.add)]
```

### Çıktı Örneği:
```
🔧 mesajlar: [] + [{'role': 'user', 'content': 'Merhaba'}]
   → [{'role': 'user', 'content': 'Merhaba'}]
```

> 👀 Artık her değişikliği görebilirsin!

---

## 🏭 Üretimde Dikkat Edilmesi Gerekenler

| Konu | Açıklama |
|------|---------|
| **Thread Güvenliği** | Immutable → eşzamanlı işlemler güvenli |
| **Bellek Kullanımı** | Yeni nesneler oluşur → biraz daha fazla bellek |
| **Performans** | Güvenlik ve debug kolaylığı, küçük maliyeti haklı çıkarır |
| **Test Etme** | Her düğüm bağımsız test edilebilir |

---

## ✅ En İyi Uygulamalar (Best Practices)

### 1. Durum Şeması İyi Tasarlanmalı
```python
# ✅ İYİ
class IyiState(TypedDict):
    user_messages: Annotated[list, operator.add]
    processing_errors: Annotated[list, operator.add]
    user_context: Annotated[dict, merge_dicts]

# ❌ KÖTÜ
class KotuState(TypedDict):
     list        # Ne tür veri?
    stuff: dict       # Ne şeyi?
    things: list      # Reducer yok → ESKİSİNİ SİLER!
```

### 2. Düğüm Fonksiyonları Sadece Farkı Dönsün
```python
# ✅ DOĞRU
def dugum(state):
    return {"counter": 1}  # Sadece değişiklik

# ❌ YANLIŞ
def dugum(state):
    state["counter"] += 1  # ORJİNALİ DEĞİŞTİRİYOR!
    return state
```

---

## 🎯 Temel Çıkarımlar

| Gerçek | Anlamı |
|-------|--------|
| **State şeması = sistem mimarisi** | İyi tanımlanmazsa, sistem çöker |
| **Reducer = birleştirme kuralı** | Nasıl birleşeceğini belirler |
| **Immutable = güven** | Veri kaybolmaz, hata olmaz |
| **LangGraph, karmaşıklığı üstlenir** | Sen sadece iş mantığına odaklan |

---

## 📋 Uygulama Kontrol Listesi

- [ ] ✅ Durum şemasını `TypedDict` ile tanımladım
- [ ] ✅ Her alan için doğru `reducer` seçtim
- [ ] ✅ Düğün fonksiyonları sadece **değişiklik** döndürüyor
- [ ] ✅ `Annotated` kullandım (tip + reducer)
- [ ] ✅ Mutable (doğrudan değiştirme) yapmadım
- [ ] ✅ Hata ayıklama için log ekledim
- [ ] ✅ Eşzamanlı test ettim
- [ ] ✅ Takım arkadaşlarımla paylaştım 😊

---

## 🙌 Son Söz

> **LangGraph’ta iyi bir sistem inşa etmenin sırrı, "durum şemasını iyi tasarlamak"tır.**  
> Gerisini LangGraph halleder. Sen, sadece **iş mantığına** odaklan.

Bu rehberi bir PDF’e dönüştürmek, takım içi eğitimde kullanmak veya dökümantasyonunuza eklemek istersen, memnuniyetle yardımcı olurum.

Hazır mısın? Şimdi ilk `WorkflowState`'ini tanımlamaya başlayabilirsin! 🚀

```python
class MyFirstState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: Annotated[int, operator.add]
```
