# LangGraph State Yönetimi: Kapsamlı Teknik Rehber

**LangGraph State nedir?** LangGraph State, bir workflow'daki tüm node'ların erişebildiği **merkezi, tip-güvenli, immutable veri yapısıdır**. Geleneksel chain-based yaklaşımların aksine, LangGraph State tüm adımlar arasında **kalıcı, erişilebilir context** sağlar.

---

## Başlamadan Önce Bilmeniz Gerekenler

### Temel Python Yapıları
- **TypedDict:** Dictionary'nin tip-güvenli versiyonu, field'ların tiplerini belirtir
- **Annotated[Type, metadata]:** Bir tip + o tip hakkında ek bilgi (bizim durumda reducer function)
- **operator.add:** Python'un built-in toplama operatörü, farklı tiplerle farklı davranır

### Fonksiyonel Programlama Temelleri
- **Pure Function:** Aynı input her zaman aynı output verir, yan etkisi yoktur
- **Immutable:** Değiştirilemez, original data hiç modify edilmez
- **Side Effect:** Fonksiyonun ana işi dışında external state'i değiştirmesi
- **Reducer:** İki değeri alıp birleştiren pure function

---

## Temel Kavramlar

### State Tanımlama
LangGraph'ta state, reducer function'larla birlikte `TypedDict` ve `Annotated` kullanılarak tanımlanır:

```python
def add_messages(current: list, new: list) -> list:
    return current + new  # Basit birleştirme

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]    # Mesajları otomatik birleştir
    counter: Annotated[int, operator.add]     # Sayıları otomatik topla
    context: Annotated[dict, merge_dicts]     # Dictionary'leri otomatik merge et
```

Bu tanımlar LangGraph'a her field'ın nasıl güncelleneceğini söyler.

**Temel Özellikler:**
* **Tip-Güvenli:** Runtime tip kontrolü hataları önler
* **Immutable:** Direct mutation yasak, sadece merge operasyonları
* **Kalıcı:** Data tüm node execution'ları boyunca yaşar
* **İzlenebilir:** Tüm state değişikliklerinin tam geçmişi

---

## State Yaşam Döngüsü

### Node'lar Arasındaki Data Akışı

**Geleneksel Chain Pattern (Sınırlı):**
```python
input → node1 → output1 → node2 → output2 → node3 → final
```
Her node sadece bir önceki node'un çıktısını görür.

**LangGraph State Pattern (Kapsamlı):**
```python
                    Shared State Pool
                         ↕  ↕  ↕
input → node1 → state → node2 → state → node3 → final
```
Her node complete shared state'i görebilir ve değiştirebilir.

### Execution Akışı
1. **State Input:** Node, current state'in read-only kopyasını alır
2. **Processing:** Node, state data'sını kullanarak business logic yapar
3. **Changes Return:** Node sadece değişen field'ları return eder
4. **Reducer Application:** LangGraph otomatik olarak reducer function'ları uygular
5. **State Update:** Tüm değişiklikler uygulanmış yeni immutable state oluşur

```python
def my_node(state: WorkflowState) -> dict:
    # 1. Current state'i oku (read-only)
    current_count = state["counter"]
    
    # 2. Business logic
    processed_data = process(state["messages"])
    
    # 3. Sadece değişiklikleri return et
    return {"counter": 5, "messages": [new_message]}
    
# 4. LangGraph reducer'ları otomatik uygular:
# new_counter = current_counter + 5
# new_messages = current_messages + [new_message]
```

Sen sadece değişenleri return et, LangGraph merge'i halleder.

---

## Reducer Functions: State Merge Mantığı

### Reducer Nedir?
**Reducer'lar, yeni data'nın mevcut data ile nasıl birleştirileceğini tanımlayan pure function'lardır.** İki parametre alır: `current_value` ve `new_value`, birleştirilmiş sonucu return eder.

```python
def reducer_function(current, new):
    return merged_result  # Her zaman yeni değer return eder, input'ları hiç modify etmez
```

Reducer = "İki değeri nasıl birleştireceğini bilen fonksiyon"

### Built-in Reducer'lar

**operator.add - En Yaygın**
```python
# Sayılar için: matematik toplama
counter: Annotated[int, operator.add]
# Kullanım: return {"counter": 5}  → current + 5
```
Sayılar için toplama işlemi yapar.

```python
# Listeler için: birleştirme  
messages: Annotated[list, operator.add]
# Kullanım: return {"messages": [new_msg]}  → current + [new_msg]
```
Listeler için birleştirme (append) yapar.

```python
# String'ler için: birleştirme
log: Annotated[str, operator.add]  
# Kullanım: return {"log": " new entry"}  → current + " new entry"
```
String'ler için ekleme yapar.

### Custom Reducer'lar

**Dictionary Merge:**
```python
def merge_dicts(current: dict, new: dict) -> dict:
    return {**current, **new}  # Spread operator ile merge

user_data: Annotated[dict, merge_dicts]
```
Dictionary'leri birleştirir, aynı key varsa yeni değer kazanır.

**Unique List (Duplicate Yok):**
```python
def unique_append(current: list, new: list) -> list:
    result = current[:]  # Current list'i kopyala
    for item in new:
        if item not in result:
            result.append(item)
    return result

tags: Annotated[list, unique_append]
```
Sadece unique elementleri ekler, duplicate'leri engeller.

**Bounded List (Max Boyut):**
```python
def bounded_list(max_size: int):
    def reducer(current: list, new: list) -> list:
        combined = current + new
        return combined[-max_size:] if len(combined) > max_size else combined
    return reducer

recent_items: Annotated[list, bounded_list(10)]
```
Maximum 10 element tutar, fazlası eski olanları siler.

---

## Mutable vs Immutable Pattern'ler

### Mutable Yaklaşım (Problemli)
```python
def bad_node(state):
    state["counter"] += 1                    # ❌ Direct mutation
    state["messages"].append(new_message)    # ❌ Side effect
    return state                             # ❌ Modified original
```
Original state object'i değiştiriyor - bu LangGraph'ta çalışmaz.

**Problemler:**
* **Data Kaybı:** Overwrite eder, merge etmez
* **Race Condition:** Concurrent execution'da güvenli değil
* **Kayıp Geçmiş:** Önceki değerler yok edilir
* **Zor Debug:** Değişikliklerin izi yok

### Immutable Yaklaşım (Önerilen)
```python
def good_node(state):
    return {
        "counter": 1,              # Reducer handles: current + 1
        "messages": [new_message]  # Reducer handles: current + [new_message]
    }
```
Sadece değişiklikler return ediliyor, merge işi reducer'a bırakılıyor.

**Faydalar:**
* **Data Korunması:** Tüm önceki data muhafaza edilir
* **Thread Safety:** Concurrent modification sorunları yok
* **Tam Geçmiş:** Tam audit trail mevcut
* **Kolay Debug:** Net değişiklik izleme

---

## Yaygın State Pattern'leri

### Chat Uygulaması State'i
```python
def add_message(current: list, new: list) -> list:
    """Yeni mesajlara timestamp ekle"""
    timestamped = []
    for msg in new:
        if "timestamp" not in msg:
            msg["timestamp"] = datetime.now().isoformat()
        timestamped.append(msg)
    return current + timestamped

class ChatState(TypedDict):
    messages: Annotated[list, add_message]
    user_id: str
    session_data: Annotated[dict, lambda c, n: {**c, **n}]
```
Her mesaja otomatik timestamp ekleyen chat state.

### Data Processing Pipeline State'i
```python
class ProcessingState(TypedDict):
    raw_data: list                                    # Input data (reducer yok)
    processed_items: Annotated[list, operator.add]   # Sonuçları biriktir
    errors: Annotated[list, operator.add]            # Hataları topla
    stats: Annotated[dict, lambda c, n: {**c, **n}]  # İstatistikleri merge et
    current_stage: str                               # İlerlemeyi takip et (reducer yok)
```
Data pipeline için tipik state yapısı.

### Error Tracking State'i
```python
def collect_errors(current: list, new: list) -> list:
    """Hatalara metadata ekle"""
    enhanced = []
    for error in new:
        if isinstance(error, str):
            error = {"message": error, "timestamp": datetime.now().isoformat()}
        enhanced.append(error)
    return current + enhanced

class ErrorAwareState(TypedDict):
    results: Annotated[list, operator.add]
    errors: Annotated[list, collect_errors]
    error_count: Annotated[int, operator.add]
```
Error'lara otomatik metadata ekleyen state.

---

## Debug Teknikleri

### State Değişiklik Takibi
```python
def debug_reducer(field_name: str, original_reducer):
    """Tüm değişiklikleri loglamak için reducer wrapper'ı"""
    def wrapper(current, new):
        print(f"🔧 {field_name}: {current} + {new}")
        result = original_reducer(current, new)
        print(f"   → {result}")
        return result
    return wrapper

# Kullanım
messages: Annotated[list, debug_reducer("messages", operator.add)]
```
Her reducer call'unu loglar, debugging için çok faydalı.

### State Geçmiş İncelemesi
**LangGraph internal state version'ları tutar:**
```python
# LangGraph'ın internal state geçmişinin conceptual görünümü
state_history = [
    {"version": 1, "node": "start", "state": initial_state},
    {"version": 2, "node": "node1", "state": after_node1}, 
    {"version": 3, "node": "node2", "state": after_node2}
]
```
LangGraph internal olarak tüm state değişikliklerini tutar.

Bu şunları sağlar:
* **Step-by-step debugging:** Ne zaman ne değişti tam görülür
* **Rollback yeteneği:** Önceki state version'larına dönüş
* **Audit trail'ler:** Workflow execution'ının tam geçmişi

---

## Production Hususları

### Thread Safety
**Immutable state update'ler race condition'ları ortadan kaldırır:**
```python
# Birden fazla node güvenle parallel çalışabilir
def node_a(state): return {"counter": 5}
def node_b(state): return {"counter": 3}
# Son sonuç: counter = initial + 5 + 3 (deterministik)
```
Race condition imkansız çünkü state immutable.

### Memory Trade-off'ları
**Memory Kullanımı:** Immutable pattern daha fazla memory kullanır (yeni objeler oluşturur)
**Maliyeti haklı çıkaran faydalar:**
* Ortadan kalkan debugging zamanı (state geçmişi korunmuş)
* Data corruption riskleri yok
* Mükemmel concurrency desteği
* Daha kolay testing ve maintenance

### Performance Karakteristikleri
* **Biraz daha yüksek memory kullanımı** object creation'dan
* **Daha iyi CPU utilization** güvenli paralellizasyon'dan
* **Azalmış debugging overhead** net state tracking'den
* **Daha hızlı development cycle'ları** kolay testing'den

---

## En İyi Uygulamalar

### Schema Tasarımı
```python
# ✅ İyi: Net isimler, uygun reducer'lar
class WellDesignedState(TypedDict):
    user_messages: Annotated[list, operator.add]       # Net amaç
    processing_errors: Annotated[list, operator.add]   # Spesifik tip
    user_context: Annotated[dict, merge_dicts]         # Mantıklı gruplama
```
İsimler açık, her field'ın amacı net.

```python
# ❌ Kötü: Belirsiz isimler, eksik reducer'lar  
class PoorState(TypedDict):
    data: list          # Hangi tür data?
    stuff: dict         # Hangi stuff?
    things: list        # Reducer yok = tamamen overwrite eder!
```
Belirsiz isimler, reducer eksikliği tehlikeli.

### Node Function Tasarımı
```python
# ✅ İyi: State'i oku, sadece değişiklikleri return et
def good_node(state):
    current_count = state["counter"]           # Current state'i oku
    processed = process_data(state["data"])    # Business logic
    return {"counter": 1, "data": [processed]} # Sadece değişiklikleri return et
```
State'i oku, business logic yap, sadece değişiklikleri return et.

```python
# ❌ Kötü: State'i directly mutate etmeye çalışır
def bad_node(state):
    state["counter"] += 1     # ❌ Mutation denemesi
    return state              # ❌ Modified input'u return eder
```
State'i modify etmeye çalışıyor - bu LangGraph'ta çalışmaz.

### Reducer Function Tasarımı
```python
# ✅ İyi: Pure function, edge case'leri handle eder
def good_reducer(current, new):
    if not current:
        return new
    if not new:
        return current
    return current + new
```
Edge case'leri handle eder, yan etkisi yok.

```python
# ❌ Kötü: Side effect'ler, mutation'lar
def bad_reducer(current, new):
    print("Logging")        # ❌ Side effect
    current.extend(new)     # ❌ Input'u mutate eder
    return current          # ❌ Mutate edilmiş input'u return eder
```
Yan etkiler var, input'u mutate ediyor.

---

## Sözlük

| Terim | Tanım |
|-------|-------|
| **State** | Workflow boyunca paylaşılan merkezi memory |
| **Reducer** | İki değeri birleştiren pure function |
| **Immutable** | Değiştirilemeyen, original hiç modify edilmeyen |
| **Node** | Workflow'daki bir işlem adımı, function |
| **Merge** | İki data structure'ı birleştirme işlemi |
| **Pure Function** | Aynı input → aynı output, yan etkisi yok |
| **Side Effect** | Function'ın ana işi dışında external değişiklik |
| **Type-Safe** | Compile/runtime'da tip kontrolü yapılan |
| **Audit Trail** | Tüm değişikliklerin kaydı, geçmiş |
| **Race Condition** | Concurrent execution'da data corruption |

---

## Temel Çıkarımlar

**State Management = Sistem Tasarımı**
* State schema'nız workflow'unuzun data mimarisidir
* Reducer function'lar merge semantiklerinizi tanımlar
* Immutable update'ler sistem güvenilirliğini sağlar

**LangGraph Karmaşıklığı Üstlenir**
* Otomatik reducer uygulaması
* Thread-safe state update'leri  
* Tam execution geçmişi
* Tip güvenliği zorlaması

**Business Logic'e Odaklanın**
* Node'lar state'i okur ve değişiklikleri return eder
* LangGraph state merging'i halleder
* Temiz concern separation
* Öngörülebilir, debug edilebilir davranış

**Production Faydaları**
* Lock'suz thread safety
* Tam audit trail'ler
* Kolay testing ve debugging
* Horizontal scaling desteği

---

## Implementation Kontrol Listesi

- [ ] Uygun tiplerle state schema'yı tanımla
- [ ] Her field için doğru reducer'ı seç
- [ ] Node'ları sadece değişiklikleri return edecek şekilde tasarla
- [ ] Uygun error handling implement et
- [ ] Gerektiğinde debugging/logging ekle
- [ ] Concurrent execution ile test et
- [ ] Performance karakteristiklerini doğrula
- [ ] State schema ve reducer'ları dokümante et