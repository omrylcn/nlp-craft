# LangGraph: Zero to Hero Rehberi

## 1. Giriş: LangGraph Nedir?

LangGraph, LangChain ekosisteminin bir parçası olarak geliştirilen ve LLM (Large Language Model) tabanlı uygulamalarda akış kontrolü ve durum yönetimi için tasarlanmış bir kütüphanedir. Temelde, LLM'leri kullanarak karmaşık, çok adımlı ve duruma dayalı iş akışlarını kolayca oluşturmanıza olanak tanır.

### 1.1 LangGraph'ın Temel Özellikleri

- **Durum Yönetimi**: Uygulamanızın durumunu koruyarak, karmaşık iş akışlarını yönetebilirsiniz.
- **Görsel ve Programatik İş Akışı Oluşturma**: İş akışlarınızı hem kod üzerinden hem de görsel olarak tanımlayabilirsiniz.
- **Modülerlik**: Bileşenlerinizi modüler olarak tasarlayabilir ve yeniden kullanabilirsiniz.
- **LangChain Entegrasyonu**: LangChain'in tüm özellikleriyle sorunsuz bir şekilde çalışır.
- **Akıllı Yönlendirme**: İş akışınızdaki adımları dinamik olarak belirleyebilirsiniz.

### 1.2 Neden LangGraph Kullanmalıyız?

LLM'lerin (Büyük Dil Modelleri) genellikle tek seferlik istek-yanıt modeline dayalı çalıştığını biliyoruz. Ancak gerçek hayattaki birçok problem, çözülmesi için birden fazla adım, karar noktası ve durum bilgisi gerektirir. İşte LangGraph tam da bu noktada devreye giriyor:

- **Durum Yönetimi**: Veri ve süreç durumunu saklar ve yönetir
- **Karmaşık İş Akışları**: Birden fazla adım ve karar noktası içeren süreçleri modelleyebilir
- **Yeniden Giriş**: Süreçlere geri dönüş ve yineleme yapabilir
- **Çoklu Aktör Sistemleri**: Farklı LLM ajanlarının birlikte çalışmasını sağlar

Geleneksel zincir yapılarının aksine LangGraph, düğümler (nodes) ve kenarlar (edges) arasında çok yönlü geçişlere izin verir, böylece döngüler, koşullu dallanmalar ve çok daha karmaşık akışlar oluşturabilirsiniz.

## 2. Kurulum ve Başlangıç

### 2.1 Kurulum

İlk olarak, LangGraph kütüphanesini kuralım:

```python
pip install langgraph
```

Ayrıca LangChain'i de kurmamız gerekiyor:

```python
pip install langchain langchain-openai
```

### 2.2 Temel Kavramlar

LangGraph'i anlamak için bazı temel kavramları bilmemiz gerekiyor:

- **Düğüm (Node)**: İş akışınızdaki her bir işlem adımı
- **Kenar (Edge)**: Düğümler arasındaki bağlantılar, akışın yönünü belirler
- **Durum (State)**: İş akışı boyunca taşınan ve değiştirilen veri
- **Graph (Graf)**: Düğümler ve kenarlardan oluşan yapı

### 2.3 İlk LangGraph Uygulaması

Basit bir LangGraph uygulaması oluşturalım. Bir konuşma akışını modelleyen bir örnek:

```python
import os
from typing import TypedDict, Annotated, Sequence, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# OpenAI API anahtarınızı ayarlayın
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Durum tipini tanımlama
class ConversationState(TypedDict):
    messages: Annotated[Sequence[Dict[str, Any]], "Sohbet mesajları"]
    next_step: Annotated[str, "Sonraki adım: 'respond' veya 'end'"]

# LLM'i tanımlama
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Düğüm fonksiyonlarını tanımlama
def respond(state: ConversationState) -> ConversationState:
    """LLM'den yanıt alır"""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    # Yanıtı mesajlara ekle
    new_messages = list(messages)
    new_messages.append({"role": "assistant", "content": response.content})
    
    # Sonraki adımı belirle (burada basit olarak sohbeti sonlandırıyoruz)
    return {"messages": new_messages, "next_step": "end"}

# Grafı oluşturma
workflow = StateGraph(ConversationState)

# Düğümleri ekle
workflow.add_node("respond", respond)

# Başlangıç düğümünü ayarla
workflow.set_entry_point("respond")

# Kenarları tanımla - burada tek bir kenar var, respond'dan END'e
workflow.add_edge("respond", END)

# Grafı derle
app = workflow.compile()

# Uygulamayı çalıştırma
result = app.invoke({
    "messages": [{"role": "user", "content": "Merhaba, nasılsın?"}],
    "next_step": "respond"
})

print(result["messages"][-1]["content"])
```

Bu basit örnekte:
1. Bir durum tipi tanımladık (`ConversationState`)
2. Bir düğüm fonksiyonu oluşturduk (`respond`)
3. Bir graf yapısı oluşturduk ve düğümü ekledik
4. Giriş noktası ve kenarları tanımladık
5. Grafı derleyip çalıştırdık

## 3. LangGraph'ın Çalışma Mantığı

### 3.1 Durum ve Veri Yönetimi

LangGraph'ın en önemli özelliği, durum yönetimidir. Durum, iş akışı boyunca taşınan ve değiştirilen veridir. Her düğüm, mevcut durumu alır, işler ve yeni bir durum döndürür.

```python
# Daha karmaşık bir durum örneği
class ComplexState(TypedDict):
    messages: Annotated[Sequence[Dict[str, Any]], "Sohbet mesajları"]
    context: Annotated[Dict[str, Any], "Bağlam bilgisi"]
    user_info: Annotated[Dict[str, Any], "Kullanıcı bilgisi"]
    current_step: Annotated[str, "Şu anki adım"]
```

Durumun immutable (değiştirilemez) olduğunu unutmayın. Her düğüm, yeni bir durum nesnesi döndürmelidir, mevcut nesneyi değiştirmemelidir.

### 3.2 Koşullu Yönlendirme

LangGraph'ın güçlü yanlarından biri, koşullu yönlendirme yapabilmesidir. İş akışınızın hangi yöne gideceğini dinamik olarak belirleyebilirsiniz.

```python
# Koşullu yönlendirme örneği
def router(state: ConversationState) -> str:
    """Bir sonraki adımı belirler"""
    message = state["messages"][-1]["content"].lower()
    
    if "yardım" in message or "help" in message:
        return "help_response"
    elif "teşekkür" in message or "thanks" in message:
        return "thank_you_response"
    else:
        return "general_response"

# Graf yapısını oluşturma
workflow = StateGraph(ConversationState)

# Düğümleri ekleme
workflow.add_node("help_response", help_response_function)
workflow.add_node("thank_you_response", thank_you_function)
workflow.add_node("general_response", general_response_function)

# Router düğümünü ekle
workflow.add_node("router", router)

# Başlangıç düğümünü ayarla
workflow.set_entry_point("router")

# Kenarları tanımla - router'dan diğer düğümlere
workflow.add_conditional_edges(
    "router",
    lambda x: x,  # router fonksiyonunun döndürdüğü değeri kullan
    {
        "help_response": "help_response",
        "thank_you_response": "thank_you_response",
        "general_response": "general_response"
    }
)

# Tüm yanıt düğümlerinden END'e
workflow.add_edge("help_response", END)
workflow.add_edge("thank_you_response", END)
workflow.add_edge("general_response", END)
```

Bu örnekte, `router` fonksiyonu mesajın içeriğine göre bir sonraki adımı belirliyor ve graf bu adıma yönlendiriliyor.

### 3.3 Düğüm Tipleri

LangGraph'ta farklı düğüm tipleri bulunur:

1. **İşlem Düğümleri**: Durumu işleyen ve yeni bir durum döndüren fonksiyonlar
2. **Yönlendirme Düğümleri**: Bir sonraki adımı belirleyen fonksiyonlar
3. **Karar Düğümleri**: Koşullara göre farklı yollara yönlendiren düğümler
4. **Başlangıç ve Bitiş Düğümleri**: İş akışının başlangıç ve bitiş noktaları

Her düğüm, belirli bir işlevi yerine getirir ve durumu değiştirebilir veya akışı yönlendirebilir.

## 4. Temel LangGraph Yapıları

### 4.1 Basit Graf Oluşturma

En temel LangGraph yapısı, düğümler ve kenarlardan oluşan bir graftır:

```python
from langgraph.graph import StateGraph, END

# Graf oluşturma
workflow = StateGraph(YourStateType)

# Düğümleri ekleme
workflow.add_node("node1", node1_function)
workflow.add_node("node2", node2_function)
workflow.add_node("node3", node3_function)

# Başlangıç noktasını ayarlama
workflow.set_entry_point("node1")

# Kenarları ekleme
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", "node3")
workflow.add_edge("node3", END)

# Grafı derleme
app = workflow.compile()
```

### 4.2 Döngüler ve Yinelemeler

LangGraph'ın güçlü yanlarından biri, döngüler ve yinelemeler oluşturabilmesidir:

```python
# Döngü örneği - Bir görevi tamamlayana kadar tekrarlama
def check_completion(state: TaskState) -> str:
    """Görevin tamamlanıp tamamlanmadığını kontrol eder"""
    if state["task_completed"]:
        return "complete"
    else:
        return "continue"

workflow = StateGraph(TaskState)
workflow.add_node("process_task", process_task_function)
workflow.add_node("check_completion", check_completion)

workflow.set_entry_point("process_task")
workflow.add_edge("process_task", "check_completion")

# Koşullu kenarlar
workflow.add_conditional_edges(
    "check_completion",
    lambda x: x,
    {
        "complete": END,
        "continue": "process_task"  # Döngü: tekrar process_task'a dön
    }
)
```

Bu örnekte, `check_completion` fonksiyonu görevin tamamlanıp tamamlanmadığını kontrol eder. Eğer tamamlanmadıysa, akış tekrar `process_task` düğümüne döner ve bir döngü oluşturur.

### 4.3 Paralel İşlemler

LangGraph, paralel işlemler yapmanıza da olanak tanır:

```python
# Paralel işlem örneği
def branch(state: AnalysisState) -> Dict[str, Any]:
    """Verileri farklı analizler için ayırır"""
    return {
        "sentiment_analysis": state,
        "entity_extraction": state,
        "summarization": state
    }

def join(states: Dict[str, AnalysisState]) -> AnalysisState:
    """Farklı analizlerin sonuçlarını birleştirir"""
    combined_state = {}
    combined_state["sentiment"] = states["sentiment_analysis"]["sentiment"]
    combined_state["entities"] = states["entity_extraction"]["entities"]
    combined_state["summary"] = states["summarization"]["summary"]
    return combined_state

workflow = StateGraph(AnalysisState)
workflow.add_node("preprocess", preprocess_function)
workflow.add_node("branch", branch)
workflow.add_node("sentiment_analysis", sentiment_analysis_function)
workflow.add_node("entity_extraction", entity_extraction_function)
workflow.add_node("summarization", summarization_function)
workflow.add_node("join", join)

workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "branch")

# Paralel kenarlar
workflow.add_edge("branch", "sentiment_analysis", key="sentiment_analysis")
workflow.add_edge("branch", "entity_extraction", key="entity_extraction")
workflow.add_edge("branch", "summarization", key="summarization")

# Birleştirme
workflow.add_edge("sentiment_analysis", "join", key="sentiment_analysis")
workflow.add_edge("entity_extraction", "join", key="entity_extraction")
workflow.add_edge("summarization", "join", key="summarization")

workflow.add_edge("join", END)
```

Bu örnekte, veri önce `branch` düğümünde farklı analiz yollarına ayrılır, her analiz paralel olarak çalışır ve sonra `join` düğümünde sonuçlar birleştirilir.

## 5. İleri Düzey LangGraph Konseptleri

### 5.1 Çoklu Aktör Sistemleri

LangGraph, birden fazla LLM ajanının birlikte çalıştığı sistemler oluşturmanıza olanak tanır:

```python
# Çoklu aktör sistemi örneği
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[Dict[str, Any]], "Sohbet mesajları"]
    current_actor: Annotated[str, "Şu anda aktif olan aktör"]
    context: Annotated[Dict[str, Any], "Paylaşılan bağlam"]

def researcher(state: MultiAgentState) -> MultiAgentState:
    """Araştırmacı ajan - bilgi toplar"""
    # LLM kullanarak araştırma yap
    research_results = llm.invoke([
        {"role": "system", "content": "Sen bir araştırmacısın. Verilen konuda detaylı bilgi topla."},
        {"role": "user", "content": state["messages"][-1]["content"]}
    ])
    
    # Yeni mesajları ve bağlamı oluştur
    new_messages = list(state["messages"])
    new_messages.append({"role": "researcher", "content": research_results.content})
    
    new_context = dict(state["context"])
    new_context["research_data"] = research_results.content
    
    # Bir sonraki aktörü belirle
    return {
        "messages": new_messages,
        "current_actor": "planner",
        "context": new_context
    }

def planner(state: MultiAgentState) -> MultiAgentState:
    """Planlayıcı ajan - araştırma sonuçlarını kullanarak plan yapar"""
    # LLM kullanarak plan oluştur
    plan_results = llm.invoke([
        {"role": "system", "content": "Sen bir planlayıcısın. Araştırma verilerini kullanarak bir plan oluştur."},
        {"role": "user", "content": f"Araştırma verileri: {state['context']['research_data']}\nBu verilere dayanarak bir plan oluştur."}
    ])
    
    # Yeni mesajları ve bağlamı oluştur
    new_messages = list(state["messages"])
    new_messages.append({"role": "planner", "content": plan_results.content})
    
    new_context = dict(state["context"])
    new_context["plan"] = plan_results.content
    
    # Bir sonraki aktörü belirle
    return {
        "messages": new_messages,
        "current_actor": "executor",
        "context": new_context
    }

def executor(state: MultiAgentState) -> MultiAgentState:
    """Uygulayıcı ajan - planı yürütür"""
    # LLM kullanarak planı uygula
    execution_results = llm.invoke([
        {"role": "system", "content": "Sen bir uygulayıcısın. Verilen planı uygula ve sonuçları açıkla."},
        {"role": "user", "content": f"Plan: {state['context']['plan']}\nBu planı uygula ve sonuçları açıkla."}
    ])
    
    # Yeni mesajları ve bağlamı oluştur
    new_messages = list(state["messages"])
    new_messages.append({"role": "executor", "content": execution_results.content})
    
    new_context = dict(state["context"])
    new_context["execution_results"] = execution_results.content
    
    # Bir sonraki aktörü belirle (burada işlem tamamlandı)
    return {
        "messages": new_messages,
        "current_actor": "end",
        "context": new_context
    }

def router(state: MultiAgentState) -> str:
    """Duruma göre bir sonraki aktörü belirler"""
    return state["current_actor"]

# Graf oluşturma
workflow = StateGraph(MultiAgentState)

# Aktör düğümlerini ekleme
workflow.add_node("researcher", researcher)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)

# Router düğümünü ekleme
workflow.add_node("router", router)

# Başlangıç noktasını ayarlama
workflow.set_entry_point("router")

# Koşullu kenarlar
workflow.add_conditional_edges(
    "router",
    lambda x: x,
    {
        "researcher": "researcher",
        "planner": "planner",
        "executor": "executor",
        "end": END
    }
)

# Her aktörden router'a geri dön
workflow.add_edge("researcher", "router")
workflow.add_edge("planner", "router")
workflow.add_edge("executor", "router")

# Grafı derleme
agent_system = workflow.compile()
```

Bu örnekte, üç farklı ajan (araştırmacı, planlayıcı ve uygulayıcı) birlikte çalışarak karmaşık bir görevi tamamlar. Her ajan kendi uzmanlık alanında çalışır ve bir sonraki ajanı belirler.

### 5.2 Modülerlik ve Kompozisyon

LangGraph, modüler graf yapıları oluşturmanıza ve bunları birleştirmenize olanak tanır:

```python
# Alt graflar oluşturma
def create_research_subgraph():
    """Araştırma alt grafını oluşturur"""
    subgraph = StateGraph(ResearchState)
    # ... düğümler ve kenarlar eklenir ...
    return subgraph.compile()

def create_analysis_subgraph():
    """Analiz alt grafını oluşturur"""
    subgraph = StateGraph(AnalysisState)
    # ... düğümler ve kenarlar eklenir ...
    return subgraph.compile()

# Alt grafları ana grafa entegre etme
main_graph = StateGraph(MainState)
main_graph.add_node("research", create_research_subgraph())
main_graph.add_node("analysis", create_analysis_subgraph())

main_graph.set_entry_point("research")
main_graph.add_edge("research", "analysis")
main_graph.add_edge("analysis", END)

app = main_graph.compile()
```

Bu yaklaşım, karmaşık sistemleri daha küçük, yönetilebilir parçalara bölmenize olanak tanır.

### 5.3 Hata Yönetimi ve Dayanıklılık

LangGraph uygulamalarında hata yönetimi önemlidir:

```python
# Hata yakalama ve işleme örneği
def safe_operation(state: AppState) -> AppState:
    """Güvenli bir şekilde operasyon gerçekleştirir"""
    try:
        # Riskli operasyon
        result = perform_risky_operation(state)
        return {**state, "result": result, "error": None}
    except Exception as e:
        # Hata durumunda
        return {**state, "result": None, "error": str(e)}

def error_handler(state: AppState) -> str:
    """Hata durumunu kontrol eder"""
    if state["error"]:
        return "handle_error"
    else:
        return "continue_normal"

# Graf oluşturma
workflow = StateGraph(AppState)
workflow.add_node("operation", safe_operation)
workflow.add_node("error_check", error_handler)
workflow.add_node("handle_error", error_handling_function)
workflow.add_node("normal_flow", normal_flow_function)

workflow.set_entry_point("operation")
workflow.add_edge("operation", "error_check")

# Koşullu kenarlar
workflow.add_conditional_edges(
    "error_check",
    lambda x: x,
    {
        "handle_error": "handle_error",
        "continue_normal": "normal_flow"
    }
)

workflow.add_edge("handle_error", END)
workflow.add_edge("normal_flow", END)
```

Bu örnekte, bir operasyon güvenli bir şekilde çalıştırılır ve hata durumunda özel bir hata işleme düğümüne yönlendirilir.

## 6. LangGraph ve LangChain Entegrasyonu

LangGraph, LangChain ile sorunsuz bir şekilde çalışır ve LangChain'in tüm özelliklerini kullanabilirsiniz.

### 6.1 LangChain Bileşenlerini Kullanma

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent

# LangChain bileşenlerini oluşturma
llm = ChatOpenAI(model="gpt-3.5-turbo")
search_tool = DuckDuckGoSearchRun()
prompt = ChatPromptTemplate.from_template("""Sistem: {system}
İnsan: {human}
Asistan:""")
chain = LLMChain(llm=llm, prompt=prompt)

# LangChain bileşenlerini LangGraph düğümlerine entegre etme
def search_node(state: AgentState) -> AgentState:
    """Arama yapan düğüm"""
    query = state["query"]
    results = search_tool.run(query)
    return {**state, "search_results": results}

def generate_response(state: AgentState) -> AgentState:
    """LLM ile yanıt üreten düğüm"""
    response = chain.invoke({
        "system": "Sen yardımcı bir asistansın.",
        "human": f"Soru: {state['query']}\nArama sonuçları: {state['search_results']}"
    })
    return {**state, "response": response}

# LangGraph yapısını oluşturma
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("search")
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

### 6.2 ReAct Ajanlarını Entegre Etme

LangGraph, LangChain'in ReAct ajanlarını bir graf yapısına entegre etmenize olanak tanır:

```python
from langchain.agents import create_react_agent
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# Araçları tanımlama
tools = [DuckDuckGoSearchRun(), WikipediaQueryRun()]

# ReAct ajanını oluşturma
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools)

# Ajanı LangGraph'a entegre etme
def agent_node(state: QueryState) -> QueryState:
    """ReAct ajanını çalıştıran düğüm"""
    query = state["query"]
    response = agent_executor.invoke({"input": query})
    return {**state, "response": response}

workflow = StateGraph(QueryState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()
```

## 7. Gerçek Dünya Uygulamaları

### 7.1 Çok Adımlı Sohbet Ajanı

Gerçek bir uygulama olarak, çok adımlı bir sohbet ajanı oluşturalım:

```python
import os
import json
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Durum tipi
class ChatState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], "Tüm mesajlar"]
    current_message: Annotated[str, "Şu anki insan mesajı"]
    context: Annotated[Dict[str, Any], "Konuşma bağlamı"]
    proposed_response: Annotated[str, "Önerilen yanıt"]
    final_response: Annotated[str, "Son yanıt"]
    action: Annotated[str, "Bir sonraki eylem"]

# LLM'i oluşturma
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Düğüm fonksiyonları
def understand_query(state: ChatState) -> ChatState:
    """Kullanıcı sorgusunu anlar ve bağlamı günceller"""
    current_msg = state["current_message"]
    context = state.get("context", {})
    
    # Mesajı analiz et
    analysis = llm.invoke([
        SystemMessage(content="Kullanıcı mesajını analiz et ve anahtar bilgileri çıkar. JSON formatında döndür."),
        HumanMessage(content=current_msg)
    ])
    
    try:
        analysis_data = json.loads(analysis.content)
    except:
        analysis_data = {"topic": "genel", "sentiment": "nötr"}
    
    # Bağlamı güncelle
    updated_context = dict(context)
    updated_context["topic"] = analysis_data.get("topic", "genel")
    updated_context["sentiment"] = analysis_data.get("sentiment", "nötr")
    
    # Mesajlar listesini güncelle
    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": current_msg})
    
    return {
        **state,
        "messages": messages,
        "context": updated_context,
        "action": "generate_response"
    }

def generate_response(state: ChatState) -> ChatState:
    """LLM kullanarak yanıt oluşturur"""
    messages = state["messages"]
    context = state["context"]
    
    # Sistem mesajını hazırla
    system_content = f"""
    Sen yardımcı bir asistansın. Şu anki konu: {context.get('topic')}
    Kullanıcının duygusu: {context.get('sentiment')}
    Bilgili, yardımcı ve samimi ol.
    """
    
    # LLM'i çağır
    chain_messages = [SystemMessage(content=system_content)]
    for msg in messages:
        if msg["role"] == "user":
            chain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chain_messages.append(AIMessage(content=msg["content"]))
    
    response = llm.invoke(chain_messages)
    
    return {
        **state,
        "proposed_response": response.content,
        "action": "check_response"
    }

def check_response(state: ChatState) -> ChatState:
    """Oluşturulan yanıtı kontrol eder"""
    proposed = state["proposed_response"]
    context = state["context"]
    
    # Yanıtı değerlendir
    evaluation = llm.invoke([
        SystemMessage(content="Bu yanıtı değerlendir. Yararlı, doğru ve nazik mi? 1-10 arası puan ver."),
        HumanMessage(content=f"Yanıt: {proposed}\nKonu: {context.get('topic')}")
    ])
    
    try:
        score = int(evaluation.content.split()[0])
    except:
        score = 5
    
    if score >= 7:
        # Yanıt yeterince iyi
        return {
            **state,
            "final_response": proposed,
            "action": "deliver_response"
        }
    else:
        # Yanıtı iyileştir
        improvement = llm.invoke([
            SystemMessage(content="Bu yanıtı iyileştir. Daha yararlı, doğru ve nazik yap."),
            HumanMessage(content=f"Orijinal yanıt: {proposed}")
        ])
        
        return {
            **state,
            "final_response": improvement.content,
            "action": "deliver_response"
        }

def deliver_response(state: ChatState) -> ChatState:
    """Son yanıtı teslim eder"""
    messages = list(state["messages"])
    messages.append({"role": "assistant", "content": state["final_response"]})
    
    return {
        **state,
        "messages": messages,
        "action": "end"
    }

def router(state: ChatState) -> str:
    """Bir sonraki adımı belirler"""
    return state["action"]

# Graf oluşturma
workflow = StateGraph(ChatState)

# Düğümleri ekleme
workflow.add_node("understand_query", understand_query)
workflow.add_node("generate_response", generate_response)
workflow.add_node("check_response", check_response)
workflow.add_node("deliver_response", deliver_response)
workflow.add_node("router", router)

# Başlangıç noktasını ayarlama
workflow.set_entry_point("understand_query")

# Kenarları ekleme
workflow.add_edge("understand_query", "router")
workflow.add_edge("generate_response", "router")
workflow.add_edge("check_response", "router")
workflow.add_edge("deliver_response", "router")

# Koşullu kenarlar
workflow.add_conditional_edges(
    "router",
    lambda x: x,
    {
        "generate_response": "generate_response",
        "check_response": "check_response",
        "deliver_response": "deliver_response",
        "end": END
    }
)

# Grafı derleme
chat_agent = workflow.compile()

# Kullanım örneği
result = chat_agent.invoke({
    "messages": [],
    "current_message": "Merhaba, bugün hava nasıl?",
    "context": {},
    "action": "understand_query"
})

print(result["messages"][-1]["content"])
```

Bu örnekte, kullanıcı mesajını anlayan, yanıt üreten, yanıtı kontrol eden ve sonunda teslim eden bir sohbet ajanı oluşturduk.

### 7.2 İş Akışı Otomasyonu

LangGraph'ı iş akışı otomasyonu için kullanabiliriz:

```python
# İş akışı otomasyonu örneği
class WorkflowState(TypedDict):
    task: Annotated[Dict[str, Any], "Görev bilgisi"]
    status: Annotated[str, "Görevin durumu"]
    history: Annotated[List[Dict[str, Any]], "İşlem geçmişi"]
    next_step: Annotated[str, "Bir sonraki adım"]

def parse_request(state: WorkflowState) -> WorkflowState:
    """Gelen isteği işler"""
    task = state["task"]
    
    # İsteği analiz et
    task_type = task.get("type", "unknown")
    priority = task.get("priority", "normal")
    
    # Geçmişi güncelle
    history = list(state.get("history", []))
    history.append({
        "action": "request_received",
        "timestamp": datetime.now().isoformat(),
        "details": f"İstek alındı: {task_type}, Öncelik: {priority}"
    })
    
    # Bir sonraki adımı belirle
    if task_type == "approval":
        next_step = "approval_process"
    elif task_type == "information":
        next_step = "information_process"
    else:
        next_step = "general_process"
    
    return {
        **state,
        "status": "received",
        "history": history,
        "next_step": next_step
    }

# Diğer işlem düğümleri...

def workflow_router(state: WorkflowState) -> str:
    """İş akışı yönlendiricisi"""
    return state["next_step"]

# Graf oluşturma
workflow = StateGraph(WorkflowState)

# Düğümleri ekleme
workflow.add_node("parse_request", parse_request)
workflow.add_node("approval_process", approval_process_function)
workflow.add_node("information_process", information_process_function)
workflow.add_node("general_process", general_process_function)
workflow.add_node("router", workflow_router)

# Başlangıç noktasını ayarlama
workflow.set_entry_point("parse_request")

# Kenarları ekleme
workflow.add_edge("parse_request", "router")
workflow.add_edge("approval_process", "router")
workflow.add_edge("information_process", "router")
workflow.add_edge("general_process", "router")

# Koşullu kenarlar
workflow.add_conditional_edges(
    "router",
    lambda x: x,
    {
        "approval_process": "approval_process",
        "information_process": "information_process",
        "general_process": "general_process",
        "complete": END
    }
)

# Grafı derleme
automation_system = workflow.compile()
```

Bu örnekte, farklı türdeki iş akışı isteklerini işleyen ve yönlendiren bir otomasyon sistemi oluşturduk.

## 8. Performans ve Optimizasyon

### 8.1 Önbellekleme ve Hızlandırma

LangGraph uygulamalarında performansı artırmak için önbellekleme kullanabilirsiniz:

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

# Önbelleği ayarlama
set_llm_cache(InMemoryCache())

# Veya Redis önbelleği kullanma
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis.from_url("redis://localhost:6379")
set_llm_cache(RedisCache(redis_client))
```

### 8.2 Paralel İşleme

İşlemleri paralel olarak çalıştırarak performansı artırabilirsiniz:

```python
# Paralel işleme örneği - dikkat edin bu senkron bir örnektir,
# gerçek uygulamada async kullanabilirsiniz
def parallel_process(state: ProcessState) -> ProcessState:
    """Birden fazla işlemi paralel çalıştırır"""
    import concurrent.futures
    
    tasks = state["tasks"]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # İşlemleri paralel olarak başlat
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        
        # Sonuçları topla
        results = []
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"task": task, "error": str(e)})
    
    return {**state, "results": results, "status": "completed"}
```

## 9. Test ve Hata Ayıklama

### 9.1 LangGraph Uygulamalarını Test Etme

```python
# Test örneği
def test_chat_agent():
    """Sohbet ajanını test eder"""
    # Test durumu oluştur
    test_state = {
        "messages": [],
        "current_message": "Test mesajı",
        "context": {},
        "action": "understand_query"
    }
    
    # Ajanı çalıştır
    result = chat_agent.invoke(test_state)
    
    # Sonuçları kontrol et
    assert "final_response" in result
    assert len(result["messages"]) > 0
    assert result["messages"][-1]["role"] == "assistant"
```

### 9.2 Graf Görselleştirme

LangGraph, graflarınızı görselleştirmenize olanak tanır:

```python
# Graf görselleştirme
from langgraph.graph import draw_graph

# Grafı çiz
draw_graph(workflow)

# VEYA dosyaya kaydet
with open("workflow_graph.html", "w") as f:
    f.write(draw_graph(workflow, format="html"))
```

## 10. İleri Düzey Örnekler ve Kısıtlamalar

### 10.1 Karmaşık Bir Örnek: Çoklu Ajan Araştırma Asistanı

```python
# Çoklu ajan araştırma asistanı örneği
class ResearchState(TypedDict):
    query: Annotated[str, "Araştırma sorgusu"]
    sources: Annotated[List[Dict[str, Any]], "Toplanan kaynaklar"]
    analysis: Annotated[Dict[str, Any], "Kaynakların analizi"]
    summary: Annotated[str, "Özet rapor"]
    current_agent: Annotated[str, "Aktif ajan"]
    status: Annotated[str, "İşlem durumu"]

# Ajan fonksiyonları
def researcher(state: ResearchState) -> ResearchState:
    """Kaynakları toplayan ajan"""
    query = state["query"]
    
    # Kaynakları topla (gerçekte bir arama API'si kullanılabilir)
    sources = [
        {"title": f"Kaynak 1 - {query}", "content": "..."},
        {"title": f"Kaynak 2 - {query}", "content": "..."},
        {"title": f"Kaynak 3 - {query}", "content": "..."}
    ]
    
    return {
        **state,
        "sources": sources,
        "current_agent": "analyzer",
        "status": "sources_collected"
    }

def analyzer(state: ResearchState) -> ResearchState:
    """Kaynakları analiz eden ajan"""
    sources = state["sources"]
    
    # Kaynakları analiz et
    analysis = {
        "key_points": ["Nokta 1", "Nokta 2", "Nokta 3"],
        "reliability": [0.8, 0.7, 0.9],
        "relevance": [0.9, 0.8, 0.7]
    }
    
    return {
        **state,
        "analysis": analysis,
        "current_agent": "writer",
        "status": "analysis_completed"
    }

def writer(state: ResearchState) -> ResearchState:
    """Rapor yazan ajan"""
    query = state["query"]
    sources = state["sources"]
    analysis = state["analysis"]
    
    # Özet rapor yaz
    summary = f"""
    ## {query.title()} Araştırma Raporu
    
    ### Anahtar Noktalar
    
    {', '.join(analysis['key_points'])}
    
    ### Kaynaklar
    
    {', '.join(source['title'] for source in sources)}
    
    ### Sonuç
    
    Bu araştırma, {query} hakkında kapsamlı bir analiz sunmaktadır.
    """
    
    return {
        **state,
        "summary": summary,
        "current_agent": "end",
        "status": "completed"
    }

def agent_router(state: ResearchState) -> str:
    """Duruma göre bir sonraki ajanı belirler"""
    return state["current_agent"]

# Graf oluşturma
workflow = StateGraph(ResearchState)

# Ajan düğümlerini ekleme
workflow.add_node("researcher", researcher)
workflow.add_node("analyzer", analyzer)
workflow.add_node("writer", writer)

# Router düğümünü ekleme
workflow.add_node("router", agent_router)

# Başlangıç noktasını ayarlama
workflow.set_entry_point("router")

# Koşullu kenarlar
workflow.add_conditional_edges(
    "router",
    lambda x: x,
    {
        "researcher": "researcher",
        "analyzer": "analyzer",
        "writer": "writer",
        "end": END
    }
)

# Her ajandan router'a geri dön
workflow.add_edge("researcher", "router")
workflow.add_edge("analyzer", "router")
workflow.add_edge("writer", "router")

# Grafı derleme
research_assistant = workflow.compile()

# Kullanım örneği
result = research_assistant.invoke({
    "query": "yapay zeka ve etik",
    "sources": [],
    "analysis": {},
    "summary": "",
    "current_agent": "researcher",
    "status": "started"
})

print(result["summary"])
```

### 10.2 Kısıtlamalar ve İyileştirme Alanları

LangGraph güçlü bir kütüphane olsa da, bazı kısıtlamaları vardır:

1. **Büyük Ölçekli Sistemler**: Çok büyük ve karmaşık sistemlerde performans sorunları olabilir.
2. **Depolama**: Uzun süreli durum depolama için ek çözümler gerekebilir.
3. **Dağıtık Sistemler**: Dağıtık sistemlere entegrasyon için ek çalışma gerekebilir.
4. **Monitörleme**: Üretim ortamlarında detaylı izleme ve loglama için ek araçlar gerekebilir.

Bu kısıtlamaları aşmak için şunları yapabilirsiniz:

```python
# Daha iyi loglama için
import logging
logging.basicConfig(level=logging.DEBUG)

# Dış durumlar için Redis entegrasyonu
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def save_state(state_id, state):
    """Durumu Redis'e kaydet"""
    redis_client.set(f"state:{state_id}", json.dumps(state))

def load_state(state_id):
    """Durumu Redis'ten yükle"""
    state_json = redis_client.get(f"state:{state_id}")
    if state_json:
        return json.loads(state_json)
    return None
```

## 11. Sonuç ve İleri Adımlar

### 11.1 LangGraph'ın Geleceği

LangGraph, yapay zeka uygulamalarının karmaşık iş akışlarını yönetmek için güçlü bir araçtır. Gelecekte daha fazla özellik ve entegrasyon görmemiz muhtemeldir.

### 11.2 İleri Adımlar

LangGraph'ı öğrendikten sonra, şu adımları izleyebilirsiniz:

1. **Kendi Uygulamalarınızı Oluşturun**: Öğrendiklerinizi kendi projelerinizde uygulayın.
2. **Toplulukla Etkileşime Geçin**: GitHub, forum ve sosyal medya üzerinden LangGraph topluluğuna katılın.
3. **Açık Kaynak Katkıları**: Kütüphaneye katkıda bulunmayı düşünün.
4. **Entegrasyonlar**: LangGraph'ı diğer kütüphaneler ve araçlarla entegre edin.

### 11.3 Kaynaklar ve Referanslar

- [LangGraph Resmi Dokümantasyonu](https://python.langchain.com/v0.2/docs/langgraph/)
- [LangChain Dokümantasyonu](https://python.langchain.com/)
- [GitHub Repository](https://github.com/langchain-ai/langgraph)
- [Örnekler ve Kullanım Senaryoları](https://python.langchain.com/v0.2/docs/langgraph/tutorials/)