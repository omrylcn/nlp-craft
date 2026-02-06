# The Informed Dispatcher Pattern: Bridging the Gap Between Router and Orchestrator

*A practical architectural pattern for LLM-based systems that offers more than routing but less than full agent orchestration.*

---

## The Missing Middle Ground

If you've built LLM-based systems, you've encountered a familiar architectural tension. On one side sits the **Router** — simple, fast, deterministic. It classifies intent and routes the request. On the other side sits the **Orchestrator** — powerful, flexible, capable of decomposing complex tasks into dynamic subtasks at runtime, coordinating multiple workers.

Most real-world systems don't fit neatly into either extreme.

A Router lacks sufficient knowledge. It can tell you *where* a request should go but not *how* it should be processed. An Orchestrator knows everything but introduces non-determinism, unpredictable latency, variable cost, and unacceptable hallucination risk in domains where accuracy is critical.

What if there were something in between? A pattern that carries the Orchestrator's domain awareness while preserving the Router's deterministic execution guarantees?

I call it the **Informed Dispatcher**.

---

## Understanding the Spectrum

Before diving into the pattern itself, it helps to understand where it stands relative to its neighbors.

### Router

A Router is a classification layer. It takes input, determines intent, and routes the request to a handler. That's it. It doesn't know what the handler does, which tools are available, or whether the task is simple or complex. It makes a single decision — *where* — and moves on.

```
User Input → [Router: "Which intent?"] → Handler A / Handler B / Handler C
```

**Strengths:** Fast, cheap, predictable. Single LLM call (or even just a classifier), fixed latency, deterministic downstream execution.

**Limitations:** Unaware of system capabilities. Cannot assess task difficulty. Every request receives the same treatment regardless of complexity. Cannot extract parameters or detect missing information.

### Orchestrator (Agent-Based)

The Orchestrator — particularly the agent pattern as defined in frameworks like LangGraph, CrewAI, or Anthropic's orchestrator-workers model — is a dynamic planner. It takes input, reasons about which subtasks are needed, spawns workers, evaluates intermediate results, and can re-plan based on observations. It uses a ReAct loop or similar reasoning framework.

```
User Input → [Orchestrator: "Which subtasks?"] → Worker 1 → Observation → Worker 2 → ... → Result
```

**Strengths:** Can handle unlimited complexity. Adapts to novel requests. Can dynamically chain tools.

**Limitations:** Non-deterministic execution paths. Variable latency (seconds to minutes). Unpredictable cost. Hallucination risk in multi-step reasoning. Extremely difficult to test, debug, and provide SLA guarantees.

### The Gap

Between these two extremes lies a significant design space occupied by systems that need:

- Understanding task difficulty before choosing an execution path
- Awareness of available tools and capabilities in the system
- Extracting structured parameters from natural language
- Efficiently handling both simple and complex requests
- Maintaining deterministic, testable execution at every level

This is where the Informed Dispatcher lives.

---

## The Informed Dispatcher Pattern

### Definition

> The **Informed Dispatcher** is a **single LLM decision point** that possesses full awareness of a system's entire capabilities — its tools, handlers, and predefined workflows — uses this context to classify both intent and difficulty, extracts required information (parameters, missing fields, conversation context), and then delegates execution to the appropriate deterministic path. In a single call, it decides *where*, *how*, and *with what parameters* a request should be processed — but never executes the plan itself.

The core insight is this: **the quality of a routing decision is directly proportional to how much the router knows about the targets it routes to.**

A Router that knows nothing about downstream handlers can only decide based on surface-level intent. An Informed Dispatcher that knows the system's full capability map — what each tool does, what each handler can process, which predefined workflows are available — can make sophisticated decisions about the *best* execution strategy for every request.

### Core Principle: A Broad-Scope Single Decision Point

The Informed Dispatcher's power comes from the breadth of the decision made at the first layer. A single LLM call takes on multiple responsibilities:

| Responsibility | What It Does | Why in a Single Call? |
|---------------|-------------|----------------------|
| **Intent Classification** | Determines what the user wants | Can't route without knowing intent |
| **Complexity Assessment** | Evaluates whether the task is easy/medium/hard | Can't select the right tier without knowing difficulty |
| **Knowledge Extraction** | Extracts parameters, filters, preferences from natural language | Handlers expect structured input |
| **Missing Field Detection** | Identifies which required information is missing | Better to ask than to execute with missing data |
| **Execution Path Selection** | Selects the right tool, handler, or workflow | This decision cannot be made without all of the above |

These responsibilities are not independent — they are intertwined. To accurately assess difficulty, you need to know what the tools do. To select the right handler, you need to have extracted the parameters. To detect missing fields, you need to know what the selected handler expects.

This is why it all happens in a single call, at a single decision point.

### Why Must the Dispatcher Know All Context?

The dispatcher cannot make a "is this easy or hard" decision without knowing the tools. Consider this example:

A user asks: *"Compare my current setup with alternatives, including early termination costs."*

A naive Router might classify this as "comparison" and send it to the comparison handler. But that handler only compares alternatives — it doesn't calculate early termination costs. The request actually requires data from a termination calculator to be fed into the comparison engine.

The Informed Dispatcher knows:
- `termination_calculator` and `comparison_engine` are separate tools
- One's output needs to be the other's input
- A predefined workflow called `comparison_with_termination` matches exactly this pattern
- The user hasn't yet provided all required parameters

This is a *routing* decision — but making it correctly requires deep system knowledge.

### The Context Layers the Dispatcher Knows

The Informed Dispatcher's system prompt contains four types of context:

| Context Layer | Contents | Why It's Needed |
|--------------|----------|-----------------|
| **Tool Registry** | What each tool does, its parameters, return schema | Assessing "Is one tool enough, or do we need multiple?" |
| **Handler Definitions** | What each handler does, which tools it combines, what parameters it expects | Assessing "Does an existing handler already solve this?" |
| **Workflow Definitions** | Predefined multi-step execution plans and tool chains | Assessing "Is there a proven workflow for this complex request?" |
| **Conversation Context** | What was previously discussed, what data was already retrieved | Assessing "Can I reuse previous results?" |

---

## The Four-Tier Execution Model

The Informed Dispatcher doesn't just route — it routes requests to the *appropriate level of machinery*. The dispatcher makes all decisions and extracts parameters in a single LLM call. After that: zero LLM in Tiers 1-2, deterministic tool chains in Tier 3 (with optional formatting LLM), and a constrained agent in Tier 4. For undefined scenarios, a tightly constrained agent layer can optionally be activated.

```
                    ┌──────────────────────────────────────┐
                    │      Informed Dispatcher (LLM)       │
                    │      ═══════════════════════         │
                    │   In a single call:                   │
                    │   • Intent classification             │
                    │   • Complexity assessment             │
                    │   • Parameter extraction              │
                    │   • Missing field detection           │
                    │   • Execution path selection          │
                    │                                      │
                    │   Parameter extraction ENDS HERE.     │
                    │   Tiers below work with structured    │
                    │   data, never parse natural language. │
                    └──────────────────┬───────────────────┘
                                       │
           ┌───────────────┬───────────┼───────────┬────────────────┐
           │               │           │           │                │
           ▼               ▼           ▼           ▼                │
  ┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐    │
  │   Tier 1     │ │  Tier 2    │ │ Tier 3   │ │  Tier 4      │    │
  │   Direct     │ │  Handler   │ │ Workflow  │ │  Constrained │    │
  │   Tool Call  │ │  Execution │ │ Execution│ │  Agent       │    │
  │              │ │            │ │          │ │  (Optional)  │    │
  │   LLM: ❌   │ │  LLM: ❌   │ │ LLM: ⚠️  │ │  LLM: ✅     │    │
  │   ~100ms    │ │  ~300ms    │ │ ~800ms   │ │  ~2-8s       │    │
  │   ~$0       │ │  ~$0.001   │ │ ~$0.003  │ │  ~$0.01-0.03 │    │
  └──────────────┘ └────────────┘ └──────────┘ └──────────────┘    │
                                                                    │
  ■■■■■■■■■■■ Deterministic, No LLM ■■■■■■■■■  ⚠️ Optional       │
                                                  LLM (format)     │
                                    ┊  ░░░░░░░░░░░░░░░░░░░░░░░░░░  │
                                    ┊  ░░░ Bounded Non-det. ░░░░░  │
                                    ┊                               │
                                    │  ❓ Missing info → Ask user ◄─┘
```

**Critical rule:** Parameter extraction ends at the dispatcher. The `params` going to handlers and workflows are fully structured, validated data. There are zero LLM calls in Tiers 1-2 — only deterministic code (regex, calculations, sorting, filtering). In Tier 3 the tool chain is deterministic, but optional lightweight LLM usage (formatting, summarization) may occur in intermediate steps.

### Tier 1 — Direct Tool Call (Simple Requests)

For requests that map to a single tool with zero or minimal parameters, the dispatcher identifies the tool and it's called directly. No additional LLM inference. No handler logic.

**Characteristics:**
- Single tool, zero or minimal parameters
- No computation, transformation, or combination needed
- The tool's raw output (a URL, a list, a status) is the answer itself

**Expected distribution:** ~35-40% of traffic.

**Cost:** Effectively $0 beyond the dispatcher call. **Latency:** ~100ms. **LLM:** None.

### Tier 2 — Handler Execution (Medium Requests)

For requests requiring filtering, sorting, computation, or combining 1-2 tools with known logic, the dispatcher selects a handler and provides the extracted parameters. The handler executes deterministically — **there are no LLM calls**, only code.

**Characteristics:**
- Works with structured parameters from the dispatcher (never parses natural language)
- Contains business logic (sorting, filtering, computation, regex, formatting)
- 1-2 tool calls, deterministic combination logic
- Each handler is an independent, unit-testable module

**Expected distribution:** ~45-50% of traffic.

**Cost:** ~$0.001 per request. **Latency:** ~300ms. **LLM:** None.

### Tier 3 — Workflow Execution (Complex Requests)

This is where the pattern's power becomes most evident. Complex requests that would traditionally require an agent's multi-step reasoning are handled by **predefined, deterministic workflows** — what I call *compiled agent behavior*.

A workflow is essentially the answer to: *"If an agent flawlessly processed this request, which tool chain would it execute?"* You define that chain in code, explicitly specify the data flow between steps. The dispatcher's job is simply to select the right workflow and provide the initial parameters.

**Characteristics:**
- 2-4 tool calls, in a defined order — **the tool chain is always the same**
- Step N's output feeds into step N+1 (explicit data flow)
- Fixed iteration count (no loops, no re-planning)
- The entire chain is a single, testable unit
- **Optional:** Lightweight LLM usage may occur in intermediate steps (formatting, summarization) — but these steps are also predefined, not runtime decisions

**Expected distribution:** ~10% of traffic.

**Cost:** ~$0.003 per request (without LLM steps) / ~$0.005 (with LLM steps). **Latency:** ~800ms (deterministic, predictable).

### Tier 4 — Constrained Agent (Optional Escape Hatch)

This is the pattern's most controversial yet most pragmatic tier. The first three tiers are entirely deterministic — predefined paths, fixed tool chains, testable outputs. But in the real world, undefined scenarios are inevitable. A user may make a request that falls outside the scope of any handler or workflow. Tier 4 is a **tightly constrained, policy-managed free agent** layer for these situations.

The important thing is this: this tier is **disabled by default**. Enabling it is a conscious policy decision by the system administrator.

**Primary Purpose: Controlled Escape Hatch**

Tier 4's reason for existence is simple: **instead of responding "sorry, I can't do that" to requests outside Tiers 1-3's scope, provide a controlled attempt.** Tier 3's boundary is the scope of predefined workflows. When a request matches no workflow, the medium fallback kicks in — but this usually produces an incomplete answer. Tier 4 is a safety valve for "undefined but legitimate" requests.

Tier 4 is permanent. While its traffic share decreases over time, it never reaches zero — because even in a bounded domain, there will always be unforeseen requests. This is not a transitional tier but a **permanent safety valve.**

It doesn't take on all of an agent's risks because it operates under strict constraints:

**Constraints (Guardrails):**

| Constraint | Value | Reason |
|-----------|-------|--------|
| **Max iteration** | 3-5 | Eliminates infinite loop risk |
| **Tool whitelist** | Only existing MCP tools | Prevents agent from calling undefined tools |
| **Hard timeout** | 8-10 seconds | Guarantees SLA |
| **Token budget** | Max ~2K output tokens | Prevents cost explosion |
| **Result verification** | Tool output vs. agent response comparison | Hallucination detection |
| **Full logging** | Every thought-action-observation recorded | Audit trail + workflow discovery |

**Policy Management:**

Tier 4's most critical feature is that it can be **toggled on and off**. This is not a code change but a configuration decision:

```python
class AgentPolicy:
    enabled: bool = False              # Default: disabled
    allowed_tools: list[str] = []       # Empty = no tools available
    max_iterations: int = 3
    timeout_seconds: float = 8.0
    max_output_tokens: int = 2000
    require_approval: bool = False      # True = requires human approval
    allowed_hours: tuple = (9, 18)      # Business hours only
    daily_budget: float = 5.0           # Max daily spend ($)
    log_level: str = "verbose"          # Every step is logged
```

This policy can vary by environment:

| Environment | Policy |
|------------|--------|
| **Development** | `enabled=True, max_iterations=5` — Full exploration mode |
| **Staging** | `enabled=True, require_approval=True` — Testing with approval |
| **Production** | `enabled=False` — Disabled (initial phase) |
| **Production (mature)** | `enabled=True, max_iterations=3, daily_budget=5.0` — Limited open |

**Expected distribution:** ~1-3% of traffic (only requests Tier 3 cannot handle).

**Cost:** ~$0.01-0.03 per request. **Latency:** ~2-8s (variable but bounded).

**Secondary Benefit: Workflow Discovery Potential**

Tier 4's primary purpose is to serve as a controlled escape hatch. However, since every execution is fully logged, it can contribute to workflow discovery as a secondary benefit.

That said, it's important to understand the limits of this benefit: **automatic pattern discovery at 1-3% traffic volume is statistically inefficient.** With 10,000 daily requests, the 100-300 hitting Tier 4 may not generate sufficient data for meaningful pattern detection. Therefore, workflow discovery is not a justification for enabling Tier 4 — it's a nice side effect.

If you want to increase discovery value, you can use **shadow mode**: run Tier 4 at higher traffic (10-20%) but only log, don't generate responses for users. This collects discovery data without affecting the live service:

```python
class AgentPolicy:
    mode: Literal["off", "shadow", "active"] = "off"
    # off:    Completely disabled
    # shadow: Run at higher traffic, log only, don't produce responses (for discovery)
    # active: Normal usage, respond to undefined requests
```

Logs collected in shadow mode are evaluated through weekly manual review. If recurring tool chains are identified, they can be converted to Tier 3 workflows with engineer approval. But this process is not automatic — **it requires human judgment**, because the chain an agent discovers may not always be optimal or correct.

### Comparing the Four Tiers

| Dimension | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|-----------|--------|--------|--------|--------|
| Execution | Direct tool call | Handler (deterministic) | Workflow (deterministic chain) | Agent (bounded non-det.) |
| Tool count | 1 | 1-2 | 2-4 | N (within whitelist) |
| LLM calls (excluding dispatcher) | ❌ 0 | ❌ 0 | ⚠️ 0-1 (optional formatting) | ✅ 1-5 (depends on iterations) |
| Determinism | ✅ Full | ✅ Full | ✅ Chain fixed, optional LLM step | ⚠️ Bounded |
| Testability | Easy | Easy | Easy | Difficult (but logged) |
| Hallucination risk | None | None | None (minimal in optional LLM step) | Low (guardrails) |
| Policy required? | No | No | No | **Yes** |
| Traffic share | ~35-40% | ~45-50% | ~10% | ~1-5% |

### Workflow vs. Agent vs. Constrained Agent: The Critical Distinction

This is the heart of the pattern. A comparison of three execution strategies:

| Dimension | Free Agent | Predefined Workflow | Constrained Agent (Tier 4) |
|-----------|-----------|--------------------|-----------------------------|
| Tool ordering | Decided at runtime (non-deterministic) | Defined in code (deterministic) | Decided at runtime but within whitelist |
| Parameter flow | Agent extracts from context (error risk) | Explicitly coded (step_1.output → step_2.input) | Agent extracts but in limited steps |
| Maximum iterations | Unlimited | Fixed (number of steps in workflow) | Fixed upper bound (3-5) |
| Hallucination risk | High | None — tool outputs used directly | Low — guardrails + verification |
| Testability | Extremely difficult | Standard unit testing | Difficult but auditable via full logging |
| Adding new capability | Update prompt and hope | Write a new workflow class | Automatic — uses it if on tool whitelist |
| Cost predictability | Variable (agent can loop) | Fixed (known step count) | Bounded (max iteration + timeout + budget) |
| Management | Prompt engineering | Code | **Policy (toggle on/off)** |

Think of it this way: **A workflow is a pre-compiled agent trace.** You observe what a perfect agent *would do* for a class of requests, then codify that behavior as a deterministic pipeline. The agent's intelligence is captured at design time, not at runtime.

**The Constrained Agent is a permanent safety valve for scenarios not yet compiled.** It provides a controlled escape hatch for requests that the deterministic tiers cannot handle. Every execution is logged, and over time these logs can yield new workflow candidates — but this is a secondary benefit, not the primary purpose.

---

## Dispatcher Output Format

The Informed Dispatcher produces a single structured JSON output in a single LLM call. The `complexity` field determines which execution tier handles the request:

**Tier 1 — Direct Tool Call:**
```json
{
  "complexity": "easy",
  "tool": "status_checker",
  "params": {},
  "handler": null,
  "workflow": null
}
```

**Tier 2 — Handler:**
```json
{
  "complexity": "medium",
  "tool": null,
  "params": { "amount": 50000, "max_monthly": 2000, "term_preference": "shortest" },
  "handler": "BUDGET_CALCULATOR",
  "workflow": null
}
```

**Tier 3 — Workflow:**
```json
{
  "complexity": "hard",
  "tool": null,
  "params": { "current_rate": 3.5, "target_amount": 100000, "early_exit_month": 6 },
  "handler": null,
  "workflow": "comparison_with_early_termination",
  "agent": null
}
```

**Tier 4 — Constrained Agent (if policy permits):**
```json
{
  "complexity": "unknown",
  "tool": null,
  "params": { "user_intent": "compare with tax implications and insurance cost" },
  "handler": null,
  "workflow": null,
  "agent": {
    "goal": "Handle user's multi-domain request",
    "suggested_tools": ["loan_rates", "tax_calculator", "insurance_rates"],
    "max_iterations": 3
  }
}
```

Note: the complexity field is `"unknown"` — the dispatcher explicitly states that this request doesn't fall within the scope of existing handlers/workflows. If the agent policy is disabled, the dispatcher falls back this request to the medium fallback and indicates it cannot provide a comprehensive answer.

**Missing Information — Ask Before Executing:**
```json
{
  "complexity": "medium",
  "handler": "BUDGET_CALCULATOR",
  "params": { "max_monthly": 2000 },
  "missing_required": ["amount"],
  "question": "What should the total amount be?"
}
```

This last case is important: the dispatcher can detect missing parameters *because it knows what every handler and workflow expects*. Instead of guessing or hallucinating values, it explicitly asks.

### Runtime Parameter Validation

The dispatcher produces structured JSON, handlers expect typed params. But there's no guarantee that the LLM's extracted parameters will exactly match what the handler actually expects — especially when the prompt is updated but the handler code isn't (or vice versa). Therefore, adding a **lightweight validation layer** between dispatcher output and handler execution is recommended:

```python
def execute_with_validation(dispatcher_output: DispatcherResult) -> Result:
    handler = registry.get(dispatcher_output.handler)
    
    # Do the params extracted by the dispatcher match what the handler expects?
    validation = handler.validate_params(dispatcher_output.params)
    
    if validation.is_valid:
        return handler.execute(validation.validated_params)
    
    if validation.has_missing_required:
        # Missing required field — ask the user
        return ask_user(validation.missing_fields)
    
    # Unexpected parameter error — fallback
    log_misrouting(dispatcher_output, validation.errors)
    return default_handler.execute(dispatcher_output.params)
```

This validation is a cheap safety net against LLM errors. The additional latency impact is negligible (<5ms) but it catches prompt-code synchronization errors before they reach production. Even with an auto-generated registry, runtime validation provides defense in depth.

---

## Fallback Strategy

No routing system is perfect. The Informed Dispatcher uses a conservative fallback strategy — when uncertain, escalate to a higher tier rather than fail:

```
Easy (uncertain) → Fall back to Medium
Medium (uncertain) → Fall back to default Medium handler
Hard (no matching workflow) + Agent policy ON → Tier 4 (constrained agent)
Hard (no matching workflow) + Agent policy OFF → Medium fallback
Tier 4 failure / timeout → Medium fallback + log
Dispatcher error → Fall back to default handler
```

Tier 4's position in the fallback chain depends on policy. If the policy is disabled, the system remains fully deterministic — Tier 4 never activates. If the policy is enabled, unmatched hard queries are routed to the agent, but if the agent also fails, it falls back to medium.

### Circuit Breaker: Preventing Latency Stacking

The fallback chain's biggest risk is **latency stacking**. Consider the worst case: Dispatcher (~500ms) → Tier 4 timeout (8s) → Medium fallback (~300ms) = **~9 seconds** — and the user ends up with only a medium-level answer.

A **circuit breaker** mechanism is mandatory to eliminate this risk:

```python
class AgentCircuitBreaker:
    # Fast cut: If agent hasn't made first tool call in 3s, cut it
    first_action_timeout: float = 3.0
    
    # Total timeout: The policy's hard timeout (8s) already exists,
    # but the circuit breaker intervenes earlier
    
    # Automatic disable:
    # If success rate drops below 50% in the last hour,
    # automatically disable Tier 4, send alert
    auto_disable_threshold: float = 0.50
    auto_disable_window_minutes: int = 60
    
    # Gradual recovery:
    # Wait 10 minutes after disable, then try with 10% traffic
    recovery_delay_minutes: int = 10
    recovery_traffic_percent: float = 0.10
```

**Practical rules:**

| Scenario | Action | Latency Impact |
|----------|--------|----------------|
| Agent has no first action within 3s | Cut immediately → Medium fallback | Max 3.5s (3s + fallback) |
| Agent running but 8s timeout | Cut → Medium fallback | Max 8.5s (timeout + fallback) |
| Success rate <50% in last hour | Tier 4 automatically disabled | 0s (falls directly to medium) |
| Agent succeeds | Normal flow | 2-5s |

Do not enable Tier 4 without a circuit breaker. This is not a safety mechanism — it's a **prerequisite.**

Principle: **an over-processed request is always better than an incorrect one.** If a request classified as "easy" actually requires handler logic, the medium tier catches it gracefully. The worst case is unnecessary computation, never a wrong result.

---

## When to Use It

### Strong Fit

The Informed Dispatcher shines in these situations:

**Limited tool count.** You have a known set of 5-20 tools, not an unlimited plugin ecosystem.

**Discoverable request patterns.** You can identify the most common complex request types through log analysis and create workflows for them in advance.

**Accuracy is non-negotiable.** Finance, healthcare, legal — domains where hallucinated intermediate reasoning is unacceptable.

**SLA matters.** You need predictable latency and cost, not "usually fast but sometimes 30 seconds."

**Testability is required.** You need to write unit tests, run regression suites, and reliably reproduce errors.

**Gradual adoption is preferred.** You want to start simple and add complexity only when data proves it necessary.

### Weak Fit

A full orchestrator/agent should be considered when:

**Tasks are fundamentally unpredictable.** Coding agents that modify an unknown number of files, research agents that follow citation chains — these require dynamic planning.

**Tool combinations are unlimited.** If you can't enumerate common workflows, you can't pre-compile them.

**Latency and cost tolerance is high.** Internal tools where 10-second response times are acceptable.

**Approximate answers are sufficient.** Creative tasks, brainstorming, exploratory analysis.

---

## Implementation Guide

### Phase 1: Foundation (Week 1-2)

Start with only Tier 1 and a single Tier 2 handler.

1. **Create the dispatcher prompt** with tool registry and one handler definition
2. **Implement JSON output parser** with validation
3. **Wire up Tier 1** (direct tool call for simple requests)
4. **Migrate your most common handler** to the new architecture
5. **Add monitoring:** log every dispatcher decision, track accuracy

### Phase 2: Handler Expansion (Week 3-4)

Add remaining Tier 2 handlers one by one.

1. **Identify handler needs** from request log analysis
2. **Develop each handler** as an independent, unit-tested module
3. **Update the dispatcher prompt** with new handler definitions
4. **A/B test** against the previous architecture

### Phase 3: Workflows (Week 5-6)

Only add workflows that your data proves necessary.

1. **Analyze logs** for multi-step request patterns that Tier 2 can't handle
2. **Design workflow schemas** (tool chain, data flow, parameters)
3. **Implement the workflow executor** — a simple sequential runner
4. **Add 2-3 workflows** for the most common complex patterns
5. **Monitor and iterate**

### Phase 4: Constrained Agent (Week 7-8, Optional)

Only after Tiers 1-3 have collected sufficient data and unmet request patterns are clear:

1. **Implement the agent policy framework** (enabled flag, tool whitelist, budget, timeout)
2. **Enable Tier 4 in development**, review logs
3. **Test guardrails** — max iteration, timeout, budget limits
4. **Test in staging** with `require_approval=true`
5. **Enable in production** with low limits (`daily_budget=5.0, max_iterations=3`)
6. **Set up the workflow discovery pipeline** — recurring pattern detection from agent logs

### Phase 5: Optimization (Ongoing)

- Add new workflows driven by log analysis (never by speculation)
- Identify workflow candidates from Tier 4 logs and promote to Tier 3
- Tune the dispatcher prompt based on misclassification data
- Optimize handler/workflow performance
- Consider caching for recurring dispatcher decisions
- Monitor Tier 4 traffic share — decreasing means the system is maturing, increasing means it's approaching its limits

Core philosophy: **data-driven, gradual adoption.** Never create a workflow without seeing the pattern in production logs. Never add complexity without proof that simplicity is insufficient.

---

## Scaling the Pattern

The most common criticism of the Informed Dispatcher is: *"You say the dispatcher must know everything, but what happens when you have 50 tools and 30 workflows? What about the system prompt?"*

This is a fair criticism — and the answer is simple: **the pattern itself scales in tiers.**

### Scale Stages

```
5-20 tools          20-50 tools              50+ tools
─────────────      ──────────────          ──────────────
Single Dispatcher  Domain Dispatchers      Hierarchical Tree
                                           
┌──────────┐      ┌──────────────┐        ┌─────────────┐
│Dispatcher│      │Meta-Dispatcher│        │Meta-Router   │
│          │      └──┬───┬───┬──┘        └──┬───┬───┬──┘
│ 6 tools  │         │   │   │              │   │   │
│ 4 handler│         ▼   ▼   ▼              ▼   ▼   ▼
│ 3 wflow  │        D1  D2  D3           Domain  Domain  Domain
└──────────┘      Credit Ins. Inv.       Cluster Cluster Cluster
                                          │       │       │
                                          ▼       ▼       ▼
                                         D1..Dn  D1..Dn  D1..Dn
```

**Stage 1 — Single Dispatcher (5-20 tools):** The pattern's base form. Tool registry, handler definitions, and workflow definitions fit in a single system prompt (~1-2K tokens). Most bounded domain systems stay at this stage and should.

**Stage 2 — Domain Dispatchers (20-50 tools):** When the system spans multiple domains, a lightweight **Meta-Dispatcher** is added at the top level. This dispatcher only selects the domain (credit, insurance, investment) — each domain has its own Informed Dispatcher. The Meta-Dispatcher's prompt stays small because it only contains domain descriptions, not tool details.

**Stage 3 — Hierarchical Tree (50+ tools):** At this point you're approaching the pattern's limits. Sub-dispatchers operate under domain clusters. But to be honest, a system that's reached 50+ tools probably needs agent behavior in some of its layers.

### Prompt Management: Auto-Generated Registry

The most practical solution for scaling is to **auto-generate the dispatcher prompt from code**. You add metadata to each handler and workflow class, and the prompt is automatically generated at build time:

```python
class BudgetHandler(BaseHandler):
    """Calculates suitable options based on monthly budget."""
    
    complexity = "medium"
    required_params = ["amount", "max_monthly_payment"]
    optional_params = ["term_preference"]
    tools_used = ["loan_rates"]

# At build time, prompt is auto-generated from all handlers
registry = ToolRegistry.from_handlers([BudgetHandler, RateFilterHandler, ...])
dispatcher_prompt = registry.to_prompt()  # Compact format, ~1-2K tokens
```

This approach has two critical benefits. First, code and prompt always stay in sync — when you add a new handler, the prompt updates automatically. Second, the prompt format can be optimized — compact schemas instead of verbose descriptions.

### Token Budget Management

Concrete targets for the dispatcher prompt:

| Component | Target Tokens | Strategy |
|-----------|--------------|----------|
| Tool Registry | ~300-500 | Compact format: name, one-line description, parameter list |
| Handler Definitions | ~300-500 | Name, tools used, expected parameters |
| Workflow Definitions | ~200-400 | Name, step list, when to use |
| Routing Rules | ~200-300 | Priority order, fallback rules |
| **Total** | **~1-2K** | ~1-2% of context window |

If the total exceeds 3K tokens, this is usually a sign that definitions are too verbose or the domain needs to be split.

---

## When to Graduate to an Agent

The Informed Dispatcher is not an endpoint — it's a **starting point**. Every pattern has boundaries of validity, and knowing these boundaries is as important as knowing the pattern itself.

### Red Flags

Know you're approaching the pattern's limits when you see these signals:

**1. Workflow explosion.** When workflow count exceeds 15-20 and you're still discovering new patterns, your domain is more dynamic than you thought. At this point, writing a workflow for every new edge case becomes a maintenance nightmare.

**2. High fallback rate.** When monitoring dispatcher decisions, if more than 15% of requests fall through to fallback, this indicates the dispatcher can't handle requests with the current handler/workflow scope.

**3. Cross-domain chains.** If users are increasingly making requests that combine multiple domains — e.g., "calculate loan + show tax impact + add insurance cost" — predefined workflows can't accommodate these combinations.

**4. Shortening workflow lifespans.** If the workflows you create need to be updated or changed within a few weeks, the rate of change in your domain is incompatible with the pre-compilation approach.

### Concrete Threshold Values

Experience-based transition criteria:

| Metric | Safe Zone | Warning Zone | Consider Transitioning |
|--------|-----------|-------------|----------------------|
| Workflow count | ≤10 | 10-20 | >20 |
| Fallback rate | ≤5% | 5-15% | >15% |
| Misrouting rate | ≤3% | 3-10% | >10% |
| Edge case rate | ≤5% | 5-10% | >10% |
| Average workflow lifespan | >3 months | 1-3 months | <1 month |

These numbers are not strict rules but directional indicators. What matters is continuously monitoring your production metrics and analyzing trends.

### The Transition Doesn't Have to Be Complete

The transition to an agent is not an "all or nothing" decision. Tier 4 already embeds the first step of this transition within the pattern. The pattern's natural evolution is gradual hybridization:

**Step 1 — Enable Tier 4 (Policy: Development):** Activate Tier 4 in the development environment. Collect agent logs. Observe which tool chains recur.

**Step 2 — Move Tier 4 to Production (Policy: Limited):** First, deploy the circuit breaker (see Fallback Strategy). Then enable in production with `max_iterations=3, daily_budget=5.0, require_approval=false`. Only 1-3% of traffic will hit this tier.

**Step 3 — Log Analysis and Manual Workflow Discovery:** Review Tier 4 logs weekly. If you identify recurring tool chain patterns, convert them to Tier 3 workflows with engineer approval. This process is not automatic — it requires human judgment. Meaningful pattern detection at 1-3% traffic volume takes time; be patient.

**Step 4 — Threshold Exceeded:** If Tier 4 traffic exceeds 5% and the variety that can't be converted to workflows is increasing, this signals you're reaching the pattern's limits. At this point, two options exist:

- **Domain splitting:** Divide the system into multiple Informed Dispatchers (each with its own bounded domain)
- **Hybrid architecture:** Tiers 1-3 remain as the Informed Dispatcher (80-90% of traffic), Tier 4 operates with broader authority. But this is no longer the Informed Dispatcher pattern — it's a hybrid architecture — and you should make this transition consciously.

This gradual transition preserves the pattern's strongest advantage: **the vast majority of traffic always remains deterministic.**

---

## Monitoring and Evolution

The Informed Dispatcher's success is directly proportional to the quality of the dispatcher's decisions. Improving this quality without measuring it is impossible.

### Core Metrics

Log the following data for every dispatcher decision:

**1. Predicted vs Actual Complexity:**
```
Log entry: {
  "query": "...",
  "predicted_complexity": "easy",
  "predicted_handler": "credit_score_url",
  "actual_execution": "easy",        // What actually happened?
  "fallback_triggered": false,        // Did it fall back?
  "execution_success": true,          // Was the result successful?
  "latency_ms": 145,
  "user_satisfaction": null            // Optional: thumbs up/down
}
```

**2. Misrouting Detection:** Dispatcher said "easy" but the handler errored → misclassification. Dispatcher said "medium" but the user received incomplete information → insufficient tier selection. Automatically detect these cases and generate weekly reports.

**3. Workflow Coverage:** What percentage of incoming requests are handled by existing handlers/workflows? If this rate is declining, new handlers/workflows should be added — or you're approaching pattern limits.

### Dashboard Metrics

| Metric | Target | Warning Threshold | Action |
|--------|--------|-------------------|--------|
| Correct tier selection rate | >95% | <90% | Dispatcher prompt review |
| Fallback rate | <5% | >10% | Add new handler/workflow |
| Workflow coverage | >90% | <80% | Log analysis → new workflow |
| Average latency (easy) | <200ms | >500ms | Tool performance check |
| Average latency (medium) | <600ms | >1s | Handler optimization |
| Average latency (hard) | <1.5s | >3s | Workflow step analysis |
| Tier 4 traffic share | <3% | >5% | Agent log → workflow conversion |
| Tier 4 success rate | >80% | <60% | Guardrail/prompt review |
| Tier 4 daily spend | <$5 | >$10 | Tighten policy budget |
| Misrouting rate | <3% | >5% | Update routing rules |

### Evolution Cycle

The pattern is not static — it operates in a continuously evolving cycle:

```
    ┌─────────────────────────────────────────────┐
    │                                             │
    ▼                                             │
Monitor (Log)  →  Analyze  →  Decide  →  Implement
    │                │            │            │
    │           Misrouting    New handler    Update
    │           patterns      needed?       prompt
    │           Fallback      New workflow   or add
    │           distribution  needed?       handler/
    │           Coverage      Transition    workflow
    │           gaps          to agent?
    │                                         │
    └─────────────────────────────────────────┘
```

**Weekly:** Check misrouting and fallback rates. Sudden spikes usually indicate a new user pattern emerging.

**Monthly:** Conduct workflow coverage analysis. Identify the top 5 most frequent request types not covered by existing workflows. If their frequency justifies adding a handler/workflow, add it; if not, let the fallback continue.

**Quarterly:** Assess overall pattern health. Review threshold values. Check for agent transition signals. Measure the dispatcher prompt's token budget.

### Minimizing Maintenance Burden

One fair criticism of the pattern is: "every time you add a new tool, you have to update both the code and the prompt." This is a real trade-off — but it should be viewed in proper perspective.

In an agent-based system, when you add a new tool, you also write a tool description and *hope* the agent uses it correctly. In the Informed Dispatcher, you write a tool description + handler/workflow and *guarantee* how it will be used. The maintenance cost is similar; the assurance level is vastly different.

That said, concrete steps can be taken to reduce maintenance burden:

**Auto-generated registry:** As discussed earlier, auto-generate the dispatcher prompt from handler/workflow metadata. Synchronization errors between code and prompt drop to zero.

**Integration test suite:** Write test cases for each handler and workflow in the form of "the dispatcher should route this query to me." When a new component is added, it's automatically checked whether existing tests break.

**Prompt diff monitoring:** Version-control every change to the dispatcher prompt. Run a regression suite after changes — "how many of these 100 test queries are now routing differently?"

---

## Cost and Performance Comparison

For a system processing 10,000 daily requests with typical distribution (35% easy, 55% medium, 8% hard/workflow, 2% unknown/agent):

| Metric | Simple Router | Informed Dispatcher | Full Agent |
|--------|-------------|--------------------:|------------|
| Easy request cost | ~$0.001 (unnecessary LLM) | **~$0** (LLM bypass) | ~$0.005 |
| Medium request cost | ~$0.001 | ~$0.001 | ~$0.010 |
| Hard request cost | ❌ Not supported | ~$0.005-0.010 | ~$0.015+ |
| Unknown request cost | ❌ Not supported | ~$0.01-0.03 (Tier 4) | ~$0.015+ |
| Monthly total cost | ~$300 | **~$420** | $1,500-4,500 |
| Easy latency | ~500ms | **~100-200ms** | ~800ms |
| Medium latency | ~500ms | ~500ms | 1-3s |
| Hard latency | ❌ | ~800ms-1.5s (fixed) | 3-5s+ (variable) |
| Unknown latency | ❌ | ~2-8s (bounded) | 3-5s+ (variable) |
| Cost explosion risk | None | **None** (policy budget) | High (agent loops) |
| Hallucination risk | Low | **Low** | High |
| Testability | Easy | **Easy** (Tiers 1-3) / Audit (Tier 4) | Very difficult |
| SLA guarantee | High | **High** (Tiers 1-3) / Bounded (Tier 4) | Low |

The Informed Dispatcher costs roughly 40% more than a simple router but handles both complex and undefined requests that the router cannot. It's 70-90% cheaper than a full agent. Tier 4's added cost is minimal to the total budget since it handles only 1-3% of traffic — and in exchange for this small cost, you gain the capacity to provide controlled responses to requests that would otherwise receive "I can't do that."

---

## Design Principles

After building systems across this spectrum, I've distilled the Informed Dispatcher into several core principles:

**1. The dispatcher decides, never executes.** A single LLM call for classification, complexity assessment, and parameter extraction. Everything after that is deterministic code — or, if policy permits, a constrained agent.

**2. The first layer's scope is broad.** The dispatcher doesn't just say "where." In a single call, it determines intent, complexity, parameters, missing fields, and execution path. This broad scope determines the quality of all downstream decisions.

**3. System knowledge improves routing quality.** A dispatcher that knows what its tools, handlers, and workflows can do makes fundamentally better decisions than a blind router.

**4. Compile agent behavior into workflows.** Don't let an LLM figure out tool chains at runtime. Observe optimal chains, define them in code, and let the dispatcher choose among them.

**5. When uncertain, escalate — don't guess.** Falling back to a higher tier wastes computation but never produces wrong results. Falling back to a lower tier risks incomplete answers.

**6. Add complexity only when data demands it.** Start with Tiers 1 + 2. Add workflows only for patterns you've observed in production. Speculative complexity is the enemy of sustainable systems.

**7. Each tier is independently testable.** The dispatcher is tested with input/output pairs. Each handler is unit tested. Each workflow is integration tested. No tier depends on another for correctness.

**8. Don't unleash the agent — manage it with policy.** Tier 4 is disabled by default. Enabling it is a conscious decision. Guardrails are in code, not in the agent's conscience. Budget, timeout, and iteration limits are configuration, not prompt instructions.

**9. The agent tier is a permanent safety valve — but should never carry most of the traffic.** Tier 4's traffic share decreases over time as patterns discovered from logs can be promoted to Tier 3. But it never reaches zero — even in a bounded domain, there will always be unforeseen requests. Tier 4's health is measured by its traffic share staying low (target 1-5%).

**10. Don't trust LLM output — verify it.** The dispatcher is an LLM and can make mistakes. Runtime parameter validation (between dispatcher → handler) and circuit breaker (in the Tier 4 fallback chain) provide defense in depth. These checks are cheap (<5ms) but catch prompt-code synchronization errors and latency explosions before they reach production.

---

## Conclusion

The Informed Dispatcher pattern fills a real architectural gap. It delivers the Orchestrator's domain awareness and complexity management while preserving the Router's determinism, testability, and cost predictability.

Its four-tier structure offers the right trade-off at every level: zero cost for simple requests, deterministic handlers for medium requests, compiled workflows for complex requests, and a policy-managed constrained agent for undefined requests. The first three tiers are entirely deterministic. The fourth tier is optional — a permanent safety valve for undefined but legitimate requests. While its traffic share decreases over time, it never reaches zero; unforeseen scenarios are inevitable in every production system.

It's not the right choice for every system. If your tasks are truly unbounded — if you genuinely cannot predict which tool chains will be needed — you need a full agent or orchestrator. And the pattern has limits: when workflow count exceeds 20, when fallback rate exceeds 15%, or when cross-domain chains become dominant, gradual hybridization should be considered. But knowing these limits is not the pattern's weakness — it's a sign of its maturity.

In my experience, most production systems operate in bounded domains with discoverable patterns. For these systems, the Informed Dispatcher offers a better balance than either extreme. What matters is starting right, measuring continuously, and letting the data tell you when you need more.

**A router that knows too much. An orchestrator that refuses to improvise — but allows limited improvisation through policy when needed. That's the Informed Dispatcher.**

---

*If you find yourself oscillating between "too simple" and "too complex" while building LLM-based systems, consider whether your domain fits this pattern. Start with the simplest version — a single-handler dispatcher — and wait for your data to tell you when you need more.*
