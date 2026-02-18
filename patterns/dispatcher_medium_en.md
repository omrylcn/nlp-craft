# Informed Dispatcher Pattern: Bridging the Gap Between Router and Orchestrator

*A practical architectural pattern for LLM-based systems that offers more than routing but less than agent orchestration.*

---

## The Missing Middle Ground

If you've built LLM-based systems, you've encountered a familiar architectural tension. On one side, there's the **Router** — simple, fast, deterministic. It classifies intent and routes the request. On the other side, there's the **Orchestrator** — powerful, flexible, capable of dynamically decomposing complex tasks into subtasks at runtime and coordinating multiple workers.

Most real-world systems don't neatly fit either extreme.

The Router doesn't know enough. It can tell *where* a request should go but not *how* it should be processed. The Orchestrator knows everything but brings non-determinism, unpredictable latency, variable cost, and unacceptable hallucination risk in domains where accuracy is critical.

What if there were something in between? A pattern that carries the Orchestrator's domain awareness while preserving the Router's deterministic execution guarantees?

I call it the **Informed Dispatcher**.

---

## Understanding the Spectrum

Before diving into the pattern itself, it's helpful to understand where it sits relative to its neighbors.

### Router

A Router is a classification layer. It takes input, determines intent, and routes the request to a handler. That's it. It doesn't know what the handler does, which tools are available, or whether the task is simple or complex. It makes a single decision — *where* — and moves on.

```bash
User Input → [Router: "Which intent?"] → Handler A / Handler B / Handler C
```

**Strengths:** Fast, cheap, predictable. A single LLM call (even a classifier suffices), fixed latency, deterministic downstream execution.

**Limitations:** Unaware of system capabilities. Cannot assess task difficulty. Every request receives the same treatment regardless of complexity. Cannot extract parameters or detect missing information.

### Orchestrator (Agent-Based)

The Orchestrator — particularly the agent pattern defined in frameworks like LangGraph, CrewAI, or Anthropic's orchestrator-workers model — is a dynamic planner. It takes input, reasons about which subtasks are needed, spawns workers, evaluates intermediate results, and can re-plan based on observations. It uses a ReAct loop or similar reasoning framework.

```bash
User Input → [Orchestrator: "Which subtasks?"] → Worker 1 → Observation → Worker 2 → ... → Result
```

**Strengths:** Can handle unlimited complexity. Adapts to novel requests. Can dynamically chain tools.

**Limitations:** Non-deterministic execution paths. Variable latency (seconds to minutes). Unpredictable cost. Hallucination risk in multi-step reasoning. Extremely difficult to test, debug, and guarantee SLAs.

<!-- IMAGE 1: diagram1_spectrum.png — Architecture Spectrum: Router → Informed Dispatcher → Agent/Orchestrator -->
![Architecture Spectrum: Router → Informed Dispatcher → Agent/Orchestrator](./diagram1_spectrum.png)

### The Gap

Between these two extremes lies a significant design space occupied by systems that need:

- Understanding task difficulty before selecting an execution path
- Knowing what tools and capabilities exist in the system
- Extracting structured parameters from natural language
- Efficiently handling both simple and complex requests
- Maintaining deterministic, testable execution at every level

This is where the Informed Dispatcher lives.

---

## The Informed Dispatcher Pattern

### Definition

> The **Informed Dispatcher** is a **single LLM decision point** that has full awareness of a system's entire capabilities — its tools, handlers, and predefined workflows — uses this context to classify both intent and difficulty, extracts the necessary information (parameters, missing fields, conversation context), and then delegates execution to the appropriate deterministic path. In a single call, it decides *where*, *how*, and *with what parameters* a request should be processed — but never executes the plan itself.

The core insight is this: **the quality of a routing decision is directly proportional to how much the router knows about the targets it routes to.**

A Router that knows nothing about downstream handlers can only decide based on surface-level intent. An Informed Dispatcher that knows the system's full capability map — what each tool does, what each handler can process, which predefined workflows are available — can make sophisticated decisions about the *best* execution strategy for every request.

### Core Principle: Broad-Scoped Single Decision Point

The power of the Informed Dispatcher comes from the breadth of the first-layer decision. In a single LLM call, it takes on multiple responsibilities:

**Intent Classification** — Determines what the user wants. You can't route without knowing the intent.

**Complexity Assessment** — Evaluates whether the task is easy/medium/hard. You can't select the right tier without knowing the difficulty.

**Knowledge Extraction** — Extracts parameters, filters, and preferences from natural language. Handlers expect structured input.

**Missing Field Detection** — Identifies which required fields are missing. It's better to ask than to execute with incomplete information.

**Execution Path Selection** — Chooses the right tool, handler, or workflow. This decision cannot be made without all of the above.

These responsibilities are not independent — they're interleaved. To accurately assess difficulty, you need to know what the tools do. To select the right handler, you need to have extracted the parameters. To detect missing fields, you need to know what the selected handler expects.

That's why it all happens in a single call, at a single decision point.

### Why Must the Dispatcher Know All Context?

The Dispatcher cannot make the "is this easy or hard" decision without knowing the tools. Consider this example:

A user asks: *"Compare my current setup with alternatives, including early termination costs."*

A naive Router might classify this as "comparison" and send it to the comparison handler. But that handler only compares alternatives — it doesn't calculate early termination costs. The request actually requires data from a termination calculator to be fed into the comparison engine.

The Informed Dispatcher knows:
- `termination_calculator` and `comparison_engine` are separate tools
- The output of one needs to be the input of the other
- A predefined workflow called `comparison_with_termination` matches exactly this pattern
- The user hasn't provided all required parameters yet

This is a *routing* decision — but making it correctly requires deep system knowledge.

### Context Layers Known to the Dispatcher

The Informed Dispatcher's system prompt contains four types of context:

**Tool Registry** — What each tool does, its parameters, return schema. Enables the assessment: *"Is a single tool enough, or do I need multiple?"*

**Handler Definitions** — What each handler does, which tools it combines, which parameters it expects. Enables the assessment: *"Does an existing handler already solve this?"*

**Workflow Definitions** — Predefined multi-step execution plans and tool chains. Enables the assessment: *"Is there a proven workflow for this complex request?"*

**Conversation Context** — What was discussed before, which data has already been retrieved. Enables the assessment: *"Can I reuse previous results?"*

---

## Four-Tier Execution Model

The Informed Dispatcher doesn't just route — it routes requests to the *appropriate level of machinery*. The Dispatcher makes all decisions and extracts parameters in a single LLM call. After that: zero LLM in Tiers 1-2, deterministic tool chain in Tier 3 (with optional formatting LLM), and a constrained agent in Tier 4. For undefined scenarios, an optionally enabled agent tier constrained by strict rules can come into play.

<!-- IMAGE 2: diagram2_architecture.png — Four-Tier Execution Model -->
![Four-Tier Execution Model](./diagram2_architecture.png)

**Critical rule:** Parameter extraction ends at the dispatcher. The `params` going to handlers and workflows are fully structured, validated data. There are zero LLM calls in Tiers 1-2 — only deterministic code (regex, computation, sorting, filtering). In Tier 3, the tool chain is deterministic, though optional lightweight LLM usage (formatting, summarization) may occur in intermediate steps.

### Tier 1 — Direct Tool Call (Simple Requests)

For requests that map to a single tool with zero or minimal parameters, the dispatcher identifies the tool and it's called directly. No additional LLM inference. No handler logic.

**Characteristics:**
- Single tool, zero or minimal parameters
- No computation, transformation, or aggregation needed
- The tool's raw output (a URL, a list, a status) is the answer itself

**Expected distribution:** ~35-40% of traffic.

**Cost:** Effectively $0 beyond the dispatcher call. **Latency:** ~100ms. **LLM:** None.

### Tier 2 — Handler Execution (Mid-Level Requests)

For requests requiring filtering, sorting, computation, or combining 1-2 tools with known logic, the dispatcher selects a handler and provides the extracted parameters. The handler runs deterministically — **there are zero LLM calls**, only code.

**Characteristics:**
- Works with structured parameters from the dispatcher (never parses natural language)
- Contains business logic (sorting, filtering, computation, regex, formatting)
- 1-2 tool calls, deterministic combination logic
- Each handler is an independent, unit-testable module

**Expected distribution:** ~45-50% of traffic.

**Cost:** ~$0.001 per request. **Latency:** ~300ms. **LLM:** None.

### Tier 3 — Workflow Execution (Complex Requests)

This is where the pattern's power becomes most apparent. Complex requests that would traditionally require an agent's multi-step reasoning are handled by **predefined, deterministic workflows** — what I call *compiled agent behavior*.

A workflow is essentially the answer to this question: *"If an agent handled this request flawlessly, which tool chain would it execute?"* You define that chain in code, explicitly specifying the data flow between steps. The dispatcher's job is merely to select the right workflow and provide the initial parameters.

**Characteristics:**
- 2-4 tool calls in a defined sequence — **the tool chain is always the same**
- Step N's output feeds into step N+1 (explicit data flow)
- Fixed iteration count (no loops, no re-planning)
- The entire chain is a single, testable unit
- **Optional:** Lightweight LLM usage in intermediate steps (formatting, summarization) may occur — but these steps are also predefined, not runtime decisions

**Expected distribution:** ~10% of traffic.

**Cost:** ~$0.003 per request (without LLM step) / ~$0.005 (with LLM step). **Latency:** ~800ms (deterministic, predictable).

### Tier 4 — Constrained Agent (Optional Escape Hatch)

This is the pattern's most controversial yet most pragmatic tier. The first three tiers are fully deterministic — predefined paths, fixed tool chains, testable outputs. But in the real world, undefined scenarios are inevitable. A user may make a request that falls outside the scope of any handler or workflow. Tier 4 is a **free agent constrained by strict rules and governed by policy** for these situations.

The key point is this: this tier is **disabled by default**. Enabling it is a conscious policy decision by the system administrator.

**Primary Purpose: Controlled Escape Hatch**

Tier 4's reason for existence is simple: **instead of responding "sorry, I can't do that" to requests outside Tiers 1-3, provide a controlled attempt.** Tier 3's boundary is the scope of predefined workflows. When a request matches no workflow, the medium fallback kicks in — but this typically produces an incomplete answer. Tier 4 is a safety valve for "undefined but legitimate" requests.

Tier 4 is permanent. Its traffic share may decrease over time but never reaches zero — because even in a bounded domain, there will always be unforeseen requests. This is not a transitional tier but a **permanent safety valve.**

It doesn't take on all the risks of an agent because it operates with strict constraints:

**Guardrails:**

- **Max iteration: 3-5** — Eliminates the risk of infinite loops.
- **Tool whitelist: Only existing MCP tools** — Prevents the agent from calling undefined tools.
- **Hard timeout: 8-10 seconds** — Guarantees SLA.
- **Token budget: Max ~2K output tokens** — Prevents cost explosions.
- **Result validation: Tool output vs agent response comparison** — Hallucination detection.
- **Full logging: Every thought-action-observation is recorded** — Audit trail + workflow discovery.

**Policy-Based Management:**

Tier 4's most critical feature is that it can be **toggled on and off**. This is not a code change but a configuration decision:

```python
class AgentPolicy:
    enabled: bool = False              # Default: disabled
    allowed_tools: list[str] = []       # Empty = no tools available
    max_iterations: int = 3
    timeout_seconds: float = 8.0
    max_output_tokens: int = 2000
    require_approval: bool = False      # True = human approval required
    allowed_hours: tuple = (9, 18)      # Business hours only
    daily_budget: float = 5.0           # Daily max spend ($)
    log_level: str = "verbose"          # Every step is logged
```

This policy can vary by environment:

- **Development:** `enabled=True, max_iterations=5` — Full exploration mode
- **Staging:** `enabled=True, require_approval=True` — Testing with approval
- **Production:** `enabled=False` — Disabled (initial phase)
- **Production (mature):** `enabled=True, max_iterations=3, daily_budget=5.0` — Limited enabled

**Expected distribution:** ~1-3% of traffic (only requests that Tier 3 can't handle).

**Cost:** ~$0.01-0.03 per request. **Latency:** ~2-8s (variable but bounded).

**Secondary Benefit: Workflow Discovery Potential**

Tier 4's primary purpose is to be a controlled escape hatch. However, since every run is fully logged, it can contribute to workflow discovery as a secondary benefit.

That said, it's important to understand the limitations of this benefit: **automatic pattern discovery at 1-3% traffic volume is statistically inefficient.** In 10,000 daily requests, the 100-300 falling to Tier 4 may not generate enough data for meaningful pattern detection. Therefore, workflow discovery is not a justification for enabling Tier 4 — it's a nice side effect.

If you want to increase discovery value, you can use **shadow mode**: run Tier 4 on higher traffic (10-20%) but only log, don't generate user responses. This collects discovery data without affecting the active service:

```python
class AgentPolicy:
    mode: Literal["off", "shadow", "active"] = "off"
    # off:    Completely disabled
    # shadow: Run on higher traffic, log only, don't respond (for discovery)
    # active: Normal usage, respond to undefined requests
```

Logs collected in shadow mode are evaluated through weekly manual review. If recurring tool chains are detected, they can be converted to Tier 3 workflows with engineer approval. But this process is not automatic — **it requires human judgment**, because the chain the agent found may not always be optimal or correct.

### Comparing the Four Tiers

**Tier 1 — Direct Tool Call:** Direct tool call, 1 tool, no LLM. Full determinism, easy testability, zero hallucination risk. No policy required. Traffic share: ~35-40%.

**Tier 2 — Handler Execution:** Handler (deterministic), 1-2 tools, no LLM. Full determinism, easy testability, zero hallucination risk. No policy required. Traffic share: ~45-50%.

**Tier 3 — Workflow Execution:** Workflow (deterministic chain), 2-4 tools, 0-1 optional LLM (formatting). Fixed chain, easy testability, zero hallucination risk (minimal in optional LLM step). No policy required. Traffic share: ~10%.

**Tier 4 — Constrained Agent:** Agent (limited non-det.), N tools (within whitelist), 1-5 LLM calls (iteration-dependent). Limited determinism (bounded), testing difficult but logged, low hallucination risk (guardrails). **Policy required.** Traffic share: ~1-5%.

### Workflow vs. Agent vs. Constrained Agent: The Critical Distinction

This is the heart of the pattern. A comparison of three execution strategies:

**Tool ordering:**
Free Agent → Decided at runtime (non-deterministic). Workflow → Defined in code (deterministic). Constrained Agent → Decided at runtime but within whitelist.

**Parameter flow:**
Free Agent → Agent extracts from context (error-prone). Workflow → Explicitly coded (`step_1.output → step_2.input`). Constrained Agent → Agent extracts but in limited steps.

**Maximum iterations:**
Free Agent → Unlimited. Workflow → Fixed (number of steps in workflow). Constrained Agent → Fixed upper bound (3-5).

**Hallucination risk:**
Free Agent → High. Workflow → None, tool outputs are used directly. Constrained Agent → Low, guardrails + validation.

**Testability:**
Free Agent → Extremely difficult. Workflow → Standard unit testing. Constrained Agent → Difficult but auditable with full logging.

**Adding new capabilities:**
Free Agent → Update the prompt and hope. Workflow → Write a new workflow class. Constrained Agent → Automatic, uses it if it's in the tool whitelist.

**Cost predictability:**
Free Agent → Variable (agent can loop). Workflow → Fixed (known step count). Constrained Agent → Bounded (max iteration + timeout + budget).

**Management:**
Free Agent → Prompt engineering. Workflow → Code. Constrained Agent → **Policy (toggleable)**.

<!-- IMAGE 3: diagram3_workflow_vs_agent.png — Workflow vs. Agent Comparison -->
![Workflow vs. Agent Comparison](./diagram3_workflow_vs_agent.png)

Think of it this way: **A Workflow is a pre-compiled agent trace.** You observe what a perfect agent *would do* for a class of requests, then encode that behavior as a deterministic pipeline. The agent's intelligence is captured at design time, not at runtime.

**The Constrained Agent is a permanent safety valve for scenarios not yet compiled.** It provides a controlled escape hatch for requests that deterministic tiers can't handle. Every run is logged and those logs can turn into new workflow candidates over time — but this is a secondary benefit, not the primary purpose.

---

## Dispatcher Output Format

The Informed Dispatcher produces a single structured JSON output in a single LLM call. The `complexity` field determines which execution tier will handle the request:

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

**Tier 4 — Constrained Agent (if policy allows):**
```json
{
  "complexity": "unknown",
  "tool": null,
  "params": { "user_intent": "compare with tax implications and insurance cost" },
  "handler": null,
  "workflow": null,
  "agent": {
    "goal": "Fulfill the user's multi-domain request",
    "suggested_tools": ["loan_rates", "tax_calculator", "insurance_rates"],
    "max_iterations": 3
  }
}
```

Note the complexity field is `"unknown"` — the dispatcher is explicitly stating that this request doesn't fall within the scope of existing handlers/workflows. If the agent policy is disabled, the dispatcher falls back this request to the medium handler and indicates it cannot provide a comprehensive answer.

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

This last case is important: the dispatcher can detect missing parameters *because it knows what each handler and workflow expects*. Instead of guessing or hallucinating values, it explicitly asks.

### Runtime Parameter Validation

The dispatcher produces structured JSON, handlers expect typed params. But there's no guarantee that the parameters extracted by the LLM will exactly match what the handler actually expects — especially when the prompt is updated but the handler code isn't (or vice versa). For this reason, adding a **lightweight validation layer** between the dispatcher output and handler execution is recommended:

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

No routing system is perfect. The Informed Dispatcher uses a conservative fallback strategy — when uncertain, escalate rather than fail:

```
Easy (uncertain) → Fallback to Medium
Medium (uncertain) → Fallback to default Medium handler
Hard (no matching workflow) + Agent policy ON → Tier 4 (constrained agent)
Hard (no matching workflow) + Agent policy OFF → Medium fallback
Tier 4 failure / timeout → Medium fallback + log
Dispatcher error → Fallback to default handler
```

Tier 4's position in the fallback chain depends on policy. If the policy is off, the system remains fully deterministic — Tier 4 never engages. If the policy is on, unmatched hard queries are routed to the agent, but if the agent also fails, it falls back to medium.

### Circuit Breaker: Preventing Latency Stacking

The biggest risk in the fallback chain is **latency stacking**. Consider the worst case scenario: Dispatcher (~500ms) → Tier 4 timeout (8s) → Medium fallback (~300ms) = **~9 seconds** — and the user ends up with only a medium-level answer.

A **circuit breaker** mechanism is mandatory to eliminate this risk:

```python
class AgentCircuitBreaker:
    # Fast cut: If agent hasn't made first tool call in 3 seconds, cut
    first_action_timeout: float = 3.0

    # Total timeout: The hard timeout (8s) in policy already exists,
    # but circuit breaker intervenes earlier

    # Automatic disable:
    # If success rate drops below 50% in the last hour
    # Automatically disable Tier 4, send alert
    auto_disable_threshold: float = 0.50
    auto_disable_window_minutes: int = 60

    # Gradual recovery:
    # Wait 10 minutes after disable, then try with 10% traffic
    recovery_delay_minutes: int = 10
    recovery_traffic_percent: float = 0.10
```

**Practical rules:**

- **No first action within 3s** → Cut immediately, Medium fallback. Max 3.5s.
- **Agent running but 8s timeout** → Cut, Medium fallback. Max 8.5s.
- **Success <50% in last hour** → Tier 4 automatically disabled. 0s (drops straight to medium).
- **Agent successful** → Normal flow. 2-5s.

Don't enable Tier 4 without a circuit breaker. This isn't a safety mechanism — it's a **prerequisite.**

The principle: **an over-processed request is always better than a wrong one.** If a request classified as "easy" actually requires handler logic, the medium tier catches it gracefully. The worst case is unnecessary computation, never an incorrect result.

---

## When to Use It

### Strong Fit

The Informed Dispatcher shines in these situations:

**Limited tool count.** You have a known set of 5-20 tools, not an unlimited plugin ecosystem.

**Request patterns are discoverable.** You can identify the most common complex request types through log analysis and pre-build workflows for them.

**Accuracy is non-negotiable.** Finance, healthcare, legal — domains where hallucinated intermediate reasoning is unacceptable.

**SLAs matter.** You need predictable latency and cost, not "usually fast but sometimes 30 seconds."

**Testability is mandatory.** You need to write unit tests, run regression suites, and reliably reproduce bugs.

**Gradual adoption is preferred.** You want to start simple and add complexity only when data proves it's needed.

### Weak Fit

A full orchestrator/agent should be considered when:

**Tasks are fundamentally unpredictable.** Coding agents that modify an unknown number of files, research agents that follow citation chains — these require dynamic planning.

**Tool combinations are unlimited.** If you can't enumerate common workflows, you can't pre-compile them.

**Latency and cost tolerance is high.** Internal tools where 10-second response times are acceptable.

**Approximate answers suffice.** Creative tasks, brainstorming, exploratory analysis.

---

## Implementation Guide

### Phase 1: Foundation (Week 1-2)

Start with only Tier 1 and a single Tier 2 handler.

1. **Create the dispatcher prompt** with the tool registry and one handler definition
2. **Implement the JSON output parser** with validation
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

Add only the workflows that your data proves are needed.

1. **Analyze logs** for multi-step request patterns that Tier 2 can't handle
2. **Design workflow schemas** (tool chain, data flow, parameters)
3. **Implement the workflow executor** — a simple sequential runner
4. **Add 2-3 workflows** for the most common complex patterns
5. **Monitor and iterate**

### Phase 4: Constrained Agent (Week 7-8, Optional)

Only after Tiers 1-3 have collected sufficient data and unmet request patterns have become clear:

1. **Implement the agent policy framework** (enabled flag, tool whitelist, budget, timeout)
2. **Enable Tier 4 in development**, review logs
3. **Test the guardrails** — max iteration, timeout, budget limit
4. **Test in staging** with `require_approval=true`
5. **Enable in production** with low limits (`daily_budget=5.0, max_iterations=3`)
6. **Set up the workflow discovery pipeline** — recurring pattern detection from agent logs

### Phase 5: Optimization (Ongoing)

- Add new workflows driven by log analysis (never by speculation)
- Identify workflow candidates from Tier 4 logs and promote them to Tier 3
- Tune the dispatcher prompt based on misclassification data
- Optimize handler/workflow performance
- Consider caching for recurring dispatcher decisions
- Monitor Tier 4 traffic share — if decreasing, the system is maturing; if increasing, it's approaching its limits

Core philosophy: **data-driven, gradual adoption.** Never create a workflow before seeing the pattern in production logs. Never add complexity before simplicity is proven insufficient.

---

## Scaling the Pattern

The most common criticism of the Informed Dispatcher is: *"You say the dispatcher needs to know everything, but what happens when you have 50 tools and 30 workflows in the system prompt?"*

This is a valid criticism — and the answer is straightforward: **the pattern itself scales in layers.**

### Scaling Stages

<!-- IMAGE 5: diagram5_scaling.png — Scaling Stages -->
![Scaling Stages](./diagram5_scaling.png)

**Stage 1 — Single Dispatcher (5-20 tools):** The pattern's base form. The tool registry, handler definitions, and workflow definitions fit in a single system prompt (~1-2K tokens). Most bounded domain systems stay at this stage and should.

**Stage 2 — Domain Dispatchers (20-50 tools):** When the system spans multiple domains, a lightweight **Meta-Dispatcher** is added at the top level. This dispatcher only selects the domain (lending, insurance, investment) — each domain has its own Informed Dispatcher. The Meta-Dispatcher's prompt stays small because it only contains domain definitions, not tool details.

**Stage 3 — Hierarchical Tree (50+ tools):** At this point you're approaching the pattern's limits. Sub-dispatchers operate under domain clusters. But honestly, a system reaching 50+ tools likely needs agent behavior in some of its layers.

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

This approach has two critical benefits. First, code and prompt always stay in sync — when you add a new handler, the prompt updates automatically. Second, the prompt format can be optimized — compact schema instead of verbose descriptions.

### Token Budget Management

Concrete targets for the dispatcher prompt:

- **Tool Registry (~300-500 tokens):** Compact format — name, one-line description, parameter list.
- **Handler Definitions (~300-500 tokens):** Name, tools used, expected parameters.
- **Workflow Definitions (~200-400 tokens):** Name, step list, when to use.
- **Routing Rules (~200-300 tokens):** Priority order, fallback rules.
- **Total: ~1-2K tokens** — 1-2% of the context window.

If the total exceeds 3K tokens, this is usually a sign that definitions are too verbose or the domain needs splitting.

---

## When to Transition to an Agent

The Informed Dispatcher is not an endpoint — it's a **starting point**. Every pattern has validity boundaries, and knowing those boundaries is as important as knowing the pattern.

### Red Flags

When you see these signals, know that you're approaching the pattern's limits:

**1. Workflow explosion.** When workflow count exceeds 15-20 and you're still discovering new patterns, your domain is more dynamic than you thought. At this point, writing a workflow for each new edge case becomes a maintenance nightmare.

**2. High fallback rate.** When monitoring dispatcher decisions, if more than 15% of requests fall to fallback, this indicates the dispatcher can't serve requests within its current handler/workflow scope.

**3. Cross-domain chains.** If users are increasingly making requests that combine multiple domains — for example "calculate loan + show tax impact + add insurance cost" — predefined workflows can't cover these combinations.

**4. Shortening workflow lifespan.** If workflows you create need updating or replacing within a few weeks, the rate of change in your domain is incompatible with the pre-compilation approach.

### Concrete Threshold Values

Experience-based transition criteria:

- **Workflow count:** Safe ≤10 | Warning 10-20 | Consider transition >20
- **Fallback rate:** Safe ≤5% | Warning 5-15% | Consider transition >15%
- **Misrouting rate:** Safe ≤3% | Warning 3-10% | Consider transition >10%
- **Edge case rate:** Safe ≤5% | Warning 5-10% | Consider transition >10%
- **Average workflow lifespan:** Safe >3 months | Warning 1-3 months | Consider transition <1 month

These numbers are not hard rules but directional indicators. What matters is continuously monitoring your production metrics and performing trend analysis.

### The Transition Doesn't Have to Be Complete

Transitioning to an agent is not an "all or nothing" decision. Tier 4 has already embedded the first step of this transition within the pattern. The pattern's natural evolution is a gradual hybridization:

**Step 1 — Enable Tier 4 (Policy: Development):** Enable Tier 4 in the development environment. Collect agent logs. Observe which tool chains recur.

**Step 2 — Move Tier 4 to Production (Policy: Limited):** First deploy the circuit breaker (see Fallback Strategy). Then enable in production with `max_iterations=3, daily_budget=5.0, require_approval=false`. Only 1-3% of traffic will fall to this tier.

**Step 3 — Log Analysis and Manual Workflow Discovery:** Review Tier 4 logs weekly. If you detect recurring tool chain patterns, convert them to Tier 3 workflows with engineer approval. This process is not automatic — it requires human judgment. Meaningful pattern detection at 1-3% traffic volume takes time; be patient.

**Step 4 — Threshold Exceeded:** If traffic falling to Tier 4 exceeds 5% and the variety that can't be converted to workflows is increasing, this indicates you've reached the pattern's limits. At this point, two options exist:

- **Domain splitting:** Divide the system into multiple Informed Dispatchers (each with its own bounded domain)
- **Hybrid architecture:** Tiers 1-3 remain as Informed Dispatcher (80-90% of traffic), Tier 4 operates with broader authority. But this is no longer the Informed Dispatcher pattern — it's a hybrid architecture, and you should make this transition consciously.

This gradual transition preserves the pattern's strongest advantage: **the vast majority of traffic always remains deterministic.**

---

## Monitoring and Evolution

The Informed Dispatcher's success is directly proportional to the quality of the dispatcher's decisions. You cannot improve this quality without measuring it.

### Core Metrics

Log the following data for every dispatcher decision:

**1. Predicted vs Actual Complexity:**
```
Log entry: {
  "query": "...",
  "predicted_complexity": "easy",
  "predicted_handler": "credit_score_url",
  "actual_execution": "easy",        // What actually happened?
  "fallback_triggered": false,        // Did it fall to fallback?
  "execution_success": true,          // Was the result successful?
  "latency_ms": 145,
  "user_satisfaction": null            // Optional: thumbs up/down
}
```

**2. Misrouting Detection:** The dispatcher said "easy" but the handler threw an error → misclassification. The dispatcher said "medium" but the user received incomplete information → insufficient tier selection. Automatically detect these cases and generate weekly reports.

**3. Workflow Coverage:** What percentage of incoming requests are served by existing handlers/workflows? If this rate is declining, new handlers/workflows need to be added or the pattern is approaching its limits.

### Dashboard Metrics

**Accuracy and Routing:**
- **Correct tier selection rate:** Target >95%, warning <90% → Dispatcher prompt review
- **Fallback rate:** Target <5%, warning >10% → Add new handler/workflow
- **Workflow coverage:** Target >90%, warning <80% → Log analysis, new workflow
- **Misrouting rate:** Target <3%, warning >5% → Update routing rules

**Latency:**
- **Easy:** Target <200ms, warning >500ms → Tool performance check
- **Medium:** Target <600ms, warning >1s → Handler optimization
- **Hard:** Target <1.5s, warning >3s → Workflow step analysis

**Tier 4:**
- **Traffic share:** Target <3%, warning >5% → Agent log, workflow conversion
- **Success rate:** Target >80%, warning <60% → Guardrail/prompt review
- **Daily spend:** Target <$5, warning >$10 → Tighten policy budget

### Evolution Cycle

The pattern is not static — it operates in a continuously evolving cycle:

<!-- IMAGE 4: diagram4_evolution.png — Evolution Cycle -->
![Evolution Cycle](./diagram4_evolution.png)

**Weekly:** Check misrouting and fallback rates. Sudden spikes usually indicate a new user pattern emerging.

**Monthly:** Perform workflow coverage analysis. Identify the top 5 most frequent request types not covered by existing workflows. If their frequency justifies adding a handler/workflow, add it; if not, let the fallback continue.

**Quarterly:** Assess overall pattern health. Review threshold values. Check for agent transition signals. Measure the dispatcher prompt's token budget.

### Minimizing Maintenance Overhead

One valid criticism of the pattern is "you have to update both the code and the prompt every time you add a new tool." This is a real trade-off — but it should be viewed in proper perspective.

In an agent-based system, when you add a new tool, you write a tool description and *hope* the agent uses it correctly. In the Informed Dispatcher, you write a tool description + handler/workflow and *guarantee* how it's used. The maintenance cost is similar; the assurance level is vastly different.

That said, concrete steps can be taken to reduce maintenance overhead:

**Auto-generated registry:** As mentioned earlier, auto-generate the dispatcher prompt from handler/workflow metadata. Synchronization errors between code and prompt drop to zero.

**Integration test suite:** Write test cases for each handler and workflow in the form "the dispatcher should route this query to me." When a new component is added, it's automatically checked whether existing tests break.

**Prompt diff monitoring:** Version-control every change to the dispatcher prompt. After a change, run a regression suite — "how many of these 100 test queries route differently?"

---

## Cost and Performance Comparison

For a system processing 10,000 daily requests with a typical distribution (35% easy, 55% medium, 8% hard/workflow, 2% unknown/agent):

**Cost (per request):**
- Easy → Simple Router: ~$0.001 (unnecessary LLM) | **Informed Dispatcher: ~$0** (LLM bypass) | Full Agent: ~$0.005
- Medium → Simple Router: ~$0.001 | Informed Dispatcher: ~$0.001 | Full Agent: ~$0.010
- Hard → Simple Router: not supported | Informed Dispatcher: ~$0.005-0.010 | Full Agent: ~$0.015+
- Unknown → Simple Router: not supported | Informed Dispatcher: ~$0.01-0.03 (Tier 4) | Full Agent: ~$0.015+
- **Monthly total** → Simple Router: ~$300 | **Informed Dispatcher: ~$420** | Full Agent: $1,500-4,500

**Latency:**
- Easy → Simple Router: ~500ms | **Informed Dispatcher: ~100-200ms** | Full Agent: ~800ms
- Medium → Simple Router: ~500ms | Informed Dispatcher: ~500ms | Full Agent: 1-3s
- Hard → Simple Router: not supported | Informed Dispatcher: ~800ms-1.5s (fixed) | Full Agent: 3-5s+ (variable)
- Unknown → Simple Router: not supported | Informed Dispatcher: ~2-8s (bounded) | Full Agent: 3-5s+ (variable)

**Safety and Quality:**
- Cost explosion risk → Simple Router: none | **Informed Dispatcher: none** (policy budget) | Full Agent: high
- Hallucination risk → Simple Router: low | **Informed Dispatcher: low** | Full Agent: high
- Testability → Simple Router: easy | **Informed Dispatcher: easy** (Tiers 1-3) / audit (Tier 4) | Full Agent: very difficult
- SLA guarantee → Simple Router: high | **Informed Dispatcher: high** (Tiers 1-3) / bounded (Tier 4) | Full Agent: low

The Informed Dispatcher costs roughly 40% more than a simple router but handles both complex and undefined requests that the router can't. It's 70-90% cheaper than a full agent. The cost added by Tier 4 minimally impacts the total budget since it only handles 1-3% of traffic — and in return for this small cost, you gain the capacity to provide controlled responses to requests that would otherwise get "I can't do that."

---

## Design Principles

After building systems across this spectrum, I've distilled the Informed Dispatcher into a few core principles:

**1. The dispatcher decides, never executes.** A single LLM call for classification, complexity assessment, and parameter extraction. Everything after that is deterministic code — or, if policy allows, a constrained agent.

**2. The first layer's scope is broad.** The dispatcher doesn't just say "where." In a single call, it determines intent, complexity, parameters, missing fields, and execution path. This broad scope determines the quality of all downstream decisions.

**3. System knowledge improves routing quality.** A dispatcher that knows what its tools, handlers, and workflows can do makes fundamentally better decisions than a blind router.

**4. Compile agent behavior into workflows.** Don't let an LLM figure out tool chains at runtime. Observe optimal chains, define them in code, and have the dispatcher select among them.

**5. Escalate in uncertainty — don't guess.** Fallback to an upper tier wastes computation but never produces an incorrect result. Fallback to a lower tier risks an incomplete response.

**6. Add complexity only when data demands it.** Start with Tiers 1 + 2. Add workflows only for patterns you've observed in production. Speculative complexity is the enemy of sustainable systems.

**7. Every tier is independently testable.** The dispatcher is tested with input/output pairs. Each handler is unit-tested. Each workflow is integration-tested. No tier depends on another for correctness.

**8. Don't unleash the agent — manage it with policy.** Tier 4 is disabled by default. Enabling it is a conscious decision. Guardrails are in code, not in the agent's conscience. Budget, timeout, and iteration limits are configuration, not prompt instructions.

**9. The agent tier is a permanent safety valve — but shouldn't carry most of the traffic.** Tier 4's traffic share decreases over time because patterns discovered from logs can be promoted to Tier 3. But it never reaches zero — even in a bounded domain, there will always be unforeseen requests. Tier 4's health is measured by its traffic share staying low (1-5% target).

**10. Don't trust LLM output — validate it.** The dispatcher is an LLM and can make mistakes. Runtime parameter validation (between dispatcher and handler) and circuit breaker (in the Tier 4 fallback chain) provide defense in depth. These checks are cheap (<5ms) but catch prompt-code synchronization errors and latency explosions before they reach production.

---

## Conclusion

The Informed Dispatcher pattern fills a real architectural gap. It offers the Orchestrator's domain awareness and complexity management while preserving the Router's determinism, testability, and cost predictability.

Its four-tier structure provides the right trade-off at every level: zero cost for simple requests, deterministic handlers for medium requests, compiled workflows for complex requests, and a policy-managed constrained agent for undefined requests. The first three tiers are fully deterministic. The fourth tier is optional — a permanent safety valve for undefined but legitimate requests. Its traffic share may decrease over time but never reaches zero; unforeseen scenarios are inevitable in every production system.

It's not the right choice for every system. If your tasks are truly unbounded — if you genuinely cannot predict which tool chains will be needed — you need a full agent or orchestrator. And the pattern has limits: when workflow count exceeds 20, fallback rate surpasses 15%, or cross-domain chains become dominant, a gradual hybridization should be considered. But knowing these limits is not the pattern's weakness — it's a sign of its maturity.

In my experience, most production systems operate in bounded domains with discoverable patterns. For these systems, the Informed Dispatcher offers a better balance than either extreme. What matters is starting right, measuring continuously, and letting the data tell you when you need more.

**A router that knows too much. An orchestrator that refuses to improvise — but permits limited improvisation through policy when needed. That's what the Informed Dispatcher is.**

---

*If you find yourself oscillating between "too simple" and "too complex" when building LLM-based systems, consider whether your domain fits this pattern. Start with the simplest version — a dispatcher with a single handler — and wait for your data to tell you when you need more.*
