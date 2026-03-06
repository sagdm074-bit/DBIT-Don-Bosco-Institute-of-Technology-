"""
Multi-Agent Debate System - COMPLETE SOLUTION
==============================================
7 agents, adversarial parallel debate, with:
  - Cross-examination round (agents respond to each other's arguments)
  - Multi-round debate (if judge score is close, go another round)
  - Cost tracking + budget enforcement
  - Per-agent error boundaries
  - Evaluation harness for debate quality

Architecture:
  Round 1: Planner -> [Researcher x2] -> [Debater x2] -> Judge
  Round 2 (if close): [Cross-examiner x2] -> Judge (final)
  Synthesis: Synthesizer -> Report

Usage:
  python project_debate_system.py doc.pdf "Is Python better than Java for ML?"
  python project_debate_system.py doc.pdf "Should we use microservices?" --rounds 2 --budget 0.80
"""

from helpers import (
    load_and_chunk, build_index, search,
    call_llm_cheap, call_llm_strong,
    parse_json, init_state, log_agent, print_log,
    CostTracker, EvalHarness
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys, time, argparse, traceback


# ============================================================
# AGENT PROMPTS
# ============================================================

PLANNER_SYSTEM = """You are a Debate Planner. Set up a fair debate framework.
Return JSON:
{
    "topic_restated": "clear neutral statement of the debate",
    "for_position": "what the FOR side argues (1 sentence)",
    "against_position": "what the AGAINST side argues (1 sentence)",
    "dimensions": ["dimension1", "dimension2", "dimension3"],
    "context_from_document": "what the document says about this topic"
}"""

DEBATER_SYSTEM_TEMPLATE = """You are a skilled Debater arguing {side} the position: "{position}"

Rules:
- Use ONLY evidence from the provided chunks. No training data.
- 3 clear arguments, each with: point, specific evidence quote, reasoning.
- Anticipate and preemptively counter the strongest opposing argument.
- Be persuasive but honest - no misrepresentation.

Return JSON:
{{
    "opening_statement": "1-2 sentence thesis",
    "arguments": [
        {{"point": "title (5-8 words)", "evidence": "specific quote from chunks", "reasoning": "2-3 sentences"}}
    ],
    "counter_to_opposition": "preemptive rebuttal",
    "closing_statement": "1-2 sentence conclusion"
}}"""

CROSS_EXAM_SYSTEM = """You are a Cross-Examiner. You've read the opposing argument and must challenge it.

Rules:
- Identify the WEAKEST point in the opposing argument.
- Find evidence in your chunks that directly contradicts or undermines it.
- Raise 1-2 questions the opposing side cannot easily answer.
- Stay factual - attack the argument, not the arguer.

Return JSON:
{
    "weakest_point": "which opposing argument is weakest",
    "challenge": "your evidence-based challenge to that point",
    "unanswerable_questions": ["question 1 they can't answer", "question 2"],
    "additional_evidence": "new evidence from chunks that strengthens YOUR side"
}"""

JUDGE_SYSTEM = """You are an impartial Judge. Evaluate both debate arguments fairly.

Criteria (0-10 each):
1. Evidence Quality - claims backed by specific document evidence?
2. Logical Coherence - argument flows logically, no fallacies?
3. Completeness - addresses all debate dimensions?
4. Persuasiveness - how compelling overall?
5. Honesty - fair evidence representation or cherry-picking?

Return JSON:
{
    "for_score": {"evidence_quality": 8, "logical_coherence": 7, "completeness": 8, "persuasiveness": 7, "honesty": 9, "total": 39},
    "against_score": {"evidence_quality": 7, "logical_coherence": 8, "completeness": 7, "persuasiveness": 8, "honesty": 8, "total": 38},
    "winner": "for|against|tie",
    "margin": "decisive|narrow|razor-thin",
    "reasoning": "2-3 sentences explaining the decision",
    "strongest_point_for": "...",
    "strongest_point_against": "...",
    "weakest_point_for": "...",
    "weakest_point_against": "..."
}"""

SYNTHESIZER_SYSTEM = """You are a Synthesizer. Write a balanced analysis that goes beyond "both sides have merit."
Rules:
- Acknowledge strongest points from each side
- Identify specific areas of agreement
- Name the core unresolved tension
- Nuanced conclusion (1-2 sentences)
- 150-200 words

Return JSON:
{"balanced_analysis": "150-200 words", "common_ground": ["point1", "point2"], "key_tension": "...", "nuanced_conclusion": "1-2 sentences"}"""


# ============================================================
# AGENT FUNCTIONS
# ============================================================

def debate_planner(state, tracker):
    topic = state["topic"]
    sample_text = "\n---\n".join([c["text"] for c in state["chunks"][:6]])

    result = call_llm_cheap(
        system=PLANNER_SYSTEM,
        prompt=f"Debate topic: {topic}\n\nDocument context:\n{sample_text}\n\nSet up a fair debate.",
        json_output=True
    )
    tracker.record(result, "planner")
    parsed = parse_json(result["text"])
    state["debate_plan"] = parsed
    log_agent(state, "planner", {"topic": topic}, parsed,
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Planner] FOR: {parsed.get('for_position', '?')[:60]}")
    print(f"  [Planner] AGAINST: {parsed.get('against_position', '?')[:60]}")
    return state


def researcher(state, tracker, side="for"):
    """Search for evidence supporting one side. No LLM needed - pure vector search."""
    plan = state["debate_plan"]
    topic = state["topic"]

    if side == "for":
        queries = [
            f"evidence supporting {plan['for_position']}",
            f"benefits advantages {topic}",
            f"reasons why {topic}",
        ]
    else:
        queries = [
            f"evidence against {topic}",
            f"problems disadvantages limitations {topic}",
            f"criticism concerns risks {topic}",
        ]

    # Add dimension-specific searches
    for dim in plan.get("dimensions", [])[:2]:
        term = "benefits positive" if side == "for" else "problems limitations"
        queries.append(f"{dim} {term}")

    evidence = []
    for q in queries:
        evidence.extend(search(state["_index"], state["_all_chunks"], q, k=3))

    seen = set()
    unique = [c for c in evidence if c["index"] not in seen and not seen.add(c["index"])]
    state[f"evidence_{side}"] = unique[:8]
    log_agent(state, f"researcher_{side}", {"queries": len(queries)},
              f"{len(unique[:8])} chunks")
    print(f"  [Researcher {side.upper()}] {len(unique[:8])} evidence chunks")
    return state


def debater(state, tracker, side="for"):
    """Build the strongest argument for one side."""
    plan = state["debate_plan"]
    evidence = state.get(f"evidence_{side}", [])
    evidence_text = "\n---\n".join([e["text"] for e in evidence])

    position = plan["for_position"] if side == "for" else plan["against_position"]
    side_label = "FOR" if side == "for" else "AGAINST"

    system = DEBATER_SYSTEM_TEMPLATE.format(
        side=side_label,
        position=position
    )

    result = call_llm_strong(
        system=system,
        prompt=f"Topic: {state['topic']}\nDimensions: {', '.join(plan.get('dimensions', []))}\n\nYour evidence:\n{evidence_text}\n\nBuild your strongest argument.",
        json_output=True
    )
    tracker.record(result, f"debater_{side}")
    parsed = parse_json(result["text"])
    state[f"argument_{side}"] = parsed
    log_agent(state, f"debater_{side}", {"evidence": len(evidence)},
              f"{len(parsed.get('arguments', []))} arguments",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Debater {side.upper()}] {len(parsed.get('arguments', []))} arguments")
    return state


def cross_examiner(state, tracker, my_side="for"):
    """Challenge the opposing argument with evidence-based questions."""
    opposing_side = "against" if my_side == "for" else "for"
    opposing_arg = state.get(f"argument_{opposing_side}", {})
    my_evidence = state.get(f"evidence_{my_side}", [])
    evidence_text = "\n---\n".join([e["text"] for e in my_evidence])

    result = call_llm_strong(
        system=CROSS_EXAM_SYSTEM,
        prompt=f"""You are arguing {my_side.upper()}.

OPPOSING ARGUMENT TO CHALLENGE:
Opening: {opposing_arg.get('opening_statement', '?')}
Arguments: {json.dumps(opposing_arg.get('arguments', []), indent=2)}
Rebuttal: {opposing_arg.get('counter_to_opposition', '?')}

YOUR EVIDENCE (to find counter-points):
{evidence_text}

Find the weakest point and challenge it.""",
        json_output=True
    )
    tracker.record(result, f"cross_exam_{my_side}")
    parsed = parse_json(result["text"])
    state[f"cross_exam_{my_side}"] = parsed
    log_agent(state, f"cross_exam_{my_side}", {"opposing_args": len(opposing_arg.get("arguments", []))},
              f"Challenged: {parsed.get('weakest_point', '?')[:50]}",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Cross-Exam {my_side.upper()}] Challenged: {parsed.get('weakest_point', '?')[:50]}")
    return state


def judge(state, tracker, round_num=1):
    """Impartially evaluate both arguments."""
    arg_for = state.get("argument_for", {})
    arg_against = state.get("argument_against", {})

    # Include cross-examination in round 2+
    cross_context = ""
    if round_num > 1:
        cx_for = state.get("cross_exam_for", {})
        cx_against = state.get("cross_exam_against", {})
        cross_context = f"""
CROSS-EXAMINATION (Round {round_num}):
FOR's challenge to AGAINST: {json.dumps(cx_for, indent=2)}
AGAINST's challenge to FOR: {json.dumps(cx_against, indent=2)}

Factor cross-examination into your scoring. Strong challenges should lower the challenged side's score."""

    result = call_llm_strong(
        system=JUDGE_SYSTEM,
        prompt=f"""DEBATE TOPIC: {state['topic']}
ROUND: {round_num}

FOR ARGUMENT:
Opening: {arg_for.get('opening_statement', '?')}
Arguments: {json.dumps(arg_for.get('arguments', []), indent=2)}
Rebuttal: {arg_for.get('counter_to_opposition', '?')}
Closing: {arg_for.get('closing_statement', '?')}

AGAINST ARGUMENT:
Opening: {arg_against.get('opening_statement', '?')}
Arguments: {json.dumps(arg_against.get('arguments', []), indent=2)}
Rebuttal: {arg_against.get('counter_to_opposition', '?')}
Closing: {arg_against.get('closing_statement', '?')}
{cross_context}
Judge fairly. Consider all evidence and reasoning.""",
        json_output=True
    )
    tracker.record(result, f"judge_r{round_num}")
    parsed = parse_json(result["text"])
    state["judgment"] = parsed
    log_agent(state, f"judge_r{round_num}", {"round": round_num},
              f"Winner: {parsed.get('winner', '?')} ({parsed.get('margin', '?')})",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})

    f_total = parsed.get("for_score", {}).get("total", "?")
    a_total = parsed.get("against_score", {}).get("total", "?")
    print(f"  [Judge R{round_num}] FOR: {f_total}/50 | AGAINST: {a_total}/50 | "
          f"Winner: {parsed.get('winner', '?').upper()} ({parsed.get('margin', '?')})")
    return state


def synthesizer(state, tracker):
    """Write balanced final analysis."""
    judgment = state.get("judgment", {})
    arg_for = state.get("argument_for", {})
    arg_against = state.get("argument_against", {})

    result = call_llm_strong(
        system=SYNTHESIZER_SYSTEM,
        prompt=f"""Topic: {state['topic']}
Verdict: {judgment.get('winner', '?')} ({judgment.get('margin', '?')})
Reasoning: {judgment.get('reasoning', '?')}
FOR thesis: {arg_for.get('opening_statement', '?')}
AGAINST thesis: {arg_against.get('opening_statement', '?')}
Strongest FOR: {judgment.get('strongest_point_for', '?')}
Strongest AGAINST: {judgment.get('strongest_point_against', '?')}

Synthesize beyond "both sides have merit".""",
        json_output=True
    )
    tracker.record(result, "synthesizer")
    parsed = parse_json(result["text"])
    state["synthesis"] = parsed
    log_agent(state, "synthesizer", {}, parsed.get("nuanced_conclusion", "")[:80],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Synthesizer] Done")
    return state


# ============================================================
# REPORT FORMATTING
# ============================================================

def format_report(state):
    plan = state.get("debate_plan", {})
    arg_for = state.get("argument_for", {})
    arg_against = state.get("argument_against", {})
    judgment = state.get("judgment", {})
    synthesis = state.get("synthesis", {})
    rounds = state.get("rounds_played", 1)

    r = []
    r.append("=" * 65)
    r.append("  MULTI-AGENT DEBATE REPORT")
    r.append("=" * 65)
    r.append(f"\n  Topic:  {state.get('topic', '?')}")
    r.append(f"  Rounds: {rounds}")

    r.append("\n" + "-" * 55)
    r.append("  DEBATE FRAMEWORK")
    r.append("-" * 55)
    r.append(f"  FOR:     {plan.get('for_position', '?')}")
    r.append(f"  AGAINST: {plan.get('against_position', '?')}")
    r.append(f"  Dimensions: {', '.join(plan.get('dimensions', []))}")

    # FOR argument
    r.append("\n" + "-" * 55)
    r.append("  FOR ARGUMENT")
    r.append("-" * 55)
    r.append(f"  Thesis: \"{arg_for.get('opening_statement', '?')}\"")
    for i, a in enumerate(arg_for.get("arguments", []), 1):
        r.append(f"\n    {i}. {a.get('point', '?')}")
        r.append(f"       Evidence: \"{a.get('evidence', '?')[:100]}...\"")
        r.append(f"       Why: {a.get('reasoning', '?')}")
    r.append(f"\n  Rebuttal: {arg_for.get('counter_to_opposition', '?')}")

    # AGAINST argument
    r.append("\n" + "-" * 55)
    r.append("  AGAINST ARGUMENT")
    r.append("-" * 55)
    r.append(f"  Thesis: \"{arg_against.get('opening_statement', '?')}\"")
    for i, a in enumerate(arg_against.get("arguments", []), 1):
        r.append(f"\n    {i}. {a.get('point', '?')}")
        r.append(f"       Evidence: \"{a.get('evidence', '?')[:100]}...\"")
        r.append(f"       Why: {a.get('reasoning', '?')}")
    r.append(f"\n  Rebuttal: {arg_against.get('counter_to_opposition', '?')}")

    # Cross-examination (if played)
    if state.get("cross_exam_for"):
        r.append("\n" + "-" * 55)
        r.append("  CROSS-EXAMINATION")
        r.append("-" * 55)
        cx_for = state["cross_exam_for"]
        cx_against = state["cross_exam_against"]
        r.append(f"  FOR challenges AGAINST on: {cx_for.get('weakest_point', '?')}")
        r.append(f"    Challenge: {cx_for.get('challenge', '?')[:120]}")
        r.append(f"  AGAINST challenges FOR on: {cx_against.get('weakest_point', '?')}")
        r.append(f"    Challenge: {cx_against.get('challenge', '?')[:120]}")

    # Judge verdict
    r.append("\n" + "-" * 55)
    r.append("  JUDGE'S VERDICT")
    r.append("-" * 55)
    f_s = judgment.get("for_score", {})
    a_s = judgment.get("against_score", {})

    r.append(f"\n    {'CRITERION':<22} {'FOR':>5}  {'AGAINST':>7}")
    r.append(f"    {'-' * 38}")
    for c in ["evidence_quality", "logical_coherence", "completeness", "persuasiveness", "honesty"]:
        label = c.replace("_", " ").title()
        r.append(f"    {label:<22} {f_s.get(c, '?'):>4}/10  {a_s.get(c, '?'):>5}/10")
    r.append(f"    {'-' * 38}")
    r.append(f"    {'TOTAL':<22} {f_s.get('total', '?'):>4}/50  {a_s.get('total', '?'):>5}/50")

    r.append(f"\n  WINNER: {judgment.get('winner', '?').upper()} ({judgment.get('margin', '?')})")
    r.append(f"  Reason: {judgment.get('reasoning', '?')}")

    # Synthesis
    r.append("\n" + "-" * 55)
    r.append("  BALANCED SYNTHESIS")
    r.append("-" * 55)
    r.append(f"  {synthesis.get('balanced_analysis', '?')}")
    if synthesis.get("common_ground"):
        r.append("\n  Common Ground:")
        for p in synthesis["common_ground"]:
            r.append(f"    - {p}")
    r.append(f"\n  Key Tension: {synthesis.get('key_tension', '?')}")
    r.append(f"  Conclusion: \"{synthesis.get('nuanced_conclusion', '?')}\"")

    if state.get("errors"):
        r.append("\n" + "-" * 55)
        r.append("  ERRORS")
        for err in state["errors"]:
            r.append(f"    [{err.get('agent', '?')}] {err.get('error', '?')}")

    r.append("\n" + "=" * 65)
    return "\n".join(r)


# ============================================================
# PARALLEL HELPERS
# ============================================================

def run_parallel(fns_with_args, state, tracker):
    """Run multiple (fn, side) pairs in parallel with error boundaries."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(fns_with_args)) as executor:
        futures = {}
        for fn, side in fns_with_args:
            s = {**state}
            futures[executor.submit(fn, s, tracker, side)] = (fn.__name__, side)

        for future in as_completed(futures):
            name, side = futures[future]
            try:
                result_state = future.result()
                results[side] = result_state
            except Exception as e:
                print(f"  [ERROR] {name}_{side}: {e}")
                state["errors"].append({"agent": f"{name}_{side}", "error": str(e)})

    # Merge results
    for side, rs in results.items():
        for key in [f"evidence_{side}", f"argument_{side}", f"cross_exam_{side}"]:
            if key in rs:
                state[key] = rs[key]
        state["log"].extend(rs.get("log", []))

    return state


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_debate(pdf_path, topic, budget=0.80, max_rounds=2):
    """
    Run the full debate pipeline.

    Round 1: Planner -> [Research x2] -> [Debate x2] -> Judge
    Round 2 (if margin is "razor-thin" or "narrow"):
            [Cross-examine x2] -> Judge (final)
    Then:   Synthesizer -> Report
    """
    tracker = CostTracker(budget=budget)

    print(f"\n{'=' * 65}")
    print(f"  MULTI-AGENT DEBATE SYSTEM")
    print(f"{'=' * 65}")
    print(f"  Topic:  {topic}")
    print(f"  Budget: ${budget:.2f}")
    print(f"  Max rounds: {max_rounds}")

    # Step 1: Load + index
    print("\n[1] Loading document...")
    chunks = load_and_chunk(pdf_path)
    index = build_index(chunks)

    state = init_state(topic)
    state["topic"] = topic
    state["pdf_path"] = pdf_path
    state["_index"] = index
    state["_all_chunks"] = chunks
    state["chunks"] = search(index, chunks, topic, k=8)

    # Step 2: Plan
    print("\n[2] Planning debate framework...")
    state = debate_planner(state, tracker)
    tracker.check_budget()

    # Step 3: Parallel evidence gathering
    print("\n[3] Gathering evidence (parallel)...")
    state = run_parallel([(researcher, "for"), (researcher, "against")], state, tracker)
    tracker.check_budget()

    # Step 4: Parallel debate
    print("\n[4] Building arguments (parallel)...")
    state = run_parallel([(debater, "for"), (debater, "against")], state, tracker)
    tracker.check_budget()

    # Step 5: Judge (Round 1)
    print("\n[5] Judge - Round 1...")
    state = judge(state, tracker, round_num=1)
    state["rounds_played"] = 1

    # Step 6: Cross-examination + re-judge (if debate is close and budget allows)
    margin = state.get("judgment", {}).get("margin", "decisive")
    if max_rounds > 1 and margin in ("razor-thin", "narrow") and tracker.remaining() > 0.15:
        print(f"\n[6] Debate is {margin} - triggering cross-examination round...")
        state = run_parallel(
            [(cross_examiner, "for"), (cross_examiner, "against")],
            state, tracker
        )
        tracker.check_budget()

        print("\n[6b] Judge - Round 2 (with cross-examination)...")
        state = judge(state, tracker, round_num=2)
        state["rounds_played"] = 2
    else:
        if margin not in ("razor-thin", "narrow"):
            print(f"\n  Skipping cross-examination: result was {margin}")
        else:
            print(f"\n  Skipping cross-examination: budget too low (${tracker.remaining():.4f} remaining)")

    # Step 7: Synthesis
    print("\n[7] Synthesizing...")
    state = synthesizer(state, tracker)

    # Report
    report = format_report(state)
    state["report"] = report
    print("\n" + report)

    print_log(state)
    tracker.report()

    return state


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Debate System")
    parser.add_argument("pdf", help="Path to PDF document")
    parser.add_argument("topic", help="Debate topic (in quotes)")
    parser.add_argument("--budget", type=float, default=0.80, help="Cost budget in USD (default: $0.80)")
    parser.add_argument("--rounds", type=int, default=2, help="Max debate rounds (default: 2)")

    args = parser.parse_args()
    state = run_debate(args.pdf, args.topic, budget=args.budget, max_rounds=args.rounds)
