"""
Multi-Agent Debate System - STARTER CODE
==========================================

WHAT'S GIVEN: Agent functions with prompts (they work individually).
YOUR JOB:     Build the adversarial pipeline that makes them debate.

Milestone 1 (45 min): Wire sequential pipeline - planner -> researchers -> debaters -> judge
Milestone 2 (45 min): Parallelize - researchers run together, debaters run together
Milestone 3 (30 min): Add cross-examination round when judge's margin is close
Milestone 4 (30 min): Cost tracking + budget-aware round decisions
Milestone 5 (30 min): Multi-topic tournament mode - run 3 topics, aggregate scores

Usage:
  python debate_starter.py doc.pdf "Is Python better than Java for ML?"
  python debate_starter.py doc.pdf "Should we use microservices?" --budget 0.80
"""

from helpers import (
    load_and_chunk, build_index, search,
    call_llm_cheap, call_llm_strong,
    parse_json, init_state, log_agent, print_log,
    CostTracker, EvalHarness
)
# MILESTONE 2: You'll need this
# from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys, time, argparse, traceback


# ============================================================
# AGENT FUNCTIONS (GIVEN - these work individually)
# ============================================================
# Each agent takes (state, tracker) or (state, tracker, side) and returns state.
# Your job: wire them into a pipeline.

PLANNER_SYSTEM = """You are a Debate Planner. Set up a fair debate framework.
Return JSON:
{"topic_restated": "...", "for_position": "FOR argues (1 sentence)", "against_position": "AGAINST argues (1 sentence)", "dimensions": ["dim1", "dim2", "dim3"], "context_from_document": "..."}"""

DEBATER_SYSTEM_TEMPLATE = """You are arguing {side} the position: "{position}"
Rules: Use ONLY evidence from chunks. 3 arguments with point/evidence/reasoning. Preemptive rebuttal.
Return JSON:
{{"opening_statement": "...", "arguments": [{{"point": "...", "evidence": "...", "reasoning": "..."}}], "counter_to_opposition": "...", "closing_statement": "..."}}"""

CROSS_EXAM_SYSTEM = """You are a Cross-Examiner challenging the opposing argument.
Find the weakest point. Cite evidence that contradicts it. Raise unanswerable questions.
Return JSON:
{"weakest_point": "...", "challenge": "...", "unanswerable_questions": ["q1", "q2"], "additional_evidence": "..."}"""

JUDGE_SYSTEM = """You are an impartial Judge. Score both sides on 5 criteria (0-10 each):
Evidence Quality, Logical Coherence, Completeness, Persuasiveness, Honesty.
Return JSON:
{"for_score": {"evidence_quality": 8, ..., "total": 39}, "against_score": {"evidence_quality": 7, ..., "total": 38}, "winner": "for|against|tie", "margin": "decisive|narrow|razor-thin", "reasoning": "2-3 sentences", "strongest_point_for": "...", "strongest_point_against": "...", "weakest_point_for": "...", "weakest_point_against": "..."}"""

SYNTHESIZER_SYSTEM = """You are a Synthesizer. Balanced analysis beyond "both sides have merit."
150-200 words. Common ground + key tension + nuanced conclusion.
Return JSON: {"balanced_analysis": "...", "common_ground": ["..."], "key_tension": "...", "nuanced_conclusion": "..."}"""


def debate_planner(state, tracker):
    """Sets up debate framework. Works standalone."""
    sample_text = "\n---\n".join([c["text"] for c in state["chunks"][:6]])
    result = call_llm_cheap(system=PLANNER_SYSTEM,
                            prompt=f"Topic: {state['topic']}\n\nDocument:\n{sample_text}\n\nPlan the debate.",
                            json_output=True)
    tracker.record(result, "planner")
    state["debate_plan"] = parse_json(result["text"])
    log_agent(state, "planner", {"topic": state["topic"]}, state["debate_plan"],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Planner] FOR: {state['debate_plan'].get('for_position', '?')[:60]}")
    print(f"  [Planner] AGAINST: {state['debate_plan'].get('against_position', '?')[:60]}")
    return state


def researcher(state, tracker, side="for"):
    """Gathers evidence via FAISS search. No LLM. Works standalone."""
    plan = state["debate_plan"]
    topic = state["topic"]

    if side == "for":
        queries = [f"evidence supporting {plan['for_position']}",
                   f"benefits advantages {topic}", f"reasons why {topic}"]
    else:
        queries = [f"evidence against {topic}",
                   f"problems disadvantages {topic}", f"criticism risks {topic}"]

    for dim in plan.get("dimensions", [])[:2]:
        queries.append(f"{dim} {'benefits' if side == 'for' else 'problems'}")

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
    """Builds argument for one side. Works standalone."""
    plan = state["debate_plan"]
    evidence = state.get(f"evidence_{side}", [])
    evidence_text = "\n---\n".join([e["text"] for e in evidence])
    position = plan["for_position"] if side == "for" else plan["against_position"]

    system = DEBATER_SYSTEM_TEMPLATE.format(
        side="FOR" if side == "for" else "AGAINST", position=position)

    result = call_llm_strong(system=system,
                             prompt=f"Topic: {state['topic']}\nDimensions: {', '.join(plan.get('dimensions', []))}\n\nEvidence:\n{evidence_text}",
                             json_output=True)
    tracker.record(result, f"debater_{side}")
    state[f"argument_{side}"] = parse_json(result["text"])
    log_agent(state, f"debater_{side}", {"evidence": len(evidence)},
              f"{len(state[f'argument_{side}'].get('arguments', []))} args",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Debater {side.upper()}] {len(state[f'argument_{side}'].get('arguments', []))} arguments")
    return state


def cross_examiner(state, tracker, my_side="for"):
    """Cross-examines the opposing argument. Works standalone."""
    opposing = "against" if my_side == "for" else "for"
    opp_arg = state.get(f"argument_{opposing}", {})
    my_evidence = "\n---\n".join([e["text"] for e in state.get(f"evidence_{my_side}", [])])

    result = call_llm_strong(
        system=CROSS_EXAM_SYSTEM,
        prompt=f"You argue {my_side.upper()}.\n\nOpposing argument:\n{json.dumps(opp_arg, indent=2)}\n\nYour evidence:\n{my_evidence}",
        json_output=True)
    tracker.record(result, f"cross_exam_{my_side}")
    state[f"cross_exam_{my_side}"] = parse_json(result["text"])
    log_agent(state, f"cross_exam_{my_side}", {},
              state[f"cross_exam_{my_side}"].get("weakest_point", "?")[:50],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Cross-Exam {my_side.upper()}] Target: {state[f'cross_exam_{my_side}'].get('weakest_point', '?')[:50]}")
    return state


def judge(state, tracker, round_num=1):
    """Evaluates both arguments. Works standalone."""
    arg_for = state.get("argument_for", {})
    arg_against = state.get("argument_against", {})

    cross_ctx = ""
    if round_num > 1 and state.get("cross_exam_for"):
        cross_ctx = f"\n\nCROSS-EXAMINATION:\nFOR challenges: {json.dumps(state['cross_exam_for'])}\nAGAINST challenges: {json.dumps(state['cross_exam_against'])}\n\nFactor cross-exam into scoring."

    result = call_llm_strong(
        system=JUDGE_SYSTEM,
        prompt=f"Topic: {state['topic']}\nRound: {round_num}\n\nFOR:\n{json.dumps(arg_for, indent=2)}\n\nAGAINST:\n{json.dumps(arg_against, indent=2)}{cross_ctx}",
        json_output=True)
    tracker.record(result, f"judge_r{round_num}")
    state["judgment"] = parse_json(result["text"])
    log_agent(state, f"judge_r{round_num}", {"round": round_num},
              f"Winner: {state['judgment'].get('winner', '?')} ({state['judgment'].get('margin', '?')})",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})

    f_t = state["judgment"].get("for_score", {}).get("total", "?")
    a_t = state["judgment"].get("against_score", {}).get("total", "?")
    print(f"  [Judge R{round_num}] FOR: {f_t}/50 | AGAINST: {a_t}/50 | "
          f"Winner: {state['judgment'].get('winner', '?').upper()} ({state['judgment'].get('margin', '?')})")
    return state


def synthesizer(state, tracker):
    """Writes balanced synthesis. Works standalone."""
    j = state.get("judgment", {})
    af = state.get("argument_for", {})
    aa = state.get("argument_against", {})

    result = call_llm_strong(
        system=SYNTHESIZER_SYSTEM,
        prompt=f"Topic: {state['topic']}\nVerdict: {j.get('winner', '?')} ({j.get('margin', '?')})\n"
               f"FOR thesis: {af.get('opening_statement', '?')}\nAGAINST thesis: {aa.get('opening_statement', '?')}\n"
               f"Strongest FOR: {j.get('strongest_point_for', '?')}\nStrongest AGAINST: {j.get('strongest_point_against', '?')}",
        json_output=True)
    tracker.record(result, "synthesizer")
    state["synthesis"] = parse_json(result["text"])
    log_agent(state, "synthesizer", {}, state["synthesis"].get("nuanced_conclusion", "")[:80],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Synthesizer] Done")
    return state


# ============================================================
# MILESTONE 1: WIRE THE SEQUENTIAL PIPELINE
# ============================================================
# YOUR TASK:
#   Wire the agents into a complete debate pipeline.
#   For now, run everything SEQUENTIALLY (parallel comes in Milestone 2).
#
# PIPELINE ORDER:
#   1. Load PDF + build index
#   2. debate_planner
#   3. researcher (for side) then researcher (against side)
#   4. debater (for side) then debater (against side)
#   5. judge
#   6. synthesizer
#   7. Print results
#
# IMPORTANT: State setup requirements:
#   - state["topic"] must be set
#   - state["_index"] and state["_all_chunks"] must be set (used by researcher)
#   - state["chunks"] should have initial context chunks
#
# TEST: python debate_starter.py your_doc.pdf "Your debate topic"
#       You should see all agents execute and a winner declared.

def run_debate(pdf_path, topic, budget=0.80, max_rounds=2):
    tracker = CostTracker(budget=budget)

    print(f"\n{'=' * 60}")
    print(f"  MULTI-AGENT DEBATE SYSTEM")
    print(f"{'=' * 60}")
    print(f"  Topic: {topic}")
    print(f"  Budget: ${budget:.2f}")

    # --- YOUR CODE: Step 1 - Load document, build index, init state ---
    # HINT: state needs: topic, _index, _all_chunks, chunks (initial search results)
    print("\n[1] Loading document...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 2 - Run Planner ---
    print("\n[2] Planning debate...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 3 - Run both researchers ---
    # For Milestone 1, run sequentially: researcher(state, tracker, "for"),
    #                                     researcher(state, tracker, "against")
    print("\n[3] Gathering evidence...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 4 - Run both debaters ---
    # For Milestone 1, run sequentially.
    print("\n[4] Building arguments...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 5 - Run judge ---
    print("\n[5] Judging...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 6 - Run synthesizer ---
    print("\n[6] Synthesizing...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 7 - Print results ---
    # Print at minimum: winner, scores, synthesis conclusion
    print("\n[7] Results:")
    # YOUR CODE HERE


    print_log(state)
    tracker.report()
    return state


# ============================================================
# MILESTONE 2: PARALLEL EXECUTION
# ============================================================
# YOUR TASK:
#   The researchers are independent - run them in parallel.
#   The debaters are independent - run them in parallel.
#   This makes the pipeline ~2x faster.
#
# REQUIREMENTS:
#   1. Import ThreadPoolExecutor and as_completed
#   2. Build a run_parallel() helper that runs multiple (fn, side) pairs concurrently
#   3. Each parallel task gets its OWN copy of state: {**state}
#   4. Merge results: state["evidence_for"] = result["evidence_for"], etc.
#   5. Merge logs: state["log"].extend(result_state.get("log", []))
#   6. Error boundary: wrap future.result() in try/except, log failures
#
# HINT: The tricky part is merging state from parallel tasks. Each task modifies
#       a DIFFERENT key (evidence_for vs evidence_against), so no conflicts.
#
# TEST: Time your pipeline before and after. Should be noticeably faster.

def run_parallel(fns_with_args, state, tracker):
    """
    YOUR CODE: Run [(fn, side), (fn, side)] pairs in parallel.
    Return modified state with results merged.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# MILESTONE 3: CROSS-EXAMINATION ROUND
# ============================================================
# YOUR TASK:
#   If the judge's margin is "razor-thin" or "narrow", the debate is too close.
#   Trigger a cross-examination round where each side challenges the other.
#
# REQUIREMENTS:
#   1. After the first judge() call, check state["judgment"]["margin"]
#   2. If margin is "razor-thin" or "narrow":
#      a. Run cross_examiner(state, tracker, "for") and
#         cross_examiner(state, tracker, "against") in PARALLEL
#      b. Run judge(state, tracker, round_num=2)  - judge sees cross-exam data
#   3. Track state["rounds_played"]
#
# ENGINEERING QUESTION: Should you also check budget before triggering round 2?
#   Cross-examination costs ~2 strong LLM calls + 1 judge call = ~$0.08-0.10
#   If budget remaining < $0.15, skip round 2 even if margin is close.


# ============================================================
# MILESTONE 4: COST TRACKING + BUDGET-AWARE DECISIONS
# ============================================================
# YOUR TASK:
#   1. Add tracker.check_budget() between pipeline stages
#   2. Make round 2 decision budget-aware (check tracker.remaining())
#   3. Print tracker.report() at the end
#   4. Try: python debate_starter.py doc.pdf "topic" --budget 0.10
#      Does it stop gracefully when budget runs out?
#
# HINT: Wrap tracker.check_budget() in try/except RuntimeError
#       and print a clean message instead of crashing.


# ============================================================
# MILESTONE 5: TOURNAMENT MODE
# ============================================================
# YOUR TASK:
#   Run the debate system on 3 different topics using the SAME document.
#   Aggregate results: which side wins most often? What's the average score?
#
# REQUIREMENTS:
#   1. Accept multiple topics via command line or a topics.json file
#   2. Reuse the FAISS index across debates (don't rebuild for each topic)
#   3. Track per-topic results: winner, scores, margin, cost
#   4. Print tournament summary: wins per side, average scores, total cost
#
# EXAMPLE OUTPUT:
#   Tournament Results (3 debates):
#   Topic 1: "Python vs Java" -> FOR wins (39/50 vs 34/50) - narrow
#   Topic 2: "Monolith vs Microservices" -> AGAINST wins (41/50 vs 37/50) - decisive
#   Topic 3: "SQL vs NoSQL" -> FOR wins (38/50 vs 36/50) - razor-thin
#   Overall: FOR 2-1, Avg margin: 3.3 points, Total cost: $0.45

def run_tournament(pdf_path, topics, budget_per_topic=0.30):
    """
    YOUR CODE: Run multiple debates on same document.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# REPORT FORMATTING (GIVEN)
# ============================================================

def format_report(state):
    """Format debate results. Given - don't modify."""
    j = state.get("judgment", {})
    af = state.get("argument_for", {})
    aa = state.get("argument_against", {})
    s = state.get("synthesis", {})

    r = ["\n" + "=" * 60, "  DEBATE REPORT", "=" * 60]
    r.append(f"\n  Topic: {state.get('topic', '?')}")
    r.append(f"  Winner: {j.get('winner', '?').upper()} ({j.get('margin', '?')})")
    r.append(f"  FOR: {j.get('for_score', {}).get('total', '?')}/50")
    r.append(f"  AGAINST: {j.get('against_score', {}).get('total', '?')}/50")
    r.append(f"  Reason: {j.get('reasoning', '?')}")
    r.append(f"\n  FOR thesis: {af.get('opening_statement', '?')}")
    r.append(f"  AGAINST thesis: {aa.get('opening_statement', '?')}")
    if state.get("cross_exam_for"):
        r.append(f"\n  Cross-exam FOR challenged: {state['cross_exam_for'].get('weakest_point', '?')}")
        r.append(f"  Cross-exam AGAINST challenged: {state['cross_exam_against'].get('weakest_point', '?')}")
    r.append(f"\n  Synthesis: {s.get('balanced_analysis', '?')[:200]}")
    r.append(f"  Conclusion: {s.get('nuanced_conclusion', '?')}")
    r.append("=" * 60)
    return "\n".join(r)


# ============================================================
# CLI (GIVEN)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Debate System")
    parser.add_argument("pdf", help="Path to PDF document")
    parser.add_argument("topic", help="Debate topic (in quotes)")
    parser.add_argument("--budget", type=float, default=0.80, help="Budget in USD")
    parser.add_argument("--rounds", type=int, default=2, help="Max debate rounds")
    parser.add_argument("--tournament", nargs="+", help="Multiple topics for tournament mode")

    args = parser.parse_args()

    if args.tournament:
        run_tournament(args.pdf, args.tournament, budget_per_topic=args.budget / len(args.tournament))
    else:
        state = run_debate(args.pdf, args.topic, budget=args.budget, max_rounds=args.rounds)
        print(format_report(state))
