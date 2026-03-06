"""
Document Q&A with Fact-Check - LIGHTWEIGHT STARTER CODE
=========================================================
A rate-limit-friendly alternative: only 3 agents, ~3 LLM calls per question.

WHAT'S GIVEN: Agent functions with prompts (they work individually).
YOUR JOB:     Build the pipeline that makes them answer questions accurately.

AGENTS (just 3!):
  1. Planner   (cheap LLM)  - Analyzes the question, generates smart search queries
  2. Answerer  (strong LLM) - Generates a comprehensive answer from retrieved chunks
  3. Verifier  (cheap LLM)  - Fact-checks the answer against source chunks

Milestone 1 (45 min): Wire sequential pipeline - planner -> answerer -> verifier
Milestone 2 (30 min): Multi-question mode - ask 3 questions, reuse FAISS index
Milestone 3 (30 min): Cost tracking + budget enforcement
Milestone 4 (30 min): Confidence-based retry - if verifier score < 0.7, retry with more chunks
Milestone 5 (30 min): Evaluation harness - test with 5 questions, measure quality

Usage:
  python qa_starter.py doc.pdf "What is the main argument of this paper?"
  python qa_starter.py doc.pdf "What methodology was used?" --budget 0.30
  python qa_starter.py doc.pdf --multi "Question 1?" "Question 2?" "Question 3?"
"""

from helpers import (
    load_and_chunk, build_index, search,
    call_llm_cheap, call_llm_strong,
    parse_json, init_state, log_agent, print_log,
    CostTracker, EvalHarness
)
import json, sys, time, argparse, traceback


# ============================================================
# AGENT FUNCTIONS (GIVEN - these work individually)
# ============================================================
# Each agent takes (state, tracker) and returns state.
# Only 3 agents = only 3 LLM calls per question!

PLANNER_SYSTEM = """You are a Question Planner agent.
Analyze the user's question and generate 3 targeted search queries
to find the most relevant information in the document.

Return JSON:
{
    "question_type": "factual | analytical | comparative | opinion",
    "search_queries": ["query1", "query2", "query3"],
    "what_to_look_for": "one sentence describing what a good answer needs"
}"""

ANSWERER_SYSTEM = """You are an Answerer agent. Generate a clear, accurate answer
using ONLY the provided document chunks. Do not use any prior knowledge.

Rules:
- Every claim must be traceable to a chunk
- If information is insufficient, say so explicitly
- Be concise but thorough (3-5 sentences)
- Include specific details, numbers, or quotes when available

Return JSON:
{
    "answer": "Your 3-5 sentence answer here",
    "confidence": "high | medium | low",
    "key_evidence": ["evidence1 from chunks", "evidence2 from chunks"],
    "limitations": "what the document doesn't cover (if any)"
}"""

VERIFIER_SYSTEM = """You are a Verifier agent (fact-checker). Check the answer
against the source chunks. Be STRICT - only confirm claims that are directly
supported by the chunks.

Score each dimension 0-10:
- Accuracy: Are claims supported by the chunks?
- Completeness: Does the answer cover the key info in the chunks?
- Faithfulness: Does the answer avoid adding unsupported info?

Return JSON:
{
    "accuracy_score": 8,
    "completeness_score": 7,
    "faithfulness_score": 9,
    "overall_score": 0.80,
    "verdict": "pass | fail",
    "issues": ["any specific problems found"],
    "suggestion": "one sentence on how to improve (if needed)"
}"""


def planner(state, tracker):
    """Analyzes question and generates search queries. Works standalone."""
    # Include retry hint if available (for Milestone 4)
    retry_hint = ""
    if state.get("_verifier_feedback"):
        retry_hint = f"\n\nPrevious answer was weak. Verifier said: {state['_verifier_feedback']}. Generate DIFFERENT search queries to find better evidence."

    result = call_llm_cheap(
        system=PLANNER_SYSTEM,
        prompt=f"Question: {state['question']}{retry_hint}\n\nGenerate search queries.",
        json_output=True)
    tracker.record(result, "planner")
    state["plan"] = parse_json(result["text"])
    log_agent(state, "planner", {"question": state["question"]}, state["plan"],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Planner] Type: {state['plan'].get('question_type', '?')} | "
          f"Queries: {len(state['plan'].get('search_queries', []))}")
    return state


def answerer(state, tracker):
    """Generates answer from retrieved chunks. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state.get("retrieved_chunks", [])])
    plan = state.get("plan", {})

    result = call_llm_strong(
        system=ANSWERER_SYSTEM,
        prompt=f"Question: {state['question']}\n"
               f"What to look for: {plan.get('what_to_look_for', 'relevant information')}\n\n"
               f"Document chunks:\n{chunks_text}",
        json_output=True)
    tracker.record(result, "answerer")
    state["answer"] = parse_json(result["text"])
    log_agent(state, "answerer", {"chunks": len(state.get("retrieved_chunks", []))},
              state["answer"].get("answer", "")[:80],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Answerer] Confidence: {state['answer'].get('confidence', '?')} | "
          f"Evidence: {len(state['answer'].get('key_evidence', []))} pieces")
    return state


def verifier(state, tracker):
    """Fact-checks the answer against source chunks. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state.get("retrieved_chunks", [])[:8]])
    answer_data = state.get("answer", {})

    result = call_llm_cheap(
        system=VERIFIER_SYSTEM,
        prompt=f"Question: {state['question']}\n\n"
               f"Answer to verify:\n{json.dumps(answer_data, indent=2)}\n\n"
               f"Source chunks:\n{chunks_text}",
        json_output=True)
    tracker.record(result, "verifier")
    parsed = parse_json(result["text"])
    state["verification"] = parsed
    state["verifier_score"] = parsed.get("overall_score", 0)
    state["_verifier_feedback"] = parsed.get("suggestion", "")
    log_agent(state, "verifier", {"answer_length": len(answer_data.get("answer", ""))},
              f"Score={state['verifier_score']}, {parsed.get('verdict', '?')}",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Verifier] Score: {state['verifier_score']}/1.0 - "
          f"{parsed.get('verdict', '?').upper()}")
    return state


# ============================================================
# MILESTONE 1: WIRE THE SEQUENTIAL PIPELINE
# ============================================================
# YOUR TASK:
#   Wire the 3 agents into a complete Q&A pipeline.
#   This is simpler than the other projects - just 3 agents!
#
# PIPELINE ORDER:
#   1. Load PDF + build index
#   2. planner  - analyze question, get search queries
#   3. RETRIEVE - use search queries to find relevant chunks (no LLM!)
#   4. answerer - generate answer from chunks
#   5. verifier - fact-check the answer
#   6. Print results
#
# RETRIEVE STEP (between planner and answerer):
#   The planner gives you search queries in state["plan"]["search_queries"].
#   Loop through them, call search() for each, combine + deduplicate results.
#   Store in state["retrieved_chunks"].
#   This step uses NO LLM calls - just FAISS vector search!
#
# TEST: python qa_starter.py your_doc.pdf "Your question here"
#       You should see all 3 agents execute and an answer printed.

def run_qa(pdf_path, question, budget=0.30, max_retries=2):
    tracker = CostTracker(budget=budget)

    print(f"\n{'=' * 60}")
    print(f"  DOCUMENT Q&A WITH FACT-CHECK")
    print(f"{'=' * 60}")
    print(f"  Question: {question}")
    print(f"  Budget: ${budget:.2f}")

    # --- YOUR CODE: Step 1 - Load document, build index, init state ---
    # HINT: chunks = load_and_chunk(pdf_path)
    # HINT: index = build_index(chunks)
    # HINT: state = init_state(query=question)
    # HINT: state["question"] = question
    # HINT: state["_index"] = index
    # HINT: state["_all_chunks"] = chunks
    print("\n[1] Loading document...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 2 - Run Planner ---
    print("\n[2] Planning search strategy...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 3 - Retrieve chunks using planner's queries ---
    # HINT: Loop through state["plan"]["search_queries"]
    # HINT: For each query, call search(state["_index"], state["_all_chunks"], q, k=5)
    # HINT: Combine all results, deduplicate by chunk index
    # HINT: Store in state["retrieved_chunks"]
    # NOTE: This step makes NO LLM calls! Only FAISS search.
    print("\n[3] Retrieving evidence...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 4 - Run Answerer ---
    print("\n[4] Generating answer...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 5 - Run Verifier ---
    print("\n[5] Fact-checking...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 6 - Print results ---
    # Print at minimum: the answer, confidence, verifier score, any issues
    print("\n[6] Results:")
    # YOUR CODE HERE


    print_log(state)
    tracker.report()
    return state


# ============================================================
# MILESTONE 2: MULTI-QUESTION MODE
# ============================================================
# YOUR TASK:
#   Ask multiple questions about the SAME document without
#   rebuilding the FAISS index each time. This saves time and
#   embedding API calls.
#
# REQUIREMENTS:
#   1. Load document and build index ONCE
#   2. Loop through questions, running the pipeline for each
#   3. Reuse the same index and chunks across questions
#   4. Collect results and print a summary table
#
# HINT: Factor out the "load + index" step from run_qa()
#       and pass index + chunks to a run_single_qa() function.
#
# TEST: python qa_starter.py doc.pdf --multi "Q1?" "Q2?" "Q3?"
#       Should see 3 answers, but document loaded only once.

def run_multi_qa(pdf_path, questions, budget=0.50):
    """
    YOUR CODE: Run multiple questions on same document.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# MILESTONE 3: COST TRACKING + BUDGET ENFORCEMENT
# ============================================================
# YOUR TASK:
#   1. Add tracker.check_budget() between pipeline stages
#   2. Wrap in try/except RuntimeError for clean budget-exceeded message
#   3. Print tracker.report() at the end
#   4. Try: python qa_starter.py doc.pdf "question" --budget 0.02
#      Does it stop gracefully?
#
# NOTE: With only 3 agents, this pipeline is very cheap!
#   Typical cost: ~$0.01-0.03 per question with Gemini Flash.
#   Compare: Debate system costs ~$0.10-0.20 per debate.


# ============================================================
# MILESTONE 4: CONFIDENCE-BASED RETRY
# ============================================================
# YOUR TASK:
#   When the Verifier scores < 0.7, don't accept the answer.
#   Feed the Verifier's feedback back to the Planner to generate
#   better search queries, retrieve new chunks, and try again.
#
# REQUIREMENTS:
#   1. After verifier, check state["verifier_score"]
#   2. If score < 0.7 and retries remaining and budget allows:
#      a. state["_verifier_feedback"] already has the suggestion
#         (planner() already reads this - check its code!)
#      b. Re-run: planner (with feedback) -> retrieve -> answerer -> verifier
#   3. Track retry_count in state
#   4. Budget check before each retry (3 more LLM calls = ~$0.01-0.03)
#
# ENGINEERING QUESTION: With only 3 LLM calls per retry,
#   retrying is much cheaper than in the other projects!
#   But when should you NOT retry?
#   - Score is 0.0 (question unanswerable from document)?
#   - Already retried max_retries times?
#   - Budget too low?


# ============================================================
# MILESTONE 5: EVALUATION HARNESS
# ============================================================
# YOUR TASK:
#   Build an evaluation mode that tests your pipeline on 5 questions.
#
# REQUIREMENTS:
#   1. Create an EvalHarness with 5 test cases for your document.
#      Each test case: question, expected_keywords, difficulty.
#   2. Write a pipeline_for_eval(question) function that:
#      - Reuses the cached index (don't rebuild!)
#      - Runs planner -> retrieve -> answerer -> verifier
#      - Returns {"answer": state["answer"]["answer"],
#                 "critic_score": state["verifier_score"]}
#   3. Run harness.run(pipeline_for_eval) and harness.report()
#
# NOTE: 5 questions x 3 LLM calls = only 15 API calls total!
#   Compare: Intelligence Hub eval = 5 x 6 = 30 calls.
#
# TEST: python qa_starter.py your_doc.pdf --eval
#       Target: >60% pass rate.

def run_evaluation(pdf_path):
    """
    YOUR CODE: Build and run evaluation harness.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# REPORT FORMATTING (GIVEN)
# ============================================================

def format_report(state):
    """Format Q&A results. Given - don't modify."""
    a = state.get("answer", {})
    v = state.get("verification", {})
    p = state.get("plan", {})

    r = ["\n" + "=" * 60, "  Q&A REPORT", "=" * 60]
    r.append(f"\n  Question: {state.get('question', '?')}")
    r.append(f"  Type: {p.get('question_type', '?')}")
    r.append(f"\n  Answer: {a.get('answer', '?')}")
    r.append(f"  Confidence: {a.get('confidence', '?')}")
    r.append(f"\n  Evidence:")
    for i, ev in enumerate(a.get("key_evidence", []), 1):
        r.append(f"    {i}. {ev[:100]}")
    if a.get("limitations"):
        r.append(f"\n  Limitations: {a['limitations']}")
    r.append(f"\n  Verification:")
    r.append(f"    Accuracy:     {v.get('accuracy_score', '?')}/10")
    r.append(f"    Completeness: {v.get('completeness_score', '?')}/10")
    r.append(f"    Faithfulness: {v.get('faithfulness_score', '?')}/10")
    r.append(f"    Overall:      {v.get('overall_score', '?')}/1.0 - {v.get('verdict', '?').upper()}")
    if v.get("issues"):
        r.append(f"    Issues: {', '.join(v['issues'])}")
    if state.get("retry_count", 0) > 0:
        r.append(f"\n  Retries: {state['retry_count']}")
    r.append("=" * 60)
    return "\n".join(r)


# ============================================================
# CLI (GIVEN)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Q&A with Fact-Check")
    parser.add_argument("pdf", help="Path to PDF document")
    parser.add_argument("question", nargs="?", default=None,
                        help="Question to ask (in quotes)")
    parser.add_argument("--budget", type=float, default=0.30,
                        help="Budget in USD (default: $0.30)")
    parser.add_argument("--retries", type=int, default=2,
                        help="Max retries on low confidence (default: 2)")
    parser.add_argument("--multi", nargs="+",
                        help="Multiple questions for multi-question mode")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation harness")

    args = parser.parse_args()

    if args.multi:
        run_multi_qa(args.pdf, args.multi, budget=args.budget)
    elif args.eval:
        print("\n\nRUNNING EVALUATION...")
        run_evaluation(args.pdf)
    elif args.question:
        state = run_qa(args.pdf, args.question,
                       budget=args.budget, max_retries=args.retries)
        print(format_report(state))
    else:
        parser.print_help()