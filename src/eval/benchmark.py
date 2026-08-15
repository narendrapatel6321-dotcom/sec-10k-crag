"""
SEC 10-K RAG Automated Evaluation Benchmark

Evaluates the Corrective RAG (CRAG) pipeline against a curated golden test suite
measuring routing accuracy, retrieval relevance, numerical precision, and factual faithfulness.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.graph.workflow import crag_app
from src.graph.verifier import extract_numbers_from_text


# --- Evaluation Schemas ---

class FaithfulnessJudge(BaseModel):
    """LLM-as-a-Judge schema for evaluating faithfulness to context."""
    is_faithful: bool = Field(
        ...,
        description="True if every claim and number in the answer is directly supported by the context, False if there are hallucinations or contradictions."
    )
    score: float = Field(
        ...,
        description="Faithfulness score from 0.0 (completely unsupported) to 1.0 (fully grounded)."
    )
    reasoning: str = Field(
        ...,
        description="Concise justification for the score."
    )


# --- Benchmark Golden Dataset ---

DEFAULT_BENCHMARK_CASES = [
    {
        "id": "TC-01",
        "question": "What are the primary operational and regulatory risk factors disclosed by Citigroup?",
        "expected_ticker": "C",
        "expected_category": "sec_filing",
        "expected_section": "risk_factors",
        "expected_numbers": [],
        "type": "qualitative_risk"
    },
    {
        "id": "TC-02",
        "question": "Discuss Apple's supply chain concentration and single-source component risks.",
        "expected_ticker": "AAPL",
        "expected_category": "sec_filing",
        "expected_section": "risk_factors",
        "expected_numbers": [],
        "type": "qualitative_risk"
    },
    {
        "id": "TC-03",
        "question": "What is Microsoft's MD&A commentary regarding cloud segment revenue and AI investments?",
        "expected_ticker": "MSFT",
        "expected_category": "sec_filing",
        "expected_section": "mdna",
        "expected_numbers": [],
        "type": "qualitative_mdna"
    },
    {
        "id": "TC-04",
        "question": "How is the Common Equity Tier 1 (CET1) ratio calculated under Basel III regulations?",
        "expected_ticker": None,
        "expected_category": "general_financial",
        "expected_section": None,
        "expected_numbers": [],
        "type": "general_finance"
    },
    {
        "id": "TC-05",
        "question": "Can you write Python code to train a Convolutional Neural Network on CIFAR-10?",
        "expected_ticker": None,
        "expected_category": "out_of_scope",
        "expected_section": None,
        "expected_numbers": [],
        "type": "out_of_scope"
    },
    {
        # NOTE: fill in a real figure pulled from the target filing's XBRL income
        # statement (e.g. total revenue for the most recent fiscal year) before
        # running this case for real. Left as a clearly-labeled placeholder here
        # since the actual value depends on which filing snapshot is indexed.
        "id": "TC-06",
        "question": "What was Apple's total net sales for the most recent fiscal year in the filing?",
        "expected_ticker": "AAPL",
        "expected_category": "sec_filing",
        "expected_section": "mdna",
        "expected_numbers": [391035],  # REPLACE with the real XBRL ground-truth figure (millions USD)
        "type": "quantitative_mdna"
    },
    {
        # Same caveat as TC-06 — replace with the real CET1 ratio disclosed in
        # Citigroup's indexed 10-K before treating this as a passing benchmark.
        "id": "TC-07",
        "question": "What was Citigroup's Common Equity Tier 1 (CET1) capital ratio disclosed in the filing?",
        "expected_ticker": "C",
        "expected_category": "sec_filing",
        "expected_section": "mdna",
        "expected_numbers": [13.4],  # REPLACE with the real XBRL/filing ground-truth figure (%)
        "type": "quantitative_mdna"
    }
]


# --- Evaluator Functions ---

def evaluate_faithfulness(question: str, context: str, answer: str, llm: ChatGroq) -> FaithfulnessJudge:
    """Evaluates factual grounding of the generated answer against the retrieved context."""
    if not context.strip():
        return FaithfulnessJudge(is_faithful=True, score=1.0, reasoning="No context required for query route.")

    structured_llm = llm.with_structured_output(FaithfulnessJudge)
    
    system = """You are an impartial financial compliance auditor.
Assess whether the provided answer is strictly faithful to and fully supported by the retrieved context.
Penalize fabricated numbers, unsubstantiated claims, or outside extrapolation."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nGenerated Answer:\n{answer}")
    ])

    judge_chain = prompt | structured_llm
    return judge_chain.invoke({"question": question, "context": context, "answer": answer})


def evaluate_numerical_accuracy(generation: str, expected_numbers: List[float], tolerance: float = 0.05) -> float:
    """Computes precision against expected ground-truth numerical targets."""
    if not expected_numbers:
        return 1.0

    generated_numbers = extract_numbers_from_text(generation)
    if not generated_numbers:
        return 0.0

    matches = 0
    for exp in expected_numbers:
        for gen in generated_numbers:
            if exp == 0:
                if abs(gen) <= tolerance:
                    matches += 1
                    break
            elif abs(gen - exp) / abs(exp) <= tolerance:
                matches += 1
                break

    return matches / len(expected_numbers)


# --- Main Benchmark Execution ---

def run_benchmark(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = "benchmark_results.csv"
) -> pd.DataFrame:
    """
    Executes the automated benchmark across test cases and logs performance metrics.

    Example:
        >>> from src.eval.benchmark import run_benchmark
        >>> results_df = run_benchmark()

    Args:
        test_cases: Optional custom list of test case dictionaries.
        output_path: File path to export results CSV.

    Returns:
        A pandas DataFrame summarizing evaluation metrics.
    """
    cases = test_cases or DEFAULT_BENCHMARK_CASES
    judge_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    results = []

    print("=================================================================")
    print(f"Starting Benchmark Evaluation ({len(cases)} test cases)...")
    print("=================================================================\n")

    for case in cases:
        case_id = case["id"]
        question = case["question"]
        print(f"[{case_id}] Evaluating: '{question[:60]}...'")

        start_time = time.time()

        initial_state = {
            "question": question,
            "route_category": None,
            "ticker": None,
            "target_section": None,
            "documents": [],
            "generation": "",
            "verification_feedback": "",
            "rewrite_count": 0,
            "verify_count": 0,
            "retry_count": 0
        }

        final_state = crag_app.invoke(initial_state)
        latency = round(time.time() - start_time, 2)

        # 1. Routing Evaluation
        route_passed = True
        if case["expected_ticker"] and final_state.get("ticker") != case["expected_ticker"]:
            route_passed = False
        if case["expected_category"] and final_state.get("route_category") != case["expected_category"]:
            route_passed = False

        # 2. Retrieval Evaluation
        retrieved_docs = final_state.get("documents", [])
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        retrieval_count = len(retrieved_docs)

        # 3. Numerical Accuracy
        generation = final_state.get("generation", "")
        num_acc = evaluate_numerical_accuracy(generation, case["expected_numbers"])

        # 4. LLM-as-a-Judge Faithfulness
        judge_result = evaluate_faithfulness(question, context_text, generation, judge_llm)

        record = {
            "id": case_id,
            "type": case["type"],
            "question": question,
            "route_passed": route_passed,
            "predicted_ticker": final_state.get("ticker"),
            "predicted_category": final_state.get("route_category"),
            "retrieved_chunks": retrieval_count,
            "numeric_accuracy": num_acc,
            "faithfulness_score": judge_result.score,
            "is_faithful": judge_result.is_faithful,
            "judge_reasoning": judge_result.reasoning,
            "latency_seconds": latency,
            "rewrite_retries": final_state.get("rewrite_count", 0),
            "verify_retries": final_state.get("verify_count", 0),
            "generation": generation
        }

        results.append(record)
        print(f"  -> Done in {latency}s | Route: {'PASS' if route_passed else 'FAIL'} | Faithfulness: {judge_result.score} | Numeric: {num_acc}")

    df = pd.DataFrame(results)

    print("\n=================================================================")
    print("Benchmark Evaluation Summary")
    print("=================================================================")
    print(f"Total Test Cases:       {len(df)}")
    print(f"Routing Accuracy:       {(df['route_passed'].mean() * 100):.1f}%")
    print(f"Mean Faithfulness:      {(df['faithfulness_score'].mean() * 100):.1f}%")
    print(f"Mean Numerical Acc:     {(df['numeric_accuracy'].mean() * 100):.1f}%")
    print(f"Average Latency:        {df['latency_seconds'].mean():.2f}s")
    print("=================================================================\n")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Full benchmark results exported to: {output_path}")

    return df
