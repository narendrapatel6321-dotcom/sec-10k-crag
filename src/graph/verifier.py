"""
Numerical Verifier

Extracts numeric values from the LLM's generated response and cross-references 
them against the structured XBRL Financial Statements (Income Statement, Balance Sheet, Cash Flow).
If numbers mismatch due to hallucination or unit errors (e.g., millions vs thousands), 
it returns structured feedback for the rewrite node.
"""

import re
import pandas as pd
from typing import Tuple, List
from pathlib import Path

def load_xbrl_tables(ticker: str, accession: str, xbrl_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Loads the 3 primary financial statements for a specific filing."""
    base_path = Path(xbrl_dir) / f"{ticker}_{accession}"
    
    tables = {}
    for stmt in ["income_statement", "balance_sheet", "cash_flow"]:
        file_path = base_path / f"{stmt}.csv"
        if file_path.exists():
            tables[stmt] = pd.read_csv(file_path)
        else:
            tables[stmt] = pd.DataFrame()
            
    return tables

def extract_numbers_from_text(text: str) -> List[float]:
    """Extracts raw numerical values from a text string, handling commas and decimals."""
    # Matches numbers like 1,000.50 or 5000000
    pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        clean_num = match.replace(',', '')
        try:
            numbers.append(float(clean_num))
        except ValueError:
            continue
    return numbers

def verify_numbers(
    generation: str, 
    ticker: str, 
    accession: str, 
    xbrl_dir: str | Path,
    tolerance: float = 0.05
) -> Tuple[bool, str]:
    """
    Checks if numbers in the generated text exist within the company's XBRL statements.
    Tolerates minor float discrepancies based on the tolerance parameter.
    """
    generated_numbers = extract_numbers_from_text(generation)
    
    # If no numbers were generated, skip verification and pass
    if not generated_numbers:
        return True, "No numerical values detected to verify."
        
    tables = load_xbrl_tables(ticker, accession, xbrl_dir)
    
    # Flatten all numerical values from all tables into a single list
    ground_truth_numbers = []
    for stmt, df in tables.items():
        # Only select numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        ground_truth_numbers.extend(numeric_df.values.flatten().tolist())
        
    # Drop NaNs
    ground_truth_numbers = [n for n in ground_truth_numbers if pd.notna(n)]
    
    if not ground_truth_numbers:
        return True, "No ground truth numeric data available to verify against."
        
    failed_numbers = []
    
    for gen_num in generated_numbers:
        # Skip small integers (e.g., years or bullet points) that might trigger false positives
        if gen_num < 3000 and gen_num.is_integer():
            continue
            
        # Check if the generated number exists within the tolerance of ANY ground truth number
        match_found = False
        for gt_num in ground_truth_numbers:
            # Handle division by zero
            if gt_num == 0:
                if abs(gen_num) <= tolerance:
                    match_found = True
                    break
            else:
                diff = abs(gen_num - gt_num) / abs(gt_num)
                if diff <= tolerance:
                    match_found = True
                    break
                    
        # Also check against common scale discrepancies (millions/thousands)
        if not match_found:
            for scale in [1e3, 1e6, 1e9, 1e-3, 1e-6]:
                scaled_gen = gen_num * scale
                for gt_num in ground_truth_numbers:
                    if gt_num != 0 and (abs(scaled_gen - gt_num) / abs(gt_num)) <= tolerance:
                        match_found = True
                        break
                if match_found:
                    break
                    
        if not match_found:
            failed_numbers.append(gen_num)
            
    if failed_numbers:
        feedback = f"Verification failed. The following generated numbers could not be grounded in the XBRL financial statements: {failed_numbers}. Please correct scale (e.g., millions vs thousands) or remove unverified metrics."
        return False, feedback
        
    return True, "All generated numerical values successfully verified against XBRL ground truth."
