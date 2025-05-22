#!/usr/bin/env python3
"""
Generate metrics comparison table between Kalman baseline and GNN models.
"""

import json
from pathlib import Path
from typing import Dict


def load_results(kalman_path: str = "results/kalman_baseline.json", 
                gnn_path: str = "results/gnn_test.json"):
    """Load results from both Kalman and GNN evaluation files."""
    with open(kalman_path, 'r') as f:
        kalman_results = json.load(f)
    
    with open(gnn_path, 'r') as f:
        gnn_results = json.load(f)
    
    return kalman_results, gnn_results


def normalize_gnn_results(gnn_results: Dict[str, float]) -> Dict[str, float]:
    """Normalize GNN results to match Kalman format."""
    return {
        "AUC": gnn_results["AUC"],
        "Precision": gnn_results["Precision_at_Optimal"],
        "Recall": gnn_results["Recall_at_Optimal"]
    }


def calculate_gains(kalman: Dict[str, float], gnn: Dict[str, float]) -> Dict[str, float]:
    """Calculate performance gains from Kalman to GNN."""
    return {k: gnn[k] - kalman[k] for k in kalman}


def format_markdown_table(kalman, gnn, gains, gnn_threshold):
    """Format metrics as a Markdown table."""
    suspicious_flag = " 🚨" if gnn['AUC'] >= 0.999 else ""
    realistic_flag = " ✅" if kalman['AUC'] < 0.95 else ""
    
    md = f"| Model                | AUC   | Precision@τ | Recall@τ | Status |\n"
    md += f"| -------------------- | ----- | ----------- | -------- | ------ |\n"
    md += f"| Kalman               | {kalman['AUC']:.2f}{realistic_flag}  | {kalman['Precision']:.2f}        | {kalman['Recall']:.2f}     | Realistic |\n"
    md += f"| GNN (τ = {gnn_threshold:.3f}) | {gnn['AUC']:.2f}{suspicious_flag}  | {gnn['Precision']:.2f}        | {gnn['Recall']:.2f}     | Suspicious |\n"
    md += f"| **Δ**                | **{gains['AUC']:+.2f}** | **{gains['Precision']:+.2f}**     | **{gains['Recall']:+.2f}**   | ❓ Questionable |"
    return md


def format_latex_table(kalman, gnn, gains, gnn_threshold):
    """Format metrics as a LaTeX table."""
    latex = "\\begin{tabular}{l|c|c|c|c}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Model} & \\textbf{AUC} & \\textbf{Precision@τ} & \\textbf{Recall@τ} & \\textbf{Status} \\\\\n"
    latex += "\\hline\n"
    latex += f"Kalman & {kalman['AUC']:.2f} & {kalman['Precision']:.2f} & {kalman['Recall']:.2f} & Realistic \\\\\n"
    latex += f"GNN (τ = {gnn_threshold:.3f}) & {gnn['AUC']:.2f} & {gnn['Precision']:.2f} & {gnn['Recall']:.2f} & Suspicious \\\\\n"
    latex += f"\\textbf{{Δ}} & \\textbf{{{gains['AUC']:+.2f}}} & \\textbf{{{gains['Precision']:+.2f}}} & \\textbf{{{gains['Recall']:+.2f}}} & Questionable \\\\\n"
    latex += "\\hline\n"
    latex += "\\end{tabular}"
    return latex


def main():
    """Main function to generate metrics table."""
    print("⚠️  Loading evaluation results...")
    
    # Load results
    kalman_results, gnn_results = load_results()
    
    # Normalize GNN results to match Kalman format
    gnn_normalized = normalize_gnn_results(gnn_results)
    
    # Check for suspicious results
    if gnn_normalized["AUC"] >= 0.999:
        print("🚨 WARNING: GNN AUC = 1.00 is highly suspicious!")
        print("   This likely indicates:")
        print("   - Task is trivially solvable")
        print("   - Information leakage")
        print("   - Evaluation error")
        print("   See debug_perfect_performance.py for investigation steps.")
    
    # Calculate gains
    gains = calculate_gains(kalman_results, gnn_normalized)
    
    # Build comparison table
    table = {
        "Kalman": kalman_results,
        "GNN": gnn_normalized,
        "Gain": gains,
        "Warning": "GNN perfect performance is suspicious" if gnn_normalized["AUC"] >= 0.999 else None
    }
    
    # Save to JSON
    Path("results").mkdir(exist_ok=True)
    with open("results/metrics_table.json", 'w') as f:
        json.dump(table, f, indent=2)
    
    print("Saved metrics comparison to results/metrics_table.json")
    
    # Extract threshold for display
    gnn_threshold = gnn_results["Optimal_Threshold"]
    
    # Print formatted tables
    print("\n" + "="*60)
    print("MARKDOWN TABLE (ready to paste):")
    print("="*60)
    print(format_markdown_table(kalman_results, gnn_normalized, gains, gnn_threshold))
    
    print("\n" + "="*60)
    print("LATEX TABLE (ready to paste):")
    print("="*60)
    print(format_latex_table(kalman_results, gnn_normalized, gains, gnn_threshold))
    
    # Print honest summary
    print("\n" + "="*60)
    print("HONEST ASSESSMENT:")
    print("="*60)
    print(f"Kalman Baseline: AUC {kalman_results['AUC']:.2f} (✅ Realistic)")
    print(f"GNN Performance: AUC {gnn_normalized['AUC']:.2f} (🚨 Suspicious)")
    if gnn_normalized["AUC"] >= 0.999:
        print("RECOMMENDATION: Investigate before claiming GNN superiority!")


if __name__ == "__main__":
    main()
