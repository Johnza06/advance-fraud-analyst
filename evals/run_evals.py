import json
from datetime import datetime
from pathlib import Path

# Placeholder evaluation functions
def evaluate_groundedness():
    return {"metric": "groundedness", "score": 0.95}

def evaluate_hallucination():
    return {"metric": "hallucination", "score": 0.05}

def evaluate_adversarial():
    return {
        "metric": "adversarial",
        "prompt_injection": 0.9,
        "jailbreak": 0.85,
        "toxic_input": 0.88,
    }

def evaluate_task_success():
    return {"metric": "task_success", "score": 0.92}

def main():
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "evaluations": [
            evaluate_groundedness(),
            evaluate_hallucination(),
            evaluate_adversarial(),
            evaluate_task_success(),
        ],
    }
    
    out_dir = Path(__file__).parent
    json_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    # simple HTML report
    rows = []
    for ev in results["evaluations"]:
        if ev["metric"] == "adversarial":
            rows.append(f"<tr><td>{ev['metric']}</td><td>prompt_injection: {ev['prompt_injection']}</td><td>jailbreak: {ev['jailbreak']}</td><td>toxic_input: {ev['toxic_input']}</td></tr>")
        else:
            rows.append(f"<tr><td>{ev['metric']}</td><td colspan='3'>{ev['score']}</td></tr>")
    
    html_content = f"""
<html>
<body>
<h1>Evaluation Report</h1>
<table border='1'>
<tr><th>Metric</th><th colspan='3'>Score</th></tr>
{''.join(rows)}
</table>
</body>
</html>
"""
    
    with html_path.open("w") as f:
        f.write(html_content)
    
    print(f"Wrote {json_path} and {html_path}")

if __name__ == "__main__":
    main()
