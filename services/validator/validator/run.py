import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
import requests
import yaml
from openai import AsyncOpenAI

class GraphRAGValidator:
    def __init__(self, config: dict):
        self.cfg = config
        self.lightrag_endpoint = config.get("lightrag_endpoint", "http://lightrag:9621")
        self.test_dataset = Path(config.get("test_dataset", "/data/test_qa.jsonl"))
        self.report_output = Path(config.get("report_output", "/output/validation_report.json"))
        self.report_output.parent.mkdir(parents=True, exist_ok=True)
        self.target_score = config.get("target_score", 0.75)

        api_key = os.getenv("OPENROUTER_API_KEY") or config.get("openrouter_api_key", "")
        self.llm = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        ) if api_key else None

    def load_test_cases(self) -> list:
        if not self.test_dataset.exists():
            print(f"Test dataset not found at {self.test_dataset}")
            return []
        cases = []
        with open(self.test_dataset, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
        return cases

    async def llm_judge(self, question: str, expected: str, actual: str):
        if not self.llm:
            return None

        prompt = f"""Bewerte die Qualität der erhaltenen Antwort im Vergleich zur erwarteten Antwort auf einer Skala von 0.0 bis 1.0.
Frage: {question}
Erwartet: {expected}
Erhalten: {actual}

Antworte NUR mit einer Zahl zwischen 0.0 und 1.0."""
        try:
            resp = await self.llm.chat.completions.create(
                model=self.cfg.get("judge_model", "nvidia/nemotron-3-ultra-550b-a55b:free"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            score_str = resp.choices[0].message.content.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"LLM Judge evaluation failed: {e}")
            return None

    async def validate_retrieval(self) -> dict:
        cases = self.load_test_cases()
        if not cases:
            return {"status": "NO_TEST_CASES", "avg_score": 0.0, "details": []}

        results = []
        for tc in cases:
            q = tc["question"]
            expected = tc["expected"]
            try:
                res = requests.post(
                    f"{self.lightrag_endpoint}/query",
                    json={
                        "query": q,
                        "mode": tc.get("mode", "hybrid"),
                        "top_k": tc.get("top_k", 10),
                    },
                    timeout=60,
                )
                if res.status_code == 200:
                    payload = res.json()
                    actual = payload.get("result", "")
                else:
                    payload = {}
                    actual = f"Error: HTTP {res.status_code}"
            except Exception as ex:
                payload = {}
                actual = f"Connection Error: {ex}"

            score = await self.llm_judge(q, expected, actual)
            results.append({
                "question": q,
                "expected": expected,
                "actual": actual,
                "score": score,
                "citations": payload.get("citations", []),
                "nodes": payload.get("nodes", []),
                "edges": payload.get("edges", []),
            })

        scored = [r["score"] for r in results if r["score"] is not None]
        avg_score = sum(scored) / len(scored) if scored else None
        if not results:
            status = "NO_TEST_CASES"
        elif not scored:
            status = "NO_JUDGE"
        else:
            status = "PASS" if avg_score >= self.target_score else "FAIL"

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "avg_score": round(avg_score, 4) if avg_score is not None else None,
            "target_score": self.target_score,
            "status": status,
            "details": results
        }

        self.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

async def main():
    parser = argparse.ArgumentParser(description="GraphRAG Validator")
    parser.add_argument("--config", type=str, default="config/validator.yaml", help="Path to config file")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds for continuous mode")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validator = GraphRAGValidator(config)

    if args.continuous:
        print(f"Starting continuous validation loop (interval: {args.interval}s)...")
        while True:
            rep = await validator.validate_retrieval()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Validation result: {rep['status']} (Avg Score: {rep['avg_score']})")
            await asyncio.sleep(args.interval)
    else:
        rep = await validator.validate_retrieval()
        print(json.dumps(rep, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
