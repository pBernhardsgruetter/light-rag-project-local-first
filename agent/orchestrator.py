import asyncio
import json
import os
from pathlib import Path
import yaml
from openai import AsyncOpenAI

class GraphRAGOrchestrator:
    def __init__(self, config_path: str = "config/extractor.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.max_iterations = 20
        self.target_score = 0.75
        api_key = os.getenv("OPENROUTER_API_KEY") or self.cfg.get("openrouter_key", "")
        self.llm = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        ) if api_key else None

    async def check_validation_report(self) -> dict:
        report_path = Path("/output/validation_report.json")
        if not report_path.exists():
            # Local fallback path check
            report_path = Path("output/validation_report.json")

        if report_path.exists():
            try:
                return json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"avg_score": 0.0, "status": "UNKNOWN"}

    async def run_iteration_loop(self):
        print("Starting GraphRAG Orchestrator Agent loop...")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n==========================================")
            print(f"  ORCHESTRATOR ITERATION {iteration}/{self.max_iterations}")
            print(f"==========================================")

            report = await self.check_validation_report()
            avg_score = report.get("avg_score")
            status = report.get("status", "FAIL")

            print(f"Current Validation Status: {status} (Score: {avg_score})")

            if isinstance(avg_score, (int, float)) and avg_score >= self.target_score:
                print("SUCCESS: Target retrieval score achieved! System optimized.")
                break

            if status == "NO_JUDGE":
                print("Validation cannot pass without a configured judge model.")
                break

            print("Retrieval score below target. Analyzing bottlenecks and applying auto-fixes...")
            # Run extraction pipeline step if needed
            from services.extractor.extractor.pipeline import run_pipeline
            await run_pipeline(self.config_path)

            await asyncio.sleep(10)

def main():
    orchestrator = GraphRAGOrchestrator()
    asyncio.run(orchestrator.run_iteration_loop())

if __name__ == "__main__":
    main()
