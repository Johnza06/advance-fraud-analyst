# Advanced Fraud Analyst

## What it does
This project demonstrates a fraud analysis assistant powered by large language models and external tools. It inspects transactions for anomalies, aggregates threat intelligence, and explains risk scores for investigators.

## Stack diagram
```
[User] -> [FastAPI] -> [LLM Provider] -> [Tools]
                                     |-> Threat Intel API
                                     |-> Validation Module
```

## Quickstart
```bash
make up  # or
docker compose up --build
```

## Demo
A 60–90s demo GIF or Loom video should be placed here to showcase basic usage.

## Eval results
| metric      | accuracy | groundedness | latency (ms) | cost/query | cache hit rate |
| ----------- | -------- | ------------ | ------------ | ---------- | -------------- |
| example run | 0.92     | 0.95         | 850          | $0.002    | 80%            |

## Safety
* Handles PII via mode-switching and redaction.
* Includes jailbreak and prompt-injection tests.

## Limits & next steps
Current evaluations are synthetic. Real datasets, richer adversarial prompts, and continuous monitoring are needed for production readiness.

## Metrics & speed
See [metrics/fastapi_metrics.png](metrics/fastapi_metrics.png) for p50/p95 latency, cost per query, and cache hit rate screenshots.

## Commit signal
Ship small daily. Open issues with labels (`bug`, `feature`, `eval`) and close them with PRs tied to metrics improvements.
