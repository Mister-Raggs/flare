# Flare

**LLM-powered log anomaly detection and incident summarization**

> [Read the blog post: Building Flare](https://mister-raggs.github.io/ai/building-flare-llm-powered-incident-detection/)

Flare ingests raw log data, detects anomalies using Isolation Forest, clusters related signals into incidents, and summarizes each in plain English with severity assessment and remediation steps — all accessible via a REST API and dark-themed dashboard.

## Quickstart

```bash
git clone https://github.com/mister-raggs/flare
cd flare
cp .env.example .env   # add your Anthropic API key
docker-compose up
# open http://localhost:8000/dashboard
```

Paste logs into the dashboard or hit the API directly at `http://localhost:8000/docs`.

> **Without Docker:** `pip install -e ".[all]"` then `uvicorn flare.api.main:app --reload`

## Why Flare?

On-call engineers drown in log volume during incidents. Keyword-based alerting either misses subtle anomalies or floods you with false positives. Flare takes a different approach: it uses statistical ML to identify *which* log blocks are anomalous, then uses an LLM to explain *why* in terms an engineer can act on. The result is a system that doesn't just detect — it triages, explains, and suggests next steps, with an eval harness that scores its own output.

## Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Flare                                                   v0.1.0  ● ok │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Paste log lines here...                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  [ Analyze Logs ]  [ Upload .log ]  ☐ Run quality eval                │
├──────────────────────┬──────────────────────────────────────────────────┤
│  INCIDENTS (3)       │  INCIDENT DETAIL                                │
│                      │                                                 │
│  ┌────────────────┐  │  Severity  Anomaly Score  Confidence            │
│  │ ■ HIGH   Inc 0 │◄─│  ┌──────┐ ┌──────────┐   ┌─────────┐          │
│  │ 1 block        │  │  │ HIGH │ │ -0.2310  │   │   85%   │          │
│  │ 5 log lines    │  │  └──────┘ └──────────┘   └─────────┘          │
│  └────────────────┘  │                                                 │
│  ┌────────────────┐  │  EXPLANATION                                    │
│  │ ■ MED   Inc 1  │  │  Block transfer failed due to connection        │
│  │ 1 block        │  │  reset. The DataXceiver thread encountered      │
│  │ 3 log lines    │  │  IOException while receiving block data...      │
│  └────────────────┘  │                                                 │
│  ┌────────────────┐  │  ROOT CAUSE                                     │
│  │ ■ MED   Inc 2  │  │  Network instability on the remote DataNode     │
│  │ 1 block        │  │  caused TCP connection reset during transfer.   │
│  │ 4 log lines    │  │                                                 │
│  └────────────────┘  │  REMEDIATION                                    │
│                      │  [immediate] Check network connectivity         │
│                      │  [immediate] Verify block replication factor    │
│                      │  [short-term] Review DataNode heap/threads      │
│                      │                                                 │
│                      │  ▸ Show 5 raw log lines                         │
├──────────────────────┴──────────────────────────────────────────────────┤
│ Incidents: 3  Critical: 0  High: 1  Medium: 2  Low: 0   42ms  $0.004 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │                    Flare                       │
                    ├────────────────────────────────────────────────┤
                    │                                                │
  Raw Logs ──────►  │  ingestion/     Log parsing & Drain3           │
  (text/file)       │       │         templating                     │
                    │       ▼                                        │
                    │  detection/     Isolation Forest                │
                    │       │         anomaly scoring                 │
                    │       ▼                                        │
                    │  clustering/    DBSCAN incident                 │
                    │       │         grouping + enrichment           │
                    │       ▼                                        │
                    │  llm/           Claude Sonnet                   │
                    │       │         summarization & eval            │
                    │       ▼                                        │
                    │  eval/          Precision / Recall / F1         │
                    │                 + LLM quality rubric            │
                    ├────────────────────────────────────────────────┤
                    │  api/           FastAPI REST layer              │
                    │  ├─ POST /detect     ← log text → incidents    │
                    │  ├─ POST /summarize  ← incidents → summaries   │
                    │  ├─ POST /analyze    ← log text → everything   │
                    │  └─ GET /health      ← status check            │
                    │                                                │
                    │  dashboard/     Single-file HTML + vanilla JS   │
                    │  cli/           Click + Rich CLI                │
                    └────────────────────────────────────────────────┘
```

## Pipeline

1. **Ingestion** — Parse raw HDFS logs with regex, apply [Drain3](https://github.com/logpai/Drain3) template mining to extract parameterized log templates
2. **Detection** — Build per-block feature vectors from template frequency distributions, run Isolation Forest to score anomalies
3. **Clustering** — Group anomalous blocks into incidents using DBSCAN on normalized feature vectors, enrich with log lines, templates, and time ranges
4. **LLM Summarization** — Send each incident to Claude Sonnet for plain-English explanation, severity assessment, root cause analysis, and remediation steps
5. **Evaluation** — Classical: precision/recall/F1 against ground truth. LLM: quality rubric scoring (relevance, specificity, actionability) via LLM-as-judge, plus cost/latency tracking

## API Reference

All endpoints are documented with examples at `http://localhost:8000/docs` (Swagger UI).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect` | Parse logs + detect anomalies + cluster incidents |
| `POST` | `/detect/upload` | Same, but accepts a file upload |
| `POST` | `/summarize` | LLM summarization of detected incidents |
| `POST` | `/analyze` | End-to-end: detect + summarize in one call |
| `GET`  | `/health` | API status, Anthropic connectivity, version |

### Example: Full pipeline

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"log_text": "<paste logs here>", "run_eval": false}'
```

## Benchmark Results

### Classical Detection — HDFS Dataset (LogHub)

Evaluated on the full public [HDFS log dataset](https://github.com/logpai/loghub) — 11.2M lines, 575,061 blocks, 16,838 anomalies (2.9% anomaly rate).

| Method           | Precision | Recall | F1    | TP     | FP    | FN    | Notes                   |
|------------------|-----------|--------|-------|--------|-------|-------|-------------------------|
| Isolation Forest | 0.688     | 0.601  | 0.642 | 10,119 | 4,590 | 6,719 | contamination=0.029     |

Features: per-block Drain3 template frequency vectors (47 templates learned across full dataset). Detection degrades on small slices (<50K blocks) due to insufficient template diversity — the model needs enough blocks to learn a meaningful normal distribution.

<details>
<summary>Micro-sample (86 lines, 18 blocks) — for unit test reference only</summary>

| Method           | Precision | Recall | F1    | Notes                |
|------------------|-----------|--------|-------|----------------------|
| Isolation Forest | 1.0000    | 1.0000 | 1.000 | contamination=0.15   |

</details>

### End-to-End Latency

| Stage | Time | Notes |
|-------|------|-------|
| Ingestion + Detection + Clustering | ~40ms | CPU-bound, no API calls |
| LLM Summarization (per incident) | ~1-3s | Claude Sonnet, temp=0.0 |
| Full pipeline (detect + summarize) | ~4-10s | Depends on incident count |

### LLM Quality Evaluation

Scored via LLM-as-judge (Claude Sonnet evaluating its own output on a 1–5 rubric). Run on the HDFS sample — 1 incident, 663 input / 380 output tokens, $0.0077.

| Metric          | Score | Notes                                                   |
|-----------------|-------|---------------------------------------------------------|
| Relevance       | 5/5   | Does the explanation match the log evidence?            |
| Specificity     | 5/5   | Is it specific to this incident, not generic?           |
| Actionability   | 4/5   | Are remediation steps concrete and useful?              |
| **Mean Quality**| **4.67/5** | Aggregate across all three dimensions              |

> Scores are generated by LLM-as-judge — see [the blog post](https://mister-raggs.github.io/ai/building-flare-llm-powered-incident-detection/) for methodology. Run your own eval with `flare summarize --input results.json --eval`.

## Project Structure

```
flare/
├── ingestion/        # Log parsing, Drain3 templating, structured events
│   ├── models.py     # LogEvent, ParsedLogBatch data models
│   └── parser.py     # LogParser with Drain3 template mining
├── detection/        # Classical anomaly detection
│   └── detector.py   # Isolation Forest on template frequency features
├── clustering/       # Incident grouping
│   └── clusterer.py  # DBSCAN clustering + enrichment with log context
├── eval/             # Benchmark framework
│   └── benchmark.py  # Classical metrics + LLM quality rubric + cost tracking
├── llm/              # LLM-assisted summarization
│   ├── client.py     # Anthropic API client with retry & rate limiting
│   ├── prompts.py    # All prompt templates
│   ├── schemas.py    # Pydantic models: LLMSummary, QualityScore, etc.
│   └── summarizer.py # Incident → LLMSummary pipeline
├── api/              # FastAPI REST layer
│   ├── main.py       # App, lifespan, CORS, exception handling
│   ├── models.py     # Pydantic request/response models
│   ├── deps.py       # Shared settings & dependencies
│   └── routes/       # Endpoint handlers
│       ├── health.py
│       ├── detect.py
│       └── summarize.py
├── cli/              # CLI entrypoint
│   └── main.py       # Click + Rich: detect & summarize commands
dashboard/
└── index.html        # Single-file dark-themed dashboard (no build step)
```

## Development

```bash
# Install with all dependencies
pip install -e ".[all]"

# Run tests (72 tests, no API calls — LLM tests use mocks)
pytest

# Run linter
ruff check flare/ tests/

# Type check
mypy flare/ --ignore-missing-imports

# Run API locally (with hot reload)
uvicorn flare.api.main:app --reload

# Run with Docker
docker-compose up --build
```

## Limitations & Future Work

- **Single log format.** Flare currently supports HDFS logs only. Supporting syslog, JSON-structured logs, or OpenTelemetry traces would require additional parsers but no architectural changes.
- **Bag-of-templates features.** The detection model uses template frequency histograms. It doesn't capture temporal patterns (event ordering, time deltas between templates) that would catch slow-burn anomalies.
- **DBSCAN clustering.** Grouping anomalous blocks by feature similarity is a rough heuristic. A vector store with log sequence embeddings would capture semantic similarity and enable "show me similar past incidents."
- **No streaming.** LLM responses block until complete. Streaming partial JSON (severity badge first, then explanation, then remediation) would improve dashboard responsiveness.
- **LLM-as-judge calibration.** The quality eval rubric hasn't been calibrated against human scores. A golden dataset of 50-100 human-evaluated explanations would quantify judge accuracy.
- **Static pipeline.** A production version would consume from Fluent Bit or an OTEL collector, maintain persistent template state across restarts, and push summaries to PagerDuty or Slack.

## Tech Stack

| Component        | Tool                                      |
|------------------|-------------------------------------------|
| Log parsing      | Drain3                                    |
| ML detection     | scikit-learn (Isolation Forest, DBSCAN)   |
| LLM              | Anthropic Claude Sonnet via `anthropic`   |
| Data validation  | Pydantic                                  |
| API              | FastAPI + Uvicorn                         |
| Dashboard        | Vanilla HTML/CSS/JS (no build step)       |
| CLI              | Click + Rich                              |
| Testing          | pytest (72 tests, all mocked for CI)      |
| Linting          | ruff                                      |
| CI               | GitHub Actions                            |
| Deployment       | Docker + docker-compose                   |

## License

MIT
