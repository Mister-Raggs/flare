# Positioning Statement

Use verbatim or adapt for LinkedIn, cover notes, or cold outreach.

---

I built Flare — an open-source, end-to-end log anomaly detection system that combines classical ML with LLM-powered incident summarization. It uses Isolation Forest on Drain3-extracted log template features to detect anomalies in HDFS log data, clusters them into incidents, then sends each to Claude Sonnet for a structured explanation with severity assessment, root cause analysis, and actionable remediation steps. What makes it non-trivial is the evaluation layer: a second LLM call acts as a judge, scoring each explanation on relevance, specificity, and actionability — making the system's quality measurable without human labeling. The full pipeline runs through a FastAPI REST API with a dark-themed dashboard, Docker deployment, and 72 automated tests in CI. I wrote a detailed technical post on the architecture decisions and what I'd change with more time. Take a look at the repo (github.com/mister-raggs/flare) or the blog post — I'd welcome feedback from anyone working on observability tooling or LLM evaluation.
