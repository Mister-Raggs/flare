"""Prompt templates for LLM incident summarization.

All prompt strings are defined here as constants.
No inline prompt strings should appear elsewhere in the codebase.
"""

SUMMARIZE_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) analyzing log anomalies \
from a Hadoop Distributed File System (HDFS) cluster. You will be given a set \
of anomalous log lines from an incident, along with their extracted templates \
and anomaly scores.

Your task is to analyze these logs and produce a structured JSON response.

You MUST respond with valid JSON matching this exact schema:
{
  "incident_id": <integer>,
  "explanation": "<plain-English explanation of what happened>",
  "severity": "<low|medium|high|critical>",
  "severity_reasoning": "<why this severity>",
  "remediation": [
    {"action": "<what to do>", "priority": "<immediate|short-term|long-term>"}
  ],
  "root_cause": "<likely root cause>",
  "confidence": <float 0.0-1.0>,
  "confidence_reasoning": "<why this confidence level>"
}

Severity guidelines:
- low: Informational anomaly, no service impact expected
- medium: Potential issue, may cause degraded performance
- high: Active issue, likely causing failures or data loss risk
- critical: Severe outage, immediate data loss or service unavailability

Confidence guidelines:
- 1.0: Very clear pattern, obvious root cause
- 0.7-0.9: Strong signal, likely correct but some ambiguity
- 0.4-0.6: Moderate signal, multiple possible explanations
- 0.1-0.3: Weak signal, speculative analysis"""

SUMMARIZE_USER_PROMPT = """\
Analyze the following incident and provide your assessment.

Think step by step:
1. First, identify what log patterns are present
2. Determine what normal HDFS operations look like vs what's anomalous here
3. Identify the likely root cause
4. Assess severity based on potential impact
5. Suggest concrete remediation steps

## Incident #{incident_id}

**Time Range:** {time_range_start} to {time_range_end}
**Blocks Affected:** {block_ids}
**Mean Anomaly Score:** {mean_anomaly_score:.4f}
**Numerical Severity:** {severity:.4f}

### Log Lines ({log_line_count} total)
{log_lines}

### Extracted Templates
{templates}

Respond with the JSON object only, no markdown fences or additional text."""

QUALITY_EVAL_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of an AI-generated incident \
analysis. You will be given the original log data and an LLM's analysis, and \
you must score it on three dimensions.

You MUST respond with valid JSON matching this exact schema:
{
  "relevance": <integer 1-5>,
  "specificity": <integer 1-5>,
  "actionability": <integer 1-5>,
  "reasoning": "<your reasoning>"
}

Scoring rubric:
- relevance (1-5): Does the explanation accurately describe what the logs show?
  1=completely wrong, 3=partially correct, 5=perfectly matches the evidence
- specificity (1-5): Is the explanation specific to THIS incident?
  1=generic boilerplate, 3=somewhat specific, 5=deeply specific to the logs
- actionability (1-5): Are the remediation steps concrete and useful?
  1=vague platitudes, 3=reasonable but generic, 5=specific actionable steps"""

QUALITY_EVAL_USER_PROMPT = """\
## Original Log Lines
{log_lines}

## LLM Analysis to Evaluate
**Explanation:** {explanation}
**Severity:** {severity}
**Root Cause:** {root_cause}
**Remediation Steps:**
{remediation_steps}

Score this analysis. Respond with the JSON object only."""
