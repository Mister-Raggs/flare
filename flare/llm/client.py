"""Anthropic API client with retry logic and rate limiting."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from dotenv import load_dotenv

from flare.llm.schemas import UsageStats

load_dotenv()

logger = logging.getLogger(__name__)

# Claude Sonnet pricing (per million tokens) as of 2025
_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0


@dataclass
class LLMResponse:
    """Raw response from the Anthropic API.

    Attributes:
        content: Parsed JSON content from the response.
        raw_text: Raw text of the response.
        usage: Token usage and cost statistics.
    """

    content: dict
    raw_text: str
    usage: UsageStats


class AnthropicClient:
    """Client for the Anthropic API with retry logic and cost tracking.

    Loads ANTHROPIC_API_KEY from environment variables. Implements
    exponential backoff for transient failures and rate limit errors.

    Example:
        >>> client = AnthropicClient()
        >>> response = client.complete(system="You are helpful.", user="Hello")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        api_key: str | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            model: Anthropic model ID to use.
            max_tokens: Maximum tokens in the response.
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay in seconds for exponential backoff.
            api_key: API key. If None, reads from ANTHROPIC_API_KEY env var.

        Raises:
            ValueError: If no API key is found.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your environment or .env file."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a completion request with retry logic.

        Args:
            system: System prompt.
            user: User message.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            LLMResponse with parsed JSON content and usage stats.

        Raises:
            anthropic.APIError: If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.monotonic()

                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

                latency_ms = (time.monotonic() - start_time) * 1000

                first_block = response.content[0]
                raw_text = first_block.text if hasattr(first_block, "text") else str(first_block)
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                usage = UsageStats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=round(latency_ms, 1),
                    estimated_cost_usd=self._estimate_cost(
                        input_tokens, output_tokens
                    ),
                )

                content = self._parse_json(raw_text)

                return LLMResponse(content=content, raw_text=raw_text, usage=usage)

            except anthropic.RateLimitError as e:
                last_error = e
                delay = self.base_delay * (2**attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)

            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    delay = self.base_delay * (2**attempt)
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %.1fs",
                        e.status_code,
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [line for line in lines[1:] if line.strip() != "```"]
            cleaned = "\n".join(lines)
        result: dict[Any, Any] = json.loads(cleaned)
        return result

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost based on token counts."""
        input_cost = (input_tokens / 1_000_000) * _INPUT_COST_PER_M
        output_cost = (output_tokens / 1_000_000) * _OUTPUT_COST_PER_M
        return round(input_cost + output_cost, 6)
