"""LLM Supervisor — output quality monitoring with automatic retry-with-context feedback.

When LLM JSON responses fail to parse or miss required keys, the supervisor
constructs a feedback prompt describing what went wrong and retries up to
MAX_RETRIES times. All corrections are logged to a supervisor audit log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from quantlab.config import DATA_DIR

logger = logging.getLogger(__name__)

SUPERVISOR_LOG_DIR = DATA_DIR / "scheduler" / "experiments"
SUPERVISOR_LOG_PATH = SUPERVISOR_LOG_DIR / "supervisor_log.json"


class LLMSupervisor:
    """Monitors LLM JSON output quality and retries with feedback on failure.

    When `chat_json()` returns a dict missing required keys or has a parse error,
    the supervisor constructs a specific feedback prompt describing the problem
    and retries the LLM call. After MAX_RETRIES attempts, it returns a failure
    sentinel so downstream code can fall back.

    Usage::

        supervisor = LLMSupervisor(llm_client, max_retries=2)
        result = supervisor.supervise(
            original_prompt=user_prompt,
            raw_response=raw_text,
            expected_schema_keys=["hypotheses"],
        )
        if result["status"] == "supervised_ok":
            parsed = result["corrected"]
    """

    def __init__(self, llm_client: Any, max_retries: int = 2) -> None:
        """Args:
            llm_client: An LLMClient instance (must expose .chat(system, user) -> str).
            max_retries: Maximum retry attempts before giving up.
        """
        self.llm = llm_client
        self.max_retries = max_retries

    def supervise(
        self,
        original_prompt: str,
        raw_response: str,
        expected_schema_keys: list[str],
    ) -> dict[str, Any]:
        """Attempt to repair a malformed or incomplete LLM JSON response.

        Args:
            original_prompt: The user prompt originally sent to the LLM.
            raw_response: The raw text the LLM returned.
            expected_schema_keys: Keys that must exist in the parsed JSON dict.

        Returns:
            dict with keys:
              - status: "supervised_ok" | "supervisor_failed"
              - corrected: parsed JSON dict (only on success)
              - retries_used: number of retries attempted
              - error: error description (only on failure)
              - last_response: raw last response text (only on failure)
        """
        # Try to parse the raw response first
        parsed = self._attempt_parse(raw_response)
        missing = self._find_missing_keys(parsed, expected_schema_keys)

        if isinstance(parsed, dict) and not missing and not parsed.get("parse_error"):
            # Already valid — no supervision needed
            return {"status": "supervised_ok", "corrected": parsed, "retries_used": 0}

        retries_used = 0
        last_response = raw_response

        for attempt in range(1, self.max_retries + 1):
            retries_used = attempt
            feedback_prompt = self._build_feedback_prompt(
                original_prompt, parsed, missing, last_response
            )

            logger.info(
                "Supervisor retry %d/%d: missing_keys=%s",
                attempt, self.max_retries, missing or ["(parse_failed)"],
            )

            try:
                last_response = self.llm.chat(
                    system_prompt=(
                        "You are a JSON correction assistant. "
                        "Your ONLY task is to fix the previous malformed or incomplete "
                        "JSON response and return COMPLETE, VALID JSON. "
                        "Do NOT add commentary, markdown fences, or explanations — "
                        "output ONLY the corrected JSON object."
                    ),
                    user_prompt=feedback_prompt,
                    temperature=0.1,
                )
            except Exception as exc:
                logger.warning("Supervisor LLM call failed: %s", exc)
                self._log_correction(original_prompt, expected_schema_keys, retries_used, False, str(exc))
                return {
                    "status": "supervisor_failed",
                    "error": f"LLM call failed during retry: {exc}",
                    "last_response": last_response if last_response != raw_response else raw_response,
                    "retries_used": retries_used,
                }

            parsed = self._attempt_parse(last_response)
            missing = self._find_missing_keys(parsed, expected_schema_keys)

            if isinstance(parsed, dict) and not missing and not parsed.get("parse_error"):
                self._log_correction(original_prompt, expected_schema_keys, retries_used, True)
                return {"status": "supervised_ok", "corrected": parsed, "retries_used": retries_used}

        # Exhausted retries
        self._log_correction(original_prompt, expected_schema_keys, retries_used, False, "max_retries_exhausted")
        return {
            "status": "supervisor_failed",
            "error": f"Max retries ({self.max_retries}) exhausted. Missing keys: {missing}",
            "last_response": last_response,
            "retries_used": retries_used,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attempt_parse(text: str) -> Any:
        """Attempt to parse JSON from LLM output, with markdown-fence stripping."""
        if not text:
            return {"parse_error": True, "raw": ""}
        t = text.strip()
        # Strip markdown code fences
        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in t:
            t = t.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            # Try to find the outermost { ... }
            start = t.find("{")
            end = t.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(t[start:end])
                except json.JSONDecodeError:
                    pass
            return {"parse_error": True, "raw": text[:500]}

    @staticmethod
    def _find_missing_keys(parsed: Any, expected: list[str]) -> list[str]:
        """Return keys from *expected* that are missing in *parsed*."""
        if not isinstance(parsed, dict) or parsed.get("parse_error"):
            return expected  # can't verify without a valid dict
        return [k for k in expected if k not in parsed]

    def _build_feedback_prompt(
        self,
        original_prompt: str,
        parsed: Any,
        missing_keys: list[str],
        last_response: str,
    ) -> str:
        """Construct a feedback prompt telling the LLM what went wrong."""
        parts = [
            "Your previous response had issues. Please fix and return ONLY valid JSON.",
            "",
            "=== ORIGINAL REQUEST ===",
            original_prompt,
            "",
            "=== YOUR PREVIOUS RESPONSE ===",
            last_response[:2000],
        ]

        if not isinstance(parsed, dict) or parsed.get("parse_error"):
            parts.append("")
            parts.append("=== PROBLEM ===")
            parts.append(
                "Your response was not valid JSON. "
                "Make sure to return a complete JSON object with proper syntax "
                "(commas between fields, double-quoted strings, no trailing commas)."
            )
        elif missing_keys:
            parts.append("")
            parts.append("=== PROBLEM ===")
            parts.append(
                f"Your JSON is valid but missing required keys: {', '.join(missing_keys)}. "
                f"Please include ALL of these keys in your response."
            )

        parts.append("")
        parts.append("Return ONLY the corrected JSON object — no markdown, no explanation.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    @staticmethod
    def _log_correction(
        prompt: str,
        expected_keys: list[str],
        retries_used: int,
        success: bool,
        error: str = "",
    ) -> None:
        """Append an entry to the supervisor audit log (JSON-lines file)."""
        try:
            SUPERVISOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "retries_used": retries_used,
                "expected_keys": expected_keys,
                "error": error,
                "prompt_snippet": prompt[:300],
            }
            line = json.dumps(entry, ensure_ascii=False)
            with open(SUPERVISOR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.debug("Failed to write supervisor log: %s", exc)
