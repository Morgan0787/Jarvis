"""
Message analyzer for Jarvis v2 Core (Step 4).

Flow:
- Load unanalyzed rows from `processed_messages`
- For each message, call the local Ollama model with a strict JSON prompt
- Validate and clean the JSON
- Store results back into `processed_messages`

We use `short_text` as the main input to keep prompts small.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from ..core.config import get_config

from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from ..core.logger import get_logger
from ..core.utils import get_project_root
from ..storage.repository import Repository


logger = get_logger(__name__)


ALLOWED_CATEGORIES = {
    "startup",
    "funding",
    "grant",
    "accelerator",
    "hackathon",
    "job",
    "event",
    "ecosystem_news",
    "other",
}


@dataclass
class AnalysisStats:
    """Summary of an analysis run."""

    analyzed_count: int
    failed_count: int


class MessageAnalyzer:
    """
    MessageAnalyzer sends processed messages to Ollama and stores
    structured analysis back into the database.
    """

    def __init__(self, batch_limit: int = 10) -> None:
        self.repo = Repository()
        cfg = get_config()
        configured_provider = str(getattr(cfg.ai, "provider", "ollama") or "ollama").strip().lower()

        if configured_provider == "ollama":
            provider = "ollama"
            self.client = OllamaClient()
        elif configured_provider == "gemini":
            provider = "gemini"
            self.client = GeminiClient()
        elif configured_provider == "openai":
            provider = "openai"
            self.client = OpenAIClient()
        else:
            provider = "ollama"
            logger.warning(
                "Unknown AI provider '%s'; falling back to Ollama.",
                configured_provider,
            )
            self.client = OllamaClient()

        logger.info("Using AI provider: %s", provider)
        self.provider = provider
        self.batch_limit = batch_limit

        # Load prompt template once.
        project_root = get_project_root()
        prompt_path = project_root / "prompts" / "classify_prompt.txt"
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def _default_analysis(self) -> Dict[str, Any]:
        """
        Return a safe default analysis object.

        Used when the model output is missing, invalid, or malformed.
        """
        return {
            "category": "other",
            "importance_score": 1,
            "actionability_score": 1,
            "priority_score": 1,
            "is_relevant": False,
            "summary": "",
            "why_it_matters": "",
        }

    def _clean_summary(self, summary: str, fallback_text: str = "") -> str:
        """
        Post-process model summary to make it shorter and more factual.

        Returns a Russian one-sentence summary (<= 15 words) or falls back
        to `fallback_text` if the cleaned summary is empty/too weak.
        """
        fillers = [
            "стоит отметить",
            "важная новость",
            "очередной шаг",
            "открывает прием заявок",
            "объявлен окончательный прием заявок",
            "предлагает возможность",
            "в рамках",
        ]

        def _normalize(text: str) -> str:
            text = (text or "").strip()
            text = text.replace("…", ".")
            text = re.sub(r"\s+", " ", text)
            # Remove generic filler/bureaucratic phrases.
            for f in fillers:
                text = re.sub(rf"\b{re.escape(f)}\b", "", text, flags=re.IGNORECASE)
            # Remove repeated punctuation like "!!" or "??" or "..."
            text = re.sub(r"([.!?])\1+", r"\1", text)
            # Collapse any spaces left after removals.
            text = re.sub(r"\s+", " ", text).strip()
            # Trim leftover leading/trailing punctuation.
            text = text.strip(" \t\r\n.,-–—;:!")
            text = text.strip()
            return text

        def _one_sentence_and_short(text: str) -> str:
            text = _normalize(text)
            if not text:
                return ""
            # Keep only the first sentence.
            m = re.search(r"[.!?]", text)
            if m:
                text = text[: m.end()].strip()
            words = text.split()
            if len(words) > 15:
                text = " ".join(words[:15]).strip()
            # If trimming removed punctuation, keep it as one sentence anyway.
            return text

        cleaned = _one_sentence_and_short(summary)
        cleaned_words = len(cleaned.split()) if cleaned else 0

        # Too weak: empty or extremely short.
        if not cleaned or cleaned_words < 3:
            fb = _one_sentence_and_short(fallback_text)
            fb_words = len(fb.split()) if fb else 0
            if fb and fb_words >= 3:
                return fb
            # Last resort: return the best we have (may still be empty).
            return cleaned or fb or ""

        return cleaned

    def _extract_json_object(self, text: str) -> Dict[str, Any] | None:
        """
        Try to extract a JSON object from the model output.

        Handles common issues:
        - Markdown fences like ```json ... ```
        - Extra text before/after the JSON
        """
        if not text:
            return None

        cleaned = text.strip()

        # Remove leading/trailing markdown fences if present.
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                first = lines[0].strip().lower()
                if first in {"```json", "```"}:
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

        # First attempt: parse the cleaned text directly.
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        # Fallback: find the first '{' and last '}' and try parsing that substring.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = cleaned[start : end + 1].strip()
        try:
            parsed2 = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        return parsed2 if isinstance(parsed2, dict) else None

    def _build_prompt(self, message_text: str) -> str:
        """
        Fill the classify prompt template with the message text.
        """
        return self.prompt_template.replace("{{MESSAGE}}", message_text)

    def _validate_and_normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the JSON dict from Ollama.

        If some fields are missing or invalid, we fall back to safe defaults.
        """
        # Start from default analysis and selectively override fields.
        result = self._default_analysis()

        # category
        cat = str(data.get("category", "")).strip().lower()
        if cat in ALLOWED_CATEGORIES:
            result["category"] = cat

        # importance_score
        try:
            imp = int(data.get("importance_score", 1))
        except (TypeError, ValueError):
            imp = 1
        result["importance_score"] = max(1, min(10, imp))

        # actionability_score
        try:
            act = int(data.get("actionability_score", 1))
        except (TypeError, ValueError):
            act = 1
        result["actionability_score"] = max(1, min(10, act))

        # priority_score
        try:
            pr = int(data.get("priority_score", 1))
        except (TypeError, ValueError):
            pr = 1
        result["priority_score"] = max(1, min(10, pr))

        # is_relevant (safe parsing so "false" is not treated as True)
        is_rel_value = data.get("is_relevant", False)
        if isinstance(is_rel_value, bool):
            parsed_bool = is_rel_value
        elif isinstance(is_rel_value, str):
            parsed_bool = is_rel_value.strip().lower() in {"true", "1", "yes"}
        else:
            parsed_bool = False
        result["is_relevant"] = parsed_bool

        # why_it_matters
        why = data.get("why_it_matters", "")
        if not isinstance(why, str):
            why = ""

        # summary
        summary = data.get("summary", "")
        if not isinstance(summary, str):
            summary = ""

        result["summary"] = self._clean_summary(summary, fallback_text=why)
        result["why_it_matters"] = why

        return result

    def analyze(self) -> AnalysisStats:
        """
        Run one analysis pass using Ollama.
        """
        messages = self.repo.get_unanalyzed_processed_messages(limit=self.batch_limit)
        if not messages:
            logger.info("No unanalyzed processed messages found.")
            return AnalysisStats(analyzed_count=0, failed_count=0)

        logger.info(
            "Selected %d newest unanalyzed messages for analysis with %s...", len(messages), self.provider
        )

        analyzed = 0
        failed = 0

        for msg in messages:
            try:
                prompt = self._build_prompt(msg.short_text or msg.cleaned_text)
                raw_response = self.client.generate(prompt)
                
                logger.info("Raw AI response for message_id=%s: %r", msg.id, raw_response[:500] if raw_response else "EMPTY")
                
                parsed = self._extract_json_object(raw_response or "")
                
                if parsed is None:
                    logger.warning(
                        "Invalid/empty JSON from AI for processed_message_id=%s: %r",
                        msg.id,
                        raw_response,
                    )
                    normalized = self._default_analysis()
                    metadata_json = json.dumps(normalized, ensure_ascii=False)
                    
                    logger.info("Using default analysis for message_id=%s: %s", msg.id, metadata_json[:200])
                    
                    self.repo.update_processed_message_analysis(
                        processed_message_id=int(msg.id),
                        classification=str(normalized["category"]),
                        importance_score=float(normalized["importance_score"]),
                        metadata_json=metadata_json,
                    )
                    logger.info("Saved default analysis to DB for message_id=%s", msg.id)
                    analyzed += 1
                    failed += 1
                    continue

                logger.info("Parsed JSON for message_id=%s: %s", msg.id, str(parsed)[:200])
                normalized = self._validate_and_normalize(parsed)
                
                # Store full validated JSON string in metadata_json.
                metadata_json = json.dumps(normalized, ensure_ascii=False)
                
                logger.info("Final normalized metadata for message_id=%s: %s", msg.id, metadata_json[:200])
                logger.info("Saving analysis to DB for message_id=%s", msg.id)

                self.repo.update_processed_message_analysis(
                    processed_message_id=int(msg.id),
                    classification=str(normalized["category"]),
                    importance_score=float(normalized["importance_score"]),
                    metadata_json=metadata_json,
                )
                
                logger.info("Successfully saved analysis to DB for message_id=%s", msg.id)
                analyzed += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to analyze processed_message_id=%s: %s",
                    getattr(msg, "id", None),
                    exc,
                )
                failed += 1

        logger.info(
            "Analysis finished. Messages analyzed: %d, failures: %d",
            analyzed,
            failed,
        )
        return AnalysisStats(analyzed_count=analyzed, failed_count=failed)

