"""
Deterministic digest builder for Jarvis v2 Core (Step 5).

Important:
- NO new Ollama calls here.
- We use the analysis already stored in `processed_messages.metadata_json`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.config import get_config
from ..core.logger import get_logger
from ..storage.repository import Repository


logger = get_logger(__name__)


SECTION_ORDER = ["OPPORTUNITIES", "TOP NEWS", "JOBS", "EVENTS"]

SECTION_MAPPING = {
    "grant": "OPPORTUNITIES",
    "accelerator": "OPPORTUNITIES",
    "hackathon": "OPPORTUNITIES",
    "startup": "TOP NEWS",
    "funding": "TOP NEWS",
    "ecosystem_news": "TOP NEWS",
    "job": "JOBS",
    "event": "EVENTS",
}


@dataclass
class DigestItem:
    processed_message_id: int
    category: str
    priority_score: int
    importance_score: int
    summary: str
    channel_username: str
    post_link: Optional[str]
    message_date: Optional[datetime] = None
    final_score: float = 0.0


@dataclass
class DigestBuildResult:
    digest_text: str
    included_processed_message_ids: List[int]
    items_count: int
    title: str


class DigestBuilder:
    """
    Builds a plain-text digest from analyzed messages stored in SQLite.
    """

    def __init__(self, max_items: int = 7, candidate_limit: int = 50) -> None:
        self.repo = Repository()
        self.max_items = max_items
        self.candidate_limit = candidate_limit

    def _safe_parse_metadata(self, metadata_json: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Parse metadata_json safely.
        """
        if not metadata_json:
            return None
        try:
            data = json.loads(metadata_json)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _filter_items_strict(self, rows: List[Dict[str, Any]], threshold_used: int) -> tuple[List[DigestItem], Dict[str, int]]:
        """
        Apply strict filtering to digest candidates.
        """
        rejections = {
            'metadata_missing': 0,
            'not_relevant': 0,
            'category_not_allowed': 0,
            'priority_below_threshold': 0,
            'summary_missing': 0,
            'summary_too_short': 0,
        }
        
        items: List[DigestItem] = []
        
        for row in rows:
            metadata = self._safe_parse_metadata(row.get("metadata_json"))
            if not metadata:
                rejections['metadata_missing'] += 1
                continue

            is_relevant = metadata.get("is_relevant", False)
            if isinstance(is_relevant, bool):
                relevant_bool = is_relevant
            elif isinstance(is_relevant, str):
                relevant_bool = is_relevant.strip().lower() in {"true", "1", "yes"}
            else:
                relevant_bool = False
            if not relevant_bool:
                rejections['not_relevant'] += 1
                continue

            category = str(metadata.get("category", "other")).strip().lower()
            if category not in SECTION_MAPPING:
                rejections['category_not_allowed'] += 1
                continue

            try:
                importance = int(metadata.get("importance_score", 1))
            except (TypeError, ValueError):
                importance = 1

            try:
                priority = int(metadata.get("priority_score", 1))
            except (TypeError, ValueError):
                priority = 1
            priority = max(1, min(10, priority))

            if priority < threshold_used:
                rejections['priority_below_threshold'] += 1
                continue
            if category == "ecosystem_news" and priority < 8:
                rejections['priority_below_threshold'] += 1
                continue

            summary = metadata.get("summary", "")
            if not isinstance(summary, str):
                summary = ""
            summary = summary.strip()

            if not summary:
                rejections['summary_missing'] += 1
                continue
            # Strict summary length: minimum 20 chars
            if len(summary) < 20:
                rejections['summary_too_short'] += 1
                continue

            channel_username = row.get("channel_username") or ""
            if channel_username and not channel_username.startswith("@"):
                channel_username = f"@{channel_username}"
            if not channel_username:
                channel_username = "@unknown"

            raw_message_date = row.get("message_date")
            message_date: Optional[datetime] = None
            if isinstance(raw_message_date, datetime):
                message_date = raw_message_date
            elif isinstance(raw_message_date, str):
                raw_dt = raw_message_date.strip()
                if raw_dt:
                    try:
                        message_date = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                    except ValueError:
                        message_date = None

            items.append(
                DigestItem(
                    processed_message_id=int(row["processed_message_id"]),
                    category=category,
                    priority_score=priority,
                    importance_score=importance,
                    summary=summary,
                    channel_username=channel_username,
                    post_link=row.get("post_link"),
                    message_date=message_date,
                )
            )
        
        return items, rejections

    def _filter_items_relaxed(self, rows: List[Dict[str, Any]], threshold_used: int) -> tuple[List[DigestItem], Dict[str, int]]:
        """
        Apply relaxed filtering to digest candidates.
        """
        rejections = {
            'metadata_missing': 0,
            'not_relevant': 0,
            'category_not_allowed': 0,
            'priority_below_threshold': 0,
            'summary_missing': 0,
            'summary_too_short': 0,
        }
        
        items: List[DigestItem] = []
        relaxed_categories = set(SECTION_MAPPING.keys()) | {"other", "general", "news"}
        
        for row in rows:
            metadata = self._safe_parse_metadata(row.get("metadata_json"))
            if not metadata:
                rejections['metadata_missing'] += 1
                continue

            # Relaxed relevance: allow items without explicit relevance flag
            is_relevant = metadata.get("is_relevant", False)
            if isinstance(is_relevant, bool):
                relevant_bool = is_relevant
            elif isinstance(is_relevant, str):
                relevant_bool = is_relevant.strip().lower() in {"true", "1", "yes"}
            else:
                relevant_bool = True  # Default to relevant in relaxed mode
            
            # Only reject if explicitly marked as not relevant
            if isinstance(is_relevant, str) and is_relevant.strip().lower() in {"false", "0", "no"}:
                rejections['not_relevant'] += 1
                continue

            category = str(metadata.get("category", "other")).strip().lower()
            if category not in relaxed_categories:
                rejections['category_not_allowed'] += 1
                continue

            try:
                importance = int(metadata.get("importance_score", 1))
            except (TypeError, ValueError):
                importance = 1

            try:
                priority = int(metadata.get("priority_score", 1))
            except (TypeError, ValueError):
                priority = 1
            priority = max(1, min(10, priority))

            # Relaxed priority: lower threshold by 2 points
            relaxed_threshold = max(1, threshold_used - 2)
            if priority < relaxed_threshold:
                rejections['priority_below_threshold'] += 1
                continue

            summary = metadata.get("summary", "")
            if not isinstance(summary, str):
                summary = ""
            summary = summary.strip()

            if not summary:
                rejections['summary_missing'] += 1
                continue
            # Relaxed summary length: minimum 10 chars OR 3 words
            words = summary.split()
            if len(summary) < 10 and len(words) < 3:
                rejections['summary_too_short'] += 1
                continue

            channel_username = row.get("channel_username") or ""
            if channel_username and not channel_username.startswith("@"):
                channel_username = f"@{channel_username}"
            if not channel_username:
                channel_username = "@unknown"

            raw_message_date = row.get("message_date")
            message_date: Optional[datetime] = None
            if isinstance(raw_message_date, datetime):
                message_date = raw_message_date
            elif isinstance(raw_message_date, str):
                raw_dt = raw_message_date.strip()
                if raw_dt:
                    try:
                        message_date = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                    except ValueError:
                        message_date = None

            items.append(
                DigestItem(
                    processed_message_id=int(row["processed_message_id"]),
                    category=category,
                    priority_score=priority,
                    importance_score=importance,
                    summary=summary,
                    channel_username=channel_username,
                    post_link=row.get("post_link"),
                    message_date=message_date,
                )
            )
        
        return items, rejections

    def build(self) -> DigestBuildResult:
        """
        Build a digest from repository candidates.
        """
        # Initialize rejection counters
        rejections = {
            'metadata_missing': 0,
            'not_relevant': 0,
            'category_not_allowed': 0,
            'priority_below_threshold': 0,
            'summary_missing': 0,
            'summary_too_short': 0,
            'deduplication': 0,
            'channel_diversity': 0,
        }
        cfg = get_config()
        recent_days = int(getattr(cfg, "digest_max_age_days", 3) or 3)
        recent_days = max(1, recent_days)
        reuse_mode = cfg.debug.reuse_analyzed_messages

        if reuse_mode:
            logger.info("Debug mode: reuse_analyzed_messages=ENABLED - will reuse already analyzed messages")
        else:
            logger.info("Normal mode: reuse_analyzed_messages=disabled - only new messages will be used")

        def _fetch_candidates(max_age_days: int) -> tuple[List[Dict[str, Any]], int]:
            # 7 -> if count >= 5 use them
            rows = self.repo.get_digest_candidates_with_threshold(
                min_priority=7, limit=self.candidate_limit, max_age_days=max_age_days, reuse_analyzed_messages=reuse_mode
            )
            if len(rows) >= 5:
                return rows, 7

            # Else fallback to 5
            rows = self.repo.get_digest_candidates_with_threshold(
                min_priority=5, limit=self.candidate_limit, max_age_days=max_age_days, reuse_analyzed_messages=reuse_mode
            )
            if len(rows) >= 3:
                return rows, 5

            # If still < 3, fallback to 3
            rows = self.repo.get_digest_candidates_with_threshold(
                min_priority=3, limit=self.candidate_limit, max_age_days=max_age_days, reuse_analyzed_messages=reuse_mode
            )
            return rows, 3

        rows, threshold_used = _fetch_candidates(recent_days)
        rows_loaded_initial = len(rows)
        recency_used = recent_days

        # If we don't have enough *recent* items, relax recency to 7 days.
        if len(rows) < 3 and recent_days < 7:
            rows, threshold_used = _fetch_candidates(7)
            recency_used = 7

        rows_loaded = len(rows)

        logger.info("Digest threshold used: %d", threshold_used)
        logger.info("Digest recency window used: %d days", recency_used)
        logger.info("Rows loaded from database: %d", rows_loaded)

        if not rows:
            logger.info("No digest candidates found.")
            return DigestBuildResult(
                digest_text="",
                included_processed_message_ids=[],
                items_count=0,
                title="",
            )

        # Try strict filtering first
        items: List[DigestItem] = []
        parsed_items: List[DigestItem] = []
        rejections = {
            'metadata_missing': 0,
            'not_relevant': 0,
            'category_not_allowed': 0,
            'priority_below_threshold': 0,
            'summary_missing': 0,
            'summary_too_short': 0,
            'deduplication': 0,
            'channel_diversity': 0,
        }
        
        strict_items, strict_rejections = self._filter_items_strict(rows, threshold_used)
        logger.info("Strict mode yielded %d items", len(strict_items))
        
        # If strict mode yields too few items, try relaxed mode
        if len(strict_items) < 3:
            logger.info("Too few items in strict mode, trying relaxed mode")
            relaxed_items, relaxed_rejections = self._filter_items_relaxed(rows, threshold_used)
            logger.info("Relaxed mode yielded %d items", len(relaxed_items))
            # Use relaxed items if it gives us more usable content
            if len(relaxed_items) > len(strict_items):
                parsed_items = relaxed_items
                rejections = relaxed_rejections
                logger.info("Using relaxed mode results")
            else:
                parsed_items = strict_items
                rejections = strict_rejections
                logger.info("Using strict mode results (relaxed didn't improve)")
        else:
            parsed_items = strict_items
            rejections = strict_rejections
            logger.info("Using strict mode results")

        if not parsed_items:
            logger.info("No usable digest items after filtering/validation.")
            return DigestBuildResult(
                digest_text="",
                included_processed_message_ids=[],
                items_count=0,
                title="",
            )

        # Deduplication + diversity filtering:
        # - Dedup by normalized summary similarity (substring OR >70% word overlap)
        # - Limit max 2 items per channel
        # Applied BEFORE final sort/slicing.
        original_count = len(parsed_items)

        def _normalize_summary(text: str) -> str:
            text = (text or "").lower()
            # Remove punctuation while keeping unicode letters/digits.
            text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def _word_overlap_ratio(a: str, b: str) -> float:
            wa = set(a.split())
            wb = set(b.split())
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / float(max(len(wa), len(wb)))

        def _is_similar_summary(a_norm: str, b_norm: str) -> bool:
            if not a_norm or not b_norm:
                return False
            if a_norm in b_norm or b_norm in a_norm:
                return True
            return _word_overlap_ratio(a_norm, b_norm) > 0.7

        # Sort for greedy dedup (prefer higher priority/importance).
        candidates = sorted(
            parsed_items,
            key=lambda x: (-x.priority_score, -x.importance_score),
        )

        kept: List[DigestItem] = []
        kept_norms: List[str] = []
        for it in candidates:
            it_norm = _normalize_summary(it.summary)
            is_dup = False
            for kept_norm in kept_norms:
                if _is_similar_summary(it_norm, kept_norm):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(it)
                kept_norms.append(it_norm)

        parsed_items = kept
        duplicates_removed = original_count - len(parsed_items)
        rejections['deduplication'] = duplicates_removed
        if duplicates_removed > 0:
            logger.info("Dedup removed %d duplicate/similar items.", duplicates_removed)

        # Enforce channel diversity (max 2 items per channel).
        # Greedy keeps best items per channel (priority -> importance).
        channel_kept: Dict[str, int] = {}
        channel_filtered: List[DigestItem] = []
        for it in sorted(
            parsed_items, key=lambda x: (-x.priority_score, -x.importance_score)
        ):
            ch = it.channel_username or "@unknown"
            if channel_kept.get(ch, 0) >= 2:
                continue
            channel_kept[ch] = channel_kept.get(ch, 0) + 1
            channel_filtered.append(it)

        channel_removed = len(parsed_items) - len(channel_filtered)
        parsed_items = channel_filtered
        rejections['channel_diversity'] = channel_removed
        if channel_removed > 0:
            logger.info(
                "Channel diversity removed %d excess items.",
                channel_removed,
            )

        # Composite ranking: reduce over-reliance on raw priority_score.
        # final_score = base + freshness + informativeness + uniqueness - generic_penalty
        now = datetime.utcnow()
        normalized_summaries = [_normalize_summary(it.summary) for it in parsed_items]
        generic_markers = [
            "стоит отметить",
            "важная новость",
            "очередной шаг",
            "открывает прием заявок",
            "объявлен окончательный прием заявок",
            "предлагает возможность",
            "в рамках",
        ]
        fact_keywords = [
            "компания",
            "проект",
            "вакансия",
            "грант",
            "контракт",
            "дедлайн",
            "дата",
            "срок",
        ]

        def _freshness_bonus(dt: Optional[datetime]) -> int:
            if not dt:
                return 0
            try:
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                # Newer within the window gets a higher bonus.
                return max(0, 15 - int(age_days))
            except Exception:  # noqa: BLE001
                return 0

        def _informativeness_bonus(text: str) -> int:
            bonus = 0
            lower = (text or "").lower()
            if re.search(r"\d", text or ""):
                bonus += 10
            if re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text or "") or re.search(
                r"\b(19|20)\d{2}\b", text or ""
            ):
                bonus += 10
            caps = len(
                re.findall(r"\b[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]*\b", text or "")
            )
            if caps > 0:
                bonus += 5
            if any(k in lower for k in fact_keywords):
                bonus += 5
            return bonus

        def _generic_penalty(text: str) -> int:
            penalty = 0
            lower = (text or "").lower()
            if any(marker in lower for marker in generic_markers):
                penalty += 15
            words = (text or "").split()
            if len(words) < 8:
                penalty += 8
            if not re.search(r"\d", text or "") and not re.search(
                r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text or ""
            ):
                penalty += 10
            return penalty

        def _similarity(a_norm: str, b_norm: str) -> float:
            if not a_norm or not b_norm:
                return 0.0
            if a_norm in b_norm or b_norm in a_norm:
                return 1.0
            return _word_overlap_ratio(a_norm, b_norm)

        for i, it in enumerate(parsed_items):
            base = it.priority_score * 10 + it.importance_score * 3
            freshness_bonus = _freshness_bonus(it.message_date)
            informativeness_bonus = _informativeness_bonus(it.summary)

            # Uniqueness: lower similarity to the rest => higher bonus.
            max_sim = 0.0
            for j in range(len(parsed_items)):
                if i == j:
                    continue
                sim = _similarity(normalized_summaries[i], normalized_summaries[j])
                if sim > max_sim:
                    max_sim = sim
            uniqueness_bonus = int((1.0 - max_sim) * 20)

            generic_penalty = _generic_penalty(it.summary)
            it.final_score = (
                base
                + freshness_bonus
                + informativeness_bonus
                + uniqueness_bonus
                - generic_penalty
            )

        parsed_items.sort(
            key=lambda x: (-x.final_score, -x.priority_score, -x.importance_score)
        )
        items = parsed_items[: self.max_items]

        # Log ranking score for selected items (readable, one line each).
        for it in items:
            logger.info(
                "Digest item: id=%s channel=%s final_score=%.1f priority=%d importance=%d",
                it.processed_message_id,
                it.channel_username,
                it.final_score,
                it.priority_score,
                it.importance_score,
            )

        sections: Dict[str, List[DigestItem]] = {name: [] for name in SECTION_ORDER}
        for item in items:
            section = SECTION_MAPPING.get(item.category, "TOP NEWS")
            sections.setdefault(section, []).append(item)

        today = datetime.utcnow().date().isoformat()
        title = f"Ежедневный дайджест Jarvis — {today}"

        lines: List[str] = []
        lines.append(title)
        lines.append("")

        item_no = 1
        for section_name in SECTION_ORDER:
            section_items = sections.get(section_name, [])
            if not section_items:
                continue

            section_label = {
                "OPPORTUNITIES": "ВОЗМОЖНОСТИ",
                "TOP NEWS": "ГЛАВНОЕ",
                "JOBS": "ВАКАНСИИ",
                "EVENTS": "СОБЫТИЯ",
            }.get(section_name, section_name)

            lines.append(section_label)
            lines.append("-" * len(section_label))

            for it in section_items:
                lines.append(f"{item_no}. {it.summary}")
                lines.append(f"   Источник: {it.channel_username}")
                if it.post_link:
                    lines.append(f"   Ссылка: {it.post_link}")
                lines.append("")
                item_no += 1

        digest_text = "\n".join(lines).strip()
        included_ids = [it.processed_message_id for it in items]

        # Log filtering summary
        logger.info("=== Digest Filtering Summary ===")
        logger.info("Load mode: %s", "REUSE (debug)" if reuse_mode else "NORMAL")
        logger.info("Initial rows loaded: %d", rows_loaded_initial)
        if rows_loaded_initial != rows_loaded:
            logger.info("After recency fallback to %d days: %d", recency_used, rows_loaded)
        logger.info("After metadata parsing: %d", rows_loaded - rejections['metadata_missing'])
        logger.info("After relevance filter: %d", rows_loaded - rejections['metadata_missing'] - rejections['not_relevant'])
        logger.info("After priority filter: %d", rows_loaded - rejections['metadata_missing'] - rejections['not_relevant'] - rejections['priority_below_threshold'])
        logger.info("After summary validation: %d", len(parsed_items) + rejections['deduplication'] + rejections['channel_diversity'])
        logger.info("After dedup: %d", len(parsed_items) + rejections['channel_diversity'])
        logger.info("After diversity filter: %d", len(parsed_items))
        logger.info("Final items count: %d", len(items))
        logger.info("Rejections: metadata_missing=%d, not_relevant=%d, category_not_allowed=%d, priority_below_threshold=%d, summary_missing=%d, summary_too_short=%d, deduplication=%d, channel_diversity=%d",
                   rejections['metadata_missing'], rejections['not_relevant'], rejections['category_not_allowed'], 
                   rejections['priority_below_threshold'], rejections['summary_missing'], rejections['summary_too_short'],
                   rejections['deduplication'], rejections['channel_diversity'])
        logger.info("===============================")

        return DigestBuildResult(
            digest_text=digest_text,
            included_processed_message_ids=included_ids,
            items_count=len(items),
            title=title,
        )
