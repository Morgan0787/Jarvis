"""
Repository layer for Jarvis v2 Core.

This module will contain higher-level database operations, such as:
- Inserting and querying channels
- Storing raw and processed messages
- Reading and writing daily digests

This is intentionally small and uses plain `sqlite3` with simple SQL.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, List, Any

from .database import get_connection
from .models import Channel, RawMessage, ProcessedMessage
from ..core.logger import get_logger


logger = get_logger(__name__)


def _utc_now_iso() -> str:
    """
    Return current UTC time in ISO format.

    We store timestamps as ISO strings in SQLite for simplicity.
    """
    return datetime.utcnow().isoformat(timespec="seconds")


def _normalize_channel_username(username: str) -> str:
    """
    Normalize a channel identifier into a username-like string.

    Examples:
    - "@startupuz" -> "startupuz"
    - "startupuz" -> "startupuz"
    - "https://t.me/startupuz" -> "startupuz"
    """
    value = username.strip()
    if value.startswith("https://t.me/"):
        value = value.replace("https://t.me/", "", 1)
    if value.startswith("http://t.me/"):
        value = value.replace("http://t.me/", "", 1)
    value = value.lstrip("@")
    # Remove any trailing path like https://t.me/name/123
    value = value.split("/")[0]
    return value


class Repository:
    """
    Small database helper that owns all SQL queries used in Step 2.

    Beginner-friendly rule:
    - Collector code should call these methods instead of writing SQL directly.
    """

    def get_or_create_channel(self, username: str) -> Channel:
        """
        Get an existing channel by username, or create it if missing.

        The `username` is normalized by removing leading '@' and URL prefixes.
        """
        normalized = _normalize_channel_username(username)
        existing = self.get_channel_by_username(normalized)
        if existing is not None:
            return existing

        created_at = _utc_now_iso()
        updated_at = created_at

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO channels (
                    telegram_id, last_message_id, username, title, is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (None, None, normalized, None, 1, created_at, updated_at),
            )
            conn.commit()
            channel_id = int(cur.lastrowid)
        finally:
            conn.close()

        created = self.get_channel_by_id(channel_id)
        if created is None:
            raise RuntimeError("Failed to create channel record.")
        return created

    # -------- Channel helpers --------

    def get_all_channels(self) -> List[Channel]:
        """
        Return all channels from the database.
        """
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM channels;")
            rows = cursor.fetchall()
            channels: List[Channel] = []
            for row in rows:
                channels.append(
                    Channel(
                        id=row["id"],
                        telegram_id=row["telegram_id"],
                        last_message_id=row["last_message_id"],
                        username=row["username"],
                        title=row["title"],
                        is_active=bool(row["is_active"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            return channels
        finally:
            conn.close()

    def get_channel_by_id(self, channel_id: int) -> Optional[Channel]:
        """
        Return a channel by its internal database ID.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM channels WHERE id = ?;", (channel_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return Channel(
                id=row["id"],
                telegram_id=row["telegram_id"],
                last_message_id=row["last_message_id"],
                username=row["username"],
                title=row["title"],
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        finally:
            conn.close()

    def get_channel_by_username(self, username: str) -> Optional[Channel]:
        """
        Return a channel by normalized username (without '@').
        """
        normalized = _normalize_channel_username(username)
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM channels WHERE username = ?;", (normalized,))
            row = cur.fetchone()
            if row is None:
                return None
            return Channel(
                id=row["id"],
                telegram_id=row["telegram_id"],
                last_message_id=row["last_message_id"],
                username=row["username"],
                title=row["title"],
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        finally:
            conn.close()

    def update_channel_last_message_id(self, channel_id: int, last_message_id: int) -> None:
        """
        Update `channels.last_message_id` after successfully collecting messages.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE channels
                SET last_message_id = ?, updated_at = ?
                WHERE id = ?;
                """,
                (last_message_id, _utc_now_iso(), channel_id),
            )
            conn.commit()
        finally:
            conn.close()

    # -------- Raw message helpers --------

    def raw_message_exists(self, channel_id: int, telegram_message_id: int) -> bool:
        """
        Return True if a raw message already exists in the database.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1
                FROM raw_messages
                WHERE channel_id = ? AND telegram_message_id = ?
                LIMIT 1;
                """,
                (channel_id, telegram_message_id),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    def insert_raw_message(
        self,
        channel_id: int,
        telegram_message_id: int,
        message_date: datetime,
        message_text: str,
        post_link: Optional[str],
        collected_at: datetime,
    ) -> int:
        """
        Insert a new raw Telegram message.

        Returns:
            The new database row ID.
        """
        # A simple hash helps with later deduplication.
        content_hash = hashlib.sha256(message_text.encode("utf-8")).hexdigest()

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO raw_messages (
                    channel_id,
                    telegram_message_id,
                    post_link,
                    message_text,
                    message_date,
                    collected_at,
                    content_hash,
                    is_processed
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    channel_id,
                    telegram_message_id,
                    post_link,
                    message_text,
                    message_date.isoformat(timespec="seconds"),
                    collected_at.isoformat(timespec="seconds"),
                    content_hash,
                    0,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_unprocessed_raw_messages(self, limit: int = 500) -> List[RawMessage]:
        """
        Fetch a batch of raw messages that have not been processed yet.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM raw_messages
                WHERE is_processed = 0
                ORDER BY id ASC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            result: List[RawMessage] = []
            for row in rows:
                result.append(
                    RawMessage(
                        id=row["id"],
                        channel_id=row["channel_id"],
                        telegram_message_id=row["telegram_message_id"],
                        post_link=row["post_link"],
                        message_text=row["message_text"],
                        message_date=datetime.fromisoformat(row["message_date"]),
                        collected_at=datetime.fromisoformat(row["collected_at"]),
                        content_hash=row["content_hash"],
                        is_processed=bool(row["is_processed"]),
                    )
                )
            return result
        finally:
            conn.close()

    def mark_raw_message_processed(self, raw_message_id: int) -> None:
        """
        Mark a raw message as processed.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE raw_messages
                SET is_processed = 1
                WHERE id = ?;
                """,
                (raw_message_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # -------- Processed message helpers --------

    def processed_message_exists(self, raw_message_id: int) -> bool:
        """
        Return True if a processed message already exists for this raw message.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1
                FROM processed_messages
                WHERE raw_message_id = ?
                LIMIT 1;
                """,
                (raw_message_id,),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    def find_duplicate_processed_message(
        self,
        cleaned_text: str,
    ) -> Optional[ProcessedMessage]:
        """
        Find an existing processed message with exactly the same cleaned_text.

        This is a simple and reliable way to spot duplicates across runs.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM processed_messages
                WHERE cleaned_text = ?
                LIMIT 1;
                """,
                (cleaned_text,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            return ProcessedMessage(
                id=row["id"],
                raw_message_id=row["raw_message_id"],
                cleaned_text=row["cleaned_text"],
                short_text=row["short_text"],
                is_duplicate=bool(row["is_duplicate"]),
                duplicate_of_raw_message_id=row["duplicate_of_raw_message_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                classification=row["classification"],
                importance_score=row["importance_score"],
                metadata_json=row["metadata_json"],
                processed_at=datetime.fromisoformat(row["processed_at"]),
                included_in_digest=bool(row["included_in_digest"]),
            )
        finally:
            conn.close()

    def insert_processed_message(
        self,
        raw_message_id: int,
        cleaned_text: str,
        short_text: str,
        is_duplicate: bool,
        duplicate_of_raw_message_id: Optional[int],
        processed_at: datetime,
    ) -> int:
        """
        Insert a processed message row.

        We keep some fields (classification, importance_score, metadata_json)
        for future steps, but leave them NULL for now.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO processed_messages (
                    raw_message_id,
                    cleaned_text,
                    short_text,
                    is_duplicate,
                    duplicate_of_raw_message_id,
                    created_at,
                    classification,
                    importance_score,
                    metadata_json,
                    processed_at,
                    included_in_digest
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    raw_message_id,
                    cleaned_text,
                    short_text,
                    1 if is_duplicate else 0,
                    duplicate_of_raw_message_id,
                    processed_at.isoformat(timespec="seconds"),
                    None,
                    None,
                    None,
                    processed_at.isoformat(timespec="seconds"),
                    0,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_unanalyzed_processed_messages(self, limit: int = 100) -> List[ProcessedMessage]:
        """
        Fetch processed messages that have not yet been analyzed by Ollama.

        We treat `classification IS NULL` as "not analyzed".
        Returns newest messages first by joining with raw_messages.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT pm.*
                FROM processed_messages pm
                JOIN raw_messages rm ON rm.id = pm.raw_message_id
                WHERE pm.classification IS NULL
                ORDER BY rm.message_date DESC, pm.id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            result: List[ProcessedMessage] = []
            for row in rows:
                result.append(
                    ProcessedMessage(
                        id=row["id"],
                        raw_message_id=row["raw_message_id"],
                        cleaned_text=row["cleaned_text"],
                        short_text=row["short_text"],
                        is_duplicate=bool(row["is_duplicate"]),
                        duplicate_of_raw_message_id=row["duplicate_of_raw_message_id"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        classification=row["classification"],
                        importance_score=row["importance_score"],
                        metadata_json=row["metadata_json"],
                        processed_at=datetime.fromisoformat(row["processed_at"]),
                        included_in_digest=bool(row["included_in_digest"]),
                    )
                )
            return result
        finally:
            conn.close()

    def update_processed_message_analysis(
        self,
        processed_message_id: int,
        classification: str,
        importance_score: float,
        metadata_json: str,
    ) -> None:
        """
        Update analysis fields for a processed message.

        - classification: category label
        - importance_score: numeric score from the model
        - metadata_json: full JSON response from the model
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE processed_messages
                SET classification = ?,
                    importance_score = ?,
                    metadata_json = ?
                WHERE id = ?;
                """,
                (classification, importance_score, metadata_json, processed_message_id),
            )
            rows_affected = cur.rowcount
            conn.commit()
            logger.info("Updated analysis for message_id=%s: rows_affected=%d, classification=%s, importance_score=%s", 
                       processed_message_id, rows_affected, classification, importance_score)
        finally:
            conn.close()

    # -------- Digest helpers --------

    def get_last_published_digest(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent digest that was successfully published.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM digests
                WHERE published_to IS NOT NULL
                  AND TRIM(published_to) != ''
                ORDER BY id DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_recent_published_processed_message_ids(self, days: int = 7) -> set[int]:
        """
        Return processed_message_ids used in recently published digests.
        """
        cutoff = (datetime.utcnow() - timedelta(days=max(1, days))).isoformat(
            timespec="seconds"
        )
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT metadata_json
                FROM digests
                WHERE published_to IS NOT NULL
                  AND TRIM(published_to) != ''
                  AND created_at >= ?;
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        result: set[int] = set()
        for row in rows:
            raw = row["metadata_json"]
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue

            ids = data.get("published_processed_message_ids", [])
            if not isinstance(ids, list):
                continue
            for item in ids:
                try:
                    result.add(int(item))
                except (TypeError, ValueError):
                    continue
        return result

    def get_digest_candidates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return processed messages suitable for the digest.

        Conditions:
        - classification IS NOT NULL
        - classification != 'other'
        - included_in_digest = 0
        - is_duplicate = 0
        - importance_score >= 6

        Joins:
        - raw_messages (post_link, message_date, channel_id)
        - channels (username)
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    pm.id AS processed_message_id,
                    pm.raw_message_id AS raw_message_id,
                    pm.cleaned_text AS cleaned_text,
                    pm.short_text AS short_text,
                    pm.classification AS classification,
                    pm.importance_score AS importance_score,
                    pm.metadata_json AS metadata_json,
                    rm.post_link AS post_link,
                    rm.message_date AS message_date,
                    rm.channel_id AS channel_id,
                    ch.username AS channel_username
                FROM processed_messages pm
                JOIN raw_messages rm ON rm.id = pm.raw_message_id
                LEFT JOIN channels ch ON ch.id = rm.channel_id
                WHERE pm.classification IS NOT NULL
                  AND pm.classification != 'other'
                  AND pm.included_in_digest = 0
                  AND pm.is_duplicate = 0
                  AND pm.importance_score >= 7
                ORDER BY pm.importance_score DESC, pm.id ASC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_digest_candidates_with_threshold(
        self,
        min_priority: int,
        limit: int = 50,
        max_age_days: int | None = None,
        reuse_analyzed_messages: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return processed messages suitable for the digest, using a dynamic threshold.

        This is identical to `get_digest_candidates()`, except:
        - pm.importance_score >= 7
          becomes
        - pm.importance_score >= min_priority
        - optional recency filter:
          - rm.message_date >= (now - max_age_days)
        - when reuse_analyzed_messages=True:
          - bypasses pm.included_in_digest = 0 filter to allow reusing already analyzed messages
        """
        cutoff_iso: str | None = None
        if max_age_days is not None:
            cutoff_dt = datetime.utcnow() - timedelta(days=max_age_days)
            cutoff_iso = cutoff_dt.isoformat(timespec="seconds")

        conn = get_connection()
        try:
            cur = conn.cursor()
            
            # Build query conditionally based on reuse_analyzed_messages flag
            if reuse_analyzed_messages:
                # Debug mode: load all analyzed rows broadly for testing
                query = """
                SELECT
                    pm.id AS processed_message_id,
                    pm.raw_message_id AS raw_message_id,
                    pm.cleaned_text AS cleaned_text,
                    pm.short_text AS short_text,
                    pm.classification AS classification,
                    pm.importance_score AS importance_score,
                    pm.metadata_json AS metadata_json,
                    rm.post_link AS post_link,
                    rm.message_date AS message_date,
                    rm.channel_id AS channel_id,
                    ch.username AS channel_username
                FROM processed_messages pm
                JOIN raw_messages rm ON rm.id = pm.raw_message_id
                LEFT JOIN channels ch ON ch.id = rm.channel_id
                WHERE pm.classification IS NOT NULL
                  AND pm.metadata_json IS NOT NULL
                  AND pm.metadata_json != ''
                  AND (? IS NULL OR datetime(rm.message_date) >= datetime(?))
                ORDER BY pm.importance_score DESC, pm.id ASC
                LIMIT ?;
                """
                params = (cutoff_iso, cutoff_iso, limit)
            else:
                # Normal mode: exclude messages already used in digests
                query = """
                SELECT
                    pm.id AS processed_message_id,
                    pm.raw_message_id AS raw_message_id,
                    pm.cleaned_text AS cleaned_text,
                    pm.short_text AS short_text,
                    pm.classification AS classification,
                    pm.importance_score AS importance_score,
                    pm.metadata_json AS metadata_json,
                    rm.post_link AS post_link,
                    rm.message_date AS message_date,
                    rm.channel_id AS channel_id,
                    ch.username AS channel_username
                FROM processed_messages pm
                JOIN raw_messages rm ON rm.id = pm.raw_message_id
                LEFT JOIN channels ch ON ch.id = rm.channel_id
                WHERE pm.classification IS NOT NULL
                  AND pm.classification != 'other'
                  AND pm.included_in_digest = 0
                  AND pm.is_duplicate = 0
                  AND (? IS NULL OR datetime(rm.message_date) >= datetime(?))
                ORDER BY pm.importance_score DESC, pm.id ASC
                LIMIT ?;
                """
                params = (cutoff_iso, cutoff_iso, limit)
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            # Log query results for debugging
            mode = "DEBUG/REUSE" if reuse_analyzed_messages else "NORMAL"
            logger.info("Digest candidate query (%s): loaded %d rows", mode, len(rows))
            
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def mark_processed_messages_included(self, message_ids: List[int]) -> None:
        """
        Mark processed_messages as included_in_digest = 1 for the given IDs.
        """
        if not message_ids:
            return

        placeholders = ",".join(["?"] * len(message_ids))
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                UPDATE processed_messages
                SET included_in_digest = 1
                WHERE id IN ({placeholders});
                """,
                tuple(message_ids),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_digest(
        self,
        digest_date: str,
        title: str,
        content: str,
        created_at: str,
        published_to: str | None,
        metadata_json: str | None,
    ) -> int:
        """
        Insert a digest row into `digests`.
        """
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO digests (
                    digest_date,
                    title,
                    content,
                    created_at,
                    published_to,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (digest_date, title, content, created_at, published_to, metadata_json),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

