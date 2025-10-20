"""
In-Process ACE Client

Wire-compatible client that talks directly to ACE's CuratorService.
Implements AceClient protocol using ACE Curator for semantic deduplication,
FAISS indexing, and multi-stage promotion.

Usage:
    from ee_ace_bridge.ace_client import InProcessAceClient

    client = InProcessAceClient(domain_id="agent-learning")
    client.ingest_insights_batch([
        {"insight_text": "Check availability first", "insight_kind": "rule", "tags": ["availability"]},
        {"insight_text": "Always verify payment", "insight_kind": "rule", "tags": ["payment"]},
    ])
    playbook = client.render_playbook(token_budget=3500)
"""

from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime

try:
    from ace.curator.curator_service import CuratorService
    from ace.models.playbook import PlaybookStage, PlaybookBullet
    ACE_CURATOR_AVAILABLE = True
except ImportError:
    ACE_CURATOR_AVAILABLE = False
    CuratorService = None  # type: ignore
    PlaybookStage = None  # type: ignore
    PlaybookBullet = None  # type: ignore

from .translate import bridge_batch_to_ace
from .config_extra import ACE_DOMAIN_ID, ACE_TARGET_STAGE, ACE_SIMILARITY_THRESHOLD


class InProcessAceClient:
    """
    In-process ACE client using CuratorService.

    Implements AceClient protocol with ACE's production curator logic:
    - Semantic deduplication via FAISS (0.80 cosine threshold)
    - Helpful/Harmful counters
    - Multi-stage promotion (shadow → staging → prod)
    - Domain isolation

    This client requires the ACE repository to be installed and available.
    """

    def __init__(
        self,
        domain_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize in-process ACE client.

        Args:
            domain_id: Domain namespace (defaults to ACE_DOMAIN_ID env var)
            similarity_threshold: Cosine similarity threshold (defaults to ACE_SIMILARITY_THRESHOLD env var)
            embedding_model: sentence-transformers model name

        Raises:
            ImportError: If ACE curator modules not available
        """
        if not ACE_CURATOR_AVAILABLE:
            raise ImportError(
                "ACE curator modules not available. "
                "Install ace-playbook package or ensure /Users/speed/ace-playbook is in PYTHONPATH"
            )

        self.domain_id = domain_id or ACE_DOMAIN_ID
        self.similarity_threshold = similarity_threshold or ACE_SIMILARITY_THRESHOLD
        self.curator_service = CuratorService(
            embedding_model=embedding_model,
            similarity_threshold=self.similarity_threshold,
        )

        # Insights counter for statistics
        self._insights_ingested = 0

    def ingest_insight(self, bridge_insight: Dict) -> Dict:
        """
        Ingest a single insight (wrapper around batch ingestion).

        Args:
            bridge_insight: Insight in EE bridge format

        Returns:
            Result dict with status
        """
        return self.ingest_insights_batch([bridge_insight])

    def ingest_insights_batch(self, bridge_insights: List[Dict]) -> Dict:
        """
        Ingest a batch of insights into ACE playbook.

        Args:
            bridge_insights: List of insights in EE bridge format
                [{"insight_text": "...", "insight_kind": "rule", "tags": [...]}]

        Returns:
            Dict with ingestion results:
                {
                    "added": int,
                    "incremented": int,
                    "duplicates": int,
                    "total_insights": int,
                    "added_ids": List[str],
                    "incremented_ids": List[str],
                    "quarantined_ids": List[str],
                }
        """
        if not bridge_insights:
            return {
                "added": 0,
                "incremented": 0,
                "duplicates": 0,
                "total_insights": 0,
            }

        # Translate EE bridge schema → ACE schema
        ace_insights = bridge_batch_to_ace(bridge_insights)

        # Generate unique task_id for this ingestion
        task_id = f"ee-bridge-{datetime.utcnow().isoformat()}"

        # Merge insights using CuratorService
        curator_output = self.curator_service.merge_insights(
            task_id=task_id,
            domain_id=self.domain_id,
            insights=ace_insights,
            target_stage=PlaybookStage(ACE_TARGET_STAGE.value),
            similarity_threshold=self.similarity_threshold,
        )

        # Update statistics
        self._insights_ingested += len(bridge_insights)

        added_ids: List[str] = []
        incremented_ids: List[str] = []
        quarantined_ids: List[str] = []

        for delta in curator_output.delta_updates:
            operation = (delta.operation or "").lower()
            if operation == "add":
                added_ids.append(delta.bullet_id)
            elif operation.startswith("increment_"):
                incremented_ids.append(delta.bullet_id)
            elif operation == "quarantine":
                quarantined_ids.append(delta.bullet_id)

        return {
            "added": curator_output.new_bullets_added,
            "incremented": curator_output.existing_bullets_incremented,
            "duplicates": curator_output.duplicates_detected,
            "total_insights": len(bridge_insights),
            "added_ids": added_ids,
            "incremented_ids": incremented_ids,
            "quarantined_ids": quarantined_ids,
        }

    def render_playbook(
        self,
        sections: Optional[List[str]] = None,
        token_budget: int = 3500,
        stage: Optional[str] = None,
    ) -> str:
        """
        Render playbook as formatted text from ACE database.

        Args:
            sections: Optional filter for sections (Helpful/Harmful/Neutral)
            token_budget: Maximum tokens (~4 chars per token)
            stage: Optional stage filter (shadow/staging/prod)

        Returns:
            Formatted playbook text
        """
        # Fetch bullets from ACE
        stage_enum = PlaybookStage(stage) if stage else None
        bullets = self.curator_service.get_playbook(
            domain_id=self.domain_id,
            stage=stage_enum,
            section=None,  # We'll filter sections manually
        )

        if not bullets:
            return ""

        # Group by section
        section_groups = {}
        for bullet in bullets:
            section = bullet.section
            if sections and section not in sections:
                continue
            if section not in section_groups:
                section_groups[section] = []
            section_groups[section].append(bullet)

        # Format playbook
        lines = []
        lines.append("# ACE Playbook\n")

        # Order: Helpful → Neutral → Harmful
        section_order = {"Helpful": 0, "Neutral": 1, "Harmful": 2}
        for section in sorted(section_groups.keys(), key=lambda s: section_order.get(s, 1)):
            bullets_in_section = section_groups[section]

            # Sort by helpful count (descending)
            sorted_bullets = sorted(bullets_in_section, key=lambda b: -b.helpful_count)

            lines.append(f"\n## {section}\n")
            for bullet in sorted_bullets:
                ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
                lines.append(f"- {bullet.content} (✓{bullet.helpful_count} ✗{bullet.harmful_count} ratio:{ratio:.1f})\n")

        # Join and apply token budget
        playbook_text = "".join(lines)
        max_chars = token_budget * 4

        if len(playbook_text) > max_chars:
            # Clip at line boundary
            playbook_text = playbook_text[:max_chars].rsplit("\n", 1)[0]

        return playbook_text

    def get_health(self) -> Dict:
        """
        Get health status.

        Returns:
            Dict with health metrics
        """
        # Get stage counts
        counts = self.curator_service.get_stage_counts(self.domain_id)

        return {
            "status": "healthy",
            "domain_id": self.domain_id,
            "insights_ingested": self._insights_ingested,
            "stage_counts": counts,
        }

    def get_section_count(self) -> int:
        """
        Get number of distinct sections (Helpful/Harmful/Neutral).

        Returns:
            Number of sections with bullets
        """
        bullets = self.curator_service.get_playbook(domain_id=self.domain_id)
        sections = set(b.section for b in bullets)
        return len(sections)

    def get_insight_count(self) -> int:
        """
        Get total number of insights/bullets in playbook.

        Returns:
            Total bullet count
        """
        bullets = self.curator_service.get_playbook(domain_id=self.domain_id)
        return len(bullets)

    def clear(self) -> None:
        """
        Clear playbook for this domain.

        WARNING: This is destructive and should only be used in testing.
        ACE playbook is append-only in production.
        """
        raise NotImplementedError(
            "Clear operation not supported by InProcessAceClient. "
            "ACE playbook is append-only by design. "
            "Use domain_id isolation for testing."
        )
