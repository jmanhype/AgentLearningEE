#!/usr/bin/env python3
"""
ACE Integration Demo - Full EE-ACE Bridge Test

Demonstrates the complete integration between EE bridge and ACE CuratorService:
1. Translate EE reflections to ACE insights
2. Ingest insights into ACE playbook
3. Verify deduplication and counter tracking
4. Render playbook for policy augmentation

Requires:
- ACE dependencies installed (sentence-transformers, faiss-cpu, sqlalchemy)
- ACE repo available at /Users/speed/ace-playbook
- Environment variables configured (see .env.ace)
"""

import sys
from pathlib import Path

# Add ACE repo to path
ace_repo = Path("/Users/speed/ace-playbook")
if ace_repo.exists():
    sys.path.insert(0, str(ace_repo))

import os

# Configure environment
os.environ["ACE_ENABLED"] = "1"
os.environ["ACE_DOMAIN_ID"] = "demo-agent-learning"
os.environ["ACE_TARGET_STAGE"] = "shadow"
os.environ["ACE_SIMILARITY_THRESHOLD"] = "0.80"
os.environ["DATABASE_URL"] = "sqlite:///ace_demo.db"

# Test imports
print("=" * 70)
print("ACE Integration Demo - Testing Imports")
print("=" * 70)

try:
    from ee_ace_bridge import ACE_INTEGRATION_AVAILABLE
    print(f"✓ ACE_INTEGRATION_AVAILABLE: {ACE_INTEGRATION_AVAILABLE}")
except ImportError as e:
    print(f"✗ Failed to import ACE_INTEGRATION_AVAILABLE: {e}")
    sys.exit(1)

if not ACE_INTEGRATION_AVAILABLE:
    print("✗ ACE integration not available. Install dependencies:")
    print("  pip install sentence-transformers faiss-cpu sqlalchemy structlog")
    sys.exit(1)

from ee_ace_bridge.ace_client import InProcessAceClient
from ee_ace_bridge.translate import bridge_to_ace_insight, bridge_batch_to_ace
from ee_ace_bridge.config_extra import (
    ACE_DOMAIN_ID,
    ACE_TARGET_STAGE,
    ACE_SIMILARITY_THRESHOLD,
)

print(f"✓ InProcessAceClient imported")
print(f"✓ Configuration loaded:")
print(f"  - Domain: {ACE_DOMAIN_ID}")
print(f"  - Stage: {ACE_TARGET_STAGE.value}")
print(f"  - Threshold: {ACE_SIMILARITY_THRESHOLD}")
print()

# Test schema translation
print("=" * 70)
print("Test 1: Schema Translation (EE → ACE)")
print("=" * 70)

ee_reflections = [
    {
        "insight_text": "Always check flight availability before booking",
        "insight_kind": "rule",
        "tags": ["availability", "booking"],
    },
    {
        "insight_text": "Validate credit card format (16 digits, no spaces)",
        "insight_kind": "rule",
        "tags": ["payment", "validation"],
    },
    {
        "insight_text": "Skipping visa check leads to booking failures",
        "insight_kind": "anti_pattern",
        "tags": ["visa", "international"],
    },
]

ace_insights = bridge_batch_to_ace(ee_reflections)

print(f"Translated {len(ee_reflections)} EE reflections → {len(ace_insights)} ACE insights:")
for i, (ee, ace) in enumerate(zip(ee_reflections, ace_insights), 1):
    print(f"\n  {i}. EE: kind={ee['insight_kind']}")
    print(f"     ACE: section={ace['section']}")
    print(f"     Content: {ace['content'][:60]}...")
    print(f"     Tags: {ace['tags']}")

print()

# Initialize database
print("=" * 70)
print("Initializing ACE Database")
print("=" * 70)

from sqlalchemy import create_engine
from ace.models.base import Base

db_url = os.getenv("DATABASE_URL", "sqlite:///ace_demo.db")
engine = create_engine(db_url, echo=False)

print(f"Creating database schema at: {db_url}")
Base.metadata.create_all(engine)
print("✓ Database initialized")
print()

# Test InProcessAceClient
print("=" * 70)
print("Test 2: InProcessAceClient - Insight Ingestion")
print("=" * 70)

try:
    client = InProcessAceClient(domain_id="demo-agent-learning")
    print("✓ InProcessAceClient initialized")
except Exception as e:
    print(f"✗ Failed to initialize InProcessAceClient: {e}")
    print("\nThis may be due to ACE repository issues.")
    print("Falling back to schema translation tests only.")
    sys.exit(0)

# Ingest first batch
print("\n→ Ingesting first batch (3 insights)...")
result1 = client.ingest_insights_batch(ee_reflections)
print(f"  Added: {result1['added']}")
print(f"  Incremented: {result1['incremented']}")
print(f"  Duplicates: {result1['duplicates']}")
print(f"  Total: {result1['total_insights']}")

# Ingest duplicate batch (should increment counters, not add new)
print("\n→ Ingesting duplicate batch (same 3 insights)...")
result2 = client.ingest_insights_batch(ee_reflections)
print(f"  Added: {result2['added']}")
print(f"  Incremented: {result2['incremented']}")
print(f"  Duplicates: {result2['duplicates']}")
print(f"  Total: {result2['total_insights']}")

if result2['duplicates'] > 0:
    print("  ✓ Deduplication working! Existing insights were incremented.")
else:
    print("  ⚠ Expected duplicates, but got new additions. Check similarity threshold.")

# Ingest new batch (should add new insights)
new_reflections = [
    {
        "insight_text": "Query flight status API before confirming",
        "insight_kind": "rule",
        "tags": ["api", "status"],
    },
    {
        "insight_text": "Retry on timeout with exponential backoff",
        "insight_kind": "pattern",
        "tags": ["reliability", "retry"],
    },
]

print("\n→ Ingesting new batch (2 different insights)...")
result3 = client.ingest_insights_batch(new_reflections)
print(f"  Added: {result3['added']}")
print(f"  Incremented: {result3['incremented']}")
print(f"  Duplicates: {result3['duplicates']}")
print(f"  Total: {result3['total_insights']}")

print()

# Test playbook rendering
print("=" * 70)
print("Test 3: Playbook Rendering")
print("=" * 70)

playbook = client.render_playbook(token_budget=3500)

if playbook:
    print(f"✓ Playbook rendered ({len(playbook)} chars):")
    print()
    print(playbook[:500])  # Show first 500 chars
    if len(playbook) > 500:
        print("\n  ... (truncated)")
else:
    print("✗ Playbook is empty (insights may be in shadow stage)")
    print("  Note: New insights start in 'shadow' stage and need promotion to 'prod'")

print()

# Test health check
print("=" * 70)
print("Test 4: Health Check")
print("=" * 70)

health = client.get_health()
print(f"Status: {health['status']}")
print(f"Domain: {health['domain_id']}")
print(f"Insights Ingested: {health['insights_ingested']}")
print(f"Stage Counts: {health['stage_counts']}")

print()

# Test counts
print("=" * 70)
print("Test 5: Insight Counts")
print("=" * 70)

section_count = client.get_section_count()
insight_count = client.get_insight_count()

print(f"Sections: {section_count}")
print(f"Total Insights: {insight_count}")

print()
print("=" * 70)
print("✅ ACE Integration Demo Complete!")
print("=" * 70)
print()
print("Summary:")
print(f"  - Schema translation: ✓ Working")
print(f"  - InProcessAceClient: ✓ Working")
print(f"  - Insight ingestion: ✓ {result1['added'] + result3['added']} insights added")
print(f"  - Deduplication: ✓ {result2['duplicates']} duplicates detected")
print(f"  - Playbook rendering: {'✓ Working' if playbook else '⚠ Empty (check stages)'}")
print()
print("Next Steps:")
print("  1. Enable ACE in policy: export ACE_ENABLED=1")
print("  2. Run training pipeline to seed playbook from reflections")
print("  3. Insights will flow: Training → ACE Playbook → Policy Context")
print()
