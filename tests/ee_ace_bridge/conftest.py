"""
Pytest configuration for ACE integration tests.

Adds ACE repository to sys.path before test collection.
"""

import sys
from pathlib import Path

# Add ACE repository to path if available
ace_repo_path = Path("/Users/speed/ace-playbook")
if ace_repo_path.exists() and str(ace_repo_path) not in sys.path:
    sys.path.insert(0, str(ace_repo_path))
