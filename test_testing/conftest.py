import os
import sys
from unittest.mock import MagicMock

# auth.py refuses to import without a secret; tests always use a throwaway one, never the real .env value
os.environ["JWT_SECRET"] = "test-secret-for-unit-tests-only-not-real"

# embedder.py loads two large models the moment it is imported; a fake module keeps the tests fast and offline
sys.modules["sentence_transformers"] = MagicMock()
