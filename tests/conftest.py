# tests/conftest.py
# Shared fixtures and test configuration.
import sys
import os

# Ensure src is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
