"""
api/analyze.py

Vercel Serverless entrypoint (see pyproject.toml [tool.vercel]).

All routes live in backend/app.py:
  GET  /api/health
  GET  /api/capabilities
  POST /api/analyze
  POST /api/diagnostics
  POST /api/stac/inspect
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable so `backend.*` resolves on Vercel.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app  # noqa: E402,F401
