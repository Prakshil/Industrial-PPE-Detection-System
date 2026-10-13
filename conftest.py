"""Root conftest so pytest adds the repository root to sys.path.

Required so that ``pytest tests/`` (as run in CI) can import the
``ppe_monitor`` package without an editable install.
"""
