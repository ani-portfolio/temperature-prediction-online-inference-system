"""Mark each test by the folder it lives in, so `pytest -m unit` (or component, system) selects one level."""

import pytest

LEVELS = ("unit", "component", "system")


def pytest_collection_modifyitems(items):
    for item in items:
        level = next((part for part in item.path.parts if part in LEVELS), None)
        if level:
            item.add_marker(getattr(pytest.mark, level))
