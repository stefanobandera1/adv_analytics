"""
Pytest configuration — runs before any test module is imported.
Sets the matplotlib backend to Agg (non-interactive) so plot methods
do not attempt to open display windows during CI or headless test runs.
"""

import matplotlib

matplotlib.use("Agg")
