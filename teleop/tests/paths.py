"""Filesystem locations for Portal mocks and unit tests (not production)."""
from __future__ import annotations

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TELEOP_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(TELEOP_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PORTAL_YAML = os.path.join(TELEOP_DIR, "portal.yaml")
PORTAL_MAPPING = os.path.join(TELEOP_DIR, "portal_mapping.yaml")
ENV_FILE = os.path.join(TELEOP_DIR, ".env")
ROBOT_MOCK = os.path.join(TESTS_DIR, "portal_robot_mock.py")
OPERATOR_MOCK = os.path.join(TESTS_DIR, "portal_operator_mock.py")
TELEOP_ROBOT = os.path.join(TELEOP_DIR, "teleop_robot.py")
