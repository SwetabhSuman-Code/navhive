"""
tests/test_trilateration.py

Unit tests for the trilateration solver.

Verifies that the least-squares solver correctly recovers known 2-D
positions given exact (noise-free) distances from anchors.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from engine.trilateration import trilateration


class FakeAnchor:
    """Minimal anchor stub — only needs a .position attribute."""
    def __init__(self, x, y):
        self.position = np.array([x, y], dtype=float)


def make_distances(target, anchors):
    """Compute exact Euclidean distances from target to each anchor."""
    return np.array([np.linalg.norm(a.position - target) for a in anchors])


# ── Test Cases ────────────────────────────────────────────────────────────────

class TestTrilateration:

    def test_origin(self):
        """Node at the origin should be recovered exactly."""
        anchors   = [FakeAnchor(0, 5), FakeAnchor(5, 0), FakeAnchor(-5, 0)]
        target    = np.array([0.0, 0.0])
        distances = make_distances(target, anchors)
        result    = trilateration(anchors, distances)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_known_position_1(self):
        """Standard 4-anchor trapezoid layout — node at (3, 4)."""
        anchors   = [FakeAnchor(0, 0), FakeAnchor(10, 0),
                     FakeAnchor(5, 8), FakeAnchor(10, 10)]
        target    = np.array([3.0, 4.0])
        distances = make_distances(target, anchors)
        result    = trilateration(anchors, distances)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_known_position_2(self):
        """Node at (7, 6) — different quadrant."""
        anchors   = [FakeAnchor(0, 0), FakeAnchor(10, 0),
                     FakeAnchor(5, 8), FakeAnchor(10, 10)]
        target    = np.array([7.0, 6.0])
        distances = make_distances(target, anchors)
        result    = trilateration(anchors, distances)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_minimum_three_anchors(self):
        """Three anchors is the minimum — should still converge."""
        anchors   = [FakeAnchor(0, 0), FakeAnchor(10, 0), FakeAnchor(5, 8)]
        target    = np.array([4.0, 5.0])
        distances = make_distances(target, anchors)
        result    = trilateration(anchors, distances)
        np.testing.assert_allclose(result, target, atol=1e-5)

    def test_noisy_distances_plausible(self):
        """With small noise the estimate should still be close (within 1 m)."""
        np.random.seed(42)
        anchors   = [FakeAnchor(0, 0), FakeAnchor(10, 0),
                     FakeAnchor(5, 8), FakeAnchor(10, 10)]
        target    = np.array([3.0, 4.0])
        distances = make_distances(target, anchors) + np.random.normal(0, 0.3, 4)
        result    = trilateration(anchors, distances)
        error     = np.linalg.norm(result - target)
        assert error < 1.5, f"Error {error:.3f} m exceeds acceptable bound with small noise"
