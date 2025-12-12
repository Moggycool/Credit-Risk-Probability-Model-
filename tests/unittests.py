""" Unit tests for basic functionality."""
from unittest import IsolatedAsyncioTestCase


class TestBasicMath(IsolatedAsyncioTestCase):
    """A basic unit test to verify the testing framework is working."""

    async def test_basic_math(self):
        """A basic test to verify the testing framework is working."""
        assert 2 + 2 == 4
