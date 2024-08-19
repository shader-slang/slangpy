import pytest
import copper
import copper.tests.helpers as helpers  # type: ignore (just here as example of having a test helpers module)


def test_copper():
    assert copper.my_function() == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
