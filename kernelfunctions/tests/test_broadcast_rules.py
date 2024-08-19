import pytest
import kernelfunctions as kf

# Test cases: [in shape 0, inshape 1, out shape]
SUCCESS_TESTS = [
    [(3,), (1,), (3,)],
    [(256, 256, 3), (3,), (256, 256, 3)],
    [(8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)],
    [(5, 4), (1,), (5, 4)],
    [(15, 3, 5), (15, 1, 5), (15, 3, 5)],
    [(15, 3, 5), (3, 5), (15, 3, 5)],
    [(15, 3, 5), (3, 1), (15, 3, 5)],
    [(10, 3), (5, 1, 3), (5, 10, 3)],
]

# Same but for expected failures
FAIL_TESTS = [[(3,), (2,), (0,)], [(2, 1), (8, 3, 4), (0,)]]


def TEST_ID(test: str):
    return f"B( {test[0]}, {test[1]} ) -> {test[2]}"


@pytest.mark.parametrize("test", SUCCESS_TESTS, ids=TEST_ID)
def test_broadcast(test: list[tuple[int, ...]]):
    s0 = test[0]
    s1 = test[1]
    dims = kf.calldata.calculate_broadcast_dimensions([s0, s1])
    assert dims == test[2]


@pytest.mark.parametrize("test", FAIL_TESTS, ids=TEST_ID)
def test_fail_broadcast(test: list[tuple[int, ...]]):
    with pytest.raises(ValueError):
        s0 = test[0]
        s1 = test[1]
        kf.calldata.calculate_broadcast_dimensions([s0, s1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
