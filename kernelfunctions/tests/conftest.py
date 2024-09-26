import pytest  # type: ignore
import kernelfunctions.function as kff  # type: ignore


@pytest.fixture(autouse=True)
def run_after_each_test():
    kff.ENABLE_CALLDATA_CACHE = False
    yield
