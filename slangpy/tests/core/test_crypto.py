# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from slangpy import SHA1


def test_sha1():
    assert SHA1().digest() == bytes.fromhex("da39a3ee5e6b4b0d3255bfef95601890afd80709")
    assert SHA1().hex_digest() == "da39a3ee5e6b4b0d3255bfef95601890afd80709"

    assert SHA1("hello world").digest() == bytes.fromhex("2aae6c35c94fcfb415dbe95f408b9ce91ee846ed")
    assert SHA1("hello world").hex_digest() == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"

    sha1 = SHA1()
    assert sha1.hex_digest() == "da39a3ee5e6b4b0d3255bfef95601890afd80709"
    sha1.update("hello")
    assert sha1.hex_digest() == "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
    sha1.update(" ")
    assert sha1.hex_digest() == "c4d871ad13ad00fde9a7bb7ff7ed2543aec54241"
    sha1.update("world")
    assert sha1.hex_digest() == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
