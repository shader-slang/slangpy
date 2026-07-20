# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from slangpy import SHA1


def test_sha1():
    assert SHA1().digest() == bytes.fromhex("da39a3ee5e6b4b0d3255bfef95601890afd80709")
    assert SHA1().hex_digest() == "da39a3ee5e6b4b0d3255bfef95601890afd80709"

    assert SHA1("hello world").digest() == bytes.fromhex("2aae6c35c94fcfb415dbe95f408b9ce91ee846ed")
    assert SHA1("hello world").hex_digest() == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"

    assert (
        SHA1(
            """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Integer felis ligula, sodales efficitur pellentesque quis, pretium at metus.
Nam eros dolor, auctor nec nibh non, iaculis euismod nulla.
Aliquam in nibh quis justo accumsan euismod nec fermentum purus.
In aliquam massa neque, congue faucibus nibh blandit ut.
Nulla facilisi.
Donec auctor blandit massa id fringilla.
Sed eu nisi neque.
Nulla ac blandit nulla, eu auctor eros.
Praesent ut est massa.
Etiam porttitor justo eu risus bibendum viverra.
Nam aliquet eros varius tristique mollis.
Curabitur bibendum libero in ipsum sodales, et convallis nisi ultrices.
Praesent eu erat vitae diam feugiat fringilla ac id mi.
Nunc risus ex, porttitor at imperdiet non, faucibus vitae quam.
Vestibulum sagittis odio nec dignissim iaculis.
Sed vel lacus suscipit mauris viverra dignissim vitae ac elit.
Aliquam erat volutpat.
Vivamus porttitor, mauris quis lobortis placerat, urna lorem eleifend enim, pharetra egestas elit metus condimentum nulla.
In ac sapien libero."""
        ).hex_digest()
        == "3b076d27b2a442e98c03304eae2e412a42a79ae2"
    )

    sha1 = SHA1()
    assert sha1.hex_digest() == "da39a3ee5e6b4b0d3255bfef95601890afd80709"
    sha1.update("hello")
    assert sha1.hex_digest() == "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
    sha1.update(" ")
    assert sha1.hex_digest() == "c4d871ad13ad00fde9a7bb7ff7ed2543aec54241"
    sha1.update("world")
    assert sha1.hex_digest() == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_sha1_block_boundaries():
    data = bytes(range(256))

    contiguous = SHA1(data)
    assert contiguous.hex_digest() == "4916d6bdb7f78e6803698cab32d1586ea457dfc8"

    chunked = SHA1()
    chunked.update(data[:1])
    chunked.update(data[1:64])
    chunked.update(data[64:128])
    chunked.update(data[128:])
    assert chunked.digest() == contiguous.digest()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
