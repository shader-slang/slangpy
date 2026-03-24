# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from pathlib import Path

import slangpy as spy
from slangpy.testing import helpers

from typing import Sequence, Union


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_type_conformance(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    def run(conformances: Sequence[Union[tuple[str, str], tuple[str, str, int]]]):
        module = device.load_module("test_type_conformance.slang")
        entry_point = module.entry_point("compute_main")
        program = device.link_program(
            modules=[module], entry_points=[entry_point], type_conformances=conformances
        )
        kernel = device.create_compute_kernel(program)
        result = device.create_buffer(
            element_count=4, struct_size=4, usage=spy.BufferUsage.unordered_access
        )
        kernel.dispatch(thread_count=[4, 1, 1], result=result)
        return result.to_numpy().view(np.uint32)

    # Conforming to non-existing interface type must raise an exception.
    with pytest.raises(RuntimeError, match='Interface type "IUnknown" not found'):
        run(conformances=[("IUnknown", "Foo1")])

    # Conforming to non-existing type must raise an exception.
    with pytest.raises(RuntimeError, match='Type "Unknown" not found'):
        run(conformances=[("IFoo", "Unknown")])

    # Specifying duplicate type conformances must raise an exception.
    with pytest.raises(
        RuntimeError,
        match='Duplicate type conformance entry for interface type "IFoo" and type "Foo1"',
    ):
        run(conformances=[("IFoo", "Foo1"), ("IFoo", "Foo1")])

    # Specifying duplicate type ids must raise an exception.
    with pytest.raises(
        RuntimeError,
        match='Duplicate type id 0 for interface type "IFoo"',
    ):
        run(conformances=[("IFoo", "Foo1", 0), ("IFoo", "Foo2", 0)])

    # If only one type is specified, createDynamicObject<IFoo> will always create the same type.
    assert np.all(run(conformances=[("IFoo", "Foo1", 0)]) == [1, 1, 1, 1])
    assert np.all(run(conformances=[("IFoo", "Foo2", 1)]) == [2, 2, 2, 2])

    # If multiple types are specified, createDynamicObject<IFoo> will create different types.
    # The last specified type will be used as a default for unknown type ids.
    assert np.all(
        run(
            conformances=[
                ("IFoo", "Foo1", 0),
                ("IFoo", "Foo2", 1),
            ]
        )
        == [1, 2, 2, 2]
    )

    # If no type ids are provided, they are auto-generated, starting from 0.
    assert np.all(
        run(
            conformances=[
                ("IFoo", "Foo1"),
                ("IFoo", "Foo2"),
                ("IFoo", "Foo3"),
                ("IFoo", "Foo4"),
            ]
        )
        == [1, 2, 3, 4]
    )

    # Type ids can be explicitly specified.
    assert np.all(
        run(
            conformances=[
                ("IFoo", "Foo1", 3),
                ("IFoo", "Foo2", 2),
                ("IFoo", "Foo3", 1),
                ("IFoo", "Foo4", 0),
            ]
        )
        == [4, 3, 2, 1]
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_type_conformance_module_cache(device_type: spy.DeviceType, tmpdir: str):

    # Run the same sequence of conformance tests twice, second time, slang modules will be loaded from the cache.
    for _ in [0, 1]:
        device = spy.Device(
            type=device_type,
            module_cache_path=tmpdir,
            compiler_options={"include_paths": [Path(__file__).parent]},
            label=f"type-conformance-cache-{device_type.name}",
        )

        def run(conformances: Sequence[Union[tuple[str, str], tuple[str, str, int]]]):
            module = device.load_module("test_type_conformance_module_cache.slang")
            entry_point = module.entry_point("compute_main_2")
            program = device.link_program(
                modules=[module], entry_points=[entry_point], type_conformances=conformances
            )
            kernel = device.create_compute_kernel(program)
            result = device.create_buffer(
                element_count=4, struct_size=4, usage=spy.BufferUsage.unordered_access
            )
            kernel.dispatch(thread_count=[4, 1, 1], result=result)
            return result.to_numpy().view(np.uint32)

        # Conforming to non-existing interface type must raise an exception.
        with pytest.raises(RuntimeError, match='Interface type "IUnknown" not found'):
            run(conformances=[("IUnknown", "Foo1")])

        # Conforming to non-existing type must raise an exception.
        with pytest.raises(RuntimeError, match='Type "Unknown" not found'):
            run(conformances=[("IFoo", "Unknown")])

        # Specifying duplicate type conformances must raise an exception.
        with pytest.raises(
            RuntimeError,
            match='Duplicate type conformance entry for interface type "IFoo" and type "Foo1"',
        ):
            run(conformances=[("IFoo", "Foo1"), ("IFoo", "Foo1")])

        # Specifying duplicate type ids must raise an exception.
        with pytest.raises(
            RuntimeError,
            match='Duplicate type id 0 for interface type "IFoo"',
        ):
            run(conformances=[("IFoo", "Foo1", 0), ("IFoo", "Foo2", 0)])

        # If only one type is specified, createDynamicObject<IFoo> will always create the same type.
        assert np.all(run(conformances=[("IFoo", "Foo1", 0)]) == [1, 1, 1, 1])
        assert np.all(run(conformances=[("IFoo", "Foo2", 1)]) == [2, 2, 2, 2])

        # If multiple types are specified, createDynamicObject<IFoo> will create different types.
        # The last specified type will be used as a default for unknown type ids.
        assert np.all(
            run(
                conformances=[
                    ("IFoo", "Foo1", 0),
                    ("IFoo", "Foo2", 1),
                ]
            )
            == [1, 2, 2, 2]
        )

        # If no type ids are provided, they are auto-generated, starting from 0.
        assert np.all(
            run(
                conformances=[
                    ("IFoo", "Foo1"),
                    ("IFoo", "Foo2"),
                    ("IFoo", "Foo3"),
                    ("IFoo", "Foo4"),
                ]
            )
            == [1, 2, 3, 4]
        )

        # Type ids can be explicitly specified.
        assert np.all(
            run(
                conformances=[
                    ("IFoo", "Foo1", 3),
                    ("IFoo", "Foo2", 2),
                    ("IFoo", "Foo3", 1),
                    ("IFoo", "Foo4", 0),
                ]
            )
            == [4, 3, 2, 1]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
