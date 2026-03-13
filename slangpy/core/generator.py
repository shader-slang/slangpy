# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any

from slangpy.bindings.codegen import CodeGen

if TYPE_CHECKING:
    from slangpy.core.function import FunctionBuildInfo


class KernelGenException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _is_slangpy_vector(type: Any) -> bool:
    return (
        hasattr(type, "element_type")
        and hasattr(type, "shape")
        and len(type.shape) == 1
        and type.shape[0] <= 4
    )


def generate_constants(build_info: "FunctionBuildInfo", cg: CodeGen) -> None:
    if build_info.constants is not None:
        for k, v in build_info.constants.items():
            if isinstance(v, bool):
                cg.constants.append_statement(
                    f"export static const bool {k} = {'true' if v else 'false'}"
                )
            elif isinstance(v, (int, float)):
                cg.constants.append_statement(f"export static const {type(v).__name__} {k} = {v}")
            elif _is_slangpy_vector(v):
                # Cheeky logic to take, eg, {0,0,0} -> float3(0,0,0)
                tn = type(v).__name__
                txt = f"{tn}({str(v)[1:-1]})"
                cg.constants.append_statement(f"export static const {tn} {k} = {txt}")
            else:
                raise KernelGenException(
                    f"Constant value '{k}' must be an int, float or bool, not {type(v).__name__}"
                )
