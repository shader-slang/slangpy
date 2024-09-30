from typing import TYPE_CHECKING, Any, Optional, cast

from kernelfunctions.backend import Device
from kernelfunctions.core.boundvariable import BoundVariableException
from kernelfunctions.shapes import TConcreteShape, check_concrete

from .enums import AccessType

if TYPE_CHECKING:
    from .boundvariable import BoundVariable, BoundCall


class BoundCallRuntime:
    def __init__(self, call: 'BoundCall'):
        super().__init__()
        self.args = [BoundVariableRuntime(arg) for arg in call.args]
        self.kwargs = {name: BoundVariableRuntime(
            arg) for name, arg in call.kwargs.items()}


class BoundVariableRuntime:
    def __init__(self, source: 'BoundVariable'):
        super().__init__()

        # Data potentially used by type marshalls
        self.access = source.access
        self.transform: Optional[TConcreteShape] = check_concrete(
            source.transform) if source.transform is not None else None
        self.slang_shape = source.slang.primal.get_shape()

        # Temp data stored / updated each call
        self.shape: TConcreteShape = ()

        # Primal calls
        self._get_shape = source.python.primal.get_shape
        self._create_calldata = source.python.primal.create_calldata
        self._read_calldata = source.python.primal.read_calldata
        self._create_output = source.python.primal.create_output
        self._read_output = source.python.primal.read_output

        # Internal data
        self._source_for_exceptions = source
        self._name = source.python.name
        self._variable_name = source.variable_name
        self._children: Optional[dict[str, BoundVariableRuntime]] = {
            name: BoundVariableRuntime(child) for name, child in source.children.items()
        } if source.children is not None else None

    def populate_call_shape(self, call_shape: list[int], value: Any):
        """
        Recursively calculate call shape for the node
        """
        if self._children is not None:
            for name, child in self._children.items():
                child.populate_call_shape(call_shape, value[name])
        elif value is not None:
            # Get concrete primal shape
            shape = cast(TConcreteShape, self._get_shape(value))
            tf = cast(TConcreteShape, self.transform)
            csl = len(call_shape)
            self.shape = shape

            for i in range(len(tf)):
                # Get value shape and corresponding index in the overall call shape
                shape_dim = cast(int, shape[i])
                call_idx = cast(int, tf[i])

                # Not interested in dimensionality for sub-kernel elements
                if call_idx >= csl:
                    continue

                # Apply shape, failing if we find mismatch
                cs = call_shape[call_idx]
                if cs != shape_dim:
                    if cs != 1 and shape_dim != 1:
                        raise BoundVariableException(
                            f"Shape mismatch for {self._variable_name} between input and output", self._source_for_exceptions)
                    if shape_dim != 1:
                        call_shape[call_idx] = shape_dim

    def write_call_data_pre_dispatch(self, device: Device, call_shape: tuple[int], call_data: dict[str, Any], value: Any):
        """Writes value to call data dictionary pre-dispatch"""
        if self._children is not None:
            res = {}
            for name, child in self._children.items():
                child.write_call_data_pre_dispatch(device, call_shape, res, value[name])
            if len(res) > 0:
                call_data[self._variable_name] = res
        else:
            # Get concrete primal shape
            shape = self.shape

            # Get call shape + append slang primal shape
            full_cs = call_shape + self.slang_shape

            # Broadcast occurs if the shape of the input is different from the shape of the output
            broadcast = []
            transform = cast(list[int], self.transform)
            for i in range(len(transform)):
                csidx = transform[i]
                broadcast.append(full_cs[csidx] != shape[i])

            cd_val = self._create_calldata(
                device, self, broadcast, value)
            if cd_val is not None:
                call_data[self._variable_name] = cd_val

    def read_call_data_post_dispatch(self, device: Device, call_data: dict[str, Any], value: Any):
        """Reads value from call data dictionary post-dispatch"""
        if self._children is not None:
            cd_val = call_data.get(self._variable_name, None)
            for name, child in self._children.items():
                if child._variable_name in cd_val:
                    child.read_call_data_post_dispatch(device, cd_val, value[name])
        else:
            cd_val = call_data.get(self._variable_name, None)
            if cd_val is not None:
                self._read_calldata(device, self, value, cd_val)

    def read_output(self, device: Device, data: Any):
        """Reads output from function for a return value"""
        if self._children is not None:
            assert isinstance(data, dict)
            res = {}
            for name, child in self._children.items():
                child_data = data.get(child._name, None)
                if child_data is not None:
                    res[name] = child.read_output(device, child_data)
            return res
        else:
            if self.access[0] in [AccessType.write, AccessType.readwrite]:
                return self._read_output(device, data)
