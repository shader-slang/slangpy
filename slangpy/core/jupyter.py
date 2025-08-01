# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from slangpy import TextureType
from slangpy import Module, Function, Struct
from slangpy import (
    Device,
    Texture,
    Bitmap,
    ModifierID,
    FunctionReflection,
    TypeReflection,
)
from slangpy.reflection import SlangFunction, SlangType
from slangpy.types.buffer import NDBuffer, NativeNDBuffer
from slangpy.types.tensor import Tensor, NativeTensor

from IPython.core.getipython import get_ipython  # type: ignore
from IPython.core.formatters import DisplayFormatter  # type: ignore
from IPython.lib import pretty  # type: ignore
from pathlib import Path
from typing import Optional, Iterable, Any
import numpy as np


def temp_dir():
    temp_path = Path(".temp")
    temp_path.mkdir(exist_ok=True)
    return temp_path


def format_bitmap_png(bmp: Bitmap):
    # Cast component to 8 bit if the source has more bits
    # We always do gamma correction here - is there a better way?
    if bmp.component_type != Bitmap.ComponentType.uint8:
        bmp = bmp.convert(component_type=Bitmap.ComponentType.uint8, srgb_gamma=True)

    # Make sure bitmap is RGB/RGBA or saving to PNG will fail
    if bmp.channel_count not in (3, 4):
        bmp_np = np.array(bmp, copy=False)
        if bmp.channel_count == 1:
            # Single channel? -> Duplicate to get mono RGB
            bmp_np = np.stack((bmp_np, bmp_np, bmp_np), axis=-1)
        elif bmp.channel_count == 2:
            # Two channels? -> Add zero-filled blue channel
            bmp_np = np.concatenate((bmp_np, np.zeros_like(bmp_np[..., 0:1])), axis=-1)
        else:
            # More than four channels? -> Truncate to first three.
            # We truncate to three instead of four channels because interpreting
            # unknown data as an alpha channel can lead to unintuitive results
            bmp_np = bmp_np[..., :3]

        bmp = Bitmap(bmp_np, Bitmap.PixelFormat.rgb)

    # Ideally we could encode in-memory, but SGL doesn't allow this currently
    # Encode to temp directory and load from there
    file = temp_dir() / "jupyter_texture.png"
    bmp.write(file)
    return open(file, "rb").read()


def format_texture_png(tex: Texture):
    return format_bitmap_png(tex.to_bitmap())


class BreakableList:
    """
    Helper class to take a list of pretty-printable fragments and print them
    with breakable separators.
    """

    def __init__(self, args: Iterable[Any], separator: Optional[str] = None):
        super().__init__()
        self.args = args
        self.separator = separator

    def _repr_pretty_(self, p: pretty.RepresentationPrinter, cycle: bool):
        for i, arg in enumerate(self.args):
            if i > 0:
                if self.separator:
                    p.text(self.separator)
                p.breakable()
            if isinstance(arg, str):
                p.text(arg)
            else:
                p.pretty(arg)


def comma_list(args: Iterable[Any]):
    """
    Turns the argument list into a pretty-printable list, separated by commas
    """
    return BreakableList(args, separator=",")


def spaced_list(args: Iterable[Any]):
    """
    Turns the argument list into a pretty-printable list, separated by spaces
    """
    return BreakableList(args)


def pprint_all(p: pretty.RepresentationPrinter, args: Iterable[Any]):
    """
    Helper method for printing a sequence of pretty printable pieces
    """
    for arg in args:
        if isinstance(arg, str):
            p.text(arg)
        else:
            p.pretty(arg)


def _get_modifiers(refl: Any) -> list[str]:
    """
    Helper function for getting the modifiers of a reflection object as a list of strings
    """
    return [name for name, value in ModifierID.__members__.items() if refl.has_modifier(value)]


def format_scalar_type(
    scalar: TypeReflection.ScalarType, p: pretty.RepresentationPrinter, cycle: bool
):
    mapping = {
        TypeReflection.ScalarType.none: "none",
        TypeReflection.ScalarType.void: "void",
        TypeReflection.ScalarType.bool: "bool",
        TypeReflection.ScalarType.int32: "int",
        TypeReflection.ScalarType.uint32: "uint",
        TypeReflection.ScalarType.int64: "int64",
        TypeReflection.ScalarType.uint64: "uint64",
        TypeReflection.ScalarType.float16: "half",
        TypeReflection.ScalarType.float32: "float",
        TypeReflection.ScalarType.float64: "double",
        TypeReflection.ScalarType.int8: "int8",
        TypeReflection.ScalarType.uint8: "uint8",
        TypeReflection.ScalarType.int16: "int16",
        TypeReflection.ScalarType.uint16: "uint16",
    }
    p.text(mapping[scalar])


def format_type_refl(refl: TypeReflection, p: pretty.RepresentationPrinter, cycle: bool):
    if refl.kind == TypeReflection.Kind.vector:
        # Pretty print vectors to be in the form float4 instead of vector<float, 4>
        pprint_all(p, (refl.scalar_type, str(refl.col_count)))
    else:
        p.text(refl.full_name)


def format_function_refl(func: FunctionReflection, p: pretty.RepresentationPrinter, cycle: bool):
    fragments = []

    modifiers = _get_modifiers(func)
    if modifiers:
        fragments.extend(("[", comma_list(modifiers), "]"))
    if func.has_modifier(ModifierID.static):
        fragments.append("static")
    if func.has_modifier(ModifierID.nodiff):
        fragments.append("no_diff")

    fragments.append(func.return_type)
    fragments.append(f"{func.name}(")

    params = [
        spaced_list(_get_modifiers(param) + [param.type, param.name]) for param in func.parameters
    ]
    fragments.append(comma_list(params))

    fragments.append(")")

    pprint_all(p, fragments)


def format_slang_function(func: SlangFunction, p: pretty.RepresentationPrinter, cycle: bool):
    pprint_all(p, ("SlangFunction(", func.reflection, ")"))


def format_type(st: SlangType, p: pretty.RepresentationPrinter, cycle: bool):
    pprint_all(p, ("SlangType(", st.type_reflection, ")"))


def format_function(func: Function, p: pretty.RepresentationPrinter, cycle: bool):
    head = 'slangpy.Function("'
    tail = f'", module="{func.module.device_module.name}")'
    sl_func = func._slang_func
    if not sl_func.is_overloaded:
        pprint_all(p, (head, sl_func.reflection, tail))
    else:
        sl_overloads = sl_func.overloads
        with p.group(4, f"{head}{len(sl_overloads)} overloads:", tail):
            for f in sl_overloads:
                p.pretty(f.reflection)


def format_module(m: Module, p: pretty.RepresentationPrinter, cycle: bool):
    path = m.device_module.path
    p.text(f'slangpy.Module("{m.device_module.name}", path="{path.absolute()}")')


def format_struct(m: Struct, p: pretty.RepresentationPrinter, cycle: bool):
    pprint_all(
        p,
        (
            'slangpy.Struct("',
            m.struct.type_reflection,
            f'", module="{m.device_module.name}")',
        ),
    )


def format_ndbuffer(buf: NativeNDBuffer, p: pretty.RepresentationPrinter, cycle: bool):
    pprint_all(p, ("NDBuffer(shape=", buf.shape, ", dtype=", buf.dtype.type_reflection, ")"))


def format_tensor(t: NativeTensor, p: pretty.RepresentationPrinter, cycle: bool):
    pprint_all(p, ("Tensor(shape=", t.shape, ", dtype=", t.dtype.type_reflection, ")"))


def format_texture(tex: Texture, p: pretty.RepresentationPrinter, cycle: bool):
    mapping = {
        TextureType.texture_1d: ("Texture1D", 1),
        TextureType.texture_2d: ("Texture2D", 2),
        TextureType.texture_3d: ("Texture3D", 3),
        TextureType.texture_cube: ("TextureCube", 3),
    }
    name, dims = mapping[tex.desc.type]

    fragments = [f"format={tex.format}", f"width={tex.width}"]
    if dims > 1:
        fragments.append(f"height={tex.height}")
    if dims > 2:
        fragments.append(f"depth={tex.depth}")
    if tex.array_length > 1:
        fragments.append(f"array_size={tex.array_length}")
    if tex.desc.sample_count > 1:
        fragments.append(f"sample_count={tex.desc.sample_count}")

    pprint_all(p, (f"{name}(", comma_list(fragments), ")"))


def setup_in_jupyter(device: Device):
    ipython = get_ipython()
    if ipython is None:
        return

    display_formatter = ipython.display_formatter
    if display_formatter is None:
        return
    assert isinstance(display_formatter, DisplayFormatter)

    png_formatter = display_formatter.formatters["image/png"]
    png_formatter.for_type(Bitmap, format_bitmap_png)
    png_formatter.for_type(Texture, format_texture_png)

    pretty_formatter = display_formatter.formatters["text/plain"]
    pretty_formatter.for_type(TypeReflection.ScalarType, format_scalar_type)
    pretty_formatter.for_type(TypeReflection, format_type_refl)
    pretty_formatter.for_type(FunctionReflection, format_function_refl)
    pretty_formatter.for_type(SlangFunction, format_slang_function)
    pretty_formatter.for_type(SlangType, format_type)
    pretty_formatter.for_type(Function, format_function)
    pretty_formatter.for_type(Module, format_module)
    pretty_formatter.for_type(Struct, format_struct)
    pretty_formatter.for_type(NDBuffer, format_ndbuffer)
    pretty_formatter.for_type(NativeNDBuffer, format_ndbuffer)
    pretty_formatter.for_type(Tensor, format_tensor)
    pretty_formatter.for_type(NativeTensor, format_tensor)
    pretty_formatter.for_type(Texture, format_texture)

    if device.desc.enable_hot_reload:
        # Set delay to 0 to avoid evaluating cells with stale kernels
        device.set_hot_reload_delay(0)
        # Trigger a hot-reload check before a cell gets executed to make sure we have the latest changes
        ipython.events.register("pre_execute", lambda: device.hot_reload_check())
