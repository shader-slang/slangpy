Core
----

.. py:class:: slangpy.DataType

    Base class: :py:class:`enum.Enum`





----

.. py:class:: slangpy.Object

    Base class for all reference counted objects.



----

.. py:class:: slangpy.Bitmap

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, pixel_format: slangpy.Bitmap.PixelFormat, component_type: slangpy.DataStruct.Type, width: int, height: int, channel_count: int = 0, channel_names: collections.abc.Sequence[str] = [], srgb_gamma: bool | None = None) -> None

    .. py:method:: __init__(self, data: ndarray[device='cpu'], pixel_format: slangpy.Bitmap.PixelFormat | None = None, channel_names: collections.abc.Sequence[str] | None = None, srgb_gamma: bool | None = None) -> None
        :no-index:

    .. py:method:: __init__(self, path: str | os.PathLike) -> None
        :no-index:

    .. py:class:: slangpy.Bitmap.PixelFormat

        Base class: :py:class:`enum.Enum`



    .. py:class:: slangpy.Bitmap.ComponentType
        Alias class: :py:class:`slangpy.DataStruct.Type`

    .. py:class:: slangpy.Bitmap.FileFormat

        Base class: :py:class:`enum.Enum`



    .. py:staticmethod:: load_from_file(path: str | os.PathLike) -> slangpy.Bitmap

        N/A

    .. py:staticmethod:: load_from_numpy(data: ndarray[device='cpu']) -> slangpy.Bitmap

        N/A

    .. py:property:: pixel_format
        :type: slangpy.Bitmap.PixelFormat

        The pixel format.

    .. py:property:: component_type
        :type: slangpy.DataStruct.Type

        The component type.

    .. py:property:: pixel_struct
        :type: slangpy.DataStruct

        DataStruct describing the pixel layout.

    .. py:property:: width
        :type: int

        The width of the bitmap in pixels.

    .. py:property:: height
        :type: int

        The height of the bitmap in pixels.

    .. py:property:: pixel_count
        :type: int

        The total number of pixels in the bitmap.

    .. py:property:: channel_count
        :type: int

        The number of channels in the bitmap.

    .. py:property:: channel_names
        :type: list[str]

        The names of the channels in the bitmap.

    .. py:property:: srgb_gamma
        :type: bool

        True if the bitmap is in sRGB gamma space.

    .. py:method:: has_alpha(self) -> bool

        Returns true if the bitmap has an alpha channel.

    .. py:property:: bytes_per_pixel
        :type: int

        The number of bytes per pixel.

    .. py:property:: buffer_size
        :type: int

        The total size of the bitmap in bytes.

    .. py:method:: empty(self) -> bool

        True if bitmap is empty.

    .. py:method:: clear(self) -> None

        Clears the bitmap to zeros.

    .. py:method:: vflip(self) -> None

        Vertically flip the bitmap.

    .. py:method:: split(self) -> list[tuple[str, slangpy.Bitmap]]

        Split bitmap into multiple bitmaps, each containing the channels with
        the same prefix.

        For example, if the bitmap has channels `albedo.R`, `albedo.G`,
        `albedo.B`, `normal.R`, `normal.G`, `normal.B`, this function will
        return two bitmaps, one containing the channels `albedo.R`,
        `albedo.G`, `albedo.B` and the other containing the channels
        `normal.R`, `normal.G`, `normal.B`.

        Common pixel formats (e.g. `y`, `rgb`, `rgba`) are automatically
        detected and used for the split bitmaps.

        Any channels that do not have a prefix will be returned in the bitmap
        with the empty prefix.

        Returns:
            Returns a list of (prefix, bitmap) pairs.

    .. py:method:: convert(self, pixel_format: slangpy.Bitmap.PixelFormat | None = None, component_type: slangpy.DataStruct.Type | None = None, srgb_gamma: bool | None = None) -> slangpy.Bitmap

    .. py:method:: resample(self, target: slangpy.Bitmap, filter: slangpy.BoxFilter | slangpy.TentFilter | slangpy.GaussianFilter | slangpy.MitchellFilter | slangpy.LanczosFilter = ..., bc: tuple[slangpy.FilterBoundaryCondition, slangpy.FilterBoundaryCondition] = (FilterBoundaryCondition.clamp, FilterBoundaryCondition.clamp), clamp: tuple[float, float] = (-inf, inf)) -> None

        Resample into a pre-allocated target bitmap using a separable filter.
        Source and target must have the same pixel format, component type and
        channel count. Only supports float16 and float32 component types.

        Parameter ``target``:
            Pre-allocated target bitmap.

        Parameter ``filter``:
            Reconstruction filter to use.

        Parameter ``bc``:
            Horizontal and vertical boundary conditions for out-of-bounds
            lookups.

        Parameter ``clamp``:
            Optional (min, max) range to clamp output values.

    .. py:method:: resample(self, width: int, height: int, filter: slangpy.BoxFilter | slangpy.TentFilter | slangpy.GaussianFilter | slangpy.MitchellFilter | slangpy.LanczosFilter = ..., bc: tuple[slangpy.FilterBoundaryCondition, slangpy.FilterBoundaryCondition] = (FilterBoundaryCondition.clamp, FilterBoundaryCondition.clamp), clamp: tuple[float, float] = (-inf, inf)) -> slangpy.Bitmap
        :no-index:

        N/A

    .. py:method:: write(self, path: str | os.PathLike, format: slangpy.Bitmap.FileFormat = FileFormat.auto, quality: int = -1) -> None

    .. py:method:: write_async(self, path: str | os.PathLike, format: slangpy.Bitmap.FileFormat = FileFormat.auto, quality: int = -1) -> None

    .. py:staticmethod:: read_multiple(paths: collections.abc.Sequence[str | os.PathLike], format: slangpy.Bitmap.FileFormat = FileFormat.auto) -> list[slangpy.Bitmap]

        Load a list of bitmaps from multiple paths. Uses multi-threading to
        load bitmaps in parallel.



----

.. py:class:: slangpy.DataStruct

    Base class: :py:class:`slangpy.Object`

    Structured data definition.

    This class is used to describe a structured data type layout. It is
    used by the DataStructConverter class to convert between different
    layouts.

    .. py:method:: __init__(self, pack: bool = False, byte_order: slangpy.DataStruct.ByteOrder = ByteOrder.host) -> None

        Constructor.

        Parameter ``pack``:
            If true, the struct will be packed.

        Parameter ``byte_order``:
            Byte order of the struct.

    .. py:class:: slangpy.DataStruct.Type

        Base class: :py:class:`enum.Enum`

        Struct field type.

    .. py:class:: slangpy.DataStruct.Flags

        Base class: :py:class:`enum.IntFlag`

        Struct field flags.

    .. py:class:: slangpy.DataStruct.ByteOrder

        Base class: :py:class:`enum.Enum`

        Byte order.

    .. py:class:: slangpy.DataStruct.Field

        Struct field.

        .. py:property:: name
            :type: str

            Name of the field.

        .. py:property:: type
            :type: slangpy.DataStruct.Type

            Type of the field.

        .. py:property:: flags
            :type: slangpy.DataStruct.Flags

            Field flags.

        .. py:property:: size
            :type: int

            Size of the field in bytes.

        .. py:property:: offset
            :type: int

            Offset of the field in bytes.

        .. py:property:: default_value
            :type: float

            Default value.

        .. py:method:: is_integer(self) -> bool

            Check if the field is an integer type.

        .. py:method:: is_unsigned(self) -> bool

            Check if the field is an unsigned type.

        .. py:method:: is_signed(self) -> bool

            Check if the field is a signed type.

        .. py:method:: is_float(self) -> bool

            Check if the field is a floating point type.

    .. py:method:: append(self, field: slangpy.DataStruct.Field) -> slangpy.DataStruct

        Append a field to the struct.

    .. py:method:: append(self, name: str, type: slangpy.DataStruct.Type, flags: slangpy.DataStruct.Flags = 0, default_value: float = 0.0, blend: collections.abc.Sequence[tuple[float, str]] = []) -> slangpy.DataStruct
        :no-index:

        Append a field to the struct.

        Parameter ``name``:
            Name of the field.

        Parameter ``type``:
            Type of the field.

        Parameter ``flags``:
            Field flags.

        Parameter ``default_value``:
            Default value.

        Parameter ``blend``:
            List of blend weights/names.

        Returns:
            Reference to the struct.

    .. py:method:: has_field(self, name: str) -> bool

        Check if a field with the specified name exists.

    .. py:method:: field(self, name: str) -> slangpy.DataStruct.Field

        Access field by name. Throws if field is not found.

    .. py:property:: size
        :type: int

        The size of the struct in bytes (with padding).

    .. py:property:: alignment
        :type: int

        The alignment of the struct in bytes.

    .. py:property:: byte_order
        :type: slangpy.DataStruct.ByteOrder

        The byte order of the struct.

    .. py:staticmethod:: type_size(arg: slangpy.DataStruct.Type, /) -> int

        Get the size of a type in bytes.

    .. py:staticmethod:: type_range(arg: slangpy.DataStruct.Type, /) -> tuple[float, float]

        Get the numeric range of a type.

    .. py:staticmethod:: is_integer(arg: slangpy.DataStruct.Type, /) -> bool

        Check if ``type`` is an integer type.

    .. py:staticmethod:: is_unsigned(arg: slangpy.DataStruct.Type, /) -> bool

        Check if ``type`` is an unsigned type.

    .. py:staticmethod:: is_signed(arg: slangpy.DataStruct.Type, /) -> bool

        Check if ``type`` is a signed type.

    .. py:staticmethod:: is_float(arg: slangpy.DataStruct.Type, /) -> bool

        Check if ``type`` is a floating point type.



----

.. py:class:: slangpy.DataStructConverter

    Base class: :py:class:`slangpy.Object`

    Data struct converter.

    This helper class can be used to convert between structs with
    different layouts.

    .. py:method:: __init__(self, src: slangpy.DataStruct, dst: slangpy.DataStruct) -> None

        Constructor.

        Parameter ``src``:
            Source struct definition.

        Parameter ``dst``:
            Destination struct definition.

    .. py:property:: src
        :type: slangpy.DataStruct

        The source struct definition.

    .. py:property:: dst
        :type: slangpy.DataStruct

        The destination struct definition.

    .. py:method:: convert(self, input: bytes) -> bytes



----

.. py:class:: slangpy.Timer



    .. py:method:: __init__(self) -> None

    .. py:method:: reset(self) -> None

        Reset the timer.

    .. py:method:: elapsed_s(self) -> float

        Elapsed seconds since last reset.

    .. py:method:: elapsed_ms(self) -> float

        Elapsed milliseconds since last reset.

    .. py:method:: elapsed_us(self) -> float

        Elapsed microseconds since last reset.

    .. py:method:: elapsed_ns(self) -> float

        Elapsed nanoseconds since last reset.

    .. py:staticmethod:: delta_s(start: int, end: int) -> float

        Compute elapsed seconds between two time points.

    .. py:staticmethod:: delta_ms(start: int, end: int) -> float

        Compute elapsed milliseconds between two time points.

    .. py:staticmethod:: delta_us(start: int, end: int) -> float

        Compute elapsed microseconds between two time points.

    .. py:staticmethod:: delta_ns(start: int, end: int) -> float

        Compute elapsed nanoseconds between two time points.

    .. py:staticmethod:: now() -> int

        Current time point in nanoseconds since epoch.



----

.. py:class:: slangpy.SHA1

    Helper to compute SHA-1 hash.

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, data: bytes) -> None
        :no-index:

    .. py:method:: __init__(self, str: str) -> None
        :no-index:

    .. py:method:: update(self, data: bytes) -> slangpy.SHA1

        Update hash by adding the given data.

        Parameter ``data``:
            Data to hash.

        Parameter ``len``:
            Length of data in bytes.

    .. py:method:: update(self, str: str) -> slangpy.SHA1
        :no-index:

        Update hash by adding the given string.

        Parameter ``str``:
            String to hash.

    .. py:method:: digest(self) -> bytes

        Return the message digest.

    .. py:method:: hex_digest(self) -> str

        Return the message digest as a hex string.



----

Constants
---------

.. py:data:: slangpy.ALL_LAYERS
    :type: int
    :value: 4294967295



----

.. py:data:: slangpy.ALL_MIPS
    :type: int
    :value: 4294967295



----

.. py:data:: slangpy.SGL_VERSION_MAJOR
    :type: int
    :value: 0



----

.. py:data:: slangpy.SGL_VERSION_MINOR
    :type: int
    :value: 43



----

.. py:data:: slangpy.SGL_VERSION_PATCH
    :type: int
    :value: 0



----

.. py:data:: slangpy.SGL_VERSION
    :type: str
    :value: "0.43.0"



----

Logging
-------

.. py:class:: slangpy.LogLevel

    Base class: :py:class:`enum.IntEnum`

    Log level.



----

.. py:class:: slangpy.LogFrequency

    Base class: :py:class:`enum.Enum`

    Log frequency.



----

.. py:class:: slangpy.Logger

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, level: slangpy.LogLevel = LogLevel.info, name: str = '', use_default_outputs: bool = True) -> None

        Constructor.

        Parameter ``level``:
            The log level to use (messages with level >= this will be logged).

        Parameter ``name``:
            The name of the logger.

        Parameter ``use_default_outputs``:
            Whether to use the default outputs (console + debug console on
            windows).

    .. py:property:: level
        :type: slangpy.LogLevel

        The log level.

    .. py:property:: name
        :type: str

        The name of the logger.

    .. py:method:: add_console_output(self, colored: bool = True) -> slangpy.LoggerOutput

        Add a console logger output.

        Parameter ``colored``:
            Whether to use colored output.

        Returns:
            The created logger output.

    .. py:method:: add_file_output(self, path: str | os.PathLike) -> slangpy.LoggerOutput

        Add a file logger output.

        Parameter ``path``:
            The path to the log file.

        Returns:
            The created logger output.

    .. py:method:: add_debug_console_output(self) -> slangpy.LoggerOutput

        Add a debug console logger output (Windows only).

        Returns:
            The created logger output.

    .. py:method:: add_output(self, output: slangpy.LoggerOutput) -> None

        Add a logger output.

        Parameter ``output``:
            The logger output to add.

    .. py:method:: use_same_outputs(self, other: slangpy.Logger) -> None

        Use the same outputs as the given logger.

        Parameter ``other``:
            Logger to copy outputs from.

    .. py:method:: remove_output(self, output: slangpy.LoggerOutput) -> None

        Remove a logger output.

        Parameter ``output``:
            The logger output to remove.

    .. py:method:: remove_all_outputs(self) -> None

        Remove all logger outputs.

    .. py:method:: log(self, level: slangpy.LogLevel, msg: str, frequency: slangpy.LogFrequency = LogFrequency.always) -> None

        Log a message.

        Parameter ``level``:
            The log level.

        Parameter ``msg``:
            The message.

        Parameter ``frequency``:
            The log frequency.

    .. py:method:: debug(self, msg: str) -> None

    .. py:method:: info(self, msg: str) -> None

    .. py:method:: warn(self, msg: str) -> None

    .. py:method:: error(self, msg: str) -> None

    .. py:method:: fatal(self, msg: str) -> None

    .. py:method:: debug_once(self, msg: str) -> None

    .. py:method:: info_once(self, msg: str) -> None

    .. py:method:: warn_once(self, msg: str) -> None

    .. py:method:: error_once(self, msg: str) -> None

    .. py:method:: fatal_once(self, msg: str) -> None

    .. py:staticmethod:: get() -> slangpy.Logger

        Returns the global logger instance.



----

.. py:class:: slangpy.LoggerOutput

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self) -> None

    .. py:method:: write(self, level: slangpy.LogLevel, name: str, msg: str) -> None

        Write a log message.

        Parameter ``level``:
            The log level.

        Parameter ``module``:
            The module name.

        Parameter ``msg``:
            The message.



----

.. py:class:: slangpy.ConsoleLoggerOutput

    Base class: :py:class:`slangpy.LoggerOutput`



    .. py:method:: __init__(self, colored: bool = True) -> None

    .. py:attribute:: slangpy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION
        :type: bool
        :value: False



----

.. py:class:: slangpy.FileLoggerOutput

    Base class: :py:class:`slangpy.LoggerOutput`



    .. py:method:: __init__(self, path: str | os.PathLike) -> None



----

.. py:class:: slangpy.DebugConsoleLoggerOutput

    Base class: :py:class:`slangpy.LoggerOutput`



    .. py:method:: __init__(self) -> None



----

.. py:function:: slangpy.log(level: slangpy.LogLevel, msg: str, frequency: slangpy.LogFrequency = LogFrequency.always) -> None

    Log a message.

    Parameter ``level``:
        The log level.

    Parameter ``msg``:
        The message.

    Parameter ``frequency``:
        The log frequency.



----

.. py:function:: slangpy.log_debug(msg: str) -> None



----

.. py:function:: slangpy.log_debug_once(msg: str) -> None



----

.. py:function:: slangpy.log_info(msg: str) -> None



----

.. py:function:: slangpy.log_info_once(msg: str) -> None



----

.. py:function:: slangpy.log_warn(msg: str) -> None



----

.. py:function:: slangpy.log_warn_once(msg: str) -> None



----

.. py:function:: slangpy.log_error(msg: str) -> None



----

.. py:function:: slangpy.log_error_once(msg: str) -> None



----

.. py:function:: slangpy.log_fatal(msg: str) -> None



----

.. py:function:: slangpy.log_fatal_once(msg: str) -> None



----

Windowing
---------

.. py:class:: slangpy.CursorMode

    Base class: :py:class:`enum.Enum`

    Mouse cursor modes.



----

.. py:class:: slangpy.CursorShape

    Base class: :py:class:`enum.Enum`

    Mouse cursor shapes.



----

.. py:class:: slangpy.WindowMode

    Base class: :py:class:`enum.Enum`

    Window modes.



----

.. py:class:: slangpy.Window

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, width: int = 1024, height: int = 1024, title: str = 'slangpy', mode: slangpy.WindowMode = WindowMode.normal, resizable: bool = True) -> None

        Constructor.

        Parameter ``width``:
            Width of the window in pixels.

        Parameter ``height``:
            Height of the window in pixels.

        Parameter ``title``:
            Title of the window.

        Parameter ``mode``:
            Window mode.

        Parameter ``resizable``:
            Whether the window is resizable.

    .. py:property:: width
        :type: int

        The width of the window in pixels.

    .. py:property:: height
        :type: int

        The height of the window in pixels.

    .. py:property:: size
        :type: slangpy.math.uint2

        Size of the window in pixels.

    .. py:method:: resize(self, width: int, height: int) -> None

        Resize the window.

        Parameter ``width``:
            The new width of the window in pixels.

        Parameter ``height``:
            The new height of the window in pixels.

    .. py:property:: position
        :type: slangpy.math.int2

        Position of the window on the screen in pixels.

    .. py:property:: title
        :type: str

        The title of the window.

    .. py:method:: close(self) -> None

        Close the window.

    .. py:method:: should_close(self) -> bool

        True if the window should be closed.

    .. py:method:: process_events(self) -> None

        Process any pending events.

    .. py:method:: set_clipboard(self, text: str) -> None

        Set the clipboard content.

    .. py:method:: get_clipboard(self) -> str | None

        Get the clipboard content.

    .. py:property:: cursor_mode
        :type: slangpy.CursorMode

        The mouse cursor mode.

    .. py:property:: cursor_shape
        :type: slangpy.CursorShape

        The mouse cursor shape.

    .. py:property:: on_resize
        :type: collections.abc.Callable[[int, int], None]

        Event handler to be called when the window is resized.

    .. py:property:: on_refresh
        :type: collections.abc.Callable[[], None]

        Event handler to be called when the window contents need to be
        refreshed.

    .. py:property:: on_keyboard_event
        :type: collections.abc.Callable[[slangpy.KeyboardEvent], None]

        Event handler to be called when a keyboard event occurs.

    .. py:property:: on_mouse_event
        :type: collections.abc.Callable[[slangpy.MouseEvent], None]

        Event handler to be called when a mouse event occurs.

    .. py:property:: on_gamepad_event
        :type: collections.abc.Callable[[slangpy.GamepadEvent], None]

        Event handler to be called when a gamepad event occurs.

    .. py:property:: on_gamepad_state
        :type: collections.abc.Callable[[slangpy.GamepadState], None]

        Event handler to be called when the gamepad state changes.

    .. py:property:: on_drop_files
        :type: collections.abc.Callable[[list[str]], None]

        Event handler to be called when files are dropped onto the window.



----

.. py:class:: slangpy.MouseButton

    Base class: :py:class:`enum.Enum`

    Mouse buttons.



----

.. py:class:: slangpy.KeyModifierFlags

    Base class: :py:class:`enum.IntFlag`

    Keyboard modifier flags.



----

.. py:class:: slangpy.KeyModifier

    Base class: :py:class:`enum.Enum`

    Keyboard modifiers.



----

.. py:class:: slangpy.KeyCode

    Base class: :py:class:`enum.Enum`

    Keyboard key codes.



----

.. py:class:: slangpy.KeyboardEventType

    Base class: :py:class:`enum.Enum`

    Keyboard event types.



----

.. py:class:: slangpy.KeyboardEvent



    .. py:method:: __init__(self) -> None

    .. py:property:: type
        :type: slangpy.KeyboardEventType

        The event type.

    .. py:property:: key
        :type: slangpy.KeyCode

        The key that was pressed/released/repeated.

    .. py:property:: mods
        :type: slangpy.KeyModifierFlags

        Keyboard modifier flags.

    .. py:property:: codepoint
        :type: int

        UTF-32 codepoint for input events.

    .. py:method:: is_key_press(self) -> bool

        Returns true if this event is a key press event.

    .. py:method:: is_key_release(self) -> bool

        Returns true if this event is a key release event.

    .. py:method:: is_key_repeat(self) -> bool

        Returns true if this event is a key repeat event.

    .. py:method:: is_input(self) -> bool

        Returns true if this event is an input event.

    .. py:method:: has_modifier(self, arg: slangpy.KeyModifier, /) -> bool

        Returns true if the specified modifier is set.



----

.. py:class:: slangpy.MouseEventType

    Base class: :py:class:`enum.Enum`

    Mouse event types.



----

.. py:class:: slangpy.MouseEvent



    .. py:method:: __init__(self) -> None

    .. py:property:: type
        :type: slangpy.MouseEventType

        The event type.

    .. py:property:: pos
        :type: slangpy.math.float2

        The mouse position.

    .. py:property:: scroll
        :type: slangpy.math.float2

        The scroll offset.

    .. py:property:: button
        :type: slangpy.MouseButton

        The mouse button that was pressed/released.

    .. py:property:: mods
        :type: slangpy.KeyModifierFlags

        Keyboard modifier flags.

    .. py:method:: is_button_down(self) -> bool

        Returns true if this event is a mouse button down event.

    .. py:method:: is_button_up(self) -> bool

        Returns true if this event is a mouse button up event.

    .. py:method:: is_move(self) -> bool

        Returns true if this event is a mouse move event.

    .. py:method:: is_scroll(self) -> bool

        Returns true if this event is a mouse scroll event.

    .. py:method:: has_modifier(self, arg: slangpy.KeyModifier, /) -> bool

        Returns true if the specified modifier is set.



----

.. py:class:: slangpy.GamepadEventType

    Base class: :py:class:`enum.Enum`

    Gamepad event types.



----

.. py:class:: slangpy.GamepadButton

    Base class: :py:class:`enum.Enum`

    Gamepad buttons.



----

.. py:class:: slangpy.GamepadEvent



    .. py:method:: __init__(self) -> None

    .. py:property:: type
        :type: slangpy.GamepadEventType

        The event type.

    .. py:property:: button
        :type: slangpy.GamepadButton

        The gamepad button that was pressed/released.

    .. py:method:: is_button_down(self) -> bool

        Returns true if this event is a gamepad button down event.

    .. py:method:: is_button_up(self) -> bool

        Returns true if this event is a gamepad button up event.

    .. py:method:: is_connect(self) -> bool

        Returns true if this event is a gamepad connect event.

    .. py:method:: is_disconnect(self) -> bool

        Returns true if this event is a gamepad disconnect event.



----

.. py:class:: slangpy.GamepadState



    .. py:method:: __init__(self) -> None

    .. py:property:: left_x
        :type: float

        X-axis of the left analog stick.

    .. py:property:: left_y
        :type: float

        Y-axis of the left analog stick.

    .. py:property:: right_x
        :type: float

        X-axis of the right analog stick.

    .. py:property:: right_y
        :type: float

        Y-axis of the right analog stick.

    .. py:property:: left_trigger
        :type: float

        Value of the left analog trigger.

    .. py:property:: right_trigger
        :type: float

        Value of the right analog trigger.

    .. py:property:: buttons
        :type: int

        Bitfield of gamepad buttons (see GamepadButton).

    .. py:method:: is_button_down(self, arg: slangpy.GamepadButton, /) -> bool

        Returns true if the specified button is down.



----

Platform
--------

.. py:class:: slangpy.platform.FileDialogFilter



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, name: str, pattern: str) -> None
        :no-index:

    .. py:method:: __init__(self, arg: tuple[str, str], /) -> None
        :no-index:

    .. py:property:: name
        :type: str

        Readable name (e.g. "JPEG").

    .. py:property:: pattern
        :type: str

        File extension pattern (e.g. "*.jpg" or "*.jpg,*.jpeg").



----

.. py:function:: slangpy.platform.open_file_dialog(filters: collections.abc.Sequence[slangpy.platform.FileDialogFilter] = []) -> pathlib.Path | None

    Show a file open dialog.

    Parameter ``filters``:
        List of file filters.

    Returns:
        The selected file path or nothing if the dialog was cancelled.



----

.. py:function:: slangpy.platform.save_file_dialog(filters: collections.abc.Sequence[slangpy.platform.FileDialogFilter] = []) -> pathlib.Path | None

    Show a file save dialog.

    Parameter ``filters``:
        List of file filters.

    Returns:
        The selected file path or nothing if the dialog was cancelled.



----

.. py:function:: slangpy.platform.choose_folder_dialog() -> pathlib.Path | None

    Show a folder selection dialog.

    Returns:
        The selected folder path or nothing if the dialog was cancelled.



----

.. py:function:: slangpy.platform.display_scale_factor() -> float

    The pixel scale factor of the primary display.



----

.. py:function:: slangpy.platform.executable_path() -> pathlib.Path

    The full path to the current executable.



----

.. py:function:: slangpy.platform.executable_directory() -> pathlib.Path

    The current executable directory.



----

.. py:function:: slangpy.platform.executable_name() -> str

    The current executable name.



----

.. py:function:: slangpy.platform.app_data_directory() -> pathlib.Path

    The application data directory.



----

.. py:function:: slangpy.platform.home_directory() -> pathlib.Path

    The home directory.



----

.. py:function:: slangpy.platform.project_directory() -> pathlib.Path

    The project source directory. Note that this is only valid during
    development.



----

.. py:function:: slangpy.platform.runtime_directory() -> pathlib.Path

    The runtime directory. This is the path where the sgl runtime library
    (sgl.dll, libsgl.so or libsgl.dynlib) resides.



----

.. py:class:: slangpy.platform.MemoryStats



    .. py:property:: rss
        :type: int

        Current resident/working set size in bytes.

    .. py:property:: peak_rss
        :type: int

        Peak resident/working set size in bytes.



----

.. py:function:: slangpy.platform.memory_stats() -> slangpy.platform.MemoryStats

    Get the current memory stats.



----

Threading
---------

.. py:function:: slangpy.thread.wait_for_tasks() -> None

    Wait for all tasks in the global task group.



----

Device
------

.. py:class:: slangpy.AccelerationStructure

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: desc
        :type: slangpy.AccelerationStructureDesc

    .. py:property:: handle
        :type: slangpy.AccelerationStructureHandle



----

.. py:class:: slangpy.AccelerationStructureBuildDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: inputs
        :type: list[slangpy.AccelerationStructureBuildInputInstances | slangpy.AccelerationStructureBuildInputTriangles | slangpy.AccelerationStructureBuildInputProceduralPrimitives | slangpy.AccelerationStructureBuildInputSpheres | slangpy.AccelerationStructureBuildInputLinearSweptSpheres]

        List of build inputs. All inputs must be of the same type.

    .. py:property:: motion_options
        :type: slangpy.AccelerationStructureBuildInputMotionOptions

    .. py:property:: mode
        :type: slangpy.AccelerationStructureBuildMode

    .. py:property:: flags
        :type: slangpy.AccelerationStructureBuildFlags



----

.. py:class:: slangpy.AccelerationStructureBuildFlags

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.AccelerationStructureBuildInputInstances



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: instance_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: instance_stride
        :type: int

    .. py:property:: instance_count
        :type: int



----

.. py:class:: slangpy.AccelerationStructureBuildInputMotionOptions



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: key_count
        :type: int

    .. py:property:: time_start
        :type: float

    .. py:property:: time_end
        :type: float



----

.. py:class:: slangpy.AccelerationStructureBuildInputProceduralPrimitives



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: aabb_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: aabb_stride
        :type: int

    .. py:property:: primitive_count
        :type: int

    .. py:property:: flags
        :type: slangpy.AccelerationStructureGeometryFlags



----

.. py:class:: slangpy.AccelerationStructureBuildInputTriangles



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: vertex_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: vertex_format
        :type: slangpy.Format

    .. py:property:: vertex_count
        :type: int

    .. py:property:: vertex_stride
        :type: int

    .. py:property:: index_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: index_format
        :type: slangpy.IndexFormat

    .. py:property:: index_count
        :type: int

    .. py:property:: pre_transform_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: flags
        :type: slangpy.AccelerationStructureGeometryFlags

    .. py:property:: opacity_micromap
        :type: slangpy.AccelerationStructureOpacityMicromapDesc | None

        Optional opacity micromap attachment.



----

.. py:class:: slangpy.AccelerationStructureBuildMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.AccelerationStructureCopyMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.AccelerationStructureDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: kind
        :type: slangpy.AccelerationStructureKind

    .. py:property:: size
        :type: int

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.AccelerationStructureGeometryFlags

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.AccelerationStructureHandle

    Acceleration structure handle.

    .. py:method:: __init__(self) -> None



----

.. py:class:: slangpy.AccelerationStructureInstanceDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: transform
        :type: slangpy.math.float3x4

    .. py:property:: instance_id
        :type: int

    .. py:property:: instance_mask
        :type: int

    .. py:property:: instance_contribution_to_hit_group_index
        :type: int

    .. py:property:: flags
        :type: slangpy.AccelerationStructureInstanceFlags

    .. py:property:: acceleration_structure
        :type: slangpy.AccelerationStructureHandle

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=uint8, shape=(64), writable=False]



----

.. py:class:: slangpy.AccelerationStructureInstanceFlags

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.AccelerationStructureInstanceList

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: size
        :type: int

    .. py:property:: instance_stride
        :type: int

    .. py:method:: resize(self, size: int) -> None

    .. py:method:: write(self, index: int, instance: slangpy.AccelerationStructureInstanceDesc) -> None

    .. py:method:: write(self, index: int, instances: collections.abc.Sequence[slangpy.AccelerationStructureInstanceDesc]) -> None
        :no-index:

    .. py:method:: buffer(self) -> slangpy.Buffer

    .. py:method:: build_input_instances(self) -> slangpy.AccelerationStructureBuildInputInstances



----

.. py:class:: slangpy.AccelerationStructureKind

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.AccelerationStructureQueryDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: query_type
        :type: slangpy.QueryType

    .. py:property:: query_pool
        :type: slangpy.QueryPool

    .. py:property:: first_query_index
        :type: int



----

.. py:class:: slangpy.AccelerationStructureSizes



    .. py:property:: acceleration_structure_size
        :type: int

    .. py:property:: scratch_size
        :type: int

    .. py:property:: update_scratch_size
        :type: int



----

.. py:class:: slangpy.AdapterInfo



    .. py:property:: name
        :type: str

        Descriptive name of the adapter.

    .. py:property:: vendor_id
        :type: int

        Unique identifier for the vendor (only available for D3D12 and
        Vulkan).

    .. py:property:: device_id
        :type: int

        Unique identifier for the physical device among devices from the
        vendor (only available for D3D12 and Vulkan).

    .. py:property:: luid
        :type: list[int]

        Logically unique identifier of the adapter.



----

.. py:class:: slangpy.AspectBlendDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: src_factor
        :type: slangpy.BlendFactor

    .. py:property:: dst_factor
        :type: slangpy.BlendFactor

    .. py:property:: op
        :type: slangpy.BlendOp



----

.. py:class:: slangpy.BaseReflectionObject

    Base class: :py:class:`slangpy.Object`



    .. py:property:: is_valid
        :type: bool



----

.. py:class:: slangpy.BlendFactor

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.BlendOp

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.Buffer

    Base class: :py:class:`slangpy.Resource`



    .. py:property:: desc
        :type: slangpy.BufferDesc

    .. py:property:: size
        :type: int

    .. py:property:: struct_size
        :type: int

    .. py:property:: device_address
        :type: int

    .. py:property:: shared_handle
        :type: slangpy.NativeHandle

        Get the shared resource handle. Note: Buffer must be created with the
        ``BufferUsage::shared`` usage flag.

    .. py:property:: descriptor_handle_ro
        :type: slangpy.DescriptorHandle

        Get bindless descriptor handle for read access.

    .. py:property:: descriptor_handle_rw
        :type: slangpy.DescriptorHandle

        Get bindless descriptor handle for read-write access.

    .. py:method:: to_numpy(self) -> numpy.ndarray[]

    .. py:method:: copy_from_numpy(self, data: numpy.ndarray[]) -> None

    .. py:method:: to_torch(self, type: slangpy.DataType = DataType.void, shape: collections.abc.Sequence[int] = [], strides: collections.abc.Sequence[int] = [], offset: int = 0) -> torch.Tensor[device='cuda']



----

.. py:class:: slangpy.BufferCursor

    Base class: :py:class:`slangpy.Object`

    Represents a list of elements in a block of memory, and provides
    simple interface to get a BufferElementCursor for each one. As this
    can be the owner of its data, it is a ref counted object that elements
    refer to.

    .. py:method:: __init__(self, device_type: slangpy.DeviceType, element_layout: slangpy.TypeLayoutReflection, size: int) -> None

    .. py:method:: __init__(self, element_layout: slangpy.TypeLayoutReflection, buffer_resource: slangpy.Buffer, load_before_write: bool = True) -> None
        :no-index:

    .. py:method:: __init__(self, element_layout: slangpy.TypeLayoutReflection, buffer_resource: slangpy.Buffer, size: int, offset: int, load_before_write: bool = True) -> None
        :no-index:

    .. py:property:: element_type_layout
        :type: slangpy.TypeLayoutReflection

        Get type layout of an element of the cursor.

    .. py:property:: element_type
        :type: slangpy.TypeReflection

        Get type of an element of the cursor.

    .. py:method:: find_element(self, index: int) -> slangpy.BufferElementCursor

        Get element at a given index.

    .. py:property:: element_count
        :type: int

        Number of elements in the buffer.

    .. py:property:: element_size
        :type: int

        Size of element.

    .. py:property:: element_stride
        :type: int

        Stride of elements.

    .. py:property:: size
        :type: int

        Size of whole buffer.

    .. py:property:: is_loaded
        :type: bool

        Check if internal buffer exists.

    .. py:method:: load(self) -> None

        In case of GPU only buffers, loads all data from GPU.

    .. py:method:: apply(self) -> None

        In case of GPU only buffers, pushes all data to the GPU.

    .. py:property:: resource
        :type: slangpy.Buffer

        Get the resource this cursor represents (if any).

    .. py:method:: write_from_numpy(self, data: object, unchecked_copy: bool = True) -> None

    .. py:method:: to_numpy(self) -> numpy.ndarray[]

    .. py:method:: copy_from_numpy(self, data: numpy.ndarray[]) -> None



----

.. py:class:: slangpy.BufferDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: size
        :type: int

        Buffer size in bytes.

    .. py:property:: struct_size
        :type: int

        Struct size in bytes.

    .. py:property:: format
        :type: slangpy.Format

        Buffer format. Used when creating typed buffer views.

    .. py:property:: memory_type
        :type: slangpy.MemoryType

        Memory type.

    .. py:property:: usage
        :type: slangpy.BufferUsage

        Resource usage flags.

    .. py:property:: default_state
        :type: slangpy.ResourceState

        Initial resource state.

    .. py:property:: label
        :type: str

        Debug label.



----

.. py:class:: slangpy.BufferElementCursor

    Represents a single element of a given type in a block of memory, and
    provides read/write tools to access its members via reflection.

    .. py:method:: reinterpret(self, new_layout: slangpy.TypeLayoutReflection) -> slangpy.BufferElementCursor

        Reinterpret the current cursor using a different type layout.

    .. py:method:: set_data(self, data: ndarray[device='cpu']) -> None

    .. py:method:: set_data(self, data: ndarray[device='cpu']) -> None
        :no-index:

    .. py:method:: is_valid(self) -> bool

        N/A

    .. py:method:: find_field(self, name: str) -> slangpy.BufferElementCursor

        N/A

    .. py:method:: find_element(self, index: int) -> slangpy.BufferElementCursor

        N/A

    .. py:method:: has_field(self, name: str) -> bool

        N/A

    .. py:method:: has_element(self, index: int) -> bool

        N/A

    .. py:method:: read(self) -> object

        N/A

    .. py:method:: write(self, val: object) -> None

        N/A



----

.. py:class:: slangpy.BufferOffsetPair



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, buffer: slangpy.Buffer) -> None
        :no-index:

    .. py:method:: __init__(self, buffer: slangpy.Buffer, offset: int = 0) -> None
        :no-index:

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: buffer
        :type: slangpy.Buffer

    .. py:property:: offset
        :type: int



----

.. py:class:: slangpy.BufferRange



    .. py:method:: __init__(self) -> None

    .. py:property:: offset
        :type: int

    .. py:property:: size
        :type: int



----

.. py:class:: slangpy.BufferUsage

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.ColorTargetDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: format
        :type: slangpy.Format

    .. py:property:: color
        :type: slangpy.AspectBlendDesc

    .. py:property:: alpha
        :type: slangpy.AspectBlendDesc

    .. py:property:: write_mask
        :type: slangpy.RenderTargetWriteMask

    .. py:property:: enable_blend
        :type: bool

    .. py:property:: logic_op
        :type: slangpy.LogicOp



----

.. py:class:: slangpy.CommandBuffer

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: queue
        :type: slangpy.CommandQueueType

        Command queue this command buffer is recorded for.

    .. py:property:: recording_id
        :type: int

        Command recording ID, which is unique for each recording of a command
        buffer.



----

.. py:class:: slangpy.CommandEncoder

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:method:: begin_render_pass(self, desc: slangpy.RenderPassDesc) -> slangpy.RenderPassEncoder

    .. py:method:: begin_compute_pass(self) -> slangpy.ComputePassEncoder

    .. py:method:: begin_ray_tracing_pass(self) -> slangpy.RayTracingPassEncoder

    .. py:method:: copy_buffer(self, dst: slangpy.Buffer, dst_offset: int, src: slangpy.Buffer, src_offset: int, size: int) -> None

        Copy a buffer region.

        Parameter ``dst``:
            Destination buffer.

        Parameter ``dst_offset``:
            Destination offset in bytes.

        Parameter ``src``:
            Source buffer.

        Parameter ``src_offset``:
            Source offset in bytes.

        Parameter ``size``:
            Size in bytes.

    .. py:method:: copy_texture(self, dst: slangpy.Texture, dst_subresource_range: slangpy.SubresourceRange, dst_offset: slangpy.math.uint3, src: slangpy.Texture, src_subresource_range: slangpy.SubresourceRange, src_offset: slangpy.math.uint3, extent: slangpy.math.uint3 = {4294967295, 4294967295, 4294967295}) -> None

        Copy a texture region.

        Parameter ``dst``:
            Destination texture.

        Parameter ``dst_subresource_range``:
            Destination subresource range.

        Parameter ``dst_offset``:
            Destination offset in texels.

        Parameter ``src``:
            Source texture.

        Parameter ``src_subresource_range``:
            Source subresource range.

        Parameter ``src_offset``:
            Source offset in texels.

        Parameter ``extent``:
            Size in texels (-1 for maximum possible size).

    .. py:method:: copy_texture(self, dst: slangpy.Texture, dst_layer: int, dst_mip: int, dst_offset: slangpy.math.uint3, src: slangpy.Texture, src_layer: int, src_mip: int, src_offset: slangpy.math.uint3, extent: slangpy.math.uint3 = {4294967295, 4294967295, 4294967295}) -> None
        :no-index:

        Copy a texture region.

        Parameter ``dst``:
            Destination texture.

        Parameter ``dst_layer``:
            Destination layer.

        Parameter ``dst_mip``:
            Destination mip level.

        Parameter ``dst_offset``:
            Destination offset in texels.

        Parameter ``src``:
            Source texture.

        Parameter ``src_layer``:
            Source layer.

        Parameter ``src_mip``:
            Source mip level.

        Parameter ``src_offset``:
            Source offset in texels.

        Parameter ``extent``:
            Size in texels (-1 for maximum possible size).

    .. py:method:: copy_texture_to_buffer(self, dst: slangpy.Buffer, dst_offset: int, dst_size: int, dst_row_pitch: int, src: slangpy.Texture, src_layer: int, src_mip: int, src_offset: slangpy.math.uint3 = {0, 0, 0}, extent: slangpy.math.uint3 = {4294967295, 4294967295, 4294967295}) -> None

        Copy a texture to a buffer.

        Parameter ``dst``:
            Destination buffer.

        Parameter ``dst_offset``:
            Destination offset in bytes.

        Parameter ``dst_size``:
            Destination size in bytes.

        Parameter ``dst_row_pitch``:
            Destination row stride in bytes.

        Parameter ``src``:
            Source texture.

        Parameter ``src_layer``:
            Source layer.

        Parameter ``src_mip``:
            Source mip level.

        Parameter ``src_offset``:
            Source offset in texels.

        Parameter ``extent``:
            Extent in texels (-1 for maximum possible extent).

    .. py:method:: copy_buffer_to_texture(self, dst: slangpy.Texture, dst_layer: int, dst_mip: int, dst_offset: slangpy.math.uint3, src: slangpy.Buffer, src_offset: int, src_size: int, src_row_pitch: int, extent: slangpy.math.uint3 = {4294967295, 4294967295, 4294967295}) -> None

        Copy a buffer to a texture.

        Parameter ``dst``:
            Destination texture.

        Parameter ``dst_layer``:
            Destination layer.

        Parameter ``dst_mip``:
            Destination mip level.

        Parameter ``dst_offset``:
            Destination offset in texels.

        Parameter ``src``:
            Source buffer.

        Parameter ``src_offset``:
            Source offset in bytes.

        Parameter ``src_size``:
            Size in bytes.

        Parameter ``src_row_pitch``:
            Source row stride in bytes.

        Parameter ``extent``:
            Extent in texels (-1 for maximum possible extent).

    .. py:method:: upload_buffer_data(self, buffer: slangpy.Buffer, offset: int, data: numpy.ndarray[]) -> None

    .. py:method:: upload_texture_data(self, texture: slangpy.Texture, layer: int, mip: int, data: numpy.ndarray[]) -> None

    .. py:method:: upload_texture_data(self, texture: slangpy.Texture, offset: slangpy.math.uint3, extent: slangpy.math.uint3, range: slangpy.SubresourceRange, subresource_data: collections.abc.Sequence[numpy.ndarray[]]) -> None
        :no-index:

    .. py:method:: upload_texture_data(self, texture: slangpy.Texture, offset: slangpy.math.uint3, extent: slangpy.math.uint3, range: slangpy.SubresourceRange, subresource_data: collections.abc.Sequence[numpy.ndarray[]]) -> None
        :no-index:

    .. py:method:: upload_texture_data(self, texture: slangpy.Texture, range: slangpy.SubresourceRange, subresource_data: collections.abc.Sequence[numpy.ndarray[]]) -> None
        :no-index:

    .. py:method:: upload_texture_data(self, texture: slangpy.Texture, subresource_data: collections.abc.Sequence[numpy.ndarray[]]) -> None
        :no-index:

    .. py:method:: clear_buffer(self, buffer: slangpy.Buffer, range: slangpy.BufferRange = BufferRange(offset=0, size=18446744073709551615) -> None

    .. py:method:: clear_texture_float(self, texture: slangpy.Texture, range: slangpy.SubresourceRange = SubresourceRange(layer=0, layer_count=4294967295, mip=0, mip_count=4294967295, clear_value: slangpy.math.float4 = {0, 0, 0, 0}) -> None

    .. py:method:: clear_texture_uint(self, texture: slangpy.Texture, range: slangpy.SubresourceRange = SubresourceRange(layer=0, layer_count=4294967295, mip=0, mip_count=4294967295, clear_value: slangpy.math.uint4 = {0, 0, 0, 0}) -> None

    .. py:method:: clear_texture_sint(self, texture: slangpy.Texture, range: slangpy.SubresourceRange = SubresourceRange(layer=0, layer_count=4294967295, mip=0, mip_count=4294967295, clear_value: slangpy.math.int4 = {0, 0, 0, 0}) -> None

    .. py:method:: clear_texture_depth_stencil(self, texture: slangpy.Texture, range: slangpy.SubresourceRange = SubresourceRange(layer=0, layer_count=4294967295, mip=0, mip_count=4294967295, clear_depth: bool = True, depth_value: float = 0.0, clear_stencil: bool = True, stencil_value: int = 0) -> None

    .. py:method:: blit(self, dst: slangpy.TextureView, src: slangpy.TextureView, filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear) -> None

        Blit a texture view.

        Blits the full extent of the source texture to the destination
        texture.

        Parameter ``dst``:
            View of the destination texture.

        Parameter ``src``:
            View of the source texture.

        Parameter ``filter``:
            Filtering mode to use.

    .. py:method:: blit(self, dst: slangpy.Texture, src: slangpy.Texture, filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear) -> None
        :no-index:

        Blit a texture.

        Blits the full extent of the source texture to the destination
        texture.

        Parameter ``dst``:
            Destination texture.

        Parameter ``src``:
            Source texture.

        Parameter ``filter``:
            Filtering mode to use.

    .. py:method:: generate_mips(self, texture: slangpy.Texture, layer: int = 0) -> None

    .. py:method:: resolve_query(self, query_pool: slangpy.QueryPool, index: int, count: int, buffer: slangpy.Buffer, offset: int) -> None

    .. py:method:: build_acceleration_structure(self, desc: slangpy.AccelerationStructureBuildDesc, dst: slangpy.AccelerationStructure, src: slangpy.AccelerationStructure | None, scratch_buffer: slangpy.BufferOffsetPair, queries: collections.abc.Sequence[slangpy.AccelerationStructureQueryDesc] = []) -> None

    .. py:method:: build_micromap(self, desc: slangpy.MicromapBuildDesc, dst: slangpy.Micromap, scratch_buffer: slangpy.BufferOffsetPair) -> None

    .. py:method:: copy_acceleration_structure(self, dst: slangpy.AccelerationStructure, src: slangpy.AccelerationStructure, mode: slangpy.AccelerationStructureCopyMode) -> None

    .. py:method:: query_acceleration_structure_properties(self, acceleration_structures: collections.abc.Sequence[slangpy.AccelerationStructure], queries: collections.abc.Sequence[slangpy.AccelerationStructureQueryDesc]) -> None

    .. py:method:: convert_coop_vec_matrices(self, dst: slangpy.Buffer, dst_descs: collections.abc.Sequence[slangpy.CoopVecMatrixDesc], src: slangpy.Buffer, src_descs: collections.abc.Sequence[slangpy.CoopVecMatrixDesc]) -> None

    .. py:method:: convert_coop_vec_matrix(self, dst: slangpy.Buffer, dst_desc: slangpy.CoopVecMatrixDesc, src: slangpy.Buffer, src_desc: slangpy.CoopVecMatrixDesc) -> None

    .. py:method:: set_buffer_state(self, buffer: slangpy.Buffer, state: slangpy.ResourceState) -> None

        Transition resource state of a buffer and add a barrier if state has
        changed.

        Parameter ``buffer``:
            Buffer

        Parameter ``state``:
            New state

    .. py:method:: set_texture_state(self, texture: slangpy.Texture, state: slangpy.ResourceState) -> None

        Transition resource state of a texture and add a barrier if state has
        changed.

        Parameter ``texture``:
            Texture

        Parameter ``state``:
            New state

    .. py:method:: set_texture_state(self, texture: slangpy.Texture, range: slangpy.SubresourceRange, state: slangpy.ResourceState) -> None
        :no-index:

    .. py:method:: global_barrier(self) -> None

        Insert a global barrier that ensures all previous writes are visible
        to subsequent reads. Note: This is not necessary for typical bindings,
        as state management is automatic, however global barriers are useful
        for cross-api synchronization (eg 2 slangpy devices constructed from
        the same native handle), or as brute force tools for synchronizing
        pointer/bindless operations.

    .. py:method:: push_debug_group(self, name: str, color: slangpy.math.float3) -> None

        Push a debug group.

    .. py:method:: pop_debug_group(self) -> None

        Pop a debug group.

    .. py:method:: insert_debug_marker(self, name: str, color: slangpy.math.float3) -> None

        Insert a debug marker.

        Parameter ``name``:
            Name of the marker.

        Parameter ``color``:
            Color of the marker.

    .. py:method:: write_timestamp(self, query_pool: slangpy.QueryPool, index: int) -> None

        Write a timestamp.

        Parameter ``query_pool``:
            Query pool.

        Parameter ``index``:
            Index of the query.

    .. py:method:: execute_callback(self, callback: collections.abc.Callable[[slangpy.NativeHandle], None]) -> None

    .. py:method:: finish(self) -> slangpy.CommandBuffer

    .. py:property:: queue
        :type: slangpy.CommandQueueType

        Command queue this encoder is recording for.

    .. py:property:: recording_id
        :type: int

        Command recording ID, which is unique for each recording of a command
        buffer.

    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the command encoder handle.



----

.. py:class:: slangpy.CommandQueueType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.ComparisonFunc

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.ComputeKernel

    Base class: :py:class:`slangpy.Kernel`



    .. py:property:: pipeline
        :type: slangpy.ComputePipeline

    .. py:method:: dispatch(self, thread_count: slangpy.math.uint3, vars: dict = {}, command_encoder: slangpy.CommandEncoder | None = None, queue: slangpy.CommandQueueType = CommandQueueType.graphics, cuda_stream: slangpy.NativeHandle = NativeHandle(type=undefined, value=0x00000000), query_pool: slangpy.QueryPool | None = None, query_index_before: int = 0, query_index_after: int = 0, **kwargs) -> None



----

.. py:class:: slangpy.ComputeKernelDesc



    .. py:method:: __init__(self) -> None

    .. py:property:: program
        :type: slangpy.ShaderProgram



----

.. py:class:: slangpy.ComputePassEncoder

    Base class: :py:class:`slangpy.PassEncoder`



    .. py:method:: bind_pipeline(self, pipeline: slangpy.ComputePipeline) -> slangpy.ShaderObject

    .. py:method:: bind_pipeline(self, pipeline: slangpy.ComputePipeline, root_object: slangpy.ShaderObject) -> None
        :no-index:

    .. py:method:: dispatch(self, thread_count: slangpy.math.uint3) -> None

    .. py:method:: dispatch_compute(self, thread_group_count: slangpy.math.uint3) -> None

    .. py:method:: dispatch_compute_indirect(self, arg_buffer: slangpy.BufferOffsetPair) -> None



----

.. py:class:: slangpy.ComputePipeline

    Base class: :py:class:`slangpy.Pipeline`



    .. py:property:: thread_group_size
        :type: slangpy.math.uint3

        Thread group size. Used to determine the number of thread groups to
        dispatch.

    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native pipeline handle.



----

.. py:class:: slangpy.ComputePipelineDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: program
        :type: slangpy.ShaderProgram

    .. py:property:: compilation_policy
        :type: slangpy.PipelineCompilationPolicy

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.CoopVecMatrixDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: rows
        :type: int

    .. py:property:: cols
        :type: int

    .. py:property:: element_type
        :type: slangpy.DataType

    .. py:property:: layout
        :type: slangpy.CoopVecMatrixLayout

    .. py:property:: size
        :type: int

        Size (in bytes) of the matrix.

    .. py:property:: offset
        :type: int

        Offset (in bytes) from start of buffer.

    .. py:property:: row_col_stride
        :type: int

        Stride (in bytes) between rows or columns.



----

.. py:class:: slangpy.CoopVecMatrixLayout

    Base class: :py:class:`enum.Enum`





----

.. py:class:: slangpy.CullMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.DeclReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:class:: slangpy.DeclReflection.Kind

        Base class: :py:class:`enum.Enum`

        Different kinds of decl slang can return.

    .. py:property:: kind
        :type: slangpy.DeclReflection.Kind

        Decl kind (struct/function/module/generic/variable).

    .. py:property:: children
        :type: slangpy.DeclReflectionChildList

        List of children of this cursor.

    .. py:property:: child_count
        :type: int

        Get number of children.

    .. py:property:: name
        :type: str

    .. py:method:: children_of_kind(self, kind: slangpy.DeclReflection.Kind) -> slangpy.DeclReflectionIndexedChildList

        List of children of this cursor of a specific kind.

    .. py:method:: as_type(self) -> slangpy.TypeReflection

        Get type corresponding to this decl ref.

    .. py:method:: as_variable(self) -> slangpy.VariableReflection

        Get variable corresponding to this decl ref.

    .. py:method:: as_function(self) -> slangpy.FunctionReflection

        Get function corresponding to this decl ref.

    .. py:method:: find_children_of_kind(self, kind: slangpy.DeclReflection.Kind, child_name: str) -> slangpy.DeclReflectionIndexedChildList

        Finds all children of a specific kind with a given name. Note: Only
        supported for types, functions and variables.

    .. py:method:: find_first_child_of_kind(self, kind: slangpy.DeclReflection.Kind, child_name: str) -> slangpy.DeclReflection

        Finds the first child of a specific kind with a given name. Note: Only
        supported for types, functions and variables.



----

.. py:class:: slangpy.DeclReflectionChildList





----

.. py:class:: slangpy.DeclReflectionIndexedChildList





----

.. py:class:: slangpy.DepthStencilDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: format
        :type: slangpy.Format

    .. py:property:: depth_test_enable
        :type: bool

    .. py:property:: depth_write_enable
        :type: bool

    .. py:property:: depth_func
        :type: slangpy.ComparisonFunc

    .. py:property:: stencil_enable
        :type: bool

    .. py:property:: stencil_read_mask
        :type: int

    .. py:property:: stencil_write_mask
        :type: int

    .. py:property:: front_face
        :type: slangpy.DepthStencilOpDesc

    .. py:property:: back_face
        :type: slangpy.DepthStencilOpDesc



----

.. py:class:: slangpy.DepthStencilOpDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: stencil_fail_op
        :type: slangpy.StencilOp

    .. py:property:: stencil_depth_fail_op
        :type: slangpy.StencilOp

    .. py:property:: stencil_pass_op
        :type: slangpy.StencilOp

    .. py:property:: stencil_func
        :type: slangpy.ComparisonFunc



----

.. py:class:: slangpy.DescriptorHandle



    .. py:property:: type
        :type: slangpy.DescriptorHandleType

    .. py:property:: value
        :type: int



----

.. py:class:: slangpy.Device

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, type: slangpy.DeviceType = DeviceType.automatic, enable_debug_layers: bool = False, debug_layers_log_level: slangpy.LogLevel = LogLevel.warn, enable_rhi_validation: bool = True, rhi_validation_log_level: slangpy.LogLevel = LogLevel.warn, enable_ray_tracing_validation: bool = False, enable_aftermath: bool = False, enable_cuda_interop: bool = False, enable_print: bool = False, enable_hot_reload: bool = True, enable_compilation_reports: bool = False, pipeline_compilation_mode: slangpy.PipelineCompilationMode = PipelineCompilationMode.serial, adapter_luid: collections.abc.Sequence[int] | None = None, compiler_options: slangpy.SlangCompilerOptions | None = None, module_cache_path: str | os.PathLike | None = None, shader_cache_path: str | os.PathLike | None = None, shader_cache_size: int = 134217728, existing_device_handles: collections.abc.Sequence[slangpy.NativeHandle] | None = None, bindless_options: slangpy.BindlessDesc | None = None, additional_vulkan_instance_extensions: collections.abc.Sequence[str] | None = None, additional_vulkan_device_extensions: collections.abc.Sequence[str] | None = None, enable_cuda_launch_from_gfx: bool = True, enable_ray_tracing: bool = True, label: str = '') -> None

    .. py:method:: __init__(self, desc: slangpy.DeviceDesc) -> None
        :no-index:

    .. py:property:: desc
        :type: slangpy.DeviceDesc

    .. py:property:: info
        :type: slangpy.DeviceInfo

        Device information.

    .. py:property:: shader_cache_stats
        :type: slangpy.ShaderCacheStats

        Shader cache statistics.

    .. py:property:: supported_shader_model
        :type: slangpy.ShaderModel

        The highest shader model supported by the device.

    .. py:property:: features
        :type: list[slangpy.Feature]

        List of features supported by the device.

    .. py:property:: capabilities
        :type: list[str]

        List of capabilities reported by the device backend.

    .. py:property:: supports_cuda_interop
        :type: bool

        True if the device supports CUDA interoperability.

    .. py:property:: native_handles
        :type: list[slangpy.NativeHandle]

        Get the native device handles.

    .. py:method:: set_cuda_context_current(self) -> None

        Set the CUDA context current on the calling thread. No-op for non-CUDA devices.

        Use this when operating from a thread that doesn't have the CUDA context set,
        or after manually switching to a different context.

    .. py:method:: cuda_context_scope(self) -> slangpy.CudaContextScope

        Returns a context manager that pushes/pops the CUDA context.

        Usage::

            with device.cuda_context_scope():
                # CUDA context is active here
                buffer = device.create_buffer(...)
            # Previous context is restored

        This is useful for multi-GPU scenarios where you need to temporarily
        switch to a different device's context.

    .. py:method:: has_feature(self, feature: slangpy.Feature) -> bool

        Check if the device supports a given feature.

    .. py:method:: has_capability(self, capability: str) -> bool

        Check if the device supports a given capability.

    .. py:method:: get_format_support(self, format: slangpy.Format) -> slangpy.FormatSupport

        Returns the supported resource states for a given format.

    .. py:property:: slang_session
        :type: slangpy.SlangSession

        Default slang session.

    .. py:method:: close(self) -> None

        Close the device.

        This function should be called before the device is released. It waits
        for all pending work to be completed and releases internal resources,
        removing all cyclic references that might prevent the device from
        being destroyed. After closing the device, no new resources must be
        created and no new work must be submitted.

        \note The Python extension will automatically close all open devices
        when the interpreter is terminated through an `atexit` handler. If a
        device is to be destroyed at runtime, it must be closed explicitly.

    .. py:method:: create_surface(self, window: slangpy.Window) -> slangpy.Surface

        Create a new surface.

        Parameter ``window``:
            Window to create the surface for.

        Returns:
            New surface object.

    .. py:method:: create_surface(self, window_handle: slangpy.WindowHandle) -> slangpy.Surface
        :no-index:

        Create a new surface.

        Parameter ``window_handle``:
            Native window handle to create the surface for.

        Returns:
            New surface object.

    .. py:method:: create_buffer(self, size: int = 0, element_count: int = 0, struct_size: int = 0, resource_type_layout: object | None = None, format: slangpy.Format = Format.undefined, memory_type: slangpy.MemoryType = MemoryType.device_local, usage: slangpy.BufferUsage = 0, default_state: slangpy.ResourceState = ResourceState.undefined, label: str = '', data: numpy.ndarray[] | None = None) -> slangpy.Buffer

        Create a new buffer.

        Parameter ``size``:
            Buffer size in bytes.

        Parameter ``element_count``:
            Buffer size in number of struct elements. Can be used instead of
            ``size``.

        Parameter ``struct_size``:
            Struct size in bytes.

        Parameter ``resource_type_layout``:
            Resource type layout of the buffer. Can be used instead of
            ``struct_size`` to specify the size of the struct.

        Parameter ``format``:
            Buffer format. Used when creating typed buffer views.

        Parameter ``initial_state``:
            Initial resource state.

        Parameter ``usage``:
            Resource usage flags.

        Parameter ``memory_type``:
            Memory type.

        Parameter ``label``:
            Debug label.

        Parameter ``data``:
            Initial data to upload to the buffer.

        Parameter ``data_size``:
            Size of the initial data in bytes.

        Returns:
            New buffer object.

    .. py:method:: create_buffer(self, desc: slangpy.BufferDesc) -> slangpy.Buffer
        :no-index:

    .. py:method:: create_buffer_from_native_handle(self, desc: slangpy.BufferDesc, handle: slangpy.NativeHandle) -> slangpy.Buffer

        Create a new buffer wrapping an existing native buffer without
        copying.

        The expected handle type depends on the device type: ``D3D12Resource``
        on D3D12, ``VkBuffer`` on Vulkan, ``MTLBuffer`` on Metal,
        ``CUdeviceptr`` on CUDA and ``WGPUBuffer`` on WGPU.

        Parameter ``desc``:
            Buffer description. The size must not exceed the native
            allocation.

        Parameter ``handle``:
            Native buffer handle to wrap.

        Returns:
            New buffer object.

    .. py:method:: create_texture(self, type: slangpy.TextureType = TextureType.texture_2d, format: slangpy.Format = Format.undefined, width: int = 1, height: int = 1, depth: int = 1, array_length: int = 1, mip_count: int = 1, sample_count: int = 1, sample_quality: int = 0, memory_type: slangpy.MemoryType = MemoryType.device_local, usage: slangpy.TextureUsage = 0, default_state: slangpy.ResourceState = ResourceState.undefined, sampler: slangpy.Sampler | None = None, label: str = '', data: numpy.ndarray[] | None = None) -> slangpy.Texture

        Create a new texture.

        Parameter ``type``:
            Texture type.

        Parameter ``format``:
            Texture format.

        Parameter ``width``:
            Width in pixels.

        Parameter ``height``:
            Height in pixels.

        Parameter ``depth``:
            Depth in pixels.

        Parameter ``array_length``:
            Array length.

        Parameter ``mip_count``:
            Mip level count. Number of mip levels (ALL_MIPS for all mip
            levels).

        Parameter ``sample_count``:
            Number of samples for multisampled textures.

        Parameter ``quality``:
            Quality level for multisampled textures.

        Parameter ``usage``:
            Resource usage.

        Parameter ``memory_type``:
            Memory type.

        Parameter ``label``:
            Debug label.

        Parameter ``data``:
            Initial data.

        Returns:
            New texture object.

    .. py:method:: create_texture(self, desc: slangpy.TextureDesc) -> slangpy.Texture
        :no-index:

    .. py:method:: create_sampler(self, min_filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear, mag_filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear, mip_filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear, reduction_op: slangpy.TextureReductionOp = TextureReductionOp.average, address_u: slangpy.TextureAddressingMode = TextureAddressingMode.wrap, address_v: slangpy.TextureAddressingMode = TextureAddressingMode.wrap, address_w: slangpy.TextureAddressingMode = TextureAddressingMode.wrap, mip_lod_bias: float = 0.0, max_anisotropy: int = 1, comparison_func: slangpy.ComparisonFunc = ComparisonFunc.never, border_color: slangpy.math.float4 = {1, 1, 1, 1}, min_lod: float = -1000.0, max_lod: float = 1000.0, label: str = '') -> slangpy.Sampler

        Create a new sampler.

        Parameter ``min_filter``:
            Minification filter.

        Parameter ``mag_filter``:
            Magnification filter.

        Parameter ``mip_filter``:
            Mip-map filter.

        Parameter ``reduction_op``:
            Reduction operation.

        Parameter ``address_u``:
            Texture addressing mode for the U coordinate.

        Parameter ``address_v``:
            Texture addressing mode for the V coordinate.

        Parameter ``address_w``:
            Texture addressing mode for the W coordinate.

        Parameter ``mip_lod_bias``:
            Mip-map LOD bias.

        Parameter ``max_anisotropy``:
            Maximum anisotropy.

        Parameter ``comparison_func``:
            Comparison function.

        Parameter ``border_color``:
            Border color.

        Parameter ``min_lod``:
            Minimum LOD level.

        Parameter ``max_lod``:
            Maximum LOD level.

        Parameter ``label``:
            Debug label.

        Returns:
            New sampler object.

    .. py:method:: create_sampler(self, desc: slangpy.SamplerDesc) -> slangpy.Sampler
        :no-index:

    .. py:method:: create_fence(self, initial_value: int = 0, shared: bool = False) -> slangpy.Fence

        Create a new fence.

        Parameter ``initial_value``:
            Initial fence value.

        Parameter ``shared``:
            Create a shared fence.

        Returns:
            New fence object.

    .. py:method:: create_fence(self, desc: slangpy.FenceDesc) -> slangpy.Fence
        :no-index:

    .. py:method:: create_query_pool(self, type: slangpy.QueryType, count: int) -> slangpy.QueryPool

        Create a new query pool.

        Parameter ``type``:
            Query type.

        Parameter ``count``:
            Number of queries in the pool.

        Returns:
            New query pool object.

    .. py:method:: create_input_layout(self, input_elements: collections.abc.Sequence[slangpy.InputElementDesc], vertex_streams: collections.abc.Sequence[slangpy.VertexStreamDesc]) -> slangpy.InputLayout

        Create a new input layout.

        Parameter ``input_elements``:
            List of input elements (see InputElementDesc for details).

        Parameter ``vertex_streams``:
            List of vertex streams (see VertexStreamDesc for details).

        Returns:
            New input layout object.

    .. py:method:: create_input_layout(self, desc: slangpy.InputLayoutDesc) -> slangpy.InputLayout
        :no-index:

    .. py:method:: create_command_encoder(self, queue: slangpy.CommandQueueType = CommandQueueType.graphics) -> slangpy.CommandEncoder

        Create a command encoder.

    .. py:method:: submit_command_buffers(self, command_buffers: collections.abc.Sequence[slangpy.CommandBuffer], wait_fences: collections.abc.Sequence[slangpy.Fence] = [], wait_fence_values: collections.abc.Sequence[int] = [], signal_fences: collections.abc.Sequence[slangpy.Fence] = [], signal_fence_values: collections.abc.Sequence[int] = [], queue: slangpy.CommandQueueType = CommandQueueType.graphics, cuda_stream: slangpy.NativeHandle = NativeHandle(type=undefined, value=0x00000000)) -> int

        Submit a list of command buffers to the device.

        The returned submission ID can be used to wait for the submission to
        complete.

        The wait fence values are optional. If not provided, the fence values
        will be set to AUTO, which means waiting for the last signaled value.

        The signal fence values are optional. If not provided, the fence
        values will be set to AUTO, which means incrementing the last signaled
        value by 1. *

        Parameter ``command_buffers``:
            List of command buffers to submit.

        Parameter ``wait_fences``:
            List of fences to wait for before executing the command buffers.

        Parameter ``wait_fence_values``:
            List of fence values to wait for before executing the command
            buffers.

        Parameter ``signal_fences``:
            List of fences to signal after executing the command buffers.

        Parameter ``signal_fence_values``:
            List of fence values to signal after executing the command
            buffers.

        Parameter ``queue``:
            Command queue to submit to.

        Parameter ``cuda_stream``:
            On none-CUDA backends, when interop is enabled, this is the stream
            to sync with before/after submission (assuming any resources are
            shared with CUDA) and use for internal copies. If not specified,
            sync will happen with the NULL (default) CUDA stream. On CUDA
            backends, this is the CUDA stream to use for the submission. If
            not specified, the default stream of the command queue will be
            used, which for CommandQueueType::graphics is the NULL stream. It
            is an error to specify a stream for none-CUDA backends that have
            interop disabled.

        Returns:
            Submission ID.

    .. py:method:: submit_command_buffer(self, command_buffer: slangpy.CommandBuffer, queue: slangpy.CommandQueueType = CommandQueueType.graphics, cuda_stream: slangpy.NativeHandle = NativeHandle(type=undefined, value=0x00000000)) -> int

        Submit a command buffer to the device.

        The returned submission ID can be used to wait for the submission to
        complete.

        Parameter ``command_buffer``:
            Command buffer to submit.

        Parameter ``queue``:
            Command queue to submit to.

        Returns:
            Submission ID.

    .. py:method:: is_submit_finished(self, id: int) -> bool

        Check if a submission is finished executing.

        Parameter ``id``:
            Submission ID.

        Returns:
            True if the submission is finished executing.

    .. py:method:: wait_for_submit(self, id: int) -> None

        Wait for a submission to finish execution.

        Parameter ``id``:
            Submission ID.

    .. py:method:: wait_for_idle(self, queue: slangpy.CommandQueueType = CommandQueueType.graphics) -> None

        Wait for the command queue to be idle.

        Parameter ``queue``:
            Command queue to wait for.

    .. py:method:: get_timestamp_calibration(self, queue: slangpy.CommandQueueType = CommandQueueType.graphics) -> slangpy.TimestampCalibration

        Get timestamp calibration data for a queue.

        This can be used to synchronize CPU and GPU timestamps, which is
        necessary for accurate profiling and debugging.

        Parameter ``queue``:
            Command queue to get timestamp calibration data for.

        Returns:
            Timestamp calibration data

    .. py:method:: sync_to_cuda(self, cuda_stream: int = 0) -> None

        Synchronize CUDA -> device.

        This signals a shared CUDA semaphore from the CUDA stream and then
        waits for the signal on the command queue.

        Parameter ``cuda_stream``:
            CUDA stream

    .. py:method:: sync_to_device(self, cuda_stream: int = 0) -> None

        Synchronize device -> CUDA.

        This waits for a shared CUDA semaphore on the CUDA stream, making sure
        all commands on the device have completed.

        Parameter ``cuda_stream``:
            CUDA stream

    .. py:method:: get_acceleration_structure_sizes(self, desc: slangpy.AccelerationStructureBuildDesc) -> slangpy.AccelerationStructureSizes

        Query the device for buffer sizes required for acceleration structure
        builds.

        Parameter ``desc``:
            Acceleration structure build description.

        Returns:
            Acceleration structure sizes.

    .. py:method:: create_acceleration_structure(self, kind: slangpy.AccelerationStructureKind = AccelerationStructureKind.unknown, size: int = 0, label: str = '') -> slangpy.AccelerationStructure

        Create a new acceleration structure.

    .. py:method:: create_acceleration_structure(self, desc: slangpy.AccelerationStructureDesc) -> slangpy.AccelerationStructure
        :no-index:

    .. py:method:: create_acceleration_structure_instance_list(self, size: int) -> slangpy.AccelerationStructureInstanceList

        Create a new acceleration structure instance list.

    .. py:method:: get_micromap_sizes(self, desc: slangpy.MicromapBuildDesc) -> slangpy.MicromapSizes

        Query the device for buffer sizes required for a micromap build.

    .. py:method:: create_micromap(self, type: slangpy.MicromapType = MicromapType.opacity, size: int = 0, flags: slangpy.MicromapBuildFlags = 0, label: str = '') -> slangpy.Micromap

        Create a new micromap.

    .. py:method:: create_micromap(self, desc: slangpy.MicromapDesc) -> slangpy.Micromap
        :no-index:

    .. py:method:: create_shader_table(self, program: slangpy.ShaderProgram, ray_gen_entry_points: collections.abc.Sequence[str] = [], miss_entry_points: collections.abc.Sequence[str] = [], hit_group_names: collections.abc.Sequence[str] = [], callable_entry_points: collections.abc.Sequence[str] = []) -> slangpy.ShaderTable

        Create a new shader table.

    .. py:method:: create_shader_table(self, desc: slangpy.ShaderTableDesc) -> slangpy.ShaderTable
        :no-index:

    .. py:method:: get_coop_vec_matrix_size(self, rows: int, cols: int, layout: slangpy.CoopVecMatrixLayout, element_type: slangpy.DataType, row_col_stride: int = 0) -> int

        Get the size of a cooperative vector matrix in bytes.

    .. py:method:: create_coop_vec_matrix_desc(self, rows: int, cols: int, layout: slangpy.CoopVecMatrixLayout, element_type: slangpy.DataType, offset: int = 0, row_col_stride: int = 0) -> slangpy.CoopVecMatrixDesc

        Create a cooperative vector matrix descriptor.

    .. py:method:: convert_coop_vec_matrices(self, dst: bytearray, dst_descs: collections.abc.Sequence[slangpy.CoopVecMatrixDesc], src: bytes, src_descs: collections.abc.Sequence[slangpy.CoopVecMatrixDesc]) -> None

        Convert multiple cooperative vector matrices between formats.

    .. py:method:: convert_coop_vec_matrix(self, dst: bytearray, dst_descs: slangpy.CoopVecMatrixDesc, src: bytes, src_descs: slangpy.CoopVecMatrixDesc) -> None

        Convert multiple cooperative vector matrices between formats.

    .. py:method:: convert_coop_vec_matrix(self, dst: ndarray[device='cpu'], src: ndarray[device='cpu'], dst_layout: slangpy.CoopVecMatrixLayout | None = None, src_layout: slangpy.CoopVecMatrixLayout | None = None) -> None
        :no-index:

    .. py:method:: create_slang_session(self, compiler_options: slangpy.SlangCompilerOptions | None = None, add_default_include_paths: bool = True, cache_path: str | os.PathLike | None = None) -> slangpy.SlangSession

        Create a new slang session.

        Parameter ``compiler_options``:
            Compiler options (see SlangCompilerOptions for details).

        Returns:
            New slang session object.

    .. py:method:: reload_all_programs(self) -> None

        Reload all shader programs.

    .. py:method:: load_module(self, module_name: str) -> slangpy.SlangModule

        Load a slang module by name.

    .. py:method:: load_module_from_source(self, module_name: str, source: str, path: str | os.PathLike | None = None) -> slangpy.SlangModule

        Load a slang module from source code.

    .. py:method:: compose_modules(self, name: str, modules: collections.abc.Sequence[slangpy.SlangModule], type_conformances: collections.abc.Sequence[slangpy.TypeConformance] = []) -> slangpy.SlangModule

        Compose multiple slang modules into one.

    .. py:method:: link_program(self, modules: collections.abc.Sequence[slangpy.SlangModule], entry_points: collections.abc.Sequence[slangpy.SlangEntryPoint], link_options: slangpy.SlangLinkOptions | None = None) -> slangpy.ShaderProgram

        Link modules and entry points into a shader program.

    .. py:method:: load_program(self, module_name: str, entry_point_names: collections.abc.Sequence[str], additional_source: str | None = None, link_options: slangpy.SlangLinkOptions | None = None) -> slangpy.ShaderProgram

        Load a module and link a shader program in one step.

    .. py:method:: get_compilation_reports(self) -> list[CompilationReport]

        Return compilation reports for all shader programs tracked by the device.

    .. py:method:: create_root_shader_object(self, shader_program: slangpy.ShaderProgram) -> slangpy.ShaderObject

        Create a root shader object for a shader program.

    .. py:method:: create_shader_object(self, type_layout: slangpy.TypeLayoutReflection) -> slangpy.ShaderObject

        Create a shader object from a type layout.

    .. py:method:: create_shader_object(self, cursor: slangpy.ReflectionCursor) -> slangpy.ShaderObject
        :no-index:

        Create a shader object from a reflection cursor.

    .. py:method:: create_compute_pipeline(self, program: slangpy.ShaderProgram, compilation_policy: slangpy.PipelineCompilationPolicy = PipelineCompilationPolicy.default, label: str | None = None) -> slangpy.ComputePipeline

        Create a compute pipeline.

    .. py:method:: create_compute_pipeline(self, desc: slangpy.ComputePipelineDesc) -> slangpy.ComputePipeline
        :no-index:

    .. py:method:: create_render_pipeline(self, program: slangpy.ShaderProgram, input_layout: slangpy.InputLayout | None, primitive_topology: slangpy.PrimitiveTopology = PrimitiveTopology.triangle_list, targets: collections.abc.Sequence[slangpy.ColorTargetDesc] = [], depth_stencil: slangpy.DepthStencilDesc | None = None, rasterizer: slangpy.RasterizerDesc | None = None, multisample: slangpy.MultisampleDesc | None = None, compilation_policy: slangpy.PipelineCompilationPolicy = PipelineCompilationPolicy.default, label: str | None = None) -> slangpy.RenderPipeline

        Create a render pipeline.

    .. py:method:: create_render_pipeline(self, desc: slangpy.RenderPipelineDesc) -> slangpy.RenderPipeline
        :no-index:

    .. py:method:: create_ray_tracing_pipeline(self, program: slangpy.ShaderProgram, hit_groups: collections.abc.Sequence[slangpy.HitGroupDesc], max_recursion: int = 0, max_ray_payload_size: int = 0, max_attribute_size: int = 8, flags: slangpy.RayTracingPipelineFlags = 0, compilation_policy: slangpy.PipelineCompilationPolicy = PipelineCompilationPolicy.default, label: str | None = None) -> slangpy.RayTracingPipeline

        Create a ray tracing pipeline.

    .. py:method:: create_ray_tracing_pipeline(self, desc: slangpy.RayTracingPipelineDesc) -> slangpy.RayTracingPipeline
        :no-index:

    .. py:method:: create_compute_kernel(self, program: slangpy.ShaderProgram) -> slangpy.ComputeKernel

        Create a compute kernel.

    .. py:method:: create_compute_kernel(self, desc: slangpy.ComputeKernelDesc) -> slangpy.ComputeKernel
        :no-index:

    .. py:method:: flush_print(self) -> None

        Block and flush all shader side debug print output.

    .. py:method:: flush_print_to_string(self) -> str

        Block and flush all shader side debug print output to a string.

    .. py:method:: wait(self) -> None

        Wait for all device work to complete.

    .. py:method:: register_device_close_callback(self, callback: collections.abc.Callable[[slangpy.Device], None]) -> int

        Register a device close callback, called at start of device close.

    .. py:method:: unregister_device_close_callback(self, id: int) -> None

        Unregister a device close callback.

    .. py:method:: register_shader_hot_reload_callback(self, callback: collections.abc.Callable[[slangpy.ShaderHotReloadEvent], None]) -> int

        Register a hot reload hook, called immediately after any module is
        reloaded.

    .. py:method:: unregister_shader_hot_reload_callback(self, id: int) -> None

        N/A

    .. py:method:: register_command_recording_submitted_callback(self, callback: collections.abc.Callable[[slangpy.CommandRecordingSubmittedEvent], None]) -> int

        Register a callback to be called when a command recording is
        submitted.

    .. py:method:: unregister_command_recording_submitted_callback(self, id: int) -> None

        Unregister a command recording submitted callback.

    .. py:method:: register_command_recording_discarded_callback(self, callback: collections.abc.Callable[[slangpy.CommandRecordingDiscardedEvent], None]) -> int

        Register a callback to be called when a command recording is discarded
        (not submitted).

    .. py:method:: unregister_command_recording_discarded_callback(self, id: int) -> None

        Unregister a command recording discarded callback.

    .. py:method:: set_hot_reload_delay(self, timeout_ms: int) -> None

    .. py:method:: hot_reload_check(self) -> None

    .. py:staticmethod:: enumerate_adapters(type: slangpy.DeviceType = DeviceType.automatic) -> list[slangpy.AdapterInfo]

        Enumerates all available adapters of a given device type.

    .. py:staticmethod:: get_created_devices() -> list[slangpy.Device]

        Lists all created devices

    .. py:staticmethod:: report_live_objects() -> None

        Report live objects in the rhi layer. This is useful for checking
        clean shutdown with all resources released properly.

    .. py:method:: report_heaps(self) -> list[slangpy.HeapReport]

        Report status of internal heaps used by the device.

        Returns:
            List of heap reports containing heap names and allocation
            information.



----

.. py:class:: slangpy.DeviceChild

    Base class: :py:class:`slangpy.Object`



    .. py:class:: slangpy.DeviceChild.MemoryUsage



        .. py:property:: device
            :type: int

            The amount of memory in bytes used on the device.

        .. py:property:: host
            :type: int

            The amount of memory in bytes used on the host.

    .. py:property:: device
        :type: slangpy.Device

    .. py:property:: memory_usage
        :type: slangpy.DeviceChild.MemoryUsage

        The memory usage by this resource.



----

.. py:class:: slangpy.DeviceDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: type
        :type: slangpy.DeviceType

        The type of the device.

    .. py:property:: enable_debug_layers
        :type: bool

        Enable debug layers.

    .. py:property:: debug_layers_log_level
        :type: slangpy.LogLevel

        Debug layers log level (only applicable if debug layers are enabled).

    .. py:property:: enable_rhi_validation
        :type: bool

        Enable RHI validation layer.

    .. py:property:: rhi_validation_log_level
        :type: slangpy.LogLevel

        RHI validation layer log level (only applicable if RHI validation is
        enabled).

    .. py:property:: enable_ray_tracing_validation
        :type: bool

        Enable ray-tracing validation.

    .. py:property:: enable_aftermath
        :type: bool

        Enable NVIDIA Aftermath.

    .. py:property:: enable_cuda_interop
        :type: bool

        Enable CUDA interoperability.

    .. py:property:: enable_cuda_launch_from_gfx
        :type: bool

        Enable launching CUDA kernels from inside graphics command buffers
        (Vulkan only, via VK_NVX_binary_import + VK_NVX_image_view_handle). On
        by default. Set to false if the application doesn't need
        vkCmdCuLaunchKernelNVX; enabling these extensions has been observed to
        interfere with concurrent cuDNN usage on some driver/GPU pairs.

    .. py:property:: enable_ray_tracing
        :type: bool

        Enable Vulkan ray tracing extensions (acceleration_structure,
        ray_tracing_pipeline, ray_query, ray_tracing_position_fetch, plus NV
        variants). On by default. Set to false if the application doesn't use
        ray tracing; enabling these extensions has been observed to interfere
        with concurrent cuDNN usage on some driver/GPU pairs.

    .. py:property:: enable_print
        :type: bool

        Enable device side printing (adds performance overhead).

    .. py:property:: enable_hot_reload
        :type: bool

        Adapter LUID to select adapter on which the device will be created.

    .. py:property:: enable_compilation_reports
        :type: bool

        Enable compilation reports.

    .. py:property:: pipeline_compilation_mode
        :type: slangpy.PipelineCompilationMode

        Control the default pipeline compilation policy and resolution of
        deferred pipelines.

    .. py:property:: adapter_luid
        :type: list[int] | None

        Adapter LUID to select adapter on which the device will be created.

    .. py:property:: compiler_options
        :type: slangpy.SlangCompilerOptions

        Compiler options (used for default slang session).

    .. py:property:: bindless_options
        :type: slangpy.BindlessDesc

    .. py:property:: module_cache_path
        :type: pathlib.Path | None

        Path to the module cache directory (optional). If a relative path is
        used, the cache is stored in the application data directory.

    .. py:property:: shader_cache_path
        :type: pathlib.Path | None

        Path to the shader and pipeline cache directory (optional). If a
        relative path is used, the cache is stored in the application data
        directory.

    .. py:property:: shader_cache_size
        :type: int

        Maximum size of the persistent cache used to cache both shaders and
        pipelines.

    .. py:property:: existing_device_handles
        :type: list[slangpy.NativeHandle]

        Native device handles for initializing with externally created device.
        Currenlty only used for CUDA interoperability.

    .. py:property:: additional_vulkan_instance_extensions
        :type: list[str]

        Additional Vulkan instance extensions to enable when SGL creates the
        Vulkan instance.

    .. py:property:: additional_vulkan_device_extensions
        :type: list[str]

        Additional Vulkan device extensions to enable when SGL creates the
        Vulkan device.

    .. py:property:: label
        :type: str

        Debug label



----

.. py:class:: slangpy.DeviceInfo



    .. py:property:: type
        :type: slangpy.DeviceType

        The type of the device.

    .. py:property:: api_name
        :type: str

        The name of the graphics API being used by this device.

    .. py:property:: adapter_name
        :type: str

        The name of the graphics adapter.

    .. py:property:: timestamp_frequency
        :type: int

        The frequency of the timestamp counter. To resolve a timestamp to
        seconds, divide by this value.

    .. py:property:: optix_version
        :type: int

        The version of OptiX used by the device (0 if OptiX is not supported).
        The format matches the OPTIX_VERSION macro, e.g. 90000 for version
        9.0.0.

    .. py:property:: limits
        :type: slangpy.DeviceLimits

        Limits of the device.



----

.. py:class:: slangpy.DeviceLimits



    .. py:property:: max_texture_dimension_1d
        :type: int

        Maximum dimension for 1D textures.

    .. py:property:: max_texture_dimension_2d
        :type: int

        Maximum dimensions for 2D textures.

    .. py:property:: max_texture_dimension_3d
        :type: int

        Maximum dimensions for 3D textures.

    .. py:property:: max_texture_dimension_cube
        :type: int

        Maximum dimensions for cube textures.

    .. py:property:: max_texture_layers
        :type: int

        Maximum number of texture layers.

    .. py:property:: max_vertex_input_elements
        :type: int

        Maximum number of vertex input elements in a graphics pipeline.

    .. py:property:: max_vertex_input_element_offset
        :type: int

        Maximum offset of a vertex input element in the vertex stream.

    .. py:property:: max_vertex_streams
        :type: int

        Maximum number of vertex streams in a graphics pipeline.

    .. py:property:: max_vertex_stream_stride
        :type: int

        Maximum stride of a vertex stream.

    .. py:property:: max_compute_threads_per_group
        :type: int

        Maximum number of threads per thread group.

    .. py:property:: max_compute_thread_group_size
        :type: slangpy.math.uint3

        Maximum dimensions of a thread group.

    .. py:property:: max_compute_dispatch_thread_groups
        :type: slangpy.math.uint3

        Maximum number of thread groups per dimension in a single dispatch.

    .. py:property:: min_wave_size
        :type: int

        Minimum number of lanes in a wave/subgroup/warp. 0 if the size is
        unknown or not applicable.

    .. py:property:: max_wave_size
        :type: int

        Maximum number of lanes in a wave/subgroup/warp. 0 if the size is
        unknown or not applicable.

    .. py:property:: max_viewports
        :type: int

        Maximum number of viewports per pipeline.

    .. py:property:: max_viewport_dimensions
        :type: slangpy.math.uint2

        Maximum viewport dimensions.

    .. py:property:: max_framebuffer_dimensions
        :type: slangpy.math.uint3

        Maximum framebuffer dimensions.

    .. py:property:: max_shader_visible_samplers
        :type: int

        Maximum samplers visible in a shader stage.

    .. py:property:: max_entry_point_uniform_size
        :type: int

        Maximum size in bytes of inline-uniform data for entry-point
        parameters. On Vulkan this corresponds to push constant size (minimum
        128 bytes). On D3D12 this corresponds to root constant space (~256
        bytes). On CUDA this corresponds to the kernel parameter block (~4096
        bytes).



----

.. py:class:: slangpy.DeviceType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.DrawArguments



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: vertex_count
        :type: int

    .. py:property:: instance_count
        :type: int

    .. py:property:: start_vertex_location
        :type: int

    .. py:property:: start_instance_location
        :type: int

    .. py:property:: start_index_location
        :type: int



----

.. py:class:: slangpy.EntryPointLayout

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:property:: name
        :type: str

    .. py:property:: name_override
        :type: str

    .. py:property:: stage
        :type: slangpy.ShaderStage

    .. py:property:: compute_thread_group_size
        :type: slangpy.math.uint3

    .. py:property:: parameters
        :type: slangpy.EntryPointLayoutParameterList



----

.. py:class:: slangpy.EntryPointLayoutParameterList





----

.. py:class:: slangpy.Feature

    Base class: :py:class:`enum.IntEnum`



----

.. py:class:: slangpy.Fence

    Base class: :py:class:`slangpy.DeviceChild`

    Fence.

    .. py:property:: desc
        :type: slangpy.FenceDesc

    .. py:method:: signal(self, value: int = 18446744073709551615) -> int

        Signal the fence. This signals the fence from the host.

        Parameter ``value``:
            The value to signal. If ``AUTO``, the signaled value will be auto-
            incremented.

        Returns:
            The signaled value.

    .. py:method:: wait(self, value: int = 18446744073709551615, timeout_ns: int = 18446744073709551615) -> None

        Wait for the fence to be signaled on the host. Blocks the host until
        the fence reaches or exceeds the specified value.

        Parameter ``value``:
            The value to wait for. If ``AUTO``, wait for the last signaled
            value.

        Parameter ``timeout_ns``:
            The timeout in nanoseconds. If ``TIMEOUT_INFINITE``, the function
            will block indefinitely.

    .. py:property:: current_value
        :type: int

        Returns the currently signaled value on the device.

    .. py:property:: signaled_value
        :type: int

        Returns the last signaled value on the device.

    .. py:property:: shared_handle
        :type: slangpy.NativeHandle

        Get the shared fence handle. Note: Fence must be created with the
        ``FenceDesc::shared`` flag.

    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native fence handle.

    .. py:attribute:: slangpy.Fence.AUTO
        :type: int
        :value: 18446744073709551615

    .. py:attribute:: slangpy.Fence.TIMEOUT_INFINITE
        :type: int
        :value: 18446744073709551615



----

.. py:class:: slangpy.FenceDesc

    Fence descriptor.

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: initial_value
        :type: int

        Initial fence value.

    .. py:property:: shared
        :type: bool

        Create a shared fence.



----

.. py:class:: slangpy.FillMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.Format

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.FormatChannels

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.FormatInfo

    Resource format information.

    .. py:property:: format
        :type: slangpy.Format

        Resource format.

    .. py:property:: name
        :type: str

        Format name.

    .. py:property:: bytes_per_block
        :type: int

        Number of bytes per block (compressed) or pixel (uncompressed).

    .. py:property:: channel_count
        :type: int

        Number of channels.

    .. py:property:: type
        :type: slangpy.FormatType

        Format type (typeless, float, unorm, unorm_srgb, snorm, uint, sint).

    .. py:property:: is_depth
        :type: bool

        True if format has a depth component.

    .. py:property:: is_stencil
        :type: bool

        True if format has a stencil component.

    .. py:property:: is_compressed
        :type: bool

        True if format is compressed.

    .. py:property:: block_width
        :type: int

        Block width for compressed formats (1 for uncompressed formats).

    .. py:property:: block_height
        :type: int

        Block height for compressed formats (1 for uncompressed formats).

    .. py:property:: channel_bit_count
        :type: list[int]

        Number of bits per channel.

    .. py:property:: dxgi_format
        :type: int

        DXGI format.

    .. py:property:: vk_format
        :type: int

        Vulkan format.

    .. py:method:: is_depth_stencil(self) -> bool

        True if format has a depth or stencil component.

    .. py:method:: is_float_format(self) -> bool

        True if format is floating point.

    .. py:method:: is_integer_format(self) -> bool

        True if format is integer.

    .. py:method:: is_normalized_format(self) -> bool

        True if format is normalized.

    .. py:method:: is_srgb_format(self) -> bool

        True if format is sRGB.

    .. py:method:: get_channels(self) -> slangpy.FormatChannels

        Get the channels for the format (only for color formats).

    .. py:method:: get_channel_bits(self, arg: slangpy.FormatChannels, /) -> int

        Get the number of bits for the specified channels.

    .. py:method:: has_equal_channel_bits(self) -> bool

        Check if all channels have the same number of bits.



----

.. py:class:: slangpy.FormatSupport

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.FormatType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.FrontFaceMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.FunctionReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:property:: name
        :type: str

        Function name.

    .. py:property:: return_type
        :type: slangpy.TypeReflection

        Function return type.

    .. py:property:: parameters
        :type: slangpy.FunctionReflectionParameterList

        List of all function parameters.

    .. py:method:: get_user_attribute_count(self) -> int

    .. py:method:: get_user_attribute_by_index(self, index: int) -> slangpy.Attribute

    .. py:method:: find_user_attribute_by_name(self, name: str) -> slangpy.Attribute

    .. py:method:: has_modifier(self, modifier: slangpy.ModifierID) -> bool

        Check if the function has a given modifier (e.g. 'differentiable').

    .. py:method:: specialize_with_arg_types(self, types: collections.abc.Sequence[slangpy.TypeReflection]) -> slangpy.FunctionReflection

        Specialize a generic or interface based function with a set of
        concrete argument types. Calling on a none-generic/interface function
        will simply validate all argument types can be implicitly converted to
        their respective parameter types. Where a function contains multiple
        overloads, specialize will identify the correct overload based on the
        arguments.

    .. py:property:: is_overloaded
        :type: bool

        Check whether this function object represents a group of overloaded
        functions, accessible via the overloads list.

    .. py:property:: overloads
        :type: slangpy.FunctionReflectionOverloadList

        List of all overloads of this function.



----

.. py:class:: slangpy.FunctionReflectionOverloadList





----

.. py:class:: slangpy.FunctionReflectionParameterList





----

.. py:function:: slangpy.get_format_info(arg: slangpy.Format, /) -> slangpy.FormatInfo



----

.. py:class:: slangpy.HitGroupDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:method:: __init__(self, hit_group_name: str, closest_hit_entry_point: str = '', any_hit_entry_point: str = '', intersection_entry_point: str = '') -> None
        :no-index:

    .. py:property:: hit_group_name
        :type: str

    .. py:property:: closest_hit_entry_point
        :type: str

    .. py:property:: any_hit_entry_point
        :type: str

    .. py:property:: intersection_entry_point
        :type: str



----

.. py:class:: slangpy.IndexFormat

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.InputElementDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: semantic_name
        :type: str

        The name of the corresponding parameter in shader code.

    .. py:property:: semantic_index
        :type: int

        The index of the corresponding parameter in shader code. Only needed
        if multiple parameters share a semantic name.

    .. py:property:: format
        :type: slangpy.Format

        The format of the data being fetched for this element.

    .. py:property:: offset
        :type: int

        The offset in bytes of this element from the start of the
        corresponding chunk of vertex stream data.

    .. py:property:: buffer_slot_index
        :type: int

        The index of the vertex stream to fetch this element's data from.



----

.. py:class:: slangpy.InputLayout

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: desc
        :type: slangpy.InputLayoutDesc



----

.. py:class:: slangpy.InputLayoutDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: input_elements
        :type: list[slangpy.InputElementDesc]

    .. py:property:: vertex_streams
        :type: list[slangpy.VertexStreamDesc]



----

.. py:class:: slangpy.InputSlotClass

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.Kernel

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: program
        :type: slangpy.ShaderProgram

    .. py:property:: reflection
        :type: slangpy.ReflectionCursor



----

.. py:class:: slangpy.LoadOp

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.LogicOp

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.MemoryType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.ModifierID

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.MultisampleDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: sample_count
        :type: int

    .. py:property:: sample_mask
        :type: int

    .. py:property:: alpha_to_coverage_enable
        :type: bool

    .. py:property:: alpha_to_one_enable
        :type: bool



----

.. py:class:: slangpy.NativeHandle

    Represents a native graphics API handle (e.g. D3D12/Vulkan/Metal/CUDA
    etc). Native handles are expected to fit into 64 bits. Type
    information and conversion from/to native handles is done using type
    traits from native_handle_traits.h which needs to be included when
    creating and accessing NativeHandle. This separation is done so we
    don't expose the heavy backend API headers everywhere.

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg0: slangpy.NativeHandleType, arg1: int, /) -> None
        :no-index:

    .. py:property:: type
        :type: slangpy.NativeHandleType

    .. py:property:: value
        :type: int

    .. py:staticmethod:: from_cuda_stream(stream: int) -> slangpy.NativeHandle

    .. py:staticmethod:: from_cuda_device_ptr(device_ptr: int) -> slangpy.NativeHandle



----

.. py:class:: slangpy.NativeHandleType

    Base class: :py:class:`enum.Enum`





----

.. py:class:: slangpy.PassEncoder

    Base class: :py:class:`slangpy.Object`



    .. py:method:: end(self) -> None

    .. py:method:: push_debug_group(self, name: str, color: slangpy.math.float3) -> None

        Push a debug group.

    .. py:method:: pop_debug_group(self) -> None

        Pop a debug group.

    .. py:method:: insert_debug_marker(self, name: str, color: slangpy.math.float3) -> None

        Insert a debug marker.

        Parameter ``name``:
            Name of the marker.

        Parameter ``color``:
            Color of the marker.

    .. py:method:: write_timestamp(self, query_pool: slangpy.QueryPool, index: int) -> None

        Write a timestamp.

        Parameter ``query_pool``:
            Query pool.

        Parameter ``index``:
            Index of the query.



----

.. py:class:: slangpy.Pipeline

    Base class: :py:class:`slangpy.DeviceChild`





----

.. py:class:: slangpy.PrimitiveTopology

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.ProgramLayout

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:class:: slangpy.ProgramLayout.HashedString



        .. py:property:: string
            :type: str

        .. py:property:: hash
            :type: int

    .. py:property:: globals_type_layout
        :type: slangpy.TypeLayoutReflection

    .. py:property:: globals_variable_layout
        :type: slangpy.VariableLayoutReflection

    .. py:property:: parameters
        :type: slangpy.ProgramLayoutParameterList

    .. py:property:: entry_points
        :type: slangpy.ProgramLayoutEntryPointList

    .. py:method:: find_type_by_name(self, name: str) -> slangpy.TypeReflection

        Find a given type by name. Handles generic specilization if generic
        variable values are provided.

    .. py:method:: find_function_by_name(self, name: str) -> slangpy.FunctionReflection

        Find a given function by name. Handles generic specilization if
        generic variable values are provided.

    .. py:method:: find_function_by_name_in_type(self, type: slangpy.TypeReflection, name: str) -> slangpy.FunctionReflection

        Find a given function in a type by name. Handles generic specilization
        if generic variable values are provided.

    .. py:method:: get_type_layout(self, type: slangpy.TypeReflection) -> slangpy.TypeLayoutReflection

        Get corresponding type layout from a given type.

    .. py:method:: is_sub_type(self, sub_type: slangpy.TypeReflection, super_type: slangpy.TypeReflection) -> bool

        Test whether a type is a sub type of another type. Handles both struct
        inheritance and interface implementation.

    .. py:property:: hashed_strings
        :type: list[slangpy.ProgramLayout.HashedString]



----

.. py:class:: slangpy.ProgramLayoutEntryPointList





----

.. py:class:: slangpy.ProgramLayoutParameterList





----

.. py:class:: slangpy.QueryPool

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: desc
        :type: slangpy.QueryPoolDesc

    .. py:method:: reset(self) -> None

    .. py:method:: reset(self, index: int, count: int) -> None
        :no-index:

    .. py:method:: get_result_state(self, index: int, count: int) -> sgl::QueryResultState

    .. py:method:: get_result_state(self, index: int) -> sgl::QueryResultState
        :no-index:

    .. py:method:: get_result(self, index: int) -> int

    .. py:method:: get_results(self, index: int, count: int) -> list[int]

    .. py:method:: get_timestamp_result(self, index: int) -> float

    .. py:method:: get_timestamp_results(self, index: int, count: int) -> list[float]



----

.. py:class:: slangpy.QueryPoolDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: type
        :type: slangpy.QueryType

        Query type.

    .. py:property:: count
        :type: int

        Number of queries in the pool.



----

.. py:class:: slangpy.QueryType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.RasterizerDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: fill_mode
        :type: slangpy.FillMode

    .. py:property:: cull_mode
        :type: slangpy.CullMode

    .. py:property:: front_face
        :type: slangpy.FrontFaceMode

    .. py:property:: depth_bias
        :type: int

    .. py:property:: depth_bias_clamp
        :type: float

    .. py:property:: slope_scaled_depth_bias
        :type: float

    .. py:property:: depth_clip_enable
        :type: bool

    .. py:property:: scissor_enable
        :type: bool

    .. py:property:: multisample_enable
        :type: bool

    .. py:property:: antialiased_line_enable
        :type: bool

    .. py:property:: enable_conservative_rasterization
        :type: bool

    .. py:property:: forced_sample_count
        :type: int



----

.. py:class:: slangpy.RayTracingPassEncoder

    Base class: :py:class:`slangpy.PassEncoder`



    .. py:method:: bind_pipeline(self, pipeline: slangpy.RayTracingPipeline, shader_table: slangpy.ShaderTable) -> slangpy.ShaderObject

    .. py:method:: bind_pipeline(self, pipeline: slangpy.RayTracingPipeline, shader_table: slangpy.ShaderTable, root_object: slangpy.ShaderObject) -> None
        :no-index:

    .. py:method:: dispatch_rays(self, ray_gen_shader_index: int, dimensions: slangpy.math.uint3) -> None



----

.. py:class:: slangpy.RayTracingPipeline

    Base class: :py:class:`slangpy.Pipeline`



    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native pipeline handle.



----

.. py:class:: slangpy.RayTracingPipelineDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: program
        :type: slangpy.ShaderProgram

    .. py:property:: hit_groups
        :type: list[slangpy.HitGroupDesc]

    .. py:property:: max_recursion
        :type: int

    .. py:property:: max_ray_payload_size
        :type: int

    .. py:property:: max_attribute_size
        :type: int

    .. py:property:: flags
        :type: slangpy.RayTracingPipelineFlags

    .. py:property:: compilation_policy
        :type: slangpy.PipelineCompilationPolicy

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.RayTracingPipelineFlags

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.ReflectionCursor



    .. py:method:: __init__(self, shader_program: slangpy.ShaderProgram) -> None

    .. py:method:: is_valid(self) -> bool

    .. py:method:: find_field(self, name: str) -> slangpy.ReflectionCursor

    .. py:method:: find_element(self, index: int) -> slangpy.ReflectionCursor

    .. py:method:: has_field(self, name: str) -> bool

    .. py:method:: has_element(self, index: int) -> bool

    .. py:property:: type_layout
        :type: slangpy.TypeLayoutReflection

    .. py:property:: type
        :type: slangpy.TypeReflection



----

.. py:class:: slangpy.RenderPassColorAttachment



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: view
        :type: slangpy.TextureView

    .. py:property:: resolve_target
        :type: slangpy.TextureView

    .. py:property:: load_op
        :type: slangpy.LoadOp

    .. py:property:: store_op
        :type: slangpy.StoreOp

    .. py:property:: clear_value
        :type: slangpy.math.float4



----

.. py:class:: slangpy.RenderPassDepthStencilAttachment



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: view
        :type: slangpy.TextureView

    .. py:property:: depth_load_op
        :type: slangpy.LoadOp

    .. py:property:: depth_store_op
        :type: slangpy.StoreOp

    .. py:property:: depth_clear_value
        :type: float

    .. py:property:: depth_read_only
        :type: bool

    .. py:property:: stencil_load_op
        :type: slangpy.LoadOp

    .. py:property:: stencil_store_op
        :type: slangpy.StoreOp

    .. py:property:: stencil_clear_value
        :type: int

    .. py:property:: stencil_read_only
        :type: bool



----

.. py:class:: slangpy.RenderPassDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: color_attachments
        :type: list[slangpy.RenderPassColorAttachment]

    .. py:property:: depth_stencil_attachment
        :type: slangpy.RenderPassDepthStencilAttachment | None



----

.. py:class:: slangpy.RenderPassEncoder

    Base class: :py:class:`slangpy.PassEncoder`



    .. py:method:: bind_pipeline(self, pipeline: slangpy.RenderPipeline) -> slangpy.ShaderObject

    .. py:method:: bind_pipeline(self, pipeline: slangpy.RenderPipeline, root_object: slangpy.ShaderObject) -> None
        :no-index:

    .. py:method:: set_render_state(self, state: slangpy.RenderState) -> None

    .. py:method:: draw(self, args: slangpy.DrawArguments) -> None

    .. py:method:: draw_indexed(self, args: slangpy.DrawArguments) -> None

    .. py:method:: draw_indirect(self, max_draw_count: int, arg_buffer: slangpy.BufferOffsetPair, count_buffer: slangpy.BufferOffsetPair = ...) -> None

    .. py:method:: draw_indexed_indirect(self, max_draw_count: int, arg_buffer: slangpy.BufferOffsetPair, count_buffer: slangpy.BufferOffsetPair = ...) -> None

    .. py:method:: draw_mesh_tasks(self, dimensions: slangpy.math.uint3) -> None



----

.. py:class:: slangpy.RenderPipeline

    Base class: :py:class:`slangpy.Pipeline`



    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native pipeline handle.



----

.. py:class:: slangpy.RenderPipelineDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: program
        :type: slangpy.ShaderProgram

    .. py:property:: input_layout
        :type: slangpy.InputLayout

    .. py:property:: primitive_topology
        :type: slangpy.PrimitiveTopology

    .. py:property:: targets
        :type: list[slangpy.ColorTargetDesc]

    .. py:property:: depth_stencil
        :type: slangpy.DepthStencilDesc

    .. py:property:: rasterizer
        :type: slangpy.RasterizerDesc

    .. py:property:: multisample
        :type: slangpy.MultisampleDesc

    .. py:property:: compilation_policy
        :type: slangpy.PipelineCompilationPolicy

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.RenderState



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: stencil_ref
        :type: int

    .. py:property:: viewports
        :type: list[slangpy.Viewport]

    .. py:property:: scissor_rects
        :type: list[slangpy.ScissorRect]

    .. py:property:: vertex_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: index_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: index_format
        :type: slangpy.IndexFormat



----

.. py:class:: slangpy.RenderTargetWriteMask

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.Resource

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native resource handle.



----

.. py:class:: slangpy.ResourceState

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.Sampler

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: desc
        :type: slangpy.SamplerDesc

    .. py:property:: descriptor_handle
        :type: slangpy.DescriptorHandle

    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native sampler handle.



----

.. py:class:: slangpy.SamplerDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: min_filter
        :type: slangpy.TextureFilteringMode

    .. py:property:: mag_filter
        :type: slangpy.TextureFilteringMode

    .. py:property:: mip_filter
        :type: slangpy.TextureFilteringMode

    .. py:property:: reduction_op
        :type: slangpy.TextureReductionOp

    .. py:property:: address_u
        :type: slangpy.TextureAddressingMode

    .. py:property:: address_v
        :type: slangpy.TextureAddressingMode

    .. py:property:: address_w
        :type: slangpy.TextureAddressingMode

    .. py:property:: mip_lod_bias
        :type: float

    .. py:property:: max_anisotropy
        :type: int

    .. py:property:: comparison_func
        :type: slangpy.ComparisonFunc

    .. py:property:: border_color
        :type: slangpy.math.float4

    .. py:property:: min_lod
        :type: float

    .. py:property:: max_lod
        :type: float

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.ScissorRect



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:staticmethod:: from_size(width: int, height: int) -> slangpy.ScissorRect

    .. py:property:: min_x
        :type: int

    .. py:property:: min_y
        :type: int

    .. py:property:: max_x
        :type: int

    .. py:property:: max_y
        :type: int



----

.. py:class:: slangpy.ShaderCacheStats



    .. py:property:: entry_count
        :type: int

        Number of entries in the cache.

    .. py:property:: hit_count
        :type: int

        Number of hits in the cache.

    .. py:property:: miss_count
        :type: int

        Number of misses in the cache.



----

.. py:class:: slangpy.ShaderCursor



    .. py:method:: __init__(self, shader_object: slangpy.ShaderObject) -> None

    .. py:method:: reinterpret(self, new_layout: slangpy.TypeLayoutReflection) -> slangpy.ShaderCursor

        Reinterpret the current cursor using a different type layout.

    .. py:method:: dereference(self) -> slangpy.ShaderCursor

    .. py:method:: find_entry_point(self, index: int) -> slangpy.ShaderCursor

    .. py:method:: get_field_by_index(self, field_index: int) -> slangpy.ShaderCursor

        N/A

    .. py:method:: find_field_index(self, name: str) -> int

        N/A

    .. py:method:: is_valid(self) -> bool

        N/A

    .. py:method:: find_field(self, name: str) -> slangpy.ShaderCursor

        N/A

    .. py:method:: find_element(self, index: int) -> slangpy.ShaderCursor

        N/A

    .. py:method:: has_field(self, name: str) -> bool

        N/A

    .. py:method:: has_element(self, index: int) -> bool

        N/A

    .. py:method:: set_data(self, data: ndarray[device='cpu']) -> None

    .. py:method:: write(self, val: object) -> None

        N/A



----

.. py:class:: slangpy.ShaderHotReloadEvent

    Event data for hot reload hook.



----

.. py:class:: slangpy.ShaderModel

    Base class: :py:class:`enum.IntEnum`



----

.. py:class:: slangpy.ShaderObject

    Base class: :py:class:`slangpy.Object`





----

.. py:class:: slangpy.ShaderOffset

    Represents the offset of a shader variable relative to its enclosing
    type/buffer/block.

    A `ShaderOffset` can be used to store the offset of a shader variable
    that might use ordinary/uniform data, resources like
    textures/buffers/samplers, or some combination.

    A `ShaderOffset` can also encode an invalid offset, to indicate that a
    particular shader variable is not present.

    .. py:property:: uniform_offset
        :type: int

    .. py:property:: binding_range_index
        :type: int

    .. py:property:: binding_array_index
        :type: int

    .. py:method:: is_valid(self) -> bool

        Check whether this offset is valid.



----

.. py:class:: slangpy.ShaderProgram

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: layout
        :type: slangpy.ProgramLayout

    .. py:property:: reflection
        :type: slangpy.ReflectionCursor

    .. py:method:: get_compilation_report(self) -> CompilationReport

        Return this shader program's compilation report.



----

.. py:class:: slangpy.ShaderStage

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.ShaderTable

    Base class: :py:class:`slangpy.DeviceChild`





----

.. py:class:: slangpy.ShaderTableDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: program
        :type: slangpy.ShaderProgram

    .. py:property:: ray_gen_entry_points
        :type: list[str]

    .. py:property:: miss_entry_points
        :type: list[str]

    .. py:property:: hit_group_names
        :type: list[str]

    .. py:property:: callable_entry_points
        :type: list[str]



----

.. py:class:: slangpy.SlangCompileError

    Base class: :py:class:`builtins.Exception`



----

.. py:class:: slangpy.SlangCompilerOptions

    Slang compiler options. Can be set when creating a Slang session.

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: include_paths
        :type: list[pathlib.Path]

        Specifies a list of include paths to be used when resolving
        module/include paths.

    .. py:property:: defines
        :type: dict[str, str]

        Specifies a list of preprocessor defines.

    .. py:property:: shader_model
        :type: slangpy.ShaderModel

        Specifies the shader model to use. Defaults to latest available on the
        device.

    .. py:property:: matrix_layout
        :type: slangpy.SlangMatrixLayout

        Specifies the matrix layout. Defaults to row-major.

    .. py:property:: enable_warnings
        :type: list[str]

        Specifies a list of warnings to enable (warning codes or names).

    .. py:property:: disable_warnings
        :type: list[str]

        Specifies a list of warnings to disable (warning codes or names).

    .. py:property:: warnings_as_errors
        :type: list[str]

        Specifies a list of warnings to be treated as errors (warning codes or
        names, or "all" to indicate all warnings).

    .. py:property:: report_downstream_time
        :type: bool

        Turn on/off downstream compilation time report.

    .. py:property:: report_perf_benchmark
        :type: bool

        Turn on/off reporting of time spend in different parts of the
        compiler.

    .. py:property:: skip_spirv_validation
        :type: bool

        Specifies whether or not to skip the validation step after emitting
        SPIRV.

    .. py:property:: floating_point_mode
        :type: slangpy.SlangFloatingPointMode

        Specifies the floating point mode.

    .. py:property:: debug_info
        :type: slangpy.SlangDebugInfoLevel

        Specifies the level of debug information to include in the generated
        code.

    .. py:property:: optimization
        :type: slangpy.SlangOptimizationLevel

        Specifies the optimization level.

    .. py:property:: downstream_args
        :type: list[str]

        Specifies a list of additional arguments to be passed to the
        downstream compiler. Only forwarded to downstream compilers that
        accept pass-through arguments: DXC (D3D12) and NVRTC (CUDA). Ignored
        for other backends.

    .. py:property:: dump_intermediates
        :type: bool

        When set will dump the intermediate source output.

    .. py:property:: dump_intermediates_prefix
        :type: str

        The file name prefix for the intermediate source output.



----

.. py:class:: slangpy.SlangDebugInfoLevel

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.SlangEntryPoint

    Base class: :py:class:`slangpy.Object`



    .. py:property:: name
        :type: str

    .. py:property:: stage
        :type: slangpy.ShaderStage

    .. py:property:: layout
        :type: slangpy.EntryPointLayout

    .. py:method:: rename(self, new_name: str) -> slangpy.SlangEntryPoint

    .. py:method:: with_name(self, new_name: str) -> slangpy.SlangEntryPoint

        Returns a copy of the entry point with a new name.

    .. py:method:: specialize(self, specialization_args: collections.abc.Sequence[slangpy.SpecializationArg]) -> slangpy.SlangEntryPoint



----

.. py:class:: slangpy.SlangFloatingPointMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.SlangLinkOptions

    Slang link options. These can optionally be set when linking a shader
    program.

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: floating_point_mode
        :type: slangpy.SlangFloatingPointMode | None

        Specifies the floating point mode.

    .. py:property:: debug_info
        :type: slangpy.SlangDebugInfoLevel | None

        Specifies the level of debug information to include in the generated
        code.

    .. py:property:: optimization
        :type: slangpy.SlangOptimizationLevel | None

        Specifies the optimization level.

    .. py:property:: downstream_args
        :type: list[str] | None

        Specifies a list of additional arguments to be passed to the
        downstream compiler. Only forwarded to downstream compilers that
        accept pass-through arguments: DXC (D3D12) and NVRTC (CUDA). Ignored
        for other backends.

    .. py:property:: dump_intermediates
        :type: bool | None

        When set will dump the intermediate source output.

    .. py:property:: dump_intermediates_prefix
        :type: str | None

        The file name prefix for the intermediate source output.



----

.. py:class:: slangpy.SlangMatrixLayout

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.SlangModule

    Base class: :py:class:`slangpy.Object`



    .. py:property:: session
        :type: slangpy.SlangSession

        The session from which this module was built.

    .. py:property:: name
        :type: str

        Module name.

    .. py:property:: path
        :type: pathlib.Path

        Module source path. This can be empty if the module was generated from
        a string.

    .. py:property:: layout
        :type: slangpy.ProgramLayout

        Combined layout reflecting the primary module and all linked modules.

    .. py:property:: entry_points
        :type: list[slangpy.SlangEntryPoint]

        Return vector of all current entry points in the module.

    .. py:property:: module_decl
        :type: slangpy.DeclReflection

        Get root decl ref for this module. Throws for composed modules (no
        single module to reflect).

    .. py:property:: is_composed
        :type: bool

        Returns true if this is a composed module.

    .. py:property:: source_modules
        :type: list[slangpy.SlangModule]

        Source modules that make up this composed module (empty for non-
        composed modules).

    .. py:method:: entry_point(self, name: str, type_conformances: collections.abc.Sequence[slangpy.TypeConformance] = []) -> slangpy.SlangEntryPoint

        Get an entry point, optionally applying type conformances to it.



----

.. py:class:: slangpy.SlangOptimizationLevel

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.SlangSession

    Base class: :py:class:`slangpy.Object`



    .. py:property:: device
        :type: slangpy.Device

    .. py:property:: desc
        :type: slangpy.SlangSessionDesc

    .. py:method:: load_module(self, module_name: str) -> slangpy.SlangModule

        Load a module by name.

    .. py:method:: load_module_from_source(self, module_name: str, source: str, path: str | os.PathLike | None = None) -> slangpy.SlangModule

        Load a module from string source code.

    .. py:method:: compose_modules(self, name: str, modules: collections.abc.Sequence[slangpy.SlangModule], type_conformances: collections.abc.Sequence[slangpy.TypeConformance] = []) -> slangpy.SlangModule

        Compose multiple modules into a single composed module. The composed
        module provides a unified layout and entry point access across all
        source modules.

    .. py:method:: link_program(self, modules: collections.abc.Sequence[slangpy.SlangModule], entry_points: collections.abc.Sequence[slangpy.SlangEntryPoint], link_options: slangpy.SlangLinkOptions | None = None) -> slangpy.ShaderProgram

        Link a program with a set of modules and entry points.

    .. py:method:: load_program(self, module_name: str, entry_point_names: collections.abc.Sequence[str], additional_source: str | None = None, link_options: slangpy.SlangLinkOptions | None = None) -> slangpy.ShaderProgram

        Load a program from a given module with a set of entry points.
        Internally this simply wraps link_program without requiring the user
        to explicitly load modules.

    .. py:method:: load_source(self, module_name: str) -> str

        Load the source code for a given module.



----

.. py:class:: slangpy.SlangSessionDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: compiler_options
        :type: slangpy.SlangCompilerOptions

    .. py:property:: add_default_include_paths
        :type: bool

    .. py:property:: cache_path
        :type: pathlib.Path | None



----

.. py:class:: slangpy.StencilOp

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.StoreOp

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.SubresourceLayout



    .. py:method:: __init__(self) -> None

    .. py:property:: size
        :type: slangpy.math.uint3

        Dimensions of the subresource (in texels).

    .. py:property:: col_pitch
        :type: int

        Stride in bytes between columns (i.e. blocks) of the subresource
        tensor.

    .. py:property:: row_pitch
        :type: int

        Stride in bytes between rows of the subresource tensor.

    .. py:property:: slice_pitch
        :type: int

        Stride in bytes between slices of the subresource tensor.

    .. py:property:: size_in_bytes
        :type: int

        Overall size required to fit the subresource data (typically size.z *
        slice_pitch).

    .. py:property:: block_width
        :type: int

        Block width in texels (1 for uncompressed formats).

    .. py:property:: block_height
        :type: int

        Block height in texels (1 for uncompressed formats).

    .. py:property:: row_count
        :type: int

        Number of rows. For uncompressed formats this matches size.y. For
        compressed formats this matches align_up(size.y, block_height) /
        block_height.



----

.. py:class:: slangpy.SubresourceRange



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: layer
        :type: int

        First array layer.

    .. py:property:: layer_count
        :type: int

        Number of array layers.

    .. py:property:: mip
        :type: int

        Most detailed mip level.

    .. py:property:: mip_count
        :type: int

        Number of mip levels.



----

.. py:class:: slangpy.Surface

    Base class: :py:class:`slangpy.Object`



    .. py:property:: info
        :type: slangpy.SurfaceInfo

        Returns the surface info.

    .. py:property:: config
        :type: slangpy.SurfaceConfig | None

        Returns the surface config.

    .. py:method:: configure(self, width: int, height: int, format: slangpy.Format = Format.undefined, usage: slangpy.TextureUsage = 0, desired_image_count: int = 3, vsync: bool = True) -> None

        Configure the surface.

    .. py:method:: configure(self, config: slangpy.SurfaceConfig) -> None
        :no-index:

    .. py:method:: unconfigure(self) -> None

        Unconfigure the surface.

    .. py:method:: acquire_next_image(self) -> slangpy.Texture

        Acquries the next surface image.

    .. py:method:: present(self) -> None

        Present the previously acquire image.



----

.. py:class:: slangpy.SurfaceConfig



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: format
        :type: slangpy.Format

        Surface texture format.

    .. py:property:: usage
        :type: slangpy.TextureUsage

        Surface texture usage.

    .. py:property:: width
        :type: int

        Surface texture width.

    .. py:property:: height
        :type: int

        Surface texture height.

    .. py:property:: desired_image_count
        :type: int

        Desired number of images.

    .. py:property:: vsync
        :type: bool

        Enable/disable vertical synchronization.



----

.. py:class:: slangpy.SurfaceInfo



    .. py:property:: preferred_format
        :type: slangpy.Format

        Preferred format for the surface.

    .. py:property:: supported_usage
        :type: slangpy.TextureUsage

        Supported texture usages.

    .. py:property:: formats
        :type: list[slangpy.Format]

        Supported texture formats.



----

.. py:class:: slangpy.Texture

    Base class: :py:class:`slangpy.Resource`



    .. py:property:: desc
        :type: slangpy.TextureDesc

    .. py:property:: type
        :type: slangpy.TextureType

    .. py:property:: format
        :type: slangpy.Format

    .. py:property:: width
        :type: int

    .. py:property:: height
        :type: int

    .. py:property:: depth
        :type: int

    .. py:property:: array_length
        :type: int

    .. py:property:: mip_count
        :type: int

    .. py:property:: layer_count
        :type: int

    .. py:property:: subresource_count
        :type: int

    .. py:property:: descriptor_handle_ro
        :type: slangpy.DescriptorHandle

        Get bindless texture descriptor handle for read access.

    .. py:property:: descriptor_handle_rw
        :type: slangpy.DescriptorHandle

        Get bindless texture descriptor handle for read-write access.

    .. py:property:: descriptor_handle_combined
        :type: slangpy.DescriptorHandle

        Get bindless combined texture/sampler descriptor handle.

    .. py:property:: shared_handle
        :type: slangpy.NativeHandle

        Get the shared resource handle. Note: Texture must be created with the
        ``TextureUsage::shared`` usage flag.

    .. py:method:: get_mip_width(self, mip: int = 0) -> int

    .. py:method:: get_mip_height(self, mip: int = 0) -> int

    .. py:method:: get_mip_depth(self, mip: int = 0) -> int

    .. py:method:: get_mip_size(self, mip: int = 0) -> slangpy.math.uint3

    .. py:method:: get_subresource_layout(self, mip: int, row_alignment: int = 4294967295) -> slangpy.SubresourceLayout

        Get layout of a texture subresource. By default, the row alignment
        used is that required by the target for direct buffer upload/download.
        Pass in 1 for a completely packed layout.

    .. py:method:: create_view(self, desc: slangpy.TextureViewDesc) -> slangpy.TextureView

    .. py:method:: create_view(self, dict: dict) -> slangpy.TextureView
        :no-index:

    .. py:method:: create_view(self, format: slangpy.Format = Format.undefined, aspect: slangpy.TextureAspect = TextureAspect.all, layer: int = 0, layer_count: int = 4294967295, mip: int = 0, mip_count: int = 4294967295, sampler: slangpy.Sampler | None = None, label: str = '') -> slangpy.TextureView
        :no-index:

    .. py:method:: to_bitmap(self, layer: int = 0, mip: int = 0) -> slangpy.Bitmap

    .. py:method:: to_numpy(self, layer: int = 0, mip: int = 0) -> numpy.ndarray[]

    .. py:method:: copy_from_numpy(self, data: numpy.ndarray[], layer: int = 0, mip: int = 0) -> None



----

.. py:class:: slangpy.TextureAddressingMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.TextureAspect

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.TextureDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: type
        :type: slangpy.TextureType

        Texture type.

    .. py:property:: format
        :type: slangpy.Format

        Texture format.

    .. py:property:: width
        :type: int

        Width in pixels.

    .. py:property:: height
        :type: int

        Height in pixels.

    .. py:property:: depth
        :type: int

        Depth in pixels.

    .. py:property:: array_length
        :type: int

        Array length.

    .. py:property:: mip_count
        :type: int

        Number of mip levels (ALL_MIPS for all mip levels).

    .. py:property:: sample_count
        :type: int

        Number of samples per pixel.

    .. py:property:: sample_quality
        :type: int

        Quality level for multisampled textures.

    .. py:property:: memory_type
        :type: slangpy.MemoryType

    .. py:property:: usage
        :type: slangpy.TextureUsage

    .. py:property:: default_state
        :type: slangpy.ResourceState

    .. py:property:: label
        :type: str

        Debug label.



----

.. py:class:: slangpy.TextureFilteringMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.TextureReductionOp

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.TextureType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.TextureUsage

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.TextureView

    Base class: :py:class:`slangpy.DeviceChild`



    .. py:property:: texture
        :type: slangpy.Texture

    .. py:property:: desc
        :type: slangpy.TextureViewDesc

    .. py:property:: format
        :type: slangpy.Format

    .. py:property:: aspect
        :type: slangpy.TextureAspect

    .. py:property:: subresource_range
        :type: slangpy.SubresourceRange

    .. py:property:: label
        :type: str

    .. py:property:: descriptor_handle_ro
        :type: slangpy.DescriptorHandle

        Get bindless texture descriptor handle for read access.

    .. py:property:: descriptor_handle_rw
        :type: slangpy.DescriptorHandle

        Get bindless texture descriptor handle for read-write access.

    .. py:property:: descriptor_handle_combined
        :type: slangpy.DescriptorHandle

        Get bindless combined texture/sampler descriptor handle.

    .. py:property:: native_handle
        :type: slangpy.NativeHandle

        Get the native texture view handle.



----

.. py:class:: slangpy.TextureViewDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: format
        :type: slangpy.Format

    .. py:property:: aspect
        :type: slangpy.TextureAspect

    .. py:property:: subresource_range
        :type: slangpy.SubresourceRange

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.TypeConformance

    Type conformance entry. Type conformances are used to narrow the set
    of types supported by a slang interface. They can be specified on an
    entry point to omit generating code for types that do not conform.

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, interface_name: str, type_name: str, id: int = -1) -> None
        :no-index:

    .. py:method:: __init__(self, arg: tuple, /) -> None
        :no-index:

    .. py:property:: interface_name
        :type: str

        Name of the interface.

    .. py:property:: type_name
        :type: str

        Name of the concrete type.

    .. py:property:: id
        :type: int

        Unique id per type for an interface (optional).



----

.. py:class:: slangpy.TypeLayoutReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:property:: kind
        :type: slangpy.TypeReflection.Kind

    .. py:property:: name
        :type: str

    .. py:property:: size
        :type: int

    .. py:property:: stride
        :type: int

    .. py:property:: alignment
        :type: int

    .. py:property:: type
        :type: slangpy.TypeReflection

    .. py:property:: fields
        :type: slangpy.TypeLayoutReflectionFieldList

    .. py:property:: element_type_layout
        :type: slangpy.TypeLayoutReflection

    .. py:method:: unwrap_array(self) -> slangpy.TypeLayoutReflection



----

.. py:class:: slangpy.TypeLayoutReflectionFieldList





----

.. py:class:: slangpy.TypeReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:class:: slangpy.TypeReflection.Kind

        Base class: :py:class:`enum.Enum`



    .. py:class:: slangpy.TypeReflection.ScalarType

        Base class: :py:class:`enum.Enum`



    .. py:class:: slangpy.TypeReflection.ResourceShape

        Base class: :py:class:`enum.Enum`



    .. py:class:: slangpy.TypeReflection.ResourceAccess

        Base class: :py:class:`enum.Enum`



    .. py:class:: slangpy.TypeReflection.ParameterCategory

        Base class: :py:class:`enum.Enum`



    .. py:property:: kind
        :type: slangpy.TypeReflection.Kind

    .. py:property:: name
        :type: str

    .. py:property:: full_name
        :type: str

    .. py:property:: fields
        :type: slangpy.TypeReflectionFieldList

    .. py:property:: element_count
        :type: int

    .. py:property:: element_type
        :type: slangpy.TypeReflection

    .. py:property:: row_count
        :type: int

    .. py:property:: col_count
        :type: int

    .. py:property:: scalar_type
        :type: slangpy.TypeReflection.ScalarType

    .. py:property:: resource_result_type
        :type: slangpy.TypeReflection

    .. py:property:: resource_shape
        :type: slangpy.TypeReflection.ResourceShape

    .. py:property:: resource_access
        :type: slangpy.TypeReflection.ResourceAccess

    .. py:method:: get_user_attribute_count(self) -> int

    .. py:method:: get_user_attribute_by_index(self, index: int) -> slangpy.Attribute

    .. py:method:: find_user_attribute_by_name(self, name: str) -> slangpy.Attribute

    .. py:method:: unwrap_array(self) -> slangpy.TypeReflection



----

.. py:class:: slangpy.TypeReflectionFieldList





----

.. py:class:: slangpy.VariableLayoutReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:property:: name
        :type: str

    .. py:property:: variable
        :type: slangpy.VariableReflection

    .. py:property:: type_layout
        :type: slangpy.TypeLayoutReflection

    .. py:property:: offset
        :type: int



----

.. py:class:: slangpy.VariableReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`

    .. py:property:: name
        :type: str

        Variable name.

    .. py:property:: type
        :type: slangpy.TypeReflection

        Variable type reflection.

    .. py:method:: has_modifier(self, modifier: slangpy.ModifierID) -> bool

        Check if variable has a given modifier (e.g. 'inout').



----

.. py:class:: slangpy.VertexStreamDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: stride
        :type: int

        The stride in bytes for this vertex stream.

    .. py:property:: slot_class
        :type: slangpy.InputSlotClass

        Whether the stream contains per-vertex or per-instance data.

    .. py:property:: instance_data_step_rate
        :type: int

        How many instances to draw per chunk of data.



----

.. py:class:: slangpy.Viewport



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:staticmethod:: from_size(width: float, height: float) -> slangpy.Viewport

    .. py:property:: x
        :type: float

    .. py:property:: y
        :type: float

    .. py:property:: width
        :type: float

    .. py:property:: height
        :type: float

    .. py:property:: min_depth
        :type: float

    .. py:property:: max_depth
        :type: float



----

.. py:class:: slangpy.WindowHandle

    Native window handle.

    .. py:method:: __init__(self, hwnd: int) -> None



----

Application
-----------

.. py:class:: slangpy.AppDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: device
        :type: slangpy.Device

        Device to use for rendering. If not provided, a default device will be
        created.



----

.. py:class:: slangpy.App

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, arg: slangpy.AppDesc, /) -> None

    .. py:method:: __init__(self, device: slangpy.Device | None = None) -> None
        :no-index:

    .. py:property:: device
        :type: slangpy.Device

    .. py:method:: run(self) -> None

    .. py:method:: run_frame(self) -> None

    .. py:method:: terminate(self) -> None



----

.. py:class:: slangpy.AppWindowDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: width
        :type: int

        Width of the window in pixels.

    .. py:property:: height
        :type: int

        Height of the window in pixels.

    .. py:property:: title
        :type: str

        Title of the window.

    .. py:property:: mode
        :type: slangpy.WindowMode

        Window mode.

    .. py:property:: resizable
        :type: bool

        Whether the window is resizable.

    .. py:property:: surface_format
        :type: slangpy.Format

        Format of the swapchain images.

    .. py:property:: enable_vsync
        :type: bool

        Enable/disable vertical synchronization.



----

.. py:class:: slangpy.AppWindow

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, app: slangpy.App, width: int = 1920, height: int = 1280, title: str = 'slangpy', mode: slangpy.WindowMode = WindowMode.normal, resizable: bool = True, surface_format: slangpy.Format = Format.undefined, enable_vsync: bool = False) -> None

    .. py:class:: slangpy.AppWindow.RenderContext



        .. py:property:: surface_texture
            :type: slangpy.Texture

        .. py:property:: command_encoder
            :type: slangpy.CommandEncoder

    .. py:property:: device
        :type: slangpy.Device

    .. py:property:: screen
        :type: slangpy.ui.Screen

    .. py:method:: render(self, render_context: slangpy.AppWindow.RenderContext) -> None

    .. py:method:: on_resize(self, width: int, height: int) -> None

    .. py:method:: on_keyboard_event(self, event: slangpy.KeyboardEvent) -> None

    .. py:method:: on_mouse_event(self, event: slangpy.MouseEvent) -> None

    .. py:method:: on_gamepad_event(self, event: slangpy.GamepadEvent) -> None

    .. py:method:: on_drop_files(self, files: collections.abc.Sequence[str]) -> None



----

Math
----

.. py:class:: slangpy.math.float1

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: float) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[float]) -> None
        :no-index:

    .. py:property:: x
        :type: float

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.float2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: float) -> None
        :no-index:

    .. py:method:: __init__(self, x: float, y: float) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[float]) -> None
        :no-index:

    .. py:property:: x
        :type: float

    .. py:property:: y
        :type: float

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.float3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: float) -> None
        :no-index:

    .. py:method:: __init__(self, x: float, y: float, z: float) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.float2, z: float) -> None
        :no-index:

    .. py:method:: __init__(self, x: float, yz: slangpy.math.float2) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[float]) -> None
        :no-index:

    .. py:property:: x
        :type: float

    .. py:property:: y
        :type: float

    .. py:property:: z
        :type: float

    .. py:property:: xy
        :type: slangpy.math.float2

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.float4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: float) -> None
        :no-index:

    .. py:method:: __init__(self, x: float, y: float, z: float, w: float) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.float2, zw: slangpy.math.float2) -> None
        :no-index:

    .. py:method:: __init__(self, xyz: slangpy.math.float3, w: float) -> None
        :no-index:

    .. py:method:: __init__(self, x: float, yzw: slangpy.math.float3) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[float]) -> None
        :no-index:

    .. py:property:: x
        :type: float

    .. py:property:: y
        :type: float

    .. py:property:: z
        :type: float

    .. py:property:: w
        :type: float

    .. py:property:: xyz
        :type: slangpy.math.float3

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.float1
    Alias class: :py:class:`slangpy.math.float1`



----

.. py:class:: slangpy.float2
    Alias class: :py:class:`slangpy.math.float2`



----

.. py:class:: slangpy.float3
    Alias class: :py:class:`slangpy.math.float3`



----

.. py:class:: slangpy.float4
    Alias class: :py:class:`slangpy.math.float4`



----

.. py:class:: slangpy.math.int1

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.int2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, y: int) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: y
        :type: int

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.int3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, y: int, z: int) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.int2, z: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, yz: slangpy.math.int2) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: y
        :type: int

    .. py:property:: z
        :type: int

    .. py:property:: xy
        :type: slangpy.math.int2

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.int4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, y: int, z: int, w: int) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.int2, zw: slangpy.math.int2) -> None
        :no-index:

    .. py:method:: __init__(self, xyz: slangpy.math.int3, w: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, yzw: slangpy.math.int3) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: y
        :type: int

    .. py:property:: z
        :type: int

    .. py:property:: w
        :type: int

    .. py:property:: xyz
        :type: slangpy.math.int3

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.int1
    Alias class: :py:class:`slangpy.math.int1`



----

.. py:class:: slangpy.int2
    Alias class: :py:class:`slangpy.math.int2`



----

.. py:class:: slangpy.int3
    Alias class: :py:class:`slangpy.math.int3`



----

.. py:class:: slangpy.int4
    Alias class: :py:class:`slangpy.math.int4`



----

.. py:class:: slangpy.math.uint1

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.uint2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, y: int) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: y
        :type: int

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.uint3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, y: int, z: int) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.uint2, z: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, yz: slangpy.math.uint2) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: y
        :type: int

    .. py:property:: z
        :type: int

    .. py:property:: xy
        :type: slangpy.math.uint2

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.uint4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, y: int, z: int, w: int) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.uint2, zw: slangpy.math.uint2) -> None
        :no-index:

    .. py:method:: __init__(self, xyz: slangpy.math.uint3, w: int) -> None
        :no-index:

    .. py:method:: __init__(self, x: int, yzw: slangpy.math.uint3) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[int]) -> None
        :no-index:

    .. py:property:: x
        :type: int

    .. py:property:: y
        :type: int

    .. py:property:: z
        :type: int

    .. py:property:: w
        :type: int

    .. py:property:: xyz
        :type: slangpy.math.uint3

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.uint1
    Alias class: :py:class:`slangpy.math.uint1`



----

.. py:class:: slangpy.uint2
    Alias class: :py:class:`slangpy.math.uint2`



----

.. py:class:: slangpy.uint3
    Alias class: :py:class:`slangpy.math.uint3`



----

.. py:class:: slangpy.uint4
    Alias class: :py:class:`slangpy.math.uint4`



----

.. py:class:: slangpy.math.bool1

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: bool) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[bool]) -> None
        :no-index:

    .. py:property:: x
        :type: bool

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.bool2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: bool) -> None
        :no-index:

    .. py:method:: __init__(self, x: bool, y: bool) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[bool]) -> None
        :no-index:

    .. py:property:: x
        :type: bool

    .. py:property:: y
        :type: bool

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.bool3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: bool) -> None
        :no-index:

    .. py:method:: __init__(self, x: bool, y: bool, z: bool) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.bool2, z: bool) -> None
        :no-index:

    .. py:method:: __init__(self, x: bool, yz: slangpy.math.bool2) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[bool]) -> None
        :no-index:

    .. py:property:: x
        :type: bool

    .. py:property:: y
        :type: bool

    .. py:property:: z
        :type: bool

    .. py:property:: xy
        :type: slangpy.math.bool2

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.bool4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: bool) -> None
        :no-index:

    .. py:method:: __init__(self, x: bool, y: bool, z: bool, w: bool) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.bool2, zw: slangpy.math.bool2) -> None
        :no-index:

    .. py:method:: __init__(self, xyz: slangpy.math.bool3, w: bool) -> None
        :no-index:

    .. py:method:: __init__(self, x: bool, yzw: slangpy.math.bool3) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[bool]) -> None
        :no-index:

    .. py:property:: x
        :type: bool

    .. py:property:: y
        :type: bool

    .. py:property:: z
        :type: bool

    .. py:property:: w
        :type: bool

    .. py:property:: xyz
        :type: slangpy.math.bool3

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.bool1
    Alias class: :py:class:`slangpy.math.bool1`



----

.. py:class:: slangpy.bool2
    Alias class: :py:class:`slangpy.math.bool2`



----

.. py:class:: slangpy.bool3
    Alias class: :py:class:`slangpy.math.bool3`



----

.. py:class:: slangpy.bool4
    Alias class: :py:class:`slangpy.math.bool4`



----

.. py:class:: slangpy.math.float16_t

    .. py:method:: __init__(self, value: float) -> None

    .. py:method:: __init__(self, value: float) -> None
        :no-index:



----

.. py:class:: slangpy.float16_t
    Alias class: :py:class:`slangpy.math.float16_t`



----

.. py:class:: slangpy.math.float16_t1

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[slangpy.math.float16_t]) -> None
        :no-index:

    .. py:property:: x
        :type: slangpy.math.float16_t

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.float16_t2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, x: slangpy.math.float16_t, y: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[slangpy.math.float16_t]) -> None
        :no-index:

    .. py:property:: x
        :type: slangpy.math.float16_t

    .. py:property:: y
        :type: slangpy.math.float16_t

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.float16_t3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, x: slangpy.math.float16_t, y: slangpy.math.float16_t, z: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.float16_t2, z: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, x: slangpy.math.float16_t, yz: slangpy.math.float16_t2) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[slangpy.math.float16_t]) -> None
        :no-index:

    .. py:property:: x
        :type: slangpy.math.float16_t

    .. py:property:: y
        :type: slangpy.math.float16_t

    .. py:property:: z
        :type: slangpy.math.float16_t

    .. py:property:: xy
        :type: slangpy.math.float16_t2

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.math.float16_t4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, scalar: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, x: slangpy.math.float16_t, y: slangpy.math.float16_t, z: slangpy.math.float16_t, w: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, xy: slangpy.math.float16_t2, zw: slangpy.math.float16_t2) -> None
        :no-index:

    .. py:method:: __init__(self, xyz: slangpy.math.float16_t3, w: slangpy.math.float16_t) -> None
        :no-index:

    .. py:method:: __init__(self, x: slangpy.math.float16_t, yzw: slangpy.math.float16_t3) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[slangpy.math.float16_t]) -> None
        :no-index:

    .. py:property:: x
        :type: slangpy.math.float16_t

    .. py:property:: y
        :type: slangpy.math.float16_t

    .. py:property:: z
        :type: slangpy.math.float16_t

    .. py:property:: w
        :type: slangpy.math.float16_t

    .. py:property:: xyz
        :type: slangpy.math.float16_t3

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.float16_t1
    Alias class: :py:class:`slangpy.math.float16_t1`



----

.. py:class:: slangpy.float16_t2
    Alias class: :py:class:`slangpy.math.float16_t2`



----

.. py:class:: slangpy.float16_t3
    Alias class: :py:class:`slangpy.math.float16_t3`



----

.. py:class:: slangpy.float16_t4
    Alias class: :py:class:`slangpy.math.float16_t4`



----

.. py:class:: slangpy.math.float2x2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(2, 2)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float2x2

    .. py:staticmethod:: identity() -> slangpy.math.float2x2

    .. py:method:: get_row(self, row: int) -> slangpy.math.float2

    .. py:method:: set_row(self, row: int, value: slangpy.math.float2) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float2

    .. py:method:: set_col(self, col: int, value: slangpy.math.float2) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(2, 2), writable=False]



----

.. py:class:: slangpy.math.float2x3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(2, 3)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float2x3

    .. py:staticmethod:: identity() -> slangpy.math.float2x3

    .. py:method:: get_row(self, row: int) -> slangpy.math.float3

    .. py:method:: set_row(self, row: int, value: slangpy.math.float3) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float2

    .. py:method:: set_col(self, col: int, value: slangpy.math.float2) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(2, 3), writable=False]



----

.. py:class:: slangpy.math.float2x4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(2, 4)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float2x4

    .. py:staticmethod:: identity() -> slangpy.math.float2x4

    .. py:method:: get_row(self, row: int) -> slangpy.math.float4

    .. py:method:: set_row(self, row: int, value: slangpy.math.float4) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float2

    .. py:method:: set_col(self, col: int, value: slangpy.math.float2) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(2, 4), writable=False]



----

.. py:class:: slangpy.math.float3x2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(3, 2)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float3x2

    .. py:staticmethod:: identity() -> slangpy.math.float3x2

    .. py:method:: get_row(self, row: int) -> slangpy.math.float2

    .. py:method:: set_row(self, row: int, value: slangpy.math.float2) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float3

    .. py:method:: set_col(self, col: int, value: slangpy.math.float3) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(3, 2), writable=False]



----

.. py:class:: slangpy.math.float3x3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: slangpy.math.float4x4, /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: slangpy.math.float3x4, /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(3, 3)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float3x3

    .. py:staticmethod:: identity() -> slangpy.math.float3x3

    .. py:method:: get_row(self, row: int) -> slangpy.math.float3

    .. py:method:: set_row(self, row: int, value: slangpy.math.float3) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float3

    .. py:method:: set_col(self, col: int, value: slangpy.math.float3) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(3, 3), writable=False]



----

.. py:class:: slangpy.math.float3x4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: slangpy.math.float3x3, /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: slangpy.math.float4x4, /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(3, 4)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float3x4

    .. py:staticmethod:: identity() -> slangpy.math.float3x4

    .. py:method:: get_row(self, row: int) -> slangpy.math.float4

    .. py:method:: set_row(self, row: int, value: slangpy.math.float4) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float3

    .. py:method:: set_col(self, col: int, value: slangpy.math.float3) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(3, 4), writable=False]



----

.. py:class:: slangpy.math.float4x2

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(4, 2)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float4x2

    .. py:staticmethod:: identity() -> slangpy.math.float4x2

    .. py:method:: get_row(self, row: int) -> slangpy.math.float2

    .. py:method:: set_row(self, row: int, value: slangpy.math.float2) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float4

    .. py:method:: set_col(self, col: int, value: slangpy.math.float4) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(4, 2), writable=False]



----

.. py:class:: slangpy.math.float4x3

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(4, 3)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float4x3

    .. py:staticmethod:: identity() -> slangpy.math.float4x3

    .. py:method:: get_row(self, row: int) -> slangpy.math.float3

    .. py:method:: set_row(self, row: int, value: slangpy.math.float3) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float4

    .. py:method:: set_col(self, col: int, value: slangpy.math.float4) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(4, 3), writable=False]



----

.. py:class:: slangpy.math.float4x4

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: slangpy.math.float3x3, /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: slangpy.math.float3x4, /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: collections.abc.Sequence[float], /) -> None
        :no-index:

    .. py:method:: __init__(self, arg: ndarray[dtype=float32, shape=(4, 4)], /) -> None
        :no-index:

    .. py:staticmethod:: zeros() -> slangpy.math.float4x4

    .. py:staticmethod:: identity() -> slangpy.math.float4x4

    .. py:method:: get_row(self, row: int) -> slangpy.math.float4

    .. py:method:: set_row(self, row: int, value: slangpy.math.float4) -> None

    .. py:method:: get_col(self, col: int) -> slangpy.math.float4

    .. py:method:: set_col(self, col: int, value: slangpy.math.float4) -> None

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object

    .. py:method:: to_numpy(self) -> numpy.ndarray[dtype=float32, shape=(4, 4), writable=False]



----

.. py:class:: slangpy.float2x2
    Alias class: :py:class:`slangpy.math.float2x2`



----

.. py:class:: slangpy.float2x3
    Alias class: :py:class:`slangpy.math.float2x3`



----

.. py:class:: slangpy.float2x4
    Alias class: :py:class:`slangpy.math.float2x4`



----

.. py:class:: slangpy.float3x2
    Alias class: :py:class:`slangpy.math.float3x2`



----

.. py:class:: slangpy.float3x3
    Alias class: :py:class:`slangpy.math.float3x3`



----

.. py:class:: slangpy.float3x4
    Alias class: :py:class:`slangpy.math.float3x4`



----

.. py:class:: slangpy.float4x2
    Alias class: :py:class:`slangpy.math.float4x2`



----

.. py:class:: slangpy.float4x3
    Alias class: :py:class:`slangpy.math.float4x3`



----

.. py:class:: slangpy.float4x4
    Alias class: :py:class:`slangpy.math.float4x4`



----

.. py:class:: slangpy.math.quatf

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, x: float, y: float, z: float, w: float) -> None
        :no-index:

    .. py:method:: __init__(self, xyz: slangpy.math.float3, w: float) -> None
        :no-index:

    .. py:method:: __init__(self, a: collections.abc.Sequence[float]) -> None
        :no-index:

    .. py:staticmethod:: identity() -> slangpy.math.quatf

    .. py:property:: x
        :type: float

    .. py:property:: y
        :type: float

    .. py:property:: z
        :type: float

    .. py:property:: w
        :type: float

    .. py:property:: shape
        :type: tuple

    .. py:property:: element_type
        :type: object



----

.. py:class:: slangpy.quatf
    Alias class: :py:class:`slangpy.math.quatf`



----

.. py:class:: slangpy.math.Handedness

    Base class: :py:class:`enum.Enum`



----

.. py:function:: slangpy.math.isfinite(x: float) -> bool

.. py:function:: slangpy.math.isfinite(x: float) -> bool
    :no-index:

.. py:function:: slangpy.math.isfinite(x: slangpy.math.float16_t) -> bool
    :no-index:

.. py:function:: slangpy.math.isfinite(x: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.isfinite(x: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.isfinite(x: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.isfinite(x: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.isfinite(x: slangpy.math.quatf) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.isinf(x: float) -> bool

.. py:function:: slangpy.math.isinf(x: float) -> bool
    :no-index:

.. py:function:: slangpy.math.isinf(x: slangpy.math.float16_t) -> bool
    :no-index:

.. py:function:: slangpy.math.isinf(x: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.isinf(x: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.isinf(x: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.isinf(x: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.isinf(x: slangpy.math.quatf) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.isnan(x: float) -> bool

.. py:function:: slangpy.math.isnan(x: float) -> bool
    :no-index:

.. py:function:: slangpy.math.isnan(x: slangpy.math.float16_t) -> bool
    :no-index:

.. py:function:: slangpy.math.isnan(x: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.isnan(x: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.isnan(x: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.isnan(x: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.isnan(x: slangpy.math.quatf) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.floor(x: float) -> float

.. py:function:: slangpy.math.floor(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.floor(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.floor(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.floor(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.floor(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.ceil(x: float) -> float

.. py:function:: slangpy.math.ceil(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.ceil(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.ceil(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.ceil(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.ceil(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.trunc(x: float) -> float

.. py:function:: slangpy.math.trunc(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.trunc(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.trunc(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.trunc(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.trunc(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.round(x: float) -> float

.. py:function:: slangpy.math.round(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.round(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.round(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.round(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.round(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.pow(x: float, y: float) -> float

.. py:function:: slangpy.math.pow(x: float, y: float) -> float
    :no-index:

.. py:function:: slangpy.math.pow(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.pow(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.pow(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.pow(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.sqrt(x: float) -> float

.. py:function:: slangpy.math.sqrt(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.sqrt(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.sqrt(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.sqrt(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.sqrt(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.rsqrt(x: float) -> float

.. py:function:: slangpy.math.rsqrt(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.rsqrt(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.rsqrt(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.rsqrt(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.rsqrt(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.exp(x: float) -> float

.. py:function:: slangpy.math.exp(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.exp(x: slangpy.math.float16_t) -> slangpy.math.float16_t
    :no-index:

.. py:function:: slangpy.math.exp(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.exp(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.exp(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.exp(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.exp2(x: float) -> float

.. py:function:: slangpy.math.exp2(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.exp2(x: slangpy.math.float16_t) -> slangpy.math.float16_t
    :no-index:

.. py:function:: slangpy.math.exp2(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.exp2(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.exp2(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.exp2(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.log(x: float) -> float

.. py:function:: slangpy.math.log(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.log(x: slangpy.math.float16_t) -> slangpy.math.float16_t
    :no-index:

.. py:function:: slangpy.math.log(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.log(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.log(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.log(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.log2(x: float) -> float

.. py:function:: slangpy.math.log2(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.log2(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.log2(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.log2(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.log2(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.log10(x: float) -> float

.. py:function:: slangpy.math.log10(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.log10(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.log10(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.log10(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.log10(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.radians(x: float) -> float

.. py:function:: slangpy.math.radians(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.radians(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.radians(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.radians(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.radians(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.degrees(x: float) -> float

.. py:function:: slangpy.math.degrees(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.degrees(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.degrees(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.degrees(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.degrees(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.sin(x: float) -> float

.. py:function:: slangpy.math.sin(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.sin(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.sin(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.sin(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.sin(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.cos(x: float) -> float

.. py:function:: slangpy.math.cos(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.cos(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.cos(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.cos(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.cos(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.tan(x: float) -> float

.. py:function:: slangpy.math.tan(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.tan(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.tan(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.tan(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.tan(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.asin(x: float) -> float

.. py:function:: slangpy.math.asin(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.asin(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.asin(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.asin(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.asin(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.acos(x: float) -> float

.. py:function:: slangpy.math.acos(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.acos(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.acos(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.acos(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.acos(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.atan(x: float) -> float

.. py:function:: slangpy.math.atan(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.atan(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.atan(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.atan(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.atan(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.atan2(y: float, x: float) -> float

.. py:function:: slangpy.math.atan2(y: float, x: float) -> float
    :no-index:

.. py:function:: slangpy.math.atan2(y: slangpy.math.float1, x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.atan2(y: slangpy.math.float2, x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.atan2(y: slangpy.math.float3, x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.atan2(y: slangpy.math.float4, x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.sinh(x: float) -> float

.. py:function:: slangpy.math.sinh(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.sinh(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.sinh(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.sinh(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.sinh(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.cosh(x: float) -> float

.. py:function:: slangpy.math.cosh(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.cosh(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.cosh(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.cosh(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.cosh(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.tanh(x: float) -> float

.. py:function:: slangpy.math.tanh(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.tanh(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.tanh(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.tanh(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.tanh(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.fmod(x: float, y: float) -> float

.. py:function:: slangpy.math.fmod(x: float, y: float) -> float
    :no-index:

.. py:function:: slangpy.math.fmod(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.fmod(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.fmod(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.fmod(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.frac(x: float) -> float

.. py:function:: slangpy.math.frac(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.frac(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.frac(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.frac(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.frac(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.lerp(x: float, y: float, s: float) -> float

.. py:function:: slangpy.math.lerp(x: float, y: float, s: float) -> float
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float1, y: slangpy.math.float1, s: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float1, y: slangpy.math.float1, s: float) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float2, y: slangpy.math.float2, s: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float2, y: slangpy.math.float2, s: float) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float3, y: slangpy.math.float3, s: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float3, y: slangpy.math.float3, s: float) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float4, y: slangpy.math.float4, s: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.float4, y: slangpy.math.float4, s: float) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.lerp(x: slangpy.math.quatf, y: slangpy.math.quatf, s: float) -> slangpy.math.quatf
    :no-index:



----

.. py:function:: slangpy.math.rcp(x: float) -> float

.. py:function:: slangpy.math.rcp(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.rcp(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.rcp(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.rcp(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.rcp(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.saturate(x: float) -> float

.. py:function:: slangpy.math.saturate(x: float) -> float
    :no-index:

.. py:function:: slangpy.math.saturate(x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.saturate(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.saturate(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.saturate(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.step(x: float, y: float) -> float

.. py:function:: slangpy.math.step(x: float, y: float) -> float
    :no-index:

.. py:function:: slangpy.math.step(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.step(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.step(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.step(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.smoothstep(min: float, max: float, x: float) -> float

.. py:function:: slangpy.math.smoothstep(min: float, max: float, x: float) -> float
    :no-index:

.. py:function:: slangpy.math.smoothstep(min: slangpy.math.float1, max: slangpy.math.float1, x: slangpy.math.float1) -> slangpy.math.float1
    :no-index:

.. py:function:: slangpy.math.smoothstep(min: slangpy.math.float2, max: slangpy.math.float2, x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.smoothstep(min: slangpy.math.float3, max: slangpy.math.float3, x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.smoothstep(min: slangpy.math.float4, max: slangpy.math.float4, x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.f16tof32(x: int) -> float

.. py:function:: slangpy.math.f16tof32(x: slangpy.math.uint2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.f16tof32(x: slangpy.math.uint3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.f16tof32(x: slangpy.math.uint4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.f32tof16(x: float) -> int

.. py:function:: slangpy.math.f32tof16(x: slangpy.math.float2) -> slangpy.math.uint2
    :no-index:

.. py:function:: slangpy.math.f32tof16(x: slangpy.math.float3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.f32tof16(x: slangpy.math.float4) -> slangpy.math.uint4
    :no-index:



----

.. py:function:: slangpy.math.asfloat(x: int) -> float

.. py:function:: slangpy.math.asfloat(x: int) -> float
    :no-index:



----

.. py:function:: slangpy.math.asfloat16(x: int) -> slangpy.math.float16_t



----

.. py:function:: slangpy.math.asuint(x: float) -> int

.. py:function:: slangpy.math.asuint(x: slangpy.math.float2) -> slangpy.math.uint2
    :no-index:

.. py:function:: slangpy.math.asuint(x: slangpy.math.float3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.asuint(x: slangpy.math.float4) -> slangpy.math.uint4
    :no-index:



----

.. py:function:: slangpy.math.asint(x: float) -> int



----

.. py:function:: slangpy.math.asuint16(x: slangpy.math.float16_t) -> int



----

.. py:function:: slangpy.math.select(condition: slangpy.math.bool1, true_value: slangpy.math.float1, false_value: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.select(condition: slangpy.math.bool2, true_value: slangpy.math.float2, false_value: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool3, true_value: slangpy.math.float3, false_value: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool4, true_value: slangpy.math.float4, false_value: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool1, true_value: slangpy.math.uint1, false_value: slangpy.math.uint1) -> slangpy.math.uint1
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool2, true_value: slangpy.math.uint2, false_value: slangpy.math.uint2) -> slangpy.math.uint2
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool3, true_value: slangpy.math.uint3, false_value: slangpy.math.uint3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool4, true_value: slangpy.math.uint4, false_value: slangpy.math.uint4) -> slangpy.math.uint4
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool1, true_value: slangpy.math.int1, false_value: slangpy.math.int1) -> slangpy.math.int1
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool2, true_value: slangpy.math.int2, false_value: slangpy.math.int2) -> slangpy.math.int2
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool3, true_value: slangpy.math.int3, false_value: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool4, true_value: slangpy.math.int4, false_value: slangpy.math.int4) -> slangpy.math.int4
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool1, true_value: slangpy.math.bool1, false_value: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool2, true_value: slangpy.math.bool2, false_value: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool3, true_value: slangpy.math.bool3, false_value: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.select(condition: slangpy.math.bool4, true_value: slangpy.math.bool4, false_value: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.eq(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.bool1

.. py:function:: slangpy.math.eq(x: slangpy.math.float1, y: float) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: float, y: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.float2, y: float) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: float, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.float3, y: float) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: float, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.float4, y: float) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: float, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.uint4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.int4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: int, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool1, y: bool) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: bool, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool2, y: bool) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: bool, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool3, y: bool) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: bool, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.bool4, y: bool) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: bool, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.eq(x: slangpy.math.quatf, y: slangpy.math.quatf) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.ne(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.bool1

.. py:function:: slangpy.math.ne(x: slangpy.math.float1, y: float) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: float, y: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.float2, y: float) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: float, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.float3, y: float) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: float, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.float4, y: float) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: float, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.uint4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.int4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: int, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool1, y: bool) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: bool, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool2, y: bool) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: bool, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool3, y: bool) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: bool, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.bool4, y: bool) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: bool, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ne(x: slangpy.math.quatf, y: slangpy.math.quatf) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.lt(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.bool1

.. py:function:: slangpy.math.lt(x: slangpy.math.float1, y: float) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: float, y: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.float2, y: float) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: float, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.float3, y: float) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: float, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.float4, y: float) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: float, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.uint4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.int4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: int, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool1, y: bool) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: bool, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool2, y: bool) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: bool, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool3, y: bool) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: bool, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: slangpy.math.bool4, y: bool) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.lt(x: bool, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.gt(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.bool1

.. py:function:: slangpy.math.gt(x: slangpy.math.float1, y: float) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: float, y: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.float2, y: float) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: float, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.float3, y: float) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: float, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.float4, y: float) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: float, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.uint4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.int4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: int, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool1, y: bool) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: bool, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool2, y: bool) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: bool, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool3, y: bool) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: bool, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: slangpy.math.bool4, y: bool) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.gt(x: bool, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.le(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.bool1

.. py:function:: slangpy.math.le(x: slangpy.math.float1, y: float) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: float, y: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.float2, y: float) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: float, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.float3, y: float) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: float, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.float4, y: float) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: float, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.uint4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.int4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: int, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool1, y: bool) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: bool, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool2, y: bool) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: bool, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool3, y: bool) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: bool, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: slangpy.math.bool4, y: bool) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.le(x: bool, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.ge(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.bool1

.. py:function:: slangpy.math.ge(x: slangpy.math.float1, y: float) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: float, y: slangpy.math.float1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.float2, y: float) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: float, y: slangpy.math.float2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.float3, y: float) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: float, y: slangpy.math.float3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.float4, y: float) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: float, y: slangpy.math.float4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.uint1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.uint2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.uint3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.uint4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.uint4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int1, y: int) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.int1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int2, y: int) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.int2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int3, y: int) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.int3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.int4, y: int) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: int, y: slangpy.math.int4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool1, y: bool) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: bool, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool2, y: bool) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: bool, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool3, y: bool) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: bool, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: slangpy.math.bool4, y: bool) -> slangpy.math.bool4
    :no-index:

.. py:function:: slangpy.math.ge(x: bool, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.min(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.min(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.uint1
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.uint2
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.uint4
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.int1
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.int2
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.int4
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.min(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.max(x: slangpy.math.float1, y: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.max(x: slangpy.math.float2, y: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.float4, y: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.uint1, y: slangpy.math.uint1) -> slangpy.math.uint1
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.uint2, y: slangpy.math.uint2) -> slangpy.math.uint2
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.uint4, y: slangpy.math.uint4) -> slangpy.math.uint4
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.int1, y: slangpy.math.int1) -> slangpy.math.int1
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.int2, y: slangpy.math.int2) -> slangpy.math.int2
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.int4, y: slangpy.math.int4) -> slangpy.math.int4
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.bool1, y: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.bool2, y: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.bool3, y: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.max(x: slangpy.math.bool4, y: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.clamp(x: slangpy.math.float1, min: slangpy.math.float1, max: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.clamp(x: slangpy.math.float2, min: slangpy.math.float2, max: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.float3, min: slangpy.math.float3, max: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.float4, min: slangpy.math.float4, max: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.uint1, min: slangpy.math.uint1, max: slangpy.math.uint1) -> slangpy.math.uint1
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.uint2, min: slangpy.math.uint2, max: slangpy.math.uint2) -> slangpy.math.uint2
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.uint3, min: slangpy.math.uint3, max: slangpy.math.uint3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.uint4, min: slangpy.math.uint4, max: slangpy.math.uint4) -> slangpy.math.uint4
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.int1, min: slangpy.math.int1, max: slangpy.math.int1) -> slangpy.math.int1
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.int2, min: slangpy.math.int2, max: slangpy.math.int2) -> slangpy.math.int2
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.int3, min: slangpy.math.int3, max: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.int4, min: slangpy.math.int4, max: slangpy.math.int4) -> slangpy.math.int4
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.bool1, min: slangpy.math.bool1, max: slangpy.math.bool1) -> slangpy.math.bool1
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.bool2, min: slangpy.math.bool2, max: slangpy.math.bool2) -> slangpy.math.bool2
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.bool3, min: slangpy.math.bool3, max: slangpy.math.bool3) -> slangpy.math.bool3
    :no-index:

.. py:function:: slangpy.math.clamp(x: slangpy.math.bool4, min: slangpy.math.bool4, max: slangpy.math.bool4) -> slangpy.math.bool4
    :no-index:



----

.. py:function:: slangpy.math.abs(x: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.abs(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.abs(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.abs(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.abs(x: slangpy.math.int1) -> slangpy.math.int1
    :no-index:

.. py:function:: slangpy.math.abs(x: slangpy.math.int2) -> slangpy.math.int2
    :no-index:

.. py:function:: slangpy.math.abs(x: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.abs(x: slangpy.math.int4) -> slangpy.math.int4
    :no-index:



----

.. py:function:: slangpy.math.sign(x: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.sign(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.sign(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.sign(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.sign(x: slangpy.math.int1) -> slangpy.math.int1
    :no-index:

.. py:function:: slangpy.math.sign(x: slangpy.math.int2) -> slangpy.math.int2
    :no-index:

.. py:function:: slangpy.math.sign(x: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.sign(x: slangpy.math.int4) -> slangpy.math.int4
    :no-index:



----

.. py:function:: slangpy.math.dot(x: slangpy.math.float1, y: slangpy.math.float1) -> float

.. py:function:: slangpy.math.dot(x: slangpy.math.float2, y: slangpy.math.float2) -> float
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.float3, y: slangpy.math.float3) -> float
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.float4, y: slangpy.math.float4) -> float
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.uint1, y: slangpy.math.uint1) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.uint2, y: slangpy.math.uint2) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.uint3, y: slangpy.math.uint3) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.uint4, y: slangpy.math.uint4) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.int1, y: slangpy.math.int1) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.int2, y: slangpy.math.int2) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.int3, y: slangpy.math.int3) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.int4, y: slangpy.math.int4) -> int
    :no-index:

.. py:function:: slangpy.math.dot(x: slangpy.math.quatf, y: slangpy.math.quatf) -> float
    :no-index:



----

.. py:function:: slangpy.math.length(x: slangpy.math.float1) -> float

.. py:function:: slangpy.math.length(x: slangpy.math.float2) -> float
    :no-index:

.. py:function:: slangpy.math.length(x: slangpy.math.float3) -> float
    :no-index:

.. py:function:: slangpy.math.length(x: slangpy.math.float4) -> float
    :no-index:

.. py:function:: slangpy.math.length(x: slangpy.math.quatf) -> float
    :no-index:



----

.. py:function:: slangpy.math.normalize(x: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.normalize(x: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.normalize(x: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.normalize(x: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.normalize(x: slangpy.math.quatf) -> slangpy.math.quatf
    :no-index:



----

.. py:function:: slangpy.math.reflect(i: slangpy.math.float1, n: slangpy.math.float1) -> slangpy.math.float1

.. py:function:: slangpy.math.reflect(i: slangpy.math.float2, n: slangpy.math.float2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.reflect(i: slangpy.math.float3, n: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.reflect(i: slangpy.math.float4, n: slangpy.math.float4) -> slangpy.math.float4
    :no-index:



----

.. py:function:: slangpy.math.cross(x: slangpy.math.float3, y: slangpy.math.float3) -> slangpy.math.float3

.. py:function:: slangpy.math.cross(x: slangpy.math.uint3, y: slangpy.math.uint3) -> slangpy.math.uint3
    :no-index:

.. py:function:: slangpy.math.cross(x: slangpy.math.int3, y: slangpy.math.int3) -> slangpy.math.int3
    :no-index:

.. py:function:: slangpy.math.cross(x: slangpy.math.quatf, y: slangpy.math.quatf) -> slangpy.math.quatf
    :no-index:



----

.. py:function:: slangpy.math.any(x: slangpy.math.bool1) -> bool

.. py:function:: slangpy.math.any(x: slangpy.math.bool2) -> bool
    :no-index:

.. py:function:: slangpy.math.any(x: slangpy.math.bool3) -> bool
    :no-index:

.. py:function:: slangpy.math.any(x: slangpy.math.bool4) -> bool
    :no-index:



----

.. py:function:: slangpy.math.all(x: slangpy.math.bool1) -> bool

.. py:function:: slangpy.math.all(x: slangpy.math.bool2) -> bool
    :no-index:

.. py:function:: slangpy.math.all(x: slangpy.math.bool3) -> bool
    :no-index:

.. py:function:: slangpy.math.all(x: slangpy.math.bool4) -> bool
    :no-index:



----

.. py:function:: slangpy.math.none(x: slangpy.math.bool1) -> bool

.. py:function:: slangpy.math.none(x: slangpy.math.bool2) -> bool
    :no-index:

.. py:function:: slangpy.math.none(x: slangpy.math.bool3) -> bool
    :no-index:

.. py:function:: slangpy.math.none(x: slangpy.math.bool4) -> bool
    :no-index:



----

.. py:function:: slangpy.math.transpose(x: slangpy.math.float2x2) -> slangpy.math.float2x2

.. py:function:: slangpy.math.transpose(x: slangpy.math.float2x3) -> slangpy.math.float3x2
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float2x4) -> slangpy.math.float4x2
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float3x2) -> slangpy.math.float2x3
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float3x3) -> slangpy.math.float3x3
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float3x4) -> slangpy.math.float4x3
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float4x2) -> slangpy.math.float2x4
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float4x3) -> slangpy.math.float3x4
    :no-index:

.. py:function:: slangpy.math.transpose(x: slangpy.math.float4x4) -> slangpy.math.float4x4
    :no-index:



----

.. py:function:: slangpy.math.determinant(x: slangpy.math.float2x2) -> float

.. py:function:: slangpy.math.determinant(x: slangpy.math.float3x3) -> float
    :no-index:

.. py:function:: slangpy.math.determinant(x: slangpy.math.float4x4) -> float
    :no-index:



----

.. py:function:: slangpy.math.inverse(x: slangpy.math.float2x2) -> slangpy.math.float2x2

.. py:function:: slangpy.math.inverse(x: slangpy.math.float3x3) -> slangpy.math.float3x3
    :no-index:

.. py:function:: slangpy.math.inverse(x: slangpy.math.float4x4) -> slangpy.math.float4x4
    :no-index:

.. py:function:: slangpy.math.inverse(x: slangpy.math.quatf) -> slangpy.math.quatf
    :no-index:



----

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x2, y: slangpy.math.float2) -> slangpy.math.float2

.. py:function:: slangpy.math.mul(x: slangpy.math.float2, y: slangpy.math.float2x2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x2, y: slangpy.math.float2x2) -> slangpy.math.float2x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x2, y: slangpy.math.float2x3) -> slangpy.math.float2x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x2, y: slangpy.math.float2x4) -> slangpy.math.float2x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x3, y: slangpy.math.float3) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2, y: slangpy.math.float2x3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x3, y: slangpy.math.float3x2) -> slangpy.math.float2x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x3, y: slangpy.math.float3x3) -> slangpy.math.float2x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x3, y: slangpy.math.float3x4) -> slangpy.math.float2x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x4, y: slangpy.math.float4) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2, y: slangpy.math.float2x4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x4, y: slangpy.math.float4x2) -> slangpy.math.float2x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x4, y: slangpy.math.float4x3) -> slangpy.math.float2x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float2x4, y: slangpy.math.float4x4) -> slangpy.math.float2x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x2, y: slangpy.math.float2) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3, y: slangpy.math.float3x2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x2, y: slangpy.math.float2x2) -> slangpy.math.float3x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x2, y: slangpy.math.float2x3) -> slangpy.math.float3x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x2, y: slangpy.math.float2x4) -> slangpy.math.float3x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x3, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3, y: slangpy.math.float3x3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x3, y: slangpy.math.float3x2) -> slangpy.math.float3x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x3, y: slangpy.math.float3x3) -> slangpy.math.float3x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x3, y: slangpy.math.float3x4) -> slangpy.math.float3x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x4, y: slangpy.math.float4) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3, y: slangpy.math.float3x4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x4, y: slangpy.math.float4x2) -> slangpy.math.float3x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x4, y: slangpy.math.float4x3) -> slangpy.math.float3x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float3x4, y: slangpy.math.float4x4) -> slangpy.math.float3x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x2, y: slangpy.math.float2) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4, y: slangpy.math.float4x2) -> slangpy.math.float2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x2, y: slangpy.math.float2x2) -> slangpy.math.float4x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x2, y: slangpy.math.float2x3) -> slangpy.math.float4x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x2, y: slangpy.math.float2x4) -> slangpy.math.float4x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x3, y: slangpy.math.float3) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4, y: slangpy.math.float4x3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x3, y: slangpy.math.float3x2) -> slangpy.math.float4x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x3, y: slangpy.math.float3x3) -> slangpy.math.float4x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x3, y: slangpy.math.float3x4) -> slangpy.math.float4x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x4, y: slangpy.math.float4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4, y: slangpy.math.float4x4) -> slangpy.math.float4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x4, y: slangpy.math.float4x2) -> slangpy.math.float4x2
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x4, y: slangpy.math.float4x3) -> slangpy.math.float4x3
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.float4x4, y: slangpy.math.float4x4) -> slangpy.math.float4x4
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.quatf, y: slangpy.math.quatf) -> slangpy.math.quatf
    :no-index:

.. py:function:: slangpy.math.mul(x: slangpy.math.quatf, y: slangpy.math.float3) -> slangpy.math.float3
    :no-index:



----

.. py:function:: slangpy.math.transform_point(m: slangpy.math.float4x4, v: slangpy.math.float3) -> slangpy.math.float3



----

.. py:function:: slangpy.math.transform_vector(m: slangpy.math.float3x3, v: slangpy.math.float3) -> slangpy.math.float3

.. py:function:: slangpy.math.transform_vector(m: slangpy.math.float4x4, v: slangpy.math.float3) -> slangpy.math.float3
    :no-index:

.. py:function:: slangpy.math.transform_vector(q: slangpy.math.quatf, v: slangpy.math.float3) -> slangpy.math.float3
    :no-index:



----

.. py:function:: slangpy.math.translate(m: slangpy.math.float4x4, v: slangpy.math.float3) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.translate_2d(m: slangpy.math.float3x3, v: slangpy.math.float2) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.rotate(m: slangpy.math.float4x4, angle: float, axis: slangpy.math.float3) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.rotate_2d(m: slangpy.math.float3x3, angle: float) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.scale(m: slangpy.math.float4x4, v: slangpy.math.float3) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.scale_2d(m: slangpy.math.float3x3, v: slangpy.math.float2) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.perspective(fovy: float, aspect: float, z_near: float, z_far: float) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.ortho(left: float, right: float, bottom: float, top: float, z_near: float, z_far: float) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_translation(v: slangpy.math.float3) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_translation_2d(v: slangpy.math.float2) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.matrix_from_scaling(v: slangpy.math.float3) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_scaling_2d(v: slangpy.math.float2) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.matrix_from_rotation(angle: float, axis: slangpy.math.float3) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_rotation_2d(angle: float) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.matrix_from_rotation_x(angle: float) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_rotation_y(angle: float) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_rotation_z(angle: float) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_rotation_xyz(angle_x: float, angle_y: float, angle_z: float) -> slangpy.math.float4x4

.. py:function:: slangpy.math.matrix_from_rotation_xyz(angles: slangpy.math.float3) -> slangpy.math.float4x4
    :no-index:



----

.. py:function:: slangpy.math.matrix_from_look_at(eye: slangpy.math.float3, center: slangpy.math.float3, up: slangpy.math.float3, handedness: slangpy.math.Handedness = Handedness.right_handed) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.matrix_from_quat(q: slangpy.math.quatf) -> slangpy.math.float3x3



----

.. py:function:: slangpy.math.matrix_4x4_from_3x4(m: slangpy.math.float3x4) -> slangpy.math.float4x4



----

.. py:function:: slangpy.math.decompose(model_matrix: slangpy.math.float4x4, scale: slangpy.math.float3, orientation: slangpy.math.quatf, translation: slangpy.math.float3, skew: slangpy.math.float3, perspective: slangpy.math.float4) -> bool



----

.. py:function:: slangpy.math.conjugate(x: slangpy.math.quatf) -> slangpy.math.quatf



----

.. py:function:: slangpy.math.slerp(x: slangpy.math.quatf, y: slangpy.math.quatf, s: float) -> slangpy.math.quatf



----

.. py:function:: slangpy.math.pitch(x: slangpy.math.quatf) -> float



----

.. py:function:: slangpy.math.yaw(x: slangpy.math.quatf) -> float



----

.. py:function:: slangpy.math.roll(x: slangpy.math.quatf) -> float



----

.. py:function:: slangpy.math.euler_angles(x: slangpy.math.quatf) -> slangpy.math.float3



----

.. py:function:: slangpy.math.quat_from_angle_axis(angle: float, axis: slangpy.math.float3) -> slangpy.math.quatf



----

.. py:function:: slangpy.math.quat_from_rotation_between_vectors(from_: slangpy.math.float3, to: slangpy.math.float3) -> slangpy.math.quatf



----

.. py:function:: slangpy.math.quat_from_euler_angles(angles: slangpy.math.float3) -> slangpy.math.quatf



----

.. py:function:: slangpy.math.quat_from_matrix(m: slangpy.math.float3x3) -> slangpy.math.quatf



----

.. py:function:: slangpy.math.quat_from_look_at(dir: slangpy.math.float3, up: slangpy.math.float3, handedness: slangpy.math.Handedness = Handedness.right_handed) -> slangpy.math.quatf



----

UI
--

.. py:function:: slangpy.ui.render_profiler_window(profiler: slangpy.Profiler | None = None) -> None

    Render real-time frame-aligned profiler statistics from cached
    snapshots.

    Parameter ``profiler``:
        Profiler to display, or nullptr to use the current profiler.



----

.. py:class:: slangpy.ui.Context

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, device: slangpy.Device) -> None

    .. py:method:: begin_frame(self, width: int, height: int, window: slangpy.Window | None = None) -> None

        Begin a new ImGui frame and renders the main screen widget. ImGui
        widget calls are generally only valid between `begin_frame` and
        `end_frame`.

        Parameter ``width``:
            Render texture width

        Parameter ``height``:
            Render texture height

        Parameter ``window``:
            Window this UI context is rendered for (optional).

    .. py:method:: end_frame(self, texture_view: slangpy.TextureView, command_encoder: slangpy.CommandEncoder) -> None

        End the ImGui frame and renders the UI to the provided texture.

        Parameter ``texture_view``:
            Texture view to render to

        Parameter ``command_encoder``:
            Command encoder to encode commands to

    .. py:method:: end_frame(self, texture: slangpy.Texture, command_encoder: slangpy.CommandEncoder) -> None
        :no-index:

        End the ImGui frame and renders the UI to the provided texture.

        Parameter ``texture``:
            Texture to render to

        Parameter ``command_encoder``:
            Command encoder to encode commands to

    .. py:method:: render_draw_data(self, draw_data: object, texture_view: slangpy.TextureView, command_encoder: slangpy.CommandEncoder) -> None

    .. py:method:: render_draw_data(self, draw_data: object, texture: slangpy.Texture, command_encoder: slangpy.CommandEncoder) -> None
        :no-index:

    .. py:method:: handle_keyboard_event(self, event: slangpy.KeyboardEvent) -> bool

        Pass a keyboard event to the UI context.

        Parameter ``event``:
            Keyboard event

        Returns:
            Returns true if event was consumed.

    .. py:method:: handle_mouse_event(self, event: slangpy.MouseEvent) -> bool

        Pass a mouse event to the UI context.

        Parameter ``event``:
            Mouse event

        Returns:
            Returns true if event was consumed.

    .. py:property:: screen
        :type: slangpy.ui.Screen

        The main screen widget.



----

.. py:class:: slangpy.ui.Widget

    Base class: :py:class:`slangpy.Object`



    .. py:property:: parent
        :type: slangpy.ui.Widget

    .. py:property:: children
        :type: list[slangpy.ui.Widget]

    .. py:property:: visible
        :type: bool

    .. py:property:: enabled
        :type: bool

    .. py:method:: child_index(self, child: slangpy.ui.Widget) -> int

    .. py:method:: add_child(self, child: slangpy.ui.Widget) -> None

    .. py:method:: add_child_at(self, child: slangpy.ui.Widget, index: int) -> None

    .. py:method:: remove_child(self, child: slangpy.ui.Widget) -> None

    .. py:method:: remove_child_at(self, index: int) -> None

    .. py:method:: remove_all_children(self) -> None



----

.. py:class:: slangpy.ui.Screen

    Base class: :py:class:`slangpy.ui.Widget`





----

.. py:class:: slangpy.ui.Window

    Base class: :py:class:`slangpy.ui.Widget`



    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, title: str = '', position: slangpy.math.float2 = {10, 10}, size: slangpy.math.float2 = {400, 400}) -> None

    .. py:method:: show(self) -> None

    .. py:method:: close(self) -> None

    .. py:property:: title
        :type: str

    .. py:property:: position
        :type: slangpy.math.float2

    .. py:property:: size
        :type: slangpy.math.float2



----

.. py:class:: slangpy.ui.Group

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '') -> None

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.ui.Text

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, text: str = '') -> None

    .. py:property:: text
        :type: str



----

.. py:class:: slangpy.ui.ProgressBar

    Base class: :py:class:`slangpy.ui.Widget`



    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, fraction: float = 0.0) -> None

    .. py:property:: fraction
        :type: float



----

.. py:class:: slangpy.ui.Button

    Base class: :py:class:`slangpy.ui.Widget`



    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', callback: collections.abc.Callable[[], None] | None = None) -> None

    .. py:property:: label
        :type: str

    .. py:property:: callback
        :type: collections.abc.Callable[[], None]



----

.. py:class:: slangpy.ui.ValuePropertyBool

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: bool

    .. py:property:: callback
        :type: collections.abc.Callable[[bool], None]



----

.. py:class:: slangpy.ui.ValuePropertyInt

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: int

    .. py:property:: callback
        :type: collections.abc.Callable[[int], None]



----

.. py:class:: slangpy.ui.ValuePropertyInt2

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: slangpy.math.int2

    .. py:property:: callback
        :type: collections.abc.Callable[[slangpy.math.int2], None]



----

.. py:class:: slangpy.ui.ValuePropertyInt3

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: slangpy.math.int3

    .. py:property:: callback
        :type: collections.abc.Callable[[slangpy.math.int3], None]



----

.. py:class:: slangpy.ui.ValuePropertyInt4

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: slangpy.math.int4

    .. py:property:: callback
        :type: collections.abc.Callable[[slangpy.math.int4], None]



----

.. py:class:: slangpy.ui.ValuePropertyFloat

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: float

    .. py:property:: callback
        :type: collections.abc.Callable[[float], None]



----

.. py:class:: slangpy.ui.ValuePropertyFloat2

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: slangpy.math.float2

    .. py:property:: callback
        :type: collections.abc.Callable[[slangpy.math.float2], None]



----

.. py:class:: slangpy.ui.ValuePropertyFloat3

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: slangpy.math.float3

    .. py:property:: callback
        :type: collections.abc.Callable[[slangpy.math.float3], None]



----

.. py:class:: slangpy.ui.ValuePropertyFloat4

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: slangpy.math.float4

    .. py:property:: callback
        :type: collections.abc.Callable[[slangpy.math.float4], None]



----

.. py:class:: slangpy.ui.ValuePropertyString

    Base class: :py:class:`slangpy.ui.Widget`

    .. py:property:: label
        :type: str

    .. py:property:: value
        :type: str

    .. py:property:: callback
        :type: collections.abc.Callable[[str], None]



----

.. py:class:: slangpy.ui.CheckBox

    Base class: :py:class:`slangpy.ui.ValuePropertyBool`



    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: bool = False, callback: collections.abc.Callable[[bool], None] | None = None) -> None



----

.. py:class:: slangpy.ui.ComboBox

    Base class: :py:class:`slangpy.ui.ValuePropertyInt`



    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: int = 0, callback: collections.abc.Callable[[int], None] | None = None, items: collections.abc.Sequence[str] = []) -> None

    .. py:property:: items
        :type: list[str]



----

.. py:class:: slangpy.ui.ListBox

    Base class: :py:class:`slangpy.ui.ValuePropertyInt`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: int = 0, callback: collections.abc.Callable[[int], None] | None = None, items: collections.abc.Sequence[str] = [], height_in_items: int = -1) -> None

    .. py:property:: items
        :type: list[str]

    .. py:property:: height_in_items
        :type: int



----

.. py:class:: slangpy.ui.SliderFlags

    Base class: :py:class:`enum.IntFlag`





----

.. py:class:: slangpy.ui.DragFloat

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: float = 0.0, callback: collections.abc.Callable[[float], None] | None = None, speed: float = 1.0, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: float

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragFloat2

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat2`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float2 = {0, 0}, callback: collections.abc.Callable[[slangpy.math.float2], None] | None = None, speed: float = 1.0, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: float

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragFloat3

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat3`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float3 = {0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.float3], None] | None = None, speed: float = 1.0, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: float

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragFloat4

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat4`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float4 = {0, 0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.float4], None] | None = None, speed: float = 1.0, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: float

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragInt

    Base class: :py:class:`slangpy.ui.ValuePropertyInt`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: int = 0, callback: collections.abc.Callable[[int], None] | None = None, speed: float = 1.0, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: int

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragInt2

    Base class: :py:class:`slangpy.ui.ValuePropertyInt2`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int2 = {0, 0}, callback: collections.abc.Callable[[slangpy.math.int2], None] | None = None, speed: float = 1.0, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: int

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragInt3

    Base class: :py:class:`slangpy.ui.ValuePropertyInt3`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int3 = {0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.int3], None] | None = None, speed: float = 1.0, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: int

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.DragInt4

    Base class: :py:class:`slangpy.ui.ValuePropertyInt4`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int4 = {0, 0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.int4], None] | None = None, speed: float = 1.0, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: speed
        :type: int

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderFloat

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: float = 0.0, callback: collections.abc.Callable[[float], None] | None = None, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderFloat2

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat2`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float2 = {0, 0}, callback: collections.abc.Callable[[slangpy.math.float2], None] | None = None, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderFloat3

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat3`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float3 = {0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.float3], None] | None = None, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderFloat4

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat4`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float4 = {0, 0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.float4], None] | None = None, min: float = 0.0, max: float = 0.0, format: str = '%.3f', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: float

    .. py:property:: max
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderInt

    Base class: :py:class:`slangpy.ui.ValuePropertyInt`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: int = 0, callback: collections.abc.Callable[[int], None] | None = None, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderInt2

    Base class: :py:class:`slangpy.ui.ValuePropertyInt2`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int2 = {0, 0}, callback: collections.abc.Callable[[slangpy.math.int2], None] | None = None, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderInt3

    Base class: :py:class:`slangpy.ui.ValuePropertyInt3`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int3 = {0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.int3], None] | None = None, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.SliderInt4

    Base class: :py:class:`slangpy.ui.ValuePropertyInt4`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int4 = {0, 0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.int4], None] | None = None, min: int = 0, max: int = 0, format: str = '%d', flags: slangpy.ui.SliderFlags = 0) -> None

    .. py:property:: min
        :type: int

    .. py:property:: max
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.SliderFlags



----

.. py:class:: slangpy.ui.InputTextFlags

    Base class: :py:class:`enum.IntFlag`





----

.. py:class:: slangpy.ui.InputFloat

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: float = 0.0, callback: collections.abc.Callable[[float], None] | None = None, step: float = 1.0, step_fast: float = 100.0, format: str = '%.3f', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: float

    .. py:property:: step_fast
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputFloat2

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat2`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float2 = {0, 0}, callback: collections.abc.Callable[[slangpy.math.float2], None] | None = None, step: float = 1.0, step_fast: float = 100.0, format: str = '%.3f', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: float

    .. py:property:: step_fast
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputFloat3

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat3`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float3 = {0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.float3], None] | None = None, step: float = 1.0, step_fast: float = 100.0, format: str = '%.3f', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: float

    .. py:property:: step_fast
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputFloat4

    Base class: :py:class:`slangpy.ui.ValuePropertyFloat4`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.float4 = {0, 0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.float4], None] | None = None, step: float = 1.0, step_fast: float = 100.0, format: str = '%.3f', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: float

    .. py:property:: step_fast
        :type: float

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputInt

    Base class: :py:class:`slangpy.ui.ValuePropertyInt`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: int = 0, callback: collections.abc.Callable[[int], None] | None = None, step: int = 1, step_fast: int = 100, format: str = '%d', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: int

    .. py:property:: step_fast
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputInt2

    Base class: :py:class:`slangpy.ui.ValuePropertyInt2`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int2 = {0, 0}, callback: collections.abc.Callable[[slangpy.math.int2], None] | None = None, step: int = 1, step_fast: int = 100, format: str = '%d', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: int

    .. py:property:: step_fast
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputInt3

    Base class: :py:class:`slangpy.ui.ValuePropertyInt3`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int3 = {0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.int3], None] | None = None, step: int = 1, step_fast: int = 100, format: str = '%d', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: int

    .. py:property:: step_fast
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputInt4

    Base class: :py:class:`slangpy.ui.ValuePropertyInt4`

    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: slangpy.math.int4 = {0, 0, 0, 0}, callback: collections.abc.Callable[[slangpy.math.int4], None] | None = None, step: int = 1, step_fast: int = 100, format: str = '%d', flags: slangpy.ui.InputTextFlags = 0) -> None

    .. py:property:: step
        :type: int

    .. py:property:: step_fast
        :type: int

    .. py:property:: format
        :type: str

    .. py:property:: flags
        :type: slangpy.ui.InputTextFlags



----

.. py:class:: slangpy.ui.InputText

    Base class: :py:class:`slangpy.ui.ValuePropertyString`



    .. py:method:: __init__(self, parent: slangpy.ui.Widget | None, label: str = '', value: str = False, callback: collections.abc.Callable[[str], None] | None = None, multi_line: bool = False, flags: slangpy.ui.InputTextFlags = 0) -> None



----

Utilities
---------

.. py:class:: slangpy.TextureLoader

    Base class: :py:class:`slangpy.Object`



    .. py:method:: __init__(self, device: slangpy.Device) -> None

    .. py:class:: slangpy.TextureLoader.Options



        .. py:method:: __init__(self) -> None

        .. py:method:: __init__(self, arg: dict, /) -> None
            :no-index:

        .. py:property:: load_as_normalized
            :type: bool

            Load 8/16-bit integer data as normalized resource format.

        .. py:property:: load_as_srgb
            :type: bool

            Use ``Format::rgba8_unorm_srgb`` format if bitmap is 8-bit RGBA with
            sRGB gamma.

        .. py:property:: extend_alpha
            :type: bool

            Extend RGB to RGBA if the RGB texture format cannot support the
            requested usage.

        .. py:property:: allocate_mips
            :type: bool

            Allocate mip levels for the texture.

        .. py:property:: generate_mips
            :type: bool

            Generate mip levels for the texture.

        .. py:property:: usage
            :type: slangpy.TextureUsage

        .. py:property:: ya_handling
            :type: slangpy.YAHandling

    .. py:method:: load_texture(self, bitmap: slangpy.Bitmap, options: slangpy.TextureLoader.Options | None = None) -> slangpy.Texture

        Load a texture from a bitmap.

        Parameter ``bitmap``:
            Bitmap to load.

        Parameter ``options``:
            Texture loading options.

        Returns:
            New texture object.

    .. py:method:: load_texture(self, path: str | os.PathLike, options: slangpy.TextureLoader.Options | None = None) -> slangpy.Texture
        :no-index:

        Load a texture from an image stream.

        The stream must be readable and seekable. All image formats supported
        by the file-path overload, including DDS, are supported.

        Parameter ``stream``:
            Image data stream.

        Parameter ``options``:
            Texture loading options.

        Returns:
            New texture object.

    .. py:method:: load_textures(self, bitmaps: collections.abc.Sequence[slangpy.Bitmap], options: slangpy.TextureLoader.Options | None = None) -> list[slangpy.Texture]

        Load textures from a list of bitmaps.

        Parameter ``bitmaps``:
            Bitmaps to load.

        Parameter ``options``:
            Texture loading options.

        Returns:
            List of new of texture objects.

    .. py:method:: load_textures(self, bitmaps: collections.abc.Sequence[slangpy.Bitmap], options: collections.abc.Sequence[slangpy.TextureLoader.Options]) -> list[slangpy.Texture]
        :no-index:

        Load textures from a list of bitmaps.

        Parameter ``bitmaps``:
            Bitmaps to load.

        Parameter ``options``:
            Texture loading options.

        Returns:
            List of new of texture objects.

    .. py:method:: load_textures(self, paths: collections.abc.Sequence[str | os.PathLike], options: slangpy.TextureLoader.Options | None = None) -> list[slangpy.Texture]
        :no-index:

        Load textures from a list of image files.

        Parameter ``paths``:
            Image file paths.

        Parameter ``options``:
            Texture loading options.

        Returns:
            List of new texture objects.

    .. py:method:: load_textures(self, paths: collections.abc.Sequence[str | os.PathLike], options: collections.abc.Sequence[slangpy.TextureLoader.Options]) -> list[slangpy.Texture]
        :no-index:

        Load textures from a list of image files.

        Parameter ``paths``:
            Image file paths.

        Parameter ``options``:
            Texture loading options.

        Returns:
            List of new texture objects.

    .. py:method:: load_texture_array(self, bitmaps: collections.abc.Sequence[slangpy.Bitmap], options: slangpy.TextureLoader.Options | None = None) -> slangpy.Texture

        Load a texture array from a list of bitmaps.

        All bitmaps need to have the same format and dimensions.

        Parameter ``bitmaps``:
            Bitmaps to load.

        Parameter ``options``:
            Texture loading options.

        Returns:
            New texture array object.

    .. py:method:: load_texture_array(self, paths: collections.abc.Sequence[str | os.PathLike], options: slangpy.TextureLoader.Options | None = None) -> slangpy.Texture
        :no-index:

        Load a texture array from a list of image files.

        All images need to have the same format and dimensions.

        Parameter ``paths``:
            Image file paths.

        Parameter ``options``:
            Texture loading options.

        Returns:
            New texture array object.



----

.. py:function:: slangpy.tev.show(bitmap: slangpy.Bitmap, name: str = '', host: str = '127.0.0.1', port: int = 14158, max_retries: int = 3) -> bool

    Show a bitmap in the tev viewer (https://github.com/Tom94/tev).

    This will block until the image is sent over.

    Parameter ``bitmap``:
        Bitmap to show.

    Parameter ``name``:
        Name of the image in tev. If not specified, a unique name will be
        generated.

    Parameter ``host``:
        Host to connect to.

    Parameter ``port``:
        Port to connect to.

    Parameter ``max_retries``:
        Maximum number of retries.

    Returns:
        True if successful.

.. py:function:: slangpy.tev.show(texture: slangpy.Texture, name: str = '', host: str = '127.0.0.1', port: int = 14158, max_retries: int = 3) -> bool
    :no-index:

    Show texture in the tev viewer (https://github.com/Tom94/tev).

    This will block until the image is sent over.

    Parameter ``texture``:
        Texture to show.

    Parameter ``name``:
        Name of the image in tev. If not specified, a unique name will be
        generated.

    Parameter ``host``:
        Host to connect to.

    Parameter ``port``:
        Port to connect to.

    Parameter ``max_retries``:
        Maximum number of retries.

    Returns:
        True if successful.



----

.. py:function:: slangpy.tev.show_async(bitmap: slangpy.Bitmap, name: str = '', host: str = '127.0.0.1', port: int = 14158, max_retries: int = 3) -> None

    Show a bitmap in the tev viewer (https://github.com/Tom94/tev).

    This will return immediately and send the image asynchronously in the
    background.

    Parameter ``bitmap``:
        Bitmap to show.

    Parameter ``name``:
        Name of the image in tev. If not specified, a unique name will be
        generated.

    Parameter ``host``:
        Host to connect to.

    Parameter ``port``:
        Port to connect to.

    Parameter ``max_retries``:
        Maximum number of retries.

.. py:function:: slangpy.tev.show_async(texture: slangpy.Texture, name: str = '', host: str = '127.0.0.1', port: int = 14158, max_retries: int = 3) -> None
    :no-index:

    Show a texture in the tev viewer (https://github.com/Tom94/tev).

    This will return immediately and send the image asynchronously in the
    background.

    Parameter ``bitmap``:
        Texture to show.

    Parameter ``name``:
        Name of the image in tev. If not specified, a unique name will be
        generated.

    Parameter ``host``:
        Host to connect to.

    Parameter ``port``:
        Port to connect to.

    Parameter ``max_retries``:
        Maximum number of retries.



----

.. py:function:: slangpy.renderdoc.is_available() -> bool

    Check if RenderDoc is available.

    This is typically the case when the application is running under the
    RenderDoc.

    Returns:
        True if RenderDoc is available.



----

.. py:function:: slangpy.renderdoc.start_frame_capture(device: slangpy.Device, window: slangpy.Window | None = None) -> bool

    Start capturing a frame in RenderDoc.

    This function will start capturing a frame (or some partial
    compute/graphics workload) in RenderDoc.

    To end the frame capture, call ``end_frame_capture()``.

    Parameter ``device``:
        The device to capture the frame for.

    Parameter ``window``:
        The window to capture the frame for (optional).

    Returns:
        True if the frame capture was started successfully.



----

.. py:function:: slangpy.renderdoc.end_frame_capture() -> bool

    End capturing a frame in RenderDoc.

    This function will end capturing a frame (or some partial
    compute/graphics workload) in RenderDoc.

    Returns:
        True if the frame capture was ended successfully.



----

.. py:function:: slangpy.renderdoc.is_frame_capturing() -> bool

    Check if a frame is currently being captured in RenderDoc.

    Returns:
        True if a frame is currently being captured.



----

SlangPy
-------

.. py:class:: slangpy.slangpy.AccessType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.slangpy.CallMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.slangpy.AutogradAccess

    Base class: :py:class:`enum.Enum`



----

.. py:function:: slangpy.slangpy.unpack_args(*args) -> tuple

    N/A



----

.. py:function:: slangpy.slangpy.unpack_kwargs(**kwargs) -> tuple

    N/A



----

.. py:function:: slangpy.slangpy.unpack_arg(arg: object) -> object

    N/A



----

.. py:function:: slangpy.slangpy.pack_arg(arg: object, unpacked_arg: object) -> None

    N/A



----

.. py:function:: slangpy.slangpy.get_value_signature(o: object) -> str

    N/A



----

.. py:class:: slangpy.slangpy.SignatureBuilder

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:method:: add(self, value: str) -> None

        N/A

    .. py:property:: str
        :type: str

        N/A

    .. py:property:: bytes
        :type: bytes

        N/A



----

.. py:class:: slangpy.slangpy.NativeObject

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:property:: slangpy_signature
        :type: str

    .. py:method:: read_signature(self, builder: slangpy.slangpy.SignatureBuilder) -> None

        N/A



----

.. py:class:: slangpy.slangpy.NativeMarshall

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:property:: concrete_shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: match_call_shape
        :type: bool

        N/A

    .. py:method:: get_shape(self, value: object) -> slangpy.slangpy.Shape

        N/A

    .. py:property:: slang_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:method:: write_shader_cursor_pre_dispatch(self, context: slangpy.slangpy.CallContext, binding: slangpy.slangpy.NativeBoundVariableRuntime, cursor: slangpy.ShaderCursor, value: object, read_back: list) -> None

        N/A

    .. py:method:: create_calldata(self, arg0: slangpy.slangpy.CallContext, arg1: slangpy.slangpy.NativeBoundVariableRuntime, arg2: object, /) -> object

        N/A

    .. py:method:: read_calldata(self, arg0: slangpy.slangpy.CallContext, arg1: slangpy.slangpy.NativeBoundVariableRuntime, arg2: object, arg3: object, /) -> None

        N/A

    .. py:method:: create_output(self, arg0: slangpy.slangpy.CallContext, arg1: slangpy.slangpy.NativeBoundVariableRuntime, /) -> object

        N/A

    .. py:method:: read_output(self, arg0: slangpy.slangpy.CallContext, arg1: slangpy.slangpy.NativeBoundVariableRuntime, arg2: object, /) -> object

        N/A

    .. py:property:: has_derivative
        :type: bool

        N/A

    .. py:property:: is_writable
        :type: bool

        N/A

    .. py:method:: gen_calldata(self, cgb: object, context: object, binding: object) -> None

        N/A

    .. py:method:: reduce_type(self, context: object, dimensions: int) -> slangpy.native_refl.Type

        N/A

    .. py:method:: resolve_type(self, context: object, bound_type: slangpy.native_refl.Type) -> slangpy.native_refl.Type

        N/A

    .. py:method:: resolve_types(self, context: object, bound_type: slangpy.native_refl.Type) -> list[slangpy.native_refl.Type]

        N/A

    .. py:method:: resolve_dimensionality(self, context: object, binding: object, vector_target_type: slangpy.native_refl.Type) -> int

        N/A

    .. py:method:: build_shader_object(self, context: object, data: object) -> slangpy.ShaderObject

        N/A



----

.. py:class:: slangpy.slangpy.NativeBoundVariableRuntime

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:property:: access
        :type: tuple[slangpy.slangpy.AccessType, slangpy.slangpy.AccessType]

        N/A

    .. py:property:: transform
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: python_type
        :type: slangpy.slangpy.NativeMarshall

        N/A

    .. py:property:: vector_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: is_param_block
        :type: bool

        N/A

    .. py:property:: variable_name
        :type: str

        N/A

    .. py:property:: children
        :type: dict[str, slangpy.slangpy.NativeBoundVariableRuntime] | None

        N/A

    .. py:method:: populate_call_shape(self, arg0: slangpy.slangpy.Shape, arg1: object, arg2: slangpy.slangpy.NativeCallData, /) -> None

        N/A

    .. py:method:: read_call_data_post_dispatch(self, arg0: slangpy.slangpy.CallContext, arg1: dict, arg2: object, /) -> None

        N/A

    .. py:method:: write_raw_dispatch_data(self, arg0: dict, arg1: object, /) -> None

        N/A

    .. py:method:: read_output(self, arg0: slangpy.slangpy.CallContext, arg1: object, /) -> object

        N/A

    .. py:property:: direct_bind
        :type: bool

        N/A



----

.. py:class:: slangpy.slangpy.NativeBoundCallRuntime

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:property:: args
        :type: list[slangpy.slangpy.NativeBoundVariableRuntime]

        N/A

    .. py:property:: kwargs
        :type: dict[str, slangpy.slangpy.NativeBoundVariableRuntime]

        N/A

    .. py:method:: find_kwarg(self, arg: str, /) -> slangpy.slangpy.NativeBoundVariableRuntime

        N/A

    .. py:method:: calculate_call_shape(self, arg0: int, arg1: object, arg2: dict, arg3: slangpy.slangpy.NativeCallData, /) -> slangpy.slangpy.Shape

        N/A

    .. py:method:: read_call_data_post_dispatch(self, arg0: slangpy.slangpy.CallContext, arg1: dict, arg2: list, arg3: dict, /) -> None

        N/A

    .. py:method:: write_raw_dispatch_data(self, arg0: dict, arg1: dict, /) -> None

        N/A



----

.. py:class:: slangpy.slangpy.NativeCallRuntimeOptions

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:property:: uniforms
        :type: list

        N/A

    .. py:property:: cuda_stream
        :type: slangpy.NativeHandle

        N/A

    .. py:property:: thread_count
        :type: int

        N/A



----

.. py:class:: slangpy.slangpy.NativeCallData

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:property:: device
        :type: slangpy.Device

        N/A

    .. py:property:: pipeline
        :type: slangpy.Pipeline

        N/A

    .. py:property:: shader_table
        :type: slangpy.ShaderTable

        N/A

    .. py:property:: call_dimensionality
        :type: int

        N/A

    .. py:property:: runtime
        :type: slangpy.slangpy.NativeBoundCallRuntime

        N/A

    .. py:property:: call_mode
        :type: slangpy.slangpy.CallMode

        N/A

    .. py:property:: last_call_shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: debug_name
        :type: str

        N/A

    .. py:property:: logger
        :type: slangpy.Logger

        N/A

    .. py:method:: call(self, opts: slangpy.slangpy.NativeCallRuntimeOptions, *args, **kwargs) -> object

        N/A

    .. py:method:: append_to(self, opts: slangpy.slangpy.NativeCallRuntimeOptions, command_buffer: slangpy.CommandEncoder, *args, **kwargs) -> object

        N/A

    .. py:property:: call_group_shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: torch_integration
        :type: bool

        N/A

    .. py:property:: torch_autograd
        :type: bool

        N/A

    .. py:property:: needs_unpack
        :type: bool

        N/A

    .. py:property:: has_thread_count
        :type: bool

        N/A

    .. py:property:: use_entrypoint_args
        :type: bool

        N/A

    .. py:property:: autograd_access_list
        :type: list[slangpy.slangpy.AutogradAccess]

        N/A

    .. py:property:: bwds_call_data
        :type: slangpy.slangpy.NativeCallData

        N/A

    .. py:method:: find_torch_tensors(self, args: list, kwargs: dict) -> list

        N/A

    .. py:method:: autograd_forward(self, opts: slangpy.slangpy.NativeCallRuntimeOptions, args: list, kwargs: dict, pairs: list) -> tuple

        N/A

    .. py:method:: autograd_backward(self, function_node: object, pairs: list, args: list, kwargs: dict, saved_tensors: list, grad_outputs: tuple) -> tuple

        N/A

    .. py:method:: log(self, level: slangpy.LogLevel, msg: str, frequency: slangpy.LogFrequency = LogFrequency.always) -> None

        Log a message.

        Parameter ``level``:
            The log level.

        Parameter ``msg``:
            The message.

        Parameter ``frequency``:
            The log frequency.

    .. py:method:: log_debug(self, msg: str) -> None

    .. py:method:: log_info(self, msg: str) -> None

    .. py:method:: log_warn(self, msg: str) -> None

    .. py:method:: log_error(self, msg: str) -> None

    .. py:method:: log_fatal(self, msg: str) -> None



----

.. py:class:: slangpy.slangpy.NativeCallDataCache

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self) -> None

        N/A

    .. py:method:: get_value_signature(self, builder: slangpy.slangpy.SignatureBuilder, o: object) -> None

        N/A

    .. py:method:: get_args_signature(self, builder: slangpy.slangpy.SignatureBuilder, *args, **kwargs) -> None

        N/A

    .. py:method:: find_call_data(self, signature: str) -> slangpy.slangpy.NativeCallData

        N/A

    .. py:method:: add_call_data(self, signature: str, call_data: slangpy.slangpy.NativeCallData) -> None

        N/A

    .. py:method:: lookup_value_signature(self, o: object) -> str | None

        N/A



----

.. py:class:: slangpy.slangpy.Shape

    .. py:method:: __init__(self, shape: collections.abc.Sequence[int]) -> None

        N/A

    .. py:method:: __init__(self, *args) -> None
        :no-index:

    .. py:property:: valid
        :type: bool

        N/A

    .. py:property:: concrete
        :type: bool

        N/A

    .. py:method:: as_tuple(self) -> tuple

        N/A

    .. py:method:: as_list(self) -> list

        N/A

    .. py:method:: calc_contiguous_strides(self) -> slangpy.slangpy.Shape

        N/A



----

.. py:class:: slangpy.slangpy.CallContext

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self, device: slangpy.Device, call_mode: slangpy.slangpy.CallMode) -> None

        N/A

    .. py:property:: device
        :type: slangpy.Device

        N/A

    .. py:property:: call_shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: call_mode
        :type: slangpy.slangpy.CallMode

        N/A



----

.. py:class:: slangpy.slangpy.FunctionNodeType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.slangpy.NativeFunctionNode

    Base class: :py:class:`slangpy.slangpy.NativeObject`

    .. py:method:: __init__(self, parent: slangpy.slangpy.NativeFunctionNode | None, type: slangpy.slangpy.FunctionNodeType, data: object | None) -> None

        N/A

    .. py:method:: generate_call_data(self, *args, **kwargs) -> slangpy.slangpy.NativeCallData

        N/A

    .. py:method:: generate_bwds_call_data(self, fwds_call_data: slangpy.slangpy.NativeCallData, *args, **kwargs) -> slangpy.slangpy.NativeCallData

        N/A

    .. py:method:: read_signature(self, builder: slangpy.slangpy.SignatureBuilder) -> None

        N/A

    .. py:method:: gather_runtime_options(self, options: slangpy.slangpy.NativeCallRuntimeOptions) -> None

        N/A



----

.. py:class:: slangpy.slangpy.NativePackedArg

    Base class: :py:class:`slangpy.slangpy.NativeObject`

    .. py:method:: __init__(self, python: slangpy.slangpy.NativeMarshall, shader_object: slangpy.ShaderObject, python_object: object) -> None

        N/A

    .. py:property:: python
        :type: slangpy.slangpy.NativeMarshall

        N/A

    .. py:property:: shader_object
        :type: slangpy.ShaderObject

        N/A

    .. py:property:: python_object
        :type: object

        N/A



----

.. py:function:: slangpy.slangpy.get_texture_shape(texture: slangpy.Texture, mip: int = 0) -> slangpy.slangpy.Shape

    N/A



----

.. py:class:: slangpy.slangpy.NativeBufferMarshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`

    .. py:method:: __init__(self, slang_type: slangpy.native_refl.Type, usage: slangpy.BufferUsage) -> None

        N/A

    .. py:method:: write_shader_cursor_pre_dispatch(self, context: slangpy.slangpy.CallContext, binding: slangpy.slangpy.NativeBoundVariableRuntime, cursor: slangpy.ShaderCursor, value: object, read_back: list) -> None

        N/A

    .. py:method:: get_shape(self, value: object) -> slangpy.slangpy.Shape

        N/A

    .. py:property:: usage
        :type: slangpy.BufferUsage

    .. py:property:: slang_type
        :type: slangpy.native_refl.Type



----

.. py:class:: slangpy.slangpy.NativeDescriptorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`

    .. py:method:: __init__(self, slang_type: slangpy.native_refl.Type, type: slangpy.DescriptorHandleType) -> None

        N/A

    .. py:method:: write_shader_cursor_pre_dispatch(self, context: slangpy.slangpy.CallContext, binding: slangpy.slangpy.NativeBoundVariableRuntime, cursor: slangpy.ShaderCursor, value: object, read_back: list) -> None

        N/A

    .. py:method:: get_shape(self, value: object) -> slangpy.slangpy.Shape

        N/A

    .. py:property:: type
        :type: slangpy.DescriptorHandleType

    .. py:property:: slang_type
        :type: slangpy.native_refl.Type



----

.. py:class:: slangpy.slangpy.NativeTextureMarshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`

    .. py:method:: __init__(self, slang_type: slangpy.native_refl.Type, element_type: slangpy.native_refl.Type, resource_shape: slangpy.TypeReflection.ResourceShape, format: slangpy.Format, usage: slangpy.TextureUsage, dims: int) -> None

        N/A

    .. py:method:: write_shader_cursor_pre_dispatch(self, context: slangpy.slangpy.CallContext, binding: slangpy.slangpy.NativeBoundVariableRuntime, cursor: slangpy.ShaderCursor, value: object, read_back: list) -> None

        N/A

    .. py:method:: get_shape(self, value: object) -> slangpy.slangpy.Shape

        N/A

    .. py:method:: get_texture_shape(self, texture: slangpy.Texture, mip: int) -> slangpy.slangpy.Shape

        N/A

    .. py:property:: resource_shape
        :type: slangpy.TypeReflection.ResourceShape

        N/A

    .. py:property:: usage
        :type: slangpy.TextureUsage

        N/A

    .. py:property:: texture_dims
        :type: int

        N/A

    .. py:property:: slang_element_type
        :type: slangpy.native_refl.Type

        N/A



----

.. py:class:: slangpy.slangpy.TensorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`

    .. py:method:: __init__(self, dims: int, writable: bool, slang_type: slangpy.native_refl.Type, slang_element_type: slangpy.native_refl.Type, element_layout: slangpy.TypeLayoutReflection, d_in: slangpy.slangpy.TensorMarshall | None, d_out: slangpy.slangpy.TensorMarshall | None) -> None

        N/A

    .. py:property:: dims
        :type: int

    .. py:property:: writable
        :type: bool

    .. py:property:: slang_element_type
        :type: slangpy.native_refl.Type

    .. py:property:: d_in
        :type: slangpy.slangpy.TensorMarshall

    .. py:property:: d_out
        :type: slangpy.slangpy.TensorMarshall



----

.. py:class:: slangpy.slangpy.NativeNumpyMarshall

    Base class: :py:class:`slangpy.slangpy.TensorMarshall`

    .. py:method:: __init__(self, dims: int, slang_type: slangpy.native_refl.Type, slang_element_type: slangpy.native_refl.Type, element_layout: slangpy.TypeLayoutReflection, numpydtype: object) -> None

        N/A

    .. py:property:: dtype
        :type: dlpack::dtype



----

.. py:class:: slangpy.slangpy.NativeTorchTensorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`

    .. py:method:: __init__(self, dims: int, writable: bool, slang_type: slangpy.native_refl.Type, slang_element_type: slangpy.native_refl.Type, element_layout: slangpy.TypeLayoutReflection, d_in: slangpy.slangpy.NativeTorchTensorMarshall | None, d_out: slangpy.slangpy.NativeTorchTensorMarshall | None) -> None

    .. py:property:: dims
        :type: int

    .. py:property:: writable
        :type: bool

    .. py:property:: slang_element_type
        :type: slangpy.native_refl.Type

    .. py:property:: element_layout
        :type: slangpy.TypeLayoutReflection

    .. py:property:: has_derivative
        :type: bool

    .. py:property:: d_in
        :type: slangpy.slangpy.NativeTorchTensorMarshall

    .. py:property:: d_out
        :type: slangpy.slangpy.NativeTorchTensorMarshall



----

.. py:class:: slangpy.slangpy.NativeTorchTensorDiffPair

    Base class: :py:class:`slangpy.slangpy.NativeObject`

    .. py:method:: __init__(self, primal: object | None, grad: object | None, index: int = -1, is_input: bool = True) -> None

        Create a diff pair from primal and gradient tensors.

        :param primal: The primal (value) tensor. May be None for output gradients.
        :param grad: The gradient tensor.
        :param index: Index into saved tensors list for reconnecting in backward pass.
        :param is_input: True if this is an input tensor, false for output.

    .. py:property:: primal
        :type: object

        The primal (value) tensor.

    .. py:property:: grad
        :type: object

        The gradient tensor.

    .. py:property:: index
        :type: int

        Index into saved tensors list.

    .. py:property:: is_input
        :type: bool

        True if input tensor, false if output.

    .. py:method:: clear_tensors(self) -> None

        Clear tensor references (set both primal and grad to None).



----

.. py:class:: slangpy.slangpy.NativeValueMarshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`

    .. py:method:: __init__(self) -> None

        N/A



----

Miscellaneous
-------------

.. py:class:: slangpy.FilterBoundaryCondition

    Base class: :py:class:`enum.Enum`

    Filter boundary condition used for resampling images.



----

.. py:class:: slangpy.BoxFilter



    .. py:method:: __init__(self) -> None

    .. py:method:: eval(self, x: float) -> float

    .. py:property:: radius
        :type: float



----

.. py:class:: slangpy.TentFilter



    .. py:method:: __init__(self, radius: float = 1.0) -> None

    .. py:method:: eval(self, x: float) -> float

    .. py:property:: radius
        :type: float



----

.. py:class:: slangpy.GaussianFilter



    .. py:method:: __init__(self, stddev: float = 0.5) -> None

    .. py:method:: eval(self, x: float) -> float

    .. py:property:: radius
        :type: float



----

.. py:class:: slangpy.MitchellFilter



    .. py:method:: __init__(self, b: float = 0.3333333432674408, c: float = 0.3333333432674408) -> None

    .. py:method:: eval(self, x: float) -> float

    .. py:property:: radius
        :type: float



----

.. py:class:: slangpy.LanczosFilter



    .. py:method:: __init__(self, lobes: int = 3) -> None

    .. py:method:: eval(self, x: float) -> float

    .. py:property:: radius
        :type: float



----

.. py:class:: slangpy.DescriptorHandleType

    Base class: :py:class:`enum.IntEnum`



----

.. py:class:: slangpy.CpuTimestampDomain

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.TimestampCalibration



    .. py:method:: __init__(self) -> None

    .. py:property:: cpu_domain
        :type: slangpy.CpuTimestampDomain

        The domain of the CPU timestamp.

    .. py:property:: cpu_timestamp
        :type: int

        The current CPU timestamp.

    .. py:property:: cpu_frequency
        :type: int

        The frequency of the CPU timestamp in ticks per second.

    .. py:property:: gpu_timestamp
        :type: int

        The current GPU timestamp.

    .. py:property:: gpu_frequency
        :type: int

        The frequency of the GPU timestamp in ticks per second.

    .. py:property:: max_deviation_ns
        :type: int

        The maximum deviation between the CPU and GPU timestamps in
        nanoseconds.



----

.. py:class:: slangpy.PipelineCompilationPolicy

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.Attribute

    Base class: :py:class:`slangpy.BaseReflectionObject`



    .. py:property:: name
        :type: str

    .. py:property:: argument_count
        :type: int

    .. py:method:: argument_type(self, index: int) -> slangpy.TypeReflection

    .. py:method:: argument_value_int(self, index: int) -> int

    .. py:method:: argument_value_float(self, index: int) -> float

    .. py:method:: argument_value_string(self, index: int) -> str



----

.. py:class:: slangpy.SpecializationArgKind

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.SpecializationArg

    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, kind: slangpy.SpecializationArgKind, value: str) -> None
        :no-index:

    .. py:staticmethod:: from_type(type_name: str) -> slangpy.SpecializationArg

    .. py:staticmethod:: from_expr(expr: str) -> slangpy.SpecializationArg

    .. py:property:: kind
        :type: slangpy.SpecializationArgKind

    .. py:property:: value
        :type: str



----

.. py:class:: slangpy.MicromapType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.OpacityMicromapFormat

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.OpacityMicromapSpecialIndex

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.MicromapIndexingMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.MicromapIndexFormat

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.MicromapBuildFlags

    Base class: :py:class:`enum.IntFlag`



----

.. py:class:: slangpy.MicromapUsageCount



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: count
        :type: int

    .. py:property:: subdivision_level
        :type: int

    .. py:property:: format
        :type: slangpy.OpacityMicromapFormat



----

.. py:class:: slangpy.MicromapBuildDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: type
        :type: slangpy.MicromapType

    .. py:property:: flags
        :type: slangpy.MicromapBuildFlags

    .. py:property:: data_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: descriptor_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: descriptor_stride
        :type: int

    .. py:property:: histogram
        :type: list[slangpy.MicromapUsageCount]



----

.. py:class:: slangpy.MicromapSizes



    .. py:property:: micromap_size
        :type: int

    .. py:property:: scratch_size
        :type: int



----

.. py:class:: slangpy.MicromapDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: type
        :type: slangpy.MicromapType

    .. py:property:: size
        :type: int

    .. py:property:: flags
        :type: slangpy.MicromapBuildFlags

    .. py:property:: label
        :type: str



----

.. py:class:: slangpy.Micromap

    Base class: :py:class:`slangpy.Resource`



    .. py:property:: desc
        :type: slangpy.MicromapDesc

    .. py:property:: device_address
        :type: int



----

.. py:class:: slangpy.AccelerationStructureOpacityMicromapDesc



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: micromap
        :type: slangpy.Micromap

    .. py:property:: indexing_mode
        :type: slangpy.MicromapIndexingMode

    .. py:property:: index_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: index_format
        :type: slangpy.MicromapIndexFormat

    .. py:property:: index_stride
        :type: int

    .. py:property:: base_micromap_index
        :type: int

    .. py:property:: usage_counts
        :type: list[slangpy.MicromapUsageCount]



----

.. py:class:: slangpy.AccelerationStructureBuildInputSpheres



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: vertex_count
        :type: int

    .. py:property:: vertex_position_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: vertex_position_format
        :type: slangpy.Format

    .. py:property:: vertex_position_stride
        :type: int

    .. py:property:: vertex_radius_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: vertex_radius_format
        :type: slangpy.Format

    .. py:property:: vertex_radius_stride
        :type: int

    .. py:property:: index_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: index_format
        :type: slangpy.IndexFormat

    .. py:property:: index_count
        :type: int

    .. py:property:: flags
        :type: slangpy.AccelerationStructureGeometryFlags



----

.. py:class:: slangpy.LinearSweptSpheresIndexingMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.LinearSweptSpheresEndCapsMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.AccelerationStructureBuildInputLinearSweptSpheres



    .. py:method:: __init__(self) -> None

    .. py:method:: __init__(self, arg: dict, /) -> None
        :no-index:

    .. py:property:: vertex_count
        :type: int

    .. py:property:: primitive_count
        :type: int

    .. py:property:: vertex_position_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: vertex_position_format
        :type: slangpy.Format

    .. py:property:: vertex_position_stride
        :type: int

    .. py:property:: vertex_radius_buffers
        :type: list[slangpy.BufferOffsetPair]

    .. py:property:: vertex_radius_format
        :type: slangpy.Format

    .. py:property:: vertex_radius_stride
        :type: int

    .. py:property:: index_buffer
        :type: slangpy.BufferOffsetPair

    .. py:property:: index_format
        :type: slangpy.IndexFormat

    .. py:property:: index_count
        :type: int

    .. py:property:: indexing_mode
        :type: slangpy.LinearSweptSpheresIndexingMode

    .. py:property:: end_caps_mode
        :type: slangpy.LinearSweptSpheresEndCapsMode

    .. py:property:: flags
        :type: slangpy.AccelerationStructureGeometryFlags



----

.. py:class:: slangpy.BindlessDesc



    .. py:method:: __init__(self, buffer_count: int | None = None, texture_count: int | None = None, sampler_count: int | None = None, acceleration_structure_count: int | None = None) -> None

    .. py:property:: buffer_count
        :type: int

    .. py:property:: texture_count
        :type: int

    .. py:property:: sampler_count
        :type: int

    .. py:property:: acceleration_structure_count
        :type: int



----

.. py:class:: slangpy.PipelineCompilationMode

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.HeapReport

    Report information for a memory heap.

    .. py:property:: label
        :type: str

        Name of the heap.

    .. py:property:: num_pages
        :type: int

        Number of pages in the heap.

    .. py:property:: total_allocated
        :type: int

        Total allocated memory in bytes.

    .. py:property:: total_mem_usage
        :type: int

        Total memory usage in bytes.

    .. py:property:: num_allocations
        :type: int

        Number of allocations.



----

.. py:class:: slangpy.CudaContextScope

    Context manager for temporarily switching CUDA contexts.

    Do not instantiate directly; use device.cuda_context_scope() instead.



----

.. py:class:: slangpy.CommandRecordingSubmittedEvent

    Event data for command recording submission callback.

    .. py:property:: device
        :type: slangpy.Device

    .. py:property:: id
        :type: int

    .. py:property:: command_buffer
        :type: slangpy.CommandBuffer

    .. py:property:: submit_id
        :type: int



----

.. py:class:: slangpy.CommandRecordingDiscardedEvent

    Event data for command recording discarded callback.

    .. py:property:: device
        :type: slangpy.Device

    .. py:property:: id
        :type: int



----

.. py:function:: slangpy.get_cuda_current_context_native_handles() -> list[slangpy.NativeHandle]

    Gets the device and context handles for the current CUDA context. Use
    to retrieve an existing context (eg from PyTorch) to pass as the
    existing_device_handles from which to create a device in the
    DeviceDesc.



----

.. py:function:: slangpy.push_current_device(device: slangpy.Device) -> None

    Push a device onto the thread-local current device stack.

    Stores a raw pointer in the thread-local stack. The caller must ensure
    that the device (and any devices below it on the stack) outlive their
    corresponding pop_current_device() calls. Device::close() only auto-
    pops if the device is on top of the stack; a closed device lower in
    the stack remains as a stale pointer - calling current_device() or
    pop_current_device() on that entry is undefined behavior unless the
    device is kept alive (refcount > 0) until popped.

    Parameter ``device``:
        Device to push (must not be null).



----

.. py:function:: slangpy.pop_current_device() -> slangpy.Device

    Pop the top device from the thread-local device stack. Throws if the
    stack is empty.

    Returns:
        The popped device.



----

.. py:function:: slangpy.current_device() -> slangpy.Device

    Get the current device from the top of the thread-local device stack.
    Throws if the stack is empty.

    Returns:
        The current device.



----

.. py:function:: slangpy.create_surface(window: slangpy.Window) -> slangpy.Surface

    Create a new surface.

    Parameter ``window``:
        Window to create the surface for.

    Returns:
        New surface object.

.. py:function:: slangpy.create_surface(window_handle: slangpy.WindowHandle) -> slangpy.Surface
    :no-index:

    Create a new surface.

    Parameter ``window_handle``:
        Native window handle to create the surface for.

    Returns:
        New surface object.



----

.. py:function:: slangpy.create_buffer(size: int = 0, element_count: int = 0, struct_size: int = 0, resource_type_layout: object | None = None, format: slangpy.Format = Format.undefined, memory_type: slangpy.MemoryType = MemoryType.device_local, usage: slangpy.BufferUsage = 0, default_state: slangpy.ResourceState = ResourceState.undefined, label: str = '', data: numpy.ndarray[] | None = None) -> slangpy.Buffer

    Create a new buffer.

    Parameter ``size``:
        Buffer size in bytes.

    Parameter ``element_count``:
        Buffer size in number of struct elements. Can be used instead of
        ``size``.

    Parameter ``struct_size``:
        Struct size in bytes.

    Parameter ``resource_type_layout``:
        Resource type layout of the buffer. Can be used instead of
        ``struct_size`` to specify the size of the struct.

    Parameter ``format``:
        Buffer format. Used when creating typed buffer views.

    Parameter ``initial_state``:
        Initial resource state.

    Parameter ``usage``:
        Resource usage flags.

    Parameter ``memory_type``:
        Memory type.

    Parameter ``label``:
        Debug label.

    Parameter ``data``:
        Initial data to upload to the buffer.

    Parameter ``data_size``:
        Size of the initial data in bytes.

    Returns:
        New buffer object.

.. py:function:: slangpy.create_buffer(desc: slangpy.BufferDesc) -> slangpy.Buffer
    :no-index:



----

.. py:function:: slangpy.create_texture(type: slangpy.TextureType = TextureType.texture_2d, format: slangpy.Format = Format.undefined, width: int = 1, height: int = 1, depth: int = 1, array_length: int = 1, mip_count: int = 1, sample_count: int = 1, sample_quality: int = 0, memory_type: slangpy.MemoryType = MemoryType.device_local, usage: slangpy.TextureUsage = 0, default_state: slangpy.ResourceState = ResourceState.undefined, sampler: slangpy.Sampler | None = None, label: str = '', data: numpy.ndarray[] | None = None) -> slangpy.Texture

    Create a new texture.

    Parameter ``type``:
        Texture type.

    Parameter ``format``:
        Texture format.

    Parameter ``width``:
        Width in pixels.

    Parameter ``height``:
        Height in pixels.

    Parameter ``depth``:
        Depth in pixels.

    Parameter ``array_length``:
        Array length.

    Parameter ``mip_count``:
        Mip level count. Number of mip levels (ALL_MIPS for all mip
        levels).

    Parameter ``sample_count``:
        Number of samples for multisampled textures.

    Parameter ``quality``:
        Quality level for multisampled textures.

    Parameter ``usage``:
        Resource usage.

    Parameter ``memory_type``:
        Memory type.

    Parameter ``label``:
        Debug label.

    Parameter ``data``:
        Initial data.

    Returns:
        New texture object.

.. py:function:: slangpy.create_texture(desc: slangpy.TextureDesc) -> slangpy.Texture
    :no-index:



----

.. py:function:: slangpy.create_sampler(min_filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear, mag_filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear, mip_filter: slangpy.TextureFilteringMode = TextureFilteringMode.linear, reduction_op: slangpy.TextureReductionOp = TextureReductionOp.average, address_u: slangpy.TextureAddressingMode = TextureAddressingMode.wrap, address_v: slangpy.TextureAddressingMode = TextureAddressingMode.wrap, address_w: slangpy.TextureAddressingMode = TextureAddressingMode.wrap, mip_lod_bias: float = 0.0, max_anisotropy: int = 1, comparison_func: slangpy.ComparisonFunc = ComparisonFunc.never, border_color: slangpy.math.float4 = {1, 1, 1, 1}, min_lod: float = -1000.0, max_lod: float = 1000.0, label: str = '') -> slangpy.Sampler

    Create a new sampler.

    Parameter ``min_filter``:
        Minification filter.

    Parameter ``mag_filter``:
        Magnification filter.

    Parameter ``mip_filter``:
        Mip-map filter.

    Parameter ``reduction_op``:
        Reduction operation.

    Parameter ``address_u``:
        Texture addressing mode for the U coordinate.

    Parameter ``address_v``:
        Texture addressing mode for the V coordinate.

    Parameter ``address_w``:
        Texture addressing mode for the W coordinate.

    Parameter ``mip_lod_bias``:
        Mip-map LOD bias.

    Parameter ``max_anisotropy``:
        Maximum anisotropy.

    Parameter ``comparison_func``:
        Comparison function.

    Parameter ``border_color``:
        Border color.

    Parameter ``min_lod``:
        Minimum LOD level.

    Parameter ``max_lod``:
        Maximum LOD level.

    Parameter ``label``:
        Debug label.

    Returns:
        New sampler object.

.. py:function:: slangpy.create_sampler(desc: slangpy.SamplerDesc) -> slangpy.Sampler
    :no-index:



----

.. py:function:: slangpy.create_fence(initial_value: int = 0, shared: bool = False) -> slangpy.Fence

    Create a new fence.

    Parameter ``initial_value``:
        Initial fence value.

    Parameter ``shared``:
        Create a shared fence.

    Returns:
        New fence object.

.. py:function:: slangpy.create_fence(desc: slangpy.FenceDesc) -> slangpy.Fence
    :no-index:



----

.. py:function:: slangpy.create_query_pool(type: slangpy.QueryType, count: int) -> slangpy.QueryPool

    Create a new query pool.

    Parameter ``type``:
        Query type.

    Parameter ``count``:
        Number of queries in the pool.

    Returns:
        New query pool object.

.. py:function:: slangpy.create_query_pool(desc: slangpy.QueryPoolDesc) -> slangpy.QueryPool
    :no-index:



----

.. py:function:: slangpy.create_input_layout(input_elements: collections.abc.Sequence[slangpy.InputElementDesc], vertex_streams: collections.abc.Sequence[slangpy.VertexStreamDesc]) -> slangpy.InputLayout

    Create a new input layout.

    Parameter ``input_elements``:
        List of input elements (see InputElementDesc for details).

    Parameter ``vertex_streams``:
        List of vertex streams (see VertexStreamDesc for details).

    Returns:
        New input layout object.

.. py:function:: slangpy.create_input_layout(desc: slangpy.InputLayoutDesc) -> slangpy.InputLayout
    :no-index:



----

.. py:function:: slangpy.create_command_encoder(queue: slangpy.CommandQueueType = CommandQueueType.graphics) -> slangpy.CommandEncoder

    Create a command encoder.



----

.. py:function:: slangpy.get_acceleration_structure_sizes(desc: slangpy.AccelerationStructureBuildDesc) -> slangpy.AccelerationStructureSizes

    Query the device for buffer sizes required for acceleration structure
    builds.

    Parameter ``desc``:
        Acceleration structure build description.

    Returns:
        Acceleration structure sizes.



----

.. py:function:: slangpy.create_acceleration_structure(kind: slangpy.AccelerationStructureKind = AccelerationStructureKind.unknown, size: int = 0, label: str = '') -> slangpy.AccelerationStructure

    Create a new acceleration structure.

.. py:function:: slangpy.create_acceleration_structure(desc: slangpy.AccelerationStructureDesc) -> slangpy.AccelerationStructure
    :no-index:



----

.. py:function:: slangpy.create_acceleration_structure_instance_list(size: int) -> slangpy.AccelerationStructureInstanceList

    Create a new acceleration structure instance list.



----

.. py:function:: slangpy.get_micromap_sizes(desc: slangpy.MicromapBuildDesc) -> slangpy.MicromapSizes

    Query the current device for buffer sizes required for a micromap
    build.



----

.. py:function:: slangpy.create_micromap(type: slangpy.MicromapType = MicromapType.opacity, size: int = 0, flags: slangpy.MicromapBuildFlags = 0, label: str = '') -> slangpy.Micromap

    Create a new micromap on the current device.

.. py:function:: slangpy.create_micromap(desc: slangpy.MicromapDesc) -> slangpy.Micromap
    :no-index:



----

.. py:function:: slangpy.create_shader_table(program: slangpy.ShaderProgram, ray_gen_entry_points: collections.abc.Sequence[str] = [], miss_entry_points: collections.abc.Sequence[str] = [], hit_group_names: collections.abc.Sequence[str] = [], callable_entry_points: collections.abc.Sequence[str] = []) -> slangpy.ShaderTable

    Create a new shader table.

.. py:function:: slangpy.create_shader_table(desc: slangpy.ShaderTableDesc) -> slangpy.ShaderTable
    :no-index:



----

.. py:function:: slangpy.get_coop_vec_matrix_size(rows: int, cols: int, layout: slangpy.CoopVecMatrixLayout, element_type: slangpy.DataType, row_col_stride: int = 0) -> int

    Get the size of a cooperative vector matrix in bytes.



----

.. py:function:: slangpy.create_coop_vec_matrix_desc(rows: int, cols: int, layout: slangpy.CoopVecMatrixLayout, element_type: slangpy.DataType, offset: int = 0, row_col_stride: int = 0) -> slangpy.CoopVecMatrixDesc

    Create a cooperative vector matrix descriptor.



----

.. py:function:: slangpy.convert_coop_vec_matrices(dst: bytearray, dst_descs: collections.abc.Sequence[slangpy.CoopVecMatrixDesc], src: bytes, src_descs: collections.abc.Sequence[slangpy.CoopVecMatrixDesc]) -> None

    Convert multiple cooperative vector matrices between formats.



----

.. py:function:: slangpy.convert_coop_vec_matrix(dst: bytearray, dst_desc: slangpy.CoopVecMatrixDesc, src: bytes, src_desc: slangpy.CoopVecMatrixDesc) -> None

    Convert a single cooperative vector matrix between formats.

.. py:function:: slangpy.convert_coop_vec_matrix(dst: ndarray[device='cpu'], src: ndarray[device='cpu'], dst_layout: slangpy.CoopVecMatrixLayout | None = None, src_layout: slangpy.CoopVecMatrixLayout | None = None) -> None
    :no-index:



----

.. py:function:: slangpy.create_slang_session(compiler_options: slangpy.SlangCompilerOptions | None = None, add_default_include_paths: bool = True, cache_path: str | os.PathLike | None = None) -> slangpy.SlangSession

    Create a new slang session.

    Parameter ``compiler_options``:
        Compiler options (see SlangCompilerOptions for details).

    Returns:
        New slang session object.

.. py:function:: slangpy.create_slang_session(desc: slangpy.SlangSessionDesc) -> slangpy.SlangSession
    :no-index:



----

.. py:function:: slangpy.load_module(module_name: str) -> slangpy.SlangModule

    Load a slang module by name.



----

.. py:function:: slangpy.load_module_from_source(module_name: str, source: str, path: str | os.PathLike | None = None) -> slangpy.SlangModule

    Load a slang module from source code.



----

.. py:function:: slangpy.compose_modules(name: str, modules: collections.abc.Sequence[slangpy.SlangModule], type_conformances: collections.abc.Sequence[slangpy.TypeConformance] = []) -> slangpy.SlangModule

    Compose multiple slang modules into one.



----

.. py:function:: slangpy.link_program(modules: collections.abc.Sequence[slangpy.SlangModule], entry_points: collections.abc.Sequence[slangpy.SlangEntryPoint], link_options: slangpy.SlangLinkOptions | None = None) -> slangpy.ShaderProgram

    Link modules and entry points into a shader program.



----

.. py:function:: slangpy.load_program(module_name: str, entry_point_names: collections.abc.Sequence[str], additional_source: str | None = None, link_options: slangpy.SlangLinkOptions | None = None) -> slangpy.ShaderProgram

    Load a module and link a shader program in one step.



----

.. py:function:: slangpy.create_root_shader_object(shader_program: slangpy.ShaderProgram) -> slangpy.ShaderObject

    Create a root shader object for a shader program.



----

.. py:function:: slangpy.create_shader_object(type_layout: slangpy.TypeLayoutReflection) -> slangpy.ShaderObject

    Create a shader object from a type layout.

.. py:function:: slangpy.create_shader_object(cursor: slangpy.ReflectionCursor) -> slangpy.ShaderObject
    :no-index:

    Create a shader object from a reflection cursor.



----

.. py:function:: slangpy.create_compute_pipeline(program: slangpy.ShaderProgram, compilation_policy: slangpy.PipelineCompilationPolicy = PipelineCompilationPolicy.default, label: str | None = None) -> slangpy.ComputePipeline

    Create a compute pipeline.

.. py:function:: slangpy.create_compute_pipeline(desc: slangpy.ComputePipelineDesc) -> slangpy.ComputePipeline
    :no-index:



----

.. py:function:: slangpy.create_render_pipeline(program: slangpy.ShaderProgram, input_layout: slangpy.InputLayout | None, primitive_topology: slangpy.PrimitiveTopology = PrimitiveTopology.triangle_list, targets: collections.abc.Sequence[slangpy.ColorTargetDesc] = [], depth_stencil: slangpy.DepthStencilDesc | None = None, rasterizer: slangpy.RasterizerDesc | None = None, multisample: slangpy.MultisampleDesc | None = None, compilation_policy: slangpy.PipelineCompilationPolicy = PipelineCompilationPolicy.default, label: str | None = None) -> slangpy.RenderPipeline

    Create a render pipeline.

.. py:function:: slangpy.create_render_pipeline(desc: slangpy.RenderPipelineDesc) -> slangpy.RenderPipeline
    :no-index:



----

.. py:function:: slangpy.create_ray_tracing_pipeline(program: slangpy.ShaderProgram, hit_groups: collections.abc.Sequence[slangpy.HitGroupDesc], max_recursion: int = 0, max_ray_payload_size: int = 0, max_attribute_size: int = 8, flags: slangpy.RayTracingPipelineFlags = 0, compilation_policy: slangpy.PipelineCompilationPolicy = PipelineCompilationPolicy.default, label: str | None = None) -> slangpy.RayTracingPipeline

    Create a ray tracing pipeline.

.. py:function:: slangpy.create_ray_tracing_pipeline(desc: slangpy.RayTracingPipelineDesc) -> slangpy.RayTracingPipeline
    :no-index:



----

.. py:function:: slangpy.create_compute_kernel(program: slangpy.ShaderProgram) -> slangpy.ComputeKernel

    Create a compute kernel.

.. py:function:: slangpy.create_compute_kernel(desc: slangpy.ComputeKernelDesc) -> slangpy.ComputeKernel
    :no-index:



----

.. py:function:: slangpy.crashpad.is_supported() -> bool

    Returns true if Crashpad is supported in this build.



----

.. py:function:: slangpy.crashpad.start_handler(handler: str | os.PathLike = '', database: str | os.PathLike = '', annotations: collections.abc.Mapping[str, str] = {}) -> None

    Starts the Crashpad handler.

    Start the chromium Crashpad handler to capture crashes and generate
    crash reports.

    Parameter ``handler``:
        Path to the handler executable. Defaults to
        `<runtime_directory>/crashpad_handler{.exe}` if empty.

    Parameter ``database``:
        Path to the database directory. Defaults to
        `<runtime_directory>/crashpad_database` if empty.

    Parameter ``annotations``:
        Annotations to include with crash reports.



----

.. py:class:: slangpy.ProfilerTimelineType

    Base class: :py:class:`enum.Enum`

    Type of execution timeline stored in a profiler trace.



----

.. py:class:: slangpy.ProfilerCaptureStopReason

    Base class: :py:class:`enum.Enum`

    Reason why a bounded profiler capture stopped recording.



----

.. py:class:: slangpy.ProfilerFrameGpuStatus

    Base class: :py:class:`enum.Enum`

    Availability of one entry's GPU duration in one retained frame.



----

.. py:class:: slangpy.ProfilerDesc

    Configuration used to construct a Profiler.

    .. py:method:: __init__(self, thread_event_capacity: int = 8192, live_frame_count: int = 120, live_event_capacity: int = 100000, frame_stats_window_size: int = 120, gpu_query_pool_size: int = 16384, enable_auto_zones: bool = True, enable_debug_groups: bool = False) -> None

    .. py:property:: thread_event_capacity
        :type: int

        Maximum number of unread CPU events in each producer thread's queue.
        Must be a positive power of two.

    .. py:property:: live_frame_count
        :type: int

        Maximum number of completed frames retained in live trace snapshots.

    .. py:property:: live_event_capacity
        :type: int

        Maximum number of CPU and GPU zones retained in live trace snapshots.

    .. py:property:: frame_stats_window_size
        :type: int

        Maximum completed and ended GPU-pending frames retained in the global
        frame stream.

    .. py:property:: gpu_query_pool_size
        :type: int

        Number of timestamp queries shared by GPU zones. Each GPU zone uses
        two queries; the count must be positive and even.

    .. py:property:: enable_auto_zones
        :type: bool

        Enable automatic zones around SlangPy functional dispatch recording.

    .. py:property:: enable_debug_groups
        :type: bool

        Mirror command-encoder profiling zones into backend debug groups.



----

.. py:class:: slangpy.ProfilerCaptureDesc

    Configuration for a bounded profiler capture.

    .. py:method:: __init__(self, max_memory_bytes: int = 268435456) -> None

    .. py:property:: max_memory_bytes
        :type: int

        Maximum capture-owned zone and frame storage in bytes.



----

.. py:class:: slangpy.ProfilerSite

    Profiling-site metadata backed by process-lifetime interned strings.
    String views remain valid for the lifetime of the process.

    .. py:property:: id
        :type: int

        One-based site identifier referenced by zones and frames.

    .. py:property:: file
        :type: str

        Source filename supplied when the site was registered.

    .. py:property:: line
        :type: int

        One-based source line number, or zero when unavailable.

    .. py:property:: function
        :type: str

        Compact qualified function name.

    .. py:property:: name
        :type: str

        User-facing zone or frame name.



----

.. py:class:: slangpy.ProfilerTimeline

    Metadata for one CPU thread or GPU queue timeline.

    .. py:property:: type
        :type: slangpy.ProfilerTimelineType

        Execution domain represented by the timeline.

    .. py:property:: name
        :type: str

        Human-readable timeline name.

    .. py:property:: thread_id
        :type: int

        Platform thread identifier for CPU timelines, otherwise zero.

    .. py:property:: device_id
        :type: int

        Device identity for GPU timelines, otherwise zero.

    .. py:property:: queue
        :type: slangpy.CommandQueueType

        Command queue type for GPU timelines.



----

.. py:class:: slangpy.ProfilerFrame

    Completed frame boundary stored in a profiler trace.

    .. py:property:: index
        :type: int

        Monotonically increasing profiler-wide frame index.

    .. py:property:: site_id
        :type: int

        Profiling site that names this frame.

    .. py:property:: timeline_id
        :type: int

        CPU timeline on which the frame was recorded.

    .. py:property:: start_ns
        :type: int

        Frame start time in profiler-clock nanoseconds.

    .. py:property:: duration_ns
        :type: int

        Frame duration in nanoseconds.



----

.. py:class:: slangpy.ProfilerDurationStatistics

    Distribution of duration samples in milliseconds.

    .. py:property:: count
        :type: int

        Number of samples.

    .. py:property:: total_ms
        :type: float

        Sum of sample durations in milliseconds.

    .. py:property:: minimum_ms
        :type: float

        Minimum sample duration in milliseconds.

    .. py:property:: maximum_ms
        :type: float

        Maximum sample duration in milliseconds.

    .. py:property:: mean_ms
        :type: float

        Arithmetic mean duration in milliseconds.

    .. py:property:: standard_deviation_ms
        :type: float

        Population standard deviation in milliseconds.

    .. py:property:: p50_ms
        :type: float

        Linearly interpolated 50th percentile in milliseconds.

    .. py:property:: p90_ms
        :type: float

        Linearly interpolated 90th percentile in milliseconds.

    .. py:property:: p95_ms
        :type: float

        Linearly interpolated 95th percentile in milliseconds.

    .. py:property:: p99_ms
        :type: float

        Linearly interpolated 99th percentile in milliseconds.



----

.. py:class:: slangpy.ProfilerCallStatistics

    Mergeable duration summary that does not retain individual call
    samples.

    .. py:property:: count
        :type: int

        Number of calls.

    .. py:property:: total_ms
        :type: float

        Sum of call durations in milliseconds.

    .. py:property:: minimum_ms
        :type: float

        Minimum call duration in milliseconds.

    .. py:property:: maximum_ms
        :type: float

        Maximum call duration in milliseconds.

    .. py:property:: mean_ms
        :type: float

        Arithmetic mean call duration in milliseconds.

    .. py:property:: standard_deviation_ms
        :type: float

        Population standard deviation in milliseconds.



----

.. py:class:: slangpy.ProfilerDiagnostics

    Diagnostic counters shared by immutable profiler snapshots.

    .. py:property:: producer_drop_count
        :type: int

        Number of producer events dropped because a per-thread event queue or
        zone stack was full.

    .. py:property:: thread_event_queue_high_water_mark
        :type: int

        Maximum number of unread events observed in any per-thread producer
        queue.

    .. py:property:: hierarchy_loss_count
        :type: int

        Number of statistics nodes promoted to roots because their parent
        event was unavailable.

    .. py:property:: gpu_query_exhaustion_count
        :type: int

        Number of GPU zones recorded without GPU timing because no timestamp-
        query pair was available.

    .. py:property:: pending_gpu_zone_count
        :type: int

        Number of allocated GPU zones still awaiting submission or timestamp
        resolution.



----

.. py:class:: slangpy.ProfilerFrameStatsEntry

    Frame-aligned statistics for one hierarchical CPU zone path.

    CPU time per frame is the sum of inclusive occurrences at this path.
    Parent and child entries are therefore not additive. GPU time is the
    sum of directly measured occurrences and can exceed elapsed frame time
    when work overlaps.

    .. py:property:: parent_index
        :type: int

        Index of the parent entry in the same frame statistics list, or -1 for
        a root.

    .. py:property:: site_id
        :type: int

        Profiling site represented by this hierarchical path.

    .. py:property:: name
        :type: str

        User-facing profiling-site name.

    .. py:property:: cpu_time_per_frame
        :type: slangpy.ProfilerDurationStatistics

        Distribution of summed CPU duration per retained frame.

    .. py:property:: gpu_time_per_frame
        :type: slangpy.ProfilerDurationStatistics

        Distribution of summed GPU duration for frames with complete GPU
        measurements.

    .. py:property:: cpu_time_per_call
        :type: slangpy.ProfilerCallStatistics

        Mergeable statistics over individual CPU occurrences.

    .. py:property:: gpu_time_per_call
        :type: slangpy.ProfilerCallStatistics

        Mergeable statistics over individual resolved GPU occurrences.



----

.. py:class:: slangpy.ProfilerFrameStats

    Base class: :py:class:`slangpy.Object`

    Immutable snapshot of rolling statistics for the profiler's global
    frame stream.

    Sample matrices are row-major with shape (sample_count(),
    entry_count()). All columns are ordered oldest to newest.

    .. py:property:: sample_count
        :type: int

        Number of retained completed frames.

    .. py:property:: entry_count
        :type: int

        Number of hierarchical entries.

    .. py:property:: pending_frame_count
        :type: int

        Total ended frames still waiting for GPU measurements.

    .. py:property:: latest_frame_ms
        :type: float

        Duration of the most recently completed frame in milliseconds.

    .. py:property:: frame_time
        :type: slangpy.ProfilerDurationStatistics

        Distribution of completed frame durations.

    .. py:property:: entries
        :type: list[slangpy.ProfilerFrameStatsEntry]

        Hierarchical CPU-zone paths in deterministic parent-before-child
        preorder.

    .. py:property:: sample_frame_index
        :type: numpy.ndarray[dtype=uint32, writable=False]

        Profiler-wide frame index for each retained sample.

    .. py:property:: sample_frame_time_ms
        :type: numpy.ndarray[dtype=float64, writable=False]

        Completed frame duration in milliseconds for each retained sample.

    .. py:property:: sample_call_count
        :type: numpy.ndarray[dtype=uint32, writable=False]

        Row-major occurrence-count matrix with shape (sample_count(),
        entry_count()).

    .. py:property:: sample_cpu_time_ms
        :type: numpy.ndarray[dtype=float64, writable=False]

        Row-major CPU-duration matrix with shape (sample_count(),
        entry_count()).

    .. py:property:: sample_gpu_time_ms
        :type: numpy.ndarray[dtype=float64, writable=False]

        Row-major GPU-duration matrix with shape (sample_count(),
        entry_count()).

    .. py:property:: sample_gpu_status
        :type: numpy.ndarray[dtype=uint8, writable=False]

        Row-major GPU-status matrix with shape (sample_count(),
        entry_count()).

    .. py:property:: diagnostics
        :type: slangpy.ProfilerDiagnostics

        Diagnostic counters captured with this snapshot.



----

.. py:class:: slangpy.ProfilerZoneChunk

    Base class: :py:class:`slangpy.Object`

    Immutable structure-of-arrays storage for at most 4096 zones.

    Bounded chunks let trace queries and exports process large captures
    without flattening all zones into one allocation. The Python bindings
    expose each column as a zero-copy, read-only NumPy array whose
    lifetime is tied to the chunk.

    .. py:property:: count
        :type: int

        Number of zones in this chunk.

    .. py:property:: start_ns
        :type: numpy.ndarray[dtype=uint64, writable=False]

        Zone start times in profiler-clock nanoseconds.

    .. py:property:: duration_ns
        :type: numpy.ndarray[dtype=uint64, writable=False]

        Zone durations in nanoseconds.

    .. py:property:: correlation_id
        :type: numpy.ndarray[dtype=uint64, writable=False]

        CPU/GPU correlation identifiers.

    .. py:property:: timeline_id
        :type: numpy.ndarray[dtype=uint32, writable=False]

        Indices into ProfilerTrace::timelines().

    .. py:property:: site_id
        :type: numpy.ndarray[dtype=uint32, writable=False]

        One-based identifiers into ProfilerTrace::sites().

    .. py:property:: parent_index
        :type: numpy.ndarray[dtype=int32, writable=False]

        Global zone indices of parents on the same timeline, or -1 for roots.

    .. py:property:: frame_index
        :type: numpy.ndarray[dtype=uint32, writable=False]

        Profiler-wide frame indices, or UINT32_MAX for zones outside a frame.



----

.. py:class:: slangpy.ProfilerZoneSelection

    Base class: :py:class:`slangpy.Object`

    Immutable set of global zone indices selected from one ProfilerTrace.

    .. py:property:: count
        :type: int

        Number of selected zones.

    .. py:property:: indices
        :type: numpy.ndarray[dtype=uint32, writable=False]

        Sorted global zone indices into the source trace.

    .. py:method:: statistics(self) -> slangpy.ProfilerDurationStatistics

        Calculate duration statistics for the selected zones.



----

.. py:class:: slangpy.ProfilerTrace

    Base class: :py:class:`slangpy.Object`

    Immutable profiler trace containing metadata, frame boundaries, and
    chunked CPU/GPU zones.

    .. py:property:: start_ns
        :type: int

        Capture or live-history start time in profiler-clock nanoseconds.

    .. py:property:: stop_ns
        :type: int

        Capture or live-history stop time in profiler-clock nanoseconds.

    .. py:property:: stop_reason
        :type: slangpy.ProfilerCaptureStopReason

        Reason why capture recording stopped.

    .. py:property:: truncated
        :type: bool

        Whether a bounded capture stopped at its memory limit.

    .. py:property:: memory_bytes
        :type: int

        Capture-owned zone and frame storage in bytes.

    .. py:property:: diagnostics
        :type: slangpy.ProfilerDiagnostics

        Diagnostic counters captured with this snapshot.

    .. py:property:: sites
        :type: list[slangpy.ProfilerSite]

        Process-global profiling sites visible when the snapshot was built.

    .. py:property:: timelines
        :type: list[slangpy.ProfilerTimeline]

        CPU and GPU timelines referenced by this trace.

    .. py:property:: frames
        :type: list[slangpy.ProfilerFrame]

        Completed frame boundaries retained by this trace.

    .. py:property:: zone_chunks
        :type: list[slangpy.ProfilerZoneChunk]

        Immutable structure-of-arrays zone chunks.

    .. py:property:: zone_count
        :type: int

        Total number of zones across all chunks.

    .. py:method:: query_zones(self, name: str | None = None, timeline_type: slangpy.ProfilerTimelineType | None = None, frame_begin: int | None = None, frame_end: int | None = None, start_ns: int | None = None, end_ns: int | None = None) -> slangpy.ProfilerZoneSelection

        Select zones using exact-name, timeline, frame, and timestamp filters.
        Frame and timestamp ranges are half-open. Zones overlap the requested
        timestamp range rather than requiring their complete duration to be
        contained in it. Zones outside frames are excluded whenever a frame
        bound is set.

    .. py:method:: write_to_json(self, path: str | os.PathLike) -> None

        Stream this trace as Chrome trace-event JSON suitable for Perfetto.



----

.. py:class:: slangpy.ProfilerZoneScope

    N/A



----

.. py:class:: slangpy.ProfilerFrameScope

    N/A



----

.. py:class:: slangpy.Profiler

    Base class: :py:class:`slangpy.Object`

    Bounded CPU/GPU instrumentation profiler with immutable trace and
    frame-statistics snapshots.

    .. py:method:: __init__(self, desc: slangpy.ProfilerDesc = ...) -> None

    .. py:property:: enabled
        :type: bool

        Whether manual and automatic profiling zones are recorded.

    .. py:property:: enable_auto_zones
        :type: bool

        Whether SlangPy functional dispatches create automatic zones.

    .. py:property:: enable_debug_groups
        :type: bool

        Whether command-encoder profiling zones also emit backend debug
        groups.

    .. py:property:: capture_active
        :type: bool

        Whether a bounded capture is currently recording. A capture stopped by
        its memory limit is ready but inactive.

    .. py:property:: desc
        :type: slangpy.ProfilerDesc

        Constructor configuration. Python exposes this as a copy because
        changing it does not reconfigure the profiler.

    .. py:method:: start_capture(self, desc: slangpy.ProfilerCaptureDesc = ...) -> None

        Start a bounded capture after flushing events completed before this
        call.

    .. py:method:: stop_capture(self) -> slangpy.ProfilerTrace

        Stop and return the current or memory-limited capture.

        This performs one tick and flushes completed CPU producer events. It
        does not deliberately wait for submitted GPU work or unresolved
        timestamp queries, but a backend timestamp-calibration refresh
        performed by tick() may synchronize queued GPU work. Call
        Device::wait(), then tick(), before stopping when complete GPU data is
        required.

    .. py:method:: discard_capture(self) -> None

        Discard the active or completed capture without producing a trace.

    .. py:method:: live_snapshot(self) -> slangpy.ProfilerTrace

        Return the most recently published immutable live trace and request
        periodic live-trace publication.

    .. py:method:: frame_stats_snapshot(self) -> slangpy.ProfilerFrameStats

        Return the most recently published immutable frame statistics and
        request periodic statistics publication.

    .. py:method:: clear_frame_stats(self) -> None

        Clear completed and pending frame statistics after flushing current
        producer events.

    .. py:method:: tick(self) -> None

        Poll submitted GPU timestamp queries and queue resolved measurements
        for the collector.

        The poll does not deliberately wait for profiled submissions or
        unresolved queries. On CUDA, refreshing timestamp calibration
        synchronizes queued GPU work.

    .. py:method:: flush(self) -> None

        Block until a collector pass started for this request consumes inputs
        whose publication happens-before the request and publishes both
        snapshot products. Concurrent publications may be included or
        deferred. Active zones, unsealed frames, and unresolved GPU queries
        are not published inputs and are not awaited.



----

.. py:function:: slangpy.current_profiler() -> slangpy.Profiler

    Return the current thread's profiler or throw if none is active.



----

.. py:function:: slangpy.current_profiler_or_null() -> Profiler | None

    Return the current thread's profiler, or nullptr if none is active.



----

.. py:function:: slangpy.profile_zone(name: str, command_encoder: slangpy.CommandEncoder | None = None) -> slangpy.ProfilerZoneScope

    Profile a named CPU zone and optional command-encoder GPU zone using the current profiler.



----

.. py:function:: slangpy.profile_function(command_encoder: slangpy.CommandEncoder | None = None) -> slangpy.ProfilerZoneScope

    Profile the calling Python function and optional command-encoder GPU work using the current profiler.



----

.. py:function:: slangpy.profile_frame(name: str = 'frame') -> slangpy.ProfilerFrameScope

    Record a named frame boundary using the current profiler.



----

.. py:function:: slangpy.native_refl.get_builtin_layout(device: slangpy.Device) -> slangpy.native_refl.Layout

    N/A



----

.. py:function:: slangpy.native_refl.name_for_scalar_type(scalar_type: slangpy.TypeReflection.ScalarType) -> str

    N/A



----

.. py:function:: slangpy.native_refl.is_unknown(type: object | None) -> bool

    N/A



----

.. py:function:: slangpy.native_refl.is_known(type: object | None) -> bool

    N/A



----

.. py:function:: slangpy.native_refl.is_known_or_none(type: object | None) -> bool

    N/A



----

.. py:class:: slangpy.native_refl.IOType

    Base class: :py:class:`enum.Enum`

    N/A



----

.. py:function:: slangpy.native_refl.resolve_layout(device: slangpy.Device, element_type: object | None = None, layout: object | None = None) -> slangpy.native_refl.Layout

    N/A



----

.. py:function:: slangpy.native_refl.resolve_element_type(layout: slangpy.native_refl.Layout, element_type: object) -> slangpy.native_refl.Type

    N/A



----

.. py:class:: slangpy.native_refl.TypeLayout

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:property:: reflection
        :type: slangpy.TypeLayoutReflection

        N/A

    .. py:property:: size
        :type: int

        N/A

    .. py:property:: alignment
        :type: int

        N/A

    .. py:property:: stride
        :type: int

        N/A



----

.. py:class:: slangpy.native_refl.Type

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:property:: layout
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: program
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: type_reflection
        :type: slangpy.TypeReflection

        N/A

    .. py:property:: name
        :type: str

        N/A

    .. py:property:: full_name
        :type: str

        N/A

    .. py:property:: element_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: num_dims
        :type: int

        N/A

    .. py:property:: is_generic
        :type: bool

        N/A

    .. py:property:: vector_type_name
        :type: str

        N/A

    .. py:property:: uniform_layout
        :type: slangpy.native_refl.TypeLayout

        N/A

    .. py:property:: buffer_layout
        :type: slangpy.native_refl.TypeLayout

        N/A

    .. py:property:: derivative
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: differentiable
        :type: bool

        N/A

    .. py:property:: fields
        :type: dict

        N/A



----

.. py:class:: slangpy.native_refl.UnknownType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.VoidType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.PointerType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: target_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: slang_scalar_type
        :type: slangpy.TypeReflection.ScalarType

        N/A



----

.. py:class:: slangpy.native_refl.ScalarType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: slang_scalar_type
        :type: slangpy.TypeReflection.ScalarType

        N/A



----

.. py:class:: slangpy.native_refl.VectorType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: num_elements
        :type: int

        N/A

    .. py:property:: scalar_type
        :type: slangpy.native_refl.ScalarType

        N/A

    .. py:property:: slang_scalar_type
        :type: slangpy.TypeReflection.ScalarType

        N/A



----

.. py:class:: slangpy.native_refl.MatrixType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: rows
        :type: int

        N/A

    .. py:property:: cols
        :type: int

        N/A

    .. py:property:: scalar_type
        :type: slangpy.native_refl.ScalarType

        N/A

    .. py:property:: slang_scalar_type
        :type: slangpy.TypeReflection.ScalarType

        N/A

    .. py:property:: inner_element_type
        :type: slangpy.native_refl.Type

        N/A



----

.. py:class:: slangpy.native_refl.ArrayType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: num_elements
        :type: int

        N/A

    .. py:property:: array_shape
        :type: slangpy.slangpy.Shape

        N/A

    .. py:property:: any_generic_dims
        :type: bool

        N/A

    .. py:property:: inner_element_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: array_dims
        :type: int

        N/A



----

.. py:class:: slangpy.native_refl.StructType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.InterfaceType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.ResourceType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: resource_shape
        :type: slangpy.TypeReflection.ResourceShape

        N/A

    .. py:property:: resource_access
        :type: slangpy.TypeReflection.ResourceAccess

        N/A

    .. py:property:: writable
        :type: bool

        N/A



----

.. py:class:: slangpy.native_refl.TextureType

    Base class: :py:class:`slangpy.native_refl.ResourceType`

    N/A

    .. py:property:: texture_dims
        :type: int

        N/A

    .. py:property:: usage
        :type: slangpy.TextureUsage

        N/A



----

.. py:class:: slangpy.native_refl.StructuredBufferType

    Base class: :py:class:`slangpy.native_refl.ResourceType`

    N/A



----

.. py:class:: slangpy.native_refl.ByteAddressBufferType

    Base class: :py:class:`slangpy.native_refl.ResourceType`

    N/A



----

.. py:class:: slangpy.native_refl.DifferentialPairType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: primal
        :type: slangpy.native_refl.Type

        N/A



----

.. py:class:: slangpy.native_refl.RaytracingAccelerationStructureType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.SamplerStateType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.TensorType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:class:: slangpy.native_refl.TensorType.Kind

        Base class: :py:class:`enum.Enum`

        N/A

    .. py:class:: slangpy.native_refl.TensorType.Access

        Base class: :py:class:`enum.Enum`

        N/A

    .. py:property:: tensor_kind
        :type: slangpy.native_refl.TensorType.Kind

        N/A

    .. py:property:: tensor_type
        :type: slangpy.native_refl.TensorType.Kind

        N/A

    .. py:property:: access
        :type: slangpy.native_refl.TensorType.Access

        N/A

    .. py:property:: readable
        :type: bool

        N/A

    .. py:property:: writable
        :type: bool

        N/A

    .. py:property:: diff_tensor
        :type: bool

        N/A

    .. py:property:: difftensor
        :type: bool

        N/A

    .. py:property:: dims
        :type: int

        N/A

    .. py:property:: dtype
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: has_grad_in
        :type: bool

        N/A

    .. py:property:: has_grad_out
        :type: bool

        N/A

    .. py:staticmethod:: build_tensor_name(element_type: slangpy.native_refl.Type, dims: int, access: slangpy.native_refl.TensorType.Access = Access.read_write, tensor_kind: slangpy.native_refl.TensorType.Kind = Kind.tensor) -> str

        N/A



----

.. py:class:: slangpy.native_refl.TensorViewType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: dtype
        :type: slangpy.native_refl.Type

        N/A

    .. py:staticmethod:: build_tensorview_name(element_type: slangpy.native_refl.Type) -> str

        N/A



----

.. py:class:: slangpy.native_refl.DiffTensorViewType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A

    .. py:property:: dtype
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: wrapper_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:staticmethod:: build_difftensorview_name(element_type: slangpy.native_refl.Type) -> str

        N/A



----

.. py:class:: slangpy.native_refl.UnhandledType

    Base class: :py:class:`slangpy.native_refl.Type`

    N/A



----

.. py:class:: slangpy.native_refl.Variable

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:property:: layout
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: program
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: reflection
        :type: slangpy.VariableReflection

        N/A

    .. py:property:: type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: name
        :type: str

        N/A

    .. py:property:: modifiers
        :type: list[slangpy.ModifierID]

        N/A

    .. py:property:: declaration
        :type: str

        N/A

    .. py:property:: io_type
        :type: slangpy.native_refl.IOType

        N/A

    .. py:property:: no_diff
        :type: bool

        N/A

    .. py:property:: differentiable
        :type: bool

        N/A

    .. py:property:: derivative
        :type: slangpy.native_refl.Type

        N/A

    .. py:method:: has_modifier(self, modifier: slangpy.ModifierID) -> bool

        N/A



----

.. py:class:: slangpy.native_refl.Field

    Base class: :py:class:`slangpy.native_refl.Variable`

    N/A



----

.. py:class:: slangpy.native_refl.Parameter

    Base class: :py:class:`slangpy.native_refl.Variable`

    N/A

    .. py:property:: index
        :type: int

        N/A

    .. py:property:: has_default
        :type: bool

        N/A



----

.. py:class:: slangpy.native_refl.Function

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:property:: layout
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: program
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: reflection
        :type: slangpy.FunctionReflection

        N/A

    .. py:property:: name
        :type: str

        N/A

    .. py:property:: full_name
        :type: str

        N/A

    .. py:property:: this_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: this
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: return_type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: parameters
        :type: list[slangpy.native_refl.Parameter]

        N/A

    .. py:property:: have_return_value
        :type: bool

        N/A

    .. py:property:: differentiable
        :type: bool

        N/A

    .. py:property:: mutating
        :type: bool

        N/A

    .. py:property:: static
        :type: bool

        N/A

    .. py:property:: is_overloaded
        :type: bool

        N/A

    .. py:property:: overloads
        :type: list[slangpy.native_refl.Function]

        N/A

    .. py:property:: is_constructor
        :type: bool

        N/A

    .. py:method:: specialize_with_arg_types(self, types: collections.abc.Sequence[slangpy.native_refl.Type]) -> slangpy.native_refl.Function

        N/A



----

.. py:class:: slangpy.native_refl.Layout

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:method:: __init__(self, low_level_layout: object) -> None

        N/A

    .. py:property:: generation
        :type: int

        N/A

    .. py:property:: low_level_layout
        :type: slangpy.ProgramLayout

        N/A

    .. py:property:: program_layout
        :type: slangpy.ProgramLayout

        N/A

    .. py:property:: is_valid
        :type: bool

        N/A

    .. py:method:: find_type(self, type_reflection: object) -> slangpy.native_refl.Type

        N/A

    .. py:method:: find_type_by_name(self, name: str) -> slangpy.native_refl.Type

        N/A

    .. py:method:: require_type_by_name(self, name: str) -> slangpy.native_refl.Type

        N/A

    .. py:method:: find_function(self, function_reflection: object, this_type: object | None = None) -> slangpy.native_refl.Function

        N/A

    .. py:method:: find_function_by_name(self, name: str) -> slangpy.native_refl.Function

        N/A

    .. py:method:: require_function_by_name(self, name: str) -> slangpy.native_refl.Function

        N/A

    .. py:method:: find_function_by_name_in_type(self, type: slangpy.native_refl.Type, name: str) -> slangpy.native_refl.Function

        N/A

    .. py:method:: require_function_by_name_in_type(self, type: slangpy.native_refl.Type, name: str) -> slangpy.native_refl.Function

        N/A

    .. py:method:: scalar_type(self, scalar_type: slangpy.TypeReflection.ScalarType) -> slangpy.native_refl.Type

        N/A

    .. py:method:: vector_type(self, scalar_type: slangpy.TypeReflection.ScalarType, size: int) -> slangpy.native_refl.VectorType

        N/A

    .. py:method:: matrix_type(self, scalar_type: slangpy.TypeReflection.ScalarType, rows: int, cols: int) -> slangpy.native_refl.MatrixType

        N/A

    .. py:method:: array_type(self, element_type: slangpy.native_refl.Type, count: int) -> slangpy.native_refl.ArrayType

        N/A

    .. py:method:: tensor_type(self, element_type: slangpy.native_refl.Type, dims: int, access: slangpy.native_refl.TensorType.Access = Access.read_write, tensor_kind: slangpy.native_refl.TensorType.Kind = Kind.tensor) -> slangpy.native_refl.TensorType

        N/A

    .. py:method:: tensorview_type(self, element_type: slangpy.native_refl.Type) -> slangpy.native_refl.TensorViewType

        N/A

    .. py:method:: difftensorview_type(self, element_type: slangpy.native_refl.Type) -> slangpy.native_refl.DiffTensorViewType

        N/A

    .. py:method:: get_resolved_generic_args(self, type_reflection: object) -> object

        N/A

    .. py:method:: on_hot_reload(self, low_level_layout: object) -> None

        N/A



----

.. py:class:: slangpy.native_func.BaseModule

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:method:: __init__(self, module: object, layout: object) -> None

        N/A

    .. py:method:: on_hot_reload(self, module: object, low_level_layout: object) -> None

        N/A

    .. py:property:: device_module
        :type: slangpy.SlangModule

        N/A

    .. py:property:: layout
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: session
        :type: slangpy.SlangSession

        N/A

    .. py:property:: device
        :type: slangpy.Device

        N/A

    .. py:property:: name
        :type: str

        N/A



----

.. py:class:: slangpy.native_func.BaseStruct

    Base class: :py:class:`slangpy.Object`

    N/A

    .. py:method:: __init__(self, module: object, type: object) -> None

        N/A

    .. py:method:: on_hot_reload(self, type: object) -> None

        N/A

    .. py:property:: module
        :type: slangpy.native_func.BaseModule

        N/A

    .. py:property:: layout
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: program
        :type: slangpy.native_refl.Layout

        N/A

    .. py:property:: type
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: struct
        :type: slangpy.native_refl.Type

        N/A

    .. py:property:: type_reflection
        :type: slangpy.TypeReflection

        N/A

    .. py:property:: name
        :type: str

        N/A

    .. py:property:: full_name
        :type: str

        N/A

    .. py:property:: shape
        :type: slangpy.slangpy.Shape

        N/A



----

.. py:class:: slangpy.native_func.TensorDesc

    .. py:method:: __init__(self) -> None

    .. py:property:: dtype
        :type: slangpy.native_refl.Type

    .. py:property:: element_layout
        :type: slangpy.TypeLayoutReflection

    .. py:property:: offset
        :type: int

    .. py:property:: shape
        :type: slangpy.slangpy.Shape

    .. py:property:: strides
        :type: slangpy.slangpy.Shape

    .. py:property:: usage
        :type: slangpy.BufferUsage

    .. py:property:: memory_type
        :type: slangpy.MemoryType



----

.. py:class:: slangpy.native_func.Tensor

    Base class: :py:class:`slangpy.Object`

    .. py:method:: __init__(self, storage: slangpy.Buffer, dtype: slangpy.native_refl.Type, shape: slangpy.slangpy.Shape, strides: slangpy.slangpy.Shape = [invalid], offset: int = 0, grad_in: slangpy.native_func.Tensor | None = None, grad_out: slangpy.native_func.Tensor | None = None) -> None

    .. py:method:: __init__(self, desc: slangpy.native_func.TensorDesc, storage: slangpy.Buffer, grad_in: slangpy.native_func.Tensor | None = None, grad_out: slangpy.native_func.Tensor | None = None) -> None
        :no-index:

    .. py:property:: device
        :type: slangpy.Device

    .. py:property:: dtype
        :type: slangpy.native_refl.Type

    .. py:property:: offset
        :type: int

    .. py:property:: shape
        :type: slangpy.slangpy.Shape

    .. py:property:: strides
        :type: slangpy.slangpy.Shape

    .. py:property:: element_count
        :type: int

    .. py:property:: usage
        :type: slangpy.BufferUsage

    .. py:property:: memory_type
        :type: slangpy.MemoryType

    .. py:property:: storage
        :type: slangpy.Buffer

    .. py:property:: grad_in
        :type: slangpy.native_func.Tensor

    .. py:property:: grad_out
        :type: slangpy.native_func.Tensor

    .. py:property:: grad
        :type: slangpy.native_func.Tensor

    .. py:method:: clear(self, cmd: slangpy.CommandEncoder | None = None) -> None

    .. py:method:: cursor(self, start: int | None = None, count: int | None = None) -> slangpy.BufferCursor

    .. py:method:: uniforms(self) -> dict

    .. py:method:: to_numpy(self) -> numpy.ndarray[]

    .. py:method:: to_torch(self) -> torch.Tensor[]

    .. py:method:: copy_from_numpy(self, data: numpy.ndarray[]) -> None

    .. py:method:: copy_from_torch(self, tensor: object) -> None

    .. py:method:: is_contiguous(self) -> bool

    .. py:method:: point_to(self, target: slangpy.native_func.Tensor) -> None

    .. py:method:: broadcast_to(self, shape: slangpy.slangpy.Shape) -> slangpy.native_func.Tensor

    .. py:method:: view(self, shape: slangpy.slangpy.Shape, strides: slangpy.slangpy.Shape = [invalid], offset: int = 0) -> slangpy.native_func.Tensor

    .. py:method:: with_grads(self, grad_in: slangpy.native_func.Tensor | None = None, grad_out: slangpy.native_func.Tensor | None = None, zero: bool = True) -> slangpy.native_func.Tensor

    .. py:method:: detach(self) -> slangpy.native_func.Tensor

    .. py:staticmethod:: numpy(device: slangpy.Device, ndarray: object) -> slangpy.native_func.Tensor

    .. py:staticmethod:: from_numpy(device: slangpy.Device, ndarray: object, usage: slangpy.BufferUsage = 24, memory_type: slangpy.MemoryType = MemoryType.device_local, program_layout: slangpy.native_refl.Layout | None = None, target_slang_dtype: object | None = None) -> slangpy.native_func.Tensor

    .. py:staticmethod:: empty(device: slangpy.Device, shape: slangpy.slangpy.Shape, dtype: object | None = None, usage: slangpy.BufferUsage = 24, memory_type: slangpy.MemoryType = MemoryType.device_local, program_layout: slangpy.native_refl.Layout | None = None) -> slangpy.native_func.Tensor

    .. py:staticmethod:: zeros(device: slangpy.Device, shape: slangpy.slangpy.Shape, dtype: object, usage: slangpy.BufferUsage = 24, memory_type: slangpy.MemoryType = MemoryType.device_local, program_layout: slangpy.native_refl.Layout | None = None) -> slangpy.native_func.Tensor

    .. py:staticmethod:: empty_like(other: slangpy.native_func.Tensor) -> slangpy.native_func.Tensor

    .. py:staticmethod:: zeros_like(other: slangpy.native_func.Tensor) -> slangpy.native_func.Tensor

    .. py:staticmethod:: from_torch(device: slangpy.Device, tensor: object, dtype: object, usage: slangpy.BufferUsage = 24, program_layout: slangpy.native_refl.Layout | None = None) -> slangpy.native_func.Tensor

    .. py:staticmethod:: load_from_image(device: slangpy.Device, path: object, flip_y: bool = False, linearize: bool = False, scale: float = 1.0, offset: float = 0.0, grayscale: bool = False) -> slangpy.native_func.Tensor



----

.. py:class:: slangpy.YAHandling

    Base class: :py:class:`enum.Enum`



----

.. py:function:: slangpy.is_torch_bridge_available() -> bool

    N/A



----

.. py:function:: slangpy.is_torch_bridge_using_fallback() -> bool

    N/A



----

.. py:function:: slangpy.get_torch_bridge_fallback_reason() -> str

    N/A



----

.. py:function:: slangpy.set_torch_bridge_python_fallback(force: bool) -> None

    N/A



----

.. py:function:: slangpy.is_torch_tensor(obj: object) -> bool

    N/A



----

.. py:function:: slangpy.extract_torch_tensor_info(tensor: object) -> object

    N/A



----

.. py:function:: slangpy.extract_torch_tensor_signature(tensor: object) -> str

    N/A



----

.. py:function:: slangpy.copy_torch_tensor_to_buffer(tensor: object, buffer: slangpy.Buffer) -> bool

    N/A



----

.. py:function:: slangpy.copy_buffer_to_torch_tensor(buffer: slangpy.Buffer, tensor: object) -> bool

    N/A



----

.. py:function:: slangpy.create_torch_empty_tensor(shape: list, scalar_type: int, device_index: int = 0) -> object

    N/A



----

.. py:function:: slangpy.create_torch_zeros_like_tensor(tensor: object) -> object

    N/A



----

.. py:class:: slangpy.core.native.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.core.native.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.core.native.AutogradAccess
    Alias class: :py:class:`slangpy.slangpy.AutogradAccess`



----

.. py:function:: slangpy.core.native.unpack_args(*args) -> tuple

    N/A



----

.. py:function:: slangpy.core.native.unpack_kwargs(**kwargs) -> tuple

    N/A



----

.. py:function:: slangpy.core.native.unpack_arg(arg: object) -> object

    N/A



----

.. py:function:: slangpy.core.native.pack_arg(arg: object, unpacked_arg: object) -> None

    N/A



----

.. py:function:: slangpy.core.native.get_value_signature(o: object) -> str

    N/A



----

.. py:class:: slangpy.core.native.SignatureBuilder
    Alias class: :py:class:`slangpy.slangpy.SignatureBuilder`



----

.. py:class:: slangpy.core.native.NativeObject
    Alias class: :py:class:`slangpy.slangpy.NativeObject`



----

.. py:class:: slangpy.core.native.NativeMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`



----

.. py:class:: slangpy.core.native.NativeBoundVariableRuntime
    Alias class: :py:class:`slangpy.slangpy.NativeBoundVariableRuntime`



----

.. py:class:: slangpy.core.native.NativeBoundCallRuntime
    Alias class: :py:class:`slangpy.slangpy.NativeBoundCallRuntime`



----

.. py:class:: slangpy.core.native.NativeCallRuntimeOptions
    Alias class: :py:class:`slangpy.slangpy.NativeCallRuntimeOptions`



----

.. py:class:: slangpy.core.native.NativeCallData
    Alias class: :py:class:`slangpy.slangpy.NativeCallData`



----

.. py:class:: slangpy.core.native.NativeCallDataCache
    Alias class: :py:class:`slangpy.slangpy.NativeCallDataCache`



----

.. py:class:: slangpy.core.native.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.core.native.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.core.native.FunctionNodeType
    Alias class: :py:class:`slangpy.slangpy.FunctionNodeType`



----

.. py:class:: slangpy.core.native.NativeFunctionNode
    Alias class: :py:class:`slangpy.slangpy.NativeFunctionNode`



----

.. py:class:: slangpy.core.native.NativePackedArg
    Alias class: :py:class:`slangpy.slangpy.NativePackedArg`



----

.. py:function:: slangpy.core.native.get_texture_shape(texture: slangpy.Texture, mip: int = 0) -> slangpy.slangpy.Shape

    N/A



----

.. py:class:: slangpy.core.native.NativeBufferMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeBufferMarshall`



----

.. py:class:: slangpy.core.native.NativeDescriptorMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeDescriptorMarshall`



----

.. py:class:: slangpy.core.native.NativeTextureMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeTextureMarshall`



----

.. py:class:: slangpy.core.native.TensorMarshall
    Alias class: :py:class:`slangpy.slangpy.TensorMarshall`



----

.. py:class:: slangpy.core.native.NativeNumpyMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeNumpyMarshall`



----

.. py:class:: slangpy.core.native.NativeTorchTensorMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorMarshall`



----

.. py:class:: slangpy.core.native.NativeTorchTensorDiffPair
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorDiffPair`



----

.. py:class:: slangpy.core.native.NativeValueMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`



----

.. py:class:: slangpy.core.utils.PathLike
    Alias class: :py:class:`os.PathLike`



----

.. py:class:: slangpy.core.utils.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.utils.DeclReflection
    Alias class: :py:class:`slangpy.DeclReflection`



----

.. py:class:: slangpy.core.utils.ProgramLayout
    Alias class: :py:class:`slangpy.ProgramLayout`



----

.. py:class:: slangpy.core.utils.TypeLayoutReflection
    Alias class: :py:class:`slangpy.TypeLayoutReflection`



----

.. py:class:: slangpy.core.utils.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.core.utils.DeviceType
    Alias class: :py:class:`slangpy.DeviceType`



----

.. py:class:: slangpy.core.utils.PipelineCompilationMode
    Alias class: :py:class:`slangpy.PipelineCompilationMode`



----

.. py:class:: slangpy.core.utils.Device
    Alias class: :py:class:`slangpy.Device`



----

.. py:class:: slangpy.core.utils.NativeHandle
    Alias class: :py:class:`slangpy.NativeHandle`



----

.. py:function:: slangpy.core.utils.get_cuda_current_context_native_handles() -> list[slangpy.NativeHandle]

    Gets the device and context handles for the current CUDA context. Use
    to retrieve an existing context (eg from PyTorch) to pass as the
    existing_device_handles from which to create a device in the
    DeviceDesc.



----

.. py:class:: slangpy.core.utils.BindlessDesc
    Alias class: :py:class:`slangpy.BindlessDesc`



----

.. py:class:: slangpy.core.utils.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.core.utils.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.core.enums.Enum
    Alias class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.core.enums.IOType
    Alias class: :py:class:`slangpy.native_refl.IOType`



----

.. py:class:: slangpy.core.enums.PrimType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.core.logging.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.logging.FunctionReflection
    Alias class: :py:class:`slangpy.FunctionReflection`



----

.. py:class:: slangpy.core.logging.ModifierID
    Alias class: :py:class:`slangpy.ModifierID`



----

.. py:class:: slangpy.core.logging.VariableReflection
    Alias class: :py:class:`slangpy.VariableReflection`



----

.. py:class:: slangpy.core.logging.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.core.logging.TableColumn



----

.. py:class:: slangpy.core.generator.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.generator.CodeGen
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`



----

.. py:class:: slangpy.core.generator.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.core.generator.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.core.generator.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.core.generator.KernelGenException

    Base class: :py:class:`builtins.Exception`



----

.. py:class:: slangpy.core.shapes.TArgShapesResult

    Base class: :py:class:`builtins.dict`



----

.. py:class:: slangpy.core.callsignature.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.callsignature.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.core.callsignature.tr.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.callsignature.tr.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.core.callsignature.tr.NativeMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`



----

.. py:class:: slangpy.core.callsignature.ModifierID
    Alias class: :py:class:`slangpy.ModifierID`



----

.. py:class:: slangpy.core.callsignature.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.core.callsignature.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.core.callsignature.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.core.callsignature.BoundCall
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundCall`



----

.. py:class:: slangpy.core.callsignature.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.core.callsignature.NoneMarshall
    Alias class: :py:class:`slangpy.builtin.value.NoneMarshall`



----

.. py:class:: slangpy.core.callsignature.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.core.callsignature.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.core.callsignature.ResolvedParam
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolvedParam`



----

.. py:class:: slangpy.core.callsignature.ResolutionDiagnostic
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolutionDiagnostic`



----

.. py:class:: slangpy.core.callsignature.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.core.callsignature.ValueRef
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`



----

.. py:class:: slangpy.core.callsignature.MismatchReason



----

.. py:class:: slangpy.core.callsignature.ResolveException

    Base class: :py:class:`builtins.Exception`



----

.. py:class:: slangpy.core.callsignature.KernelGenException
    Alias class: :py:class:`slangpy.core.generator.KernelGenException`



----

.. py:class:: slangpy.core.function.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.function.Protocol
    Alias class: :py:class:`typing.Protocol`



----

.. py:class:: slangpy.core.function.Enum
    Alias class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.core.function.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.core.function.SignatureBuilder
    Alias class: :py:class:`slangpy.slangpy.SignatureBuilder`



----

.. py:class:: slangpy.core.function.NativeCallRuntimeOptions
    Alias class: :py:class:`slangpy.slangpy.NativeCallRuntimeOptions`



----

.. py:class:: slangpy.core.function.NativeFunctionNode
    Alias class: :py:class:`slangpy.slangpy.NativeFunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeType
    Alias class: :py:class:`slangpy.slangpy.FunctionNodeType`



----

.. py:class:: slangpy.core.function.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.core.function.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.core.function.CommandEncoder
    Alias class: :py:class:`slangpy.CommandEncoder`



----

.. py:class:: slangpy.core.function.TypeConformance
    Alias class: :py:class:`slangpy.TypeConformance`



----

.. py:class:: slangpy.core.function.uint3
    Alias class: :py:class:`slangpy.math.uint3`



----

.. py:class:: slangpy.core.function.Logger
    Alias class: :py:class:`slangpy.Logger`



----

.. py:class:: slangpy.core.function.NativeHandle
    Alias class: :py:class:`slangpy.NativeHandle`



----

.. py:class:: slangpy.core.function.NativeHandleType
    Alias class: :py:class:`slangpy.NativeHandleType`



----

.. py:class:: slangpy.core.function.RayTracingPipelineFlags
    Alias class: :py:class:`slangpy.RayTracingPipelineFlags`



----

.. py:class:: slangpy.core.function.HitGroupDesc
    Alias class: :py:class:`slangpy.HitGroupDesc`



----

.. py:class:: slangpy.core.function.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.core.function.IThis

    Base class: :py:class:`typing.Protocol`



----

.. py:class:: slangpy.core.function.PipelineType

    Base class: :py:class:`enum.Enum`



----

.. py:class:: slangpy.core.function.FunctionBuildInfo



----

.. py:class:: slangpy.core.function.FunctionNode

    Base class: :py:class:`slangpy.slangpy.NativeFunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeBind

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeMap

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeSet

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeCUDAStream

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeConstants

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodePrelude

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeTypeConformances

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeRayTracing

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeBwds

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeReturnType

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeThreadGroupSize

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeLogger

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.FunctionNodeCallGroupShape

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.function.Function

    Base class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.calldata.Path
    Alias class: :py:class:`pathlib.Path`



----

.. py:class:: slangpy.core.calldata.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.calldata.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.core.calldata.ModifierID
    Alias class: :py:class:`slangpy.ModifierID`



----

.. py:class:: slangpy.core.calldata.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.core.calldata.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.core.calldata.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.core.calldata.BoundCall
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundCall`



----

.. py:class:: slangpy.core.calldata.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.core.calldata.NoneMarshall
    Alias class: :py:class:`slangpy.builtin.value.NoneMarshall`



----

.. py:class:: slangpy.core.calldata.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.core.calldata.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.core.calldata.ResolvedParam
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolvedParam`



----

.. py:class:: slangpy.core.calldata.ResolutionDiagnostic
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolutionDiagnostic`



----

.. py:class:: slangpy.core.calldata.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.core.calldata.ValueRef
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`



----

.. py:class:: slangpy.core.calldata.MismatchReason
    Alias class: :py:class:`slangpy.core.callsignature.MismatchReason`



----

.. py:class:: slangpy.core.calldata.ResolveException
    Alias class: :py:class:`slangpy.core.callsignature.ResolveException`



----

.. py:class:: slangpy.core.calldata.KernelGenException
    Alias class: :py:class:`slangpy.core.generator.KernelGenException`



----

.. py:class:: slangpy.core.calldata.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.core.calldata.NativeCallData
    Alias class: :py:class:`slangpy.slangpy.NativeCallData`



----

.. py:function:: slangpy.core.calldata.unpack_args(*args) -> tuple

    N/A



----

.. py:function:: slangpy.core.calldata.unpack_kwargs(**kwargs) -> tuple

    N/A



----

.. py:class:: slangpy.core.calldata.PipelineType
    Alias class: :py:class:`slangpy.core.function.PipelineType`



----

.. py:class:: slangpy.core.calldata.SlangCompileError
    Alias class: :py:class:`slangpy.SlangCompileError`



----

.. py:class:: slangpy.core.calldata.SlangLinkOptions
    Alias class: :py:class:`slangpy.SlangLinkOptions`



----

.. py:class:: slangpy.core.calldata.NativeHandle
    Alias class: :py:class:`slangpy.NativeHandle`



----

.. py:class:: slangpy.core.calldata.DeviceType
    Alias class: :py:class:`slangpy.DeviceType`



----

.. py:class:: slangpy.core.calldata.PipelineCompilationPolicy
    Alias class: :py:class:`slangpy.PipelineCompilationPolicy`



----

.. py:class:: slangpy.core.calldata.TypeConformance
    Alias class: :py:class:`slangpy.TypeConformance`



----

.. py:function:: slangpy.core.calldata.is_torch_bridge_using_fallback() -> bool

    N/A



----

.. py:function:: slangpy.core.calldata.get_torch_bridge_fallback_reason() -> str

    N/A



----

.. py:class:: slangpy.core.calldata.BoundCallRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundCallRuntime`



----

.. py:class:: slangpy.core.calldata.BoundVariableException
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariableException`



----

.. py:class:: slangpy.core.calldata.CodeGen
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`



----

.. py:class:: slangpy.core.calldata.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.core.calldata.ITensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType`



----

.. py:class:: slangpy.core.calldata.TensorAccess
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`



----

.. py:class:: slangpy.core.calldata.CallData

    Base class: :py:class:`slangpy.slangpy.NativeCallData`



----

.. py:class:: slangpy.core.struct.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.struct.Function
    Alias class: :py:class:`slangpy.core.function.Function`



----

.. py:class:: slangpy.core.struct.BaseStruct
    Alias class: :py:class:`slangpy.native_func.BaseStruct`



----

.. py:class:: slangpy.core.struct.Struct

    Base class: :py:class:`slangpy.native_func.BaseStruct`


        A Slang struct, typically created by accessing it via a module or parent struct. i.e. mymodule.Foo,
        or mymodule.Foo.Bar.




----

.. py:class:: slangpy.core.module.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.module.Function
    Alias class: :py:class:`slangpy.core.function.Function`



----

.. py:class:: slangpy.core.module.Struct
    Alias class: :py:class:`slangpy.core.struct.Struct`



----

.. py:class:: slangpy.core.module.Pipeline
    Alias class: :py:class:`slangpy.Pipeline`



----

.. py:class:: slangpy.core.module.ShaderTable
    Alias class: :py:class:`slangpy.ShaderTable`



----

.. py:class:: slangpy.core.module.SlangModule
    Alias class: :py:class:`slangpy.SlangModule`



----

.. py:class:: slangpy.core.module.Device
    Alias class: :py:class:`slangpy.Device`



----

.. py:class:: slangpy.core.module.Logger
    Alias class: :py:class:`slangpy.Logger`



----

.. py:class:: slangpy.core.module.NativeCallDataCache
    Alias class: :py:class:`slangpy.slangpy.NativeCallDataCache`



----

.. py:class:: slangpy.core.module.BaseModule
    Alias class: :py:class:`slangpy.native_func.BaseModule`



----

.. py:class:: slangpy.core.module.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.core.module.CallDataCache

    Base class: :py:class:`slangpy.slangpy.NativeCallDataCache`



----

.. py:class:: slangpy.core.module.Module

    Base class: :py:class:`slangpy.native_func.BaseModule`


        A Slang module, created either by loading a slang file or providing a loaded SGL module.




----

.. py:class:: slangpy.core.instance.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.instance.FunctionNode
    Alias class: :py:class:`slangpy.core.function.FunctionNode`



----

.. py:class:: slangpy.core.instance.Struct
    Alias class: :py:class:`slangpy.core.struct.Struct`



----

.. py:class:: slangpy.core.instance.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.core.instance.InstanceList


        Represents a list of instances of a struct, either as a single buffer
        or an SOA style set of buffers for each field. data can either
        be a dictionary of field names to buffers, or a single buffer.




----

.. py:class:: slangpy.core.instance.InstanceTensor

    Base class: :py:class:`slangpy.core.instance.InstanceList`


        Simplified implementation of InstanceList that uses a single buffer for all instances and
        provides buffer convenience functions for accessing its data.




----

.. py:class:: slangpy.core.packedarg.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.core.packedarg.Module
    Alias class: :py:class:`slangpy.core.module.Module`



----

.. py:function:: slangpy.core.packedarg.get_value_signature(o: object) -> str

    N/A



----

.. py:class:: slangpy.core.packedarg.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.core.packedarg.NativePackedArg
    Alias class: :py:class:`slangpy.slangpy.NativePackedArg`



----

.. py:function:: slangpy.core.packedarg.unpack_arg(arg: object) -> object

    N/A



----

.. py:class:: slangpy.core.packedarg.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.core.packedarg.PackedArg

    Base class: :py:class:`slangpy.slangpy.NativePackedArg`


        Represents an argument that has been efficiently packed into
        a shader object for use in later functionc alls.




----

.. py:class:: slangpy.reflection.reflectiontypes.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.reflection.reflectiontypes.TR
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.reflection.reflectiontypes.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.reflection.reflectiontypes.ArrayType
    Alias class: :py:class:`slangpy.native_refl.ArrayType`



----

.. py:class:: slangpy.reflection.reflectiontypes.ByteAddressBufferType
    Alias class: :py:class:`slangpy.native_refl.ByteAddressBufferType`



----

.. py:class:: slangpy.reflection.reflectiontypes.DiffTensorViewType
    Alias class: :py:class:`slangpy.native_refl.DiffTensorViewType`



----

.. py:class:: slangpy.reflection.reflectiontypes.DifferentialPairType
    Alias class: :py:class:`slangpy.native_refl.DifferentialPairType`



----

.. py:class:: slangpy.reflection.reflectiontypes.SlangField
    Alias class: :py:class:`slangpy.native_refl.Field`



----

.. py:class:: slangpy.reflection.reflectiontypes.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.reflection.reflectiontypes.InterfaceType
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`



----

.. py:class:: slangpy.reflection.reflectiontypes.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.reflection.reflectiontypes.MatrixType
    Alias class: :py:class:`slangpy.native_refl.MatrixType`



----

.. py:class:: slangpy.reflection.reflectiontypes.SlangParameter
    Alias class: :py:class:`slangpy.native_refl.Parameter`



----

.. py:class:: slangpy.reflection.reflectiontypes.PointerType
    Alias class: :py:class:`slangpy.native_refl.PointerType`



----

.. py:class:: slangpy.reflection.reflectiontypes.RaytracingAccelerationStructureType
    Alias class: :py:class:`slangpy.native_refl.RaytracingAccelerationStructureType`



----

.. py:class:: slangpy.reflection.reflectiontypes.ResourceType
    Alias class: :py:class:`slangpy.native_refl.ResourceType`



----

.. py:class:: slangpy.reflection.reflectiontypes.SamplerStateType
    Alias class: :py:class:`slangpy.native_refl.SamplerStateType`



----

.. py:class:: slangpy.reflection.reflectiontypes.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.reflection.reflectiontypes.StructuredBufferType
    Alias class: :py:class:`slangpy.native_refl.StructuredBufferType`



----

.. py:class:: slangpy.reflection.reflectiontypes.StructType
    Alias class: :py:class:`slangpy.native_refl.StructType`



----

.. py:class:: slangpy.reflection.reflectiontypes.ITensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType`



----

.. py:class:: slangpy.reflection.reflectiontypes.TensorViewType
    Alias class: :py:class:`slangpy.native_refl.TensorViewType`



----

.. py:class:: slangpy.reflection.reflectiontypes.TextureType
    Alias class: :py:class:`slangpy.native_refl.TextureType`



----

.. py:class:: slangpy.reflection.reflectiontypes.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.reflection.reflectiontypes.SlangLayout
    Alias class: :py:class:`slangpy.native_refl.TypeLayout`



----

.. py:class:: slangpy.reflection.reflectiontypes.UnhandledType
    Alias class: :py:class:`slangpy.native_refl.UnhandledType`



----

.. py:class:: slangpy.reflection.reflectiontypes.UnknownType
    Alias class: :py:class:`slangpy.native_refl.UnknownType`



----

.. py:class:: slangpy.reflection.reflectiontypes.VectorType
    Alias class: :py:class:`slangpy.native_refl.VectorType`



----

.. py:class:: slangpy.reflection.reflectiontypes.VoidType
    Alias class: :py:class:`slangpy.native_refl.VoidType`



----

.. py:function:: slangpy.reflection.reflectiontypes.is_known(type: object | None) -> bool

    N/A



----

.. py:function:: slangpy.reflection.reflectiontypes.is_known_or_none(type: object | None) -> bool

    N/A



----

.. py:function:: slangpy.reflection.reflectiontypes.is_unknown(type: object | None) -> bool

    N/A



----

.. py:class:: slangpy.reflection.reflectiontypes.TensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`



----

.. py:class:: slangpy.reflection.reflectiontypes.TensorAccess
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`



----

.. py:class:: slangpy.reflection.SlangLayout
    Alias class: :py:class:`slangpy.native_refl.TypeLayout`



----

.. py:class:: slangpy.reflection.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.reflection.VoidType
    Alias class: :py:class:`slangpy.native_refl.VoidType`



----

.. py:class:: slangpy.reflection.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.reflection.VectorType
    Alias class: :py:class:`slangpy.native_refl.VectorType`



----

.. py:class:: slangpy.reflection.MatrixType
    Alias class: :py:class:`slangpy.native_refl.MatrixType`



----

.. py:class:: slangpy.reflection.ArrayType
    Alias class: :py:class:`slangpy.native_refl.ArrayType`



----

.. py:class:: slangpy.reflection.StructType
    Alias class: :py:class:`slangpy.native_refl.StructType`



----

.. py:class:: slangpy.reflection.InterfaceType
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`



----

.. py:class:: slangpy.reflection.TextureType
    Alias class: :py:class:`slangpy.native_refl.TextureType`



----

.. py:class:: slangpy.reflection.PointerType
    Alias class: :py:class:`slangpy.native_refl.PointerType`



----

.. py:class:: slangpy.reflection.UnknownType
    Alias class: :py:class:`slangpy.native_refl.UnknownType`



----

.. py:class:: slangpy.reflection.ResourceType
    Alias class: :py:class:`slangpy.native_refl.ResourceType`



----

.. py:class:: slangpy.reflection.StructuredBufferType
    Alias class: :py:class:`slangpy.native_refl.StructuredBufferType`



----

.. py:class:: slangpy.reflection.ByteAddressBufferType
    Alias class: :py:class:`slangpy.native_refl.ByteAddressBufferType`



----

.. py:class:: slangpy.reflection.DifferentialPairType
    Alias class: :py:class:`slangpy.native_refl.DifferentialPairType`



----

.. py:class:: slangpy.reflection.RaytracingAccelerationStructureType
    Alias class: :py:class:`slangpy.native_refl.RaytracingAccelerationStructureType`



----

.. py:class:: slangpy.reflection.SamplerStateType
    Alias class: :py:class:`slangpy.native_refl.SamplerStateType`



----

.. py:class:: slangpy.reflection.UnhandledType
    Alias class: :py:class:`slangpy.native_refl.UnhandledType`



----

.. py:class:: slangpy.reflection.ITensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType`



----

.. py:class:: slangpy.reflection.TensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`



----

.. py:class:: slangpy.reflection.TensorAccess
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`



----

.. py:class:: slangpy.reflection.TensorViewType
    Alias class: :py:class:`slangpy.native_refl.TensorViewType`



----

.. py:class:: slangpy.reflection.DiffTensorViewType
    Alias class: :py:class:`slangpy.native_refl.DiffTensorViewType`



----

.. py:class:: slangpy.reflection.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.reflection.SlangField
    Alias class: :py:class:`slangpy.native_refl.Field`



----

.. py:class:: slangpy.reflection.SlangParameter
    Alias class: :py:class:`slangpy.native_refl.Parameter`



----

.. py:class:: slangpy.reflection.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:function:: slangpy.reflection.is_unknown(type: object | None) -> bool

    N/A



----

.. py:function:: slangpy.reflection.is_known(type: object | None) -> bool

    N/A



----

.. py:function:: slangpy.reflection.is_known_or_none(type: object | None) -> bool

    N/A



----

.. py:class:: slangpy.reflection.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.reflection.typeresolution.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.reflection.typeresolution.ResolutionArg


        Holds a single argument for type resolution. Starts with the python marshall and slang type
        being resolved, and ends with the resolved vector type (if any). In the case of functions
        with optional parameters, the python variable can be None and the parameter type will be used




----

.. py:class:: slangpy.reflection.typeresolution.ResolutionDiagnostic



----

.. py:class:: slangpy.reflection.typeresolution.ResolvedParam



----

.. py:class:: slangpy.reflection.typeresolution.ResolveResult



----

.. py:class:: slangpy.reflection.lookup.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.reflection.lookup.Device
    Alias class: :py:class:`slangpy.Device`



----

.. py:class:: slangpy.reflection.lookup.TypeLayoutReflection
    Alias class: :py:class:`slangpy.TypeLayoutReflection`



----

.. py:class:: slangpy.reflection.lookup.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.reflection.lookup.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.reflection.lookup.BaseStruct
    Alias class: :py:class:`slangpy.native_func.BaseStruct`



----

.. py:function:: slangpy.reflection.lookup.get_builtin_layout(device: slangpy.Device) -> slangpy.native_refl.Layout

    N/A



----

.. py:function:: slangpy.reflection.lookup.resolve_element_type(layout: slangpy.native_refl.Layout, element_type: object) -> slangpy.native_refl.Type

    N/A



----

.. py:function:: slangpy.reflection.lookup.resolve_layout(device: slangpy.Device, element_type: object | None = None, layout: object | None = None) -> slangpy.native_refl.Layout

    N/A



----

.. py:class:: slangpy.reflection.lookup.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.reflection.lookup.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.reflection.lookup.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.reflection.lookup.ST
    Alias class: :py:class:`slangpy.TypeReflection.ScalarType`



----

.. py:class:: slangpy.bindings.codegen.CodeGenBlock



----

.. py:class:: slangpy.bindings.codegen.CodeGen


        Tool for generating the code for a SlangPy kernel. Contains a set of
        different code blocks that can be filled in and then combined to
        generate the final code.




----

.. py:class:: slangpy.bindings.marshall.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.bindings.marshall.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.bindings.marshall.NativeMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`



----

.. py:class:: slangpy.bindings.marshall.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.bindings.marshall.BindContext


        Contextual information passed around during kernel generation process.




----

.. py:class:: slangpy.bindings.marshall.ReturnContext


        Internal structure used to store information about return type of a function during generation.




----

.. py:class:: slangpy.bindings.marshall.Marshall

    Base class: :py:class:`slangpy.slangpy.NativeMarshall`


        Base class for a type marshall that describes how to pass a given type to/from a
        SlangPy kernel. When a kernel is generated, a marshall is instantiated for each
        Python value. Future calls to the kernel verify type signatures match and then
        re-use the existing marshalls.




----

.. py:class:: slangpy.bindings.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.bindings.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.bindings.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.bindings.boundvariable.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.bindings.boundvariable.IOType
    Alias class: :py:class:`slangpy.native_refl.IOType`



----

.. py:class:: slangpy.bindings.boundvariable.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.bindings.boundvariable.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.bindings.boundvariable.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.bindings.boundvariable.NativeMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`



----

.. py:class:: slangpy.bindings.boundvariable.ModifierID
    Alias class: :py:class:`slangpy.ModifierID`



----

.. py:class:: slangpy.bindings.boundvariable.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.bindings.boundvariable.CodeGen
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`



----

.. py:class:: slangpy.bindings.boundvariable.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.bindings.boundvariable.SlangField
    Alias class: :py:class:`slangpy.native_refl.Field`



----

.. py:class:: slangpy.bindings.boundvariable.SlangFunction
    Alias class: :py:class:`slangpy.native_refl.Function`



----

.. py:class:: slangpy.bindings.boundvariable.SlangParameter
    Alias class: :py:class:`slangpy.native_refl.Parameter`



----

.. py:class:: slangpy.bindings.boundvariable.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.bindings.boundvariable.ResolvedParam
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolvedParam`



----

.. py:class:: slangpy.bindings.boundvariable.BoundVariableException

    Base class: :py:class:`builtins.Exception`


        Custom exception type that carries a message and the variable that caused
        the exception.




----

.. py:class:: slangpy.bindings.boundvariable.BoundCall


        Stores the binding of python arguments to slang parameters during kernel
        generation. This is initialized purely with a set of python arguments and
        later bound to corresponding slang parameters during function resolution.




----

.. py:class:: slangpy.bindings.boundvariable.BoundVariable


        Node in a built signature tree, maintains a pairing of python+slang marshall,
        and a potential set of child nodes for use during kernel generation.




----

.. py:class:: slangpy.bindings.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.bindings.BoundCall
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundCall`



----

.. py:class:: slangpy.bindings.BoundVariableException
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariableException`



----

.. py:class:: slangpy.bindings.boundvariableruntime.NativeBoundCallRuntime
    Alias class: :py:class:`slangpy.slangpy.NativeBoundCallRuntime`



----

.. py:class:: slangpy.bindings.boundvariableruntime.NativeBoundVariableRuntime
    Alias class: :py:class:`slangpy.slangpy.NativeBoundVariableRuntime`



----

.. py:class:: slangpy.bindings.boundvariableruntime.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.bindings.boundvariableruntime.BoundCallRuntime

    Base class: :py:class:`slangpy.slangpy.NativeBoundCallRuntime`


        Minimal call data stored after kernel generation required to
        dispatch a call to a SlangPy kernel.




----

.. py:class:: slangpy.bindings.boundvariableruntime.BoundVariableRuntime

    Base class: :py:class:`slangpy.slangpy.NativeBoundVariableRuntime`


        Minimal variable data stored after kernel generation required to
        dispatch a call to a SlangPy kernel.




----

.. py:class:: slangpy.bindings.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.bindings.BoundCallRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundCallRuntime`



----

.. py:class:: slangpy.bindings.CodeGen
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`



----

.. py:class:: slangpy.bindings.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.bindings.cursor.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.bindings.cursor.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.bindings.cursor.BoundVariableException
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariableException`



----

.. py:class:: slangpy.bindings.cursor.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.bindings.cursor.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.bindings.cursor.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.bindings.cursor.NativeValueMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`



----

.. py:class:: slangpy.bindings.cursor.WriteToCursorMarshallInfo

    Metadata needed to marshal values through native cursor-writer registration.



----

.. py:class:: slangpy.bindings.cursor.WriteToCursorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeValueMarshall`

    Marshall for scalar values that are written through the native cursor fast path.



----

.. py:class:: slangpy.bindings.WriteToCursorMarshall
    Alias class: :py:class:`slangpy.bindings.cursor.WriteToCursorMarshall`



----

.. py:class:: slangpy.bindings.WriteToCursorMarshallInfo
    Alias class: :py:class:`slangpy.bindings.cursor.WriteToCursorMarshallInfo`



----

.. py:class:: slangpy.bindings.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.bindings.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.bindings.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.experimental.gridarg.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.experimental.gridarg.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.experimental.gridarg.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.experimental.gridarg.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.experimental.gridarg.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.experimental.gridarg.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.experimental.gridarg.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.experimental.gridarg.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.experimental.gridarg.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.experimental.gridarg.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.experimental.gridarg.NativeObject
    Alias class: :py:class:`slangpy.slangpy.NativeObject`



----

.. py:class:: slangpy.experimental.gridarg.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.experimental.gridarg.GridArg

    Base class: :py:class:`slangpy.slangpy.NativeObject`


        Passes the thread id as an argument to a SlangPy function.




----

.. py:class:: slangpy.experimental.gridarg.GridArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.diffpair.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.types.diffpair.PrimType
    Alias class: :py:class:`slangpy.core.enums.PrimType`



----

.. py:class:: slangpy.types.diffpair.DiffPair


        A pair of values, one representing the primal value and the other representing the gradient value.
        Typically only required when wanting to output gradients from scalar calls to a function.




----

.. py:class:: slangpy.types.DiffPair
    Alias class: :py:class:`slangpy.types.diffpair.DiffPair`



----

.. py:class:: slangpy.types.helpers.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.types.helpers.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.types.helpers.ArrayType
    Alias class: :py:class:`slangpy.native_refl.ArrayType`



----

.. py:class:: slangpy.types.helpers.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.types.helpers.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.types.helpers.VectorType
    Alias class: :py:class:`slangpy.native_refl.VectorType`



----

.. py:class:: slangpy.types.wanghasharg.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.types.wanghasharg.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.types.wanghasharg.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.wanghasharg.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.types.wanghasharg.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.types.wanghasharg.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.types.wanghasharg.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.types.wanghasharg.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.types.wanghasharg.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.types.wanghasharg.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.types.wanghasharg.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.types.wanghasharg.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.types.wanghasharg.WangHashArg


        Generates a random int/vector per thread when passed as an argument using a wang
        hash of the thread id.




----

.. py:class:: slangpy.types.wanghasharg.WangHashArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.randfloatarg.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.types.randfloatarg.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.types.randfloatarg.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.randfloatarg.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.types.randfloatarg.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.types.randfloatarg.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.types.randfloatarg.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.types.randfloatarg.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.types.randfloatarg.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.types.randfloatarg.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.types.randfloatarg.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.types.randfloatarg.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.types.randfloatarg.RandFloatArg


        Generates a random float/vector per thread when passed as an argument
        to a SlangPy function. The min and max values are inclusive.




----

.. py:class:: slangpy.types.randfloatarg.RandFloatArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.RandFloatArg
    Alias class: :py:class:`slangpy.types.randfloatarg.RandFloatArg`



----

.. py:class:: slangpy.types.threadidarg.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.types.threadidarg.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.threadidarg.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.types.threadidarg.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.types.threadidarg.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.types.threadidarg.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.types.threadidarg.NativeObject
    Alias class: :py:class:`slangpy.slangpy.NativeObject`



----

.. py:class:: slangpy.types.threadidarg.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.types.threadidarg.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.types.threadidarg.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.types.threadidarg.ThreadIdArg

    Base class: :py:class:`slangpy.slangpy.NativeObject`


        Passes the thread id as an argument to a SlangPy function.




----

.. py:class:: slangpy.types.threadidarg.ThreadIdArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.ThreadIdArg
    Alias class: :py:class:`slangpy.types.threadidarg.ThreadIdArg`



----

.. py:class:: slangpy.types.callidarg.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.types.callidarg.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.callidarg.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.types.callidarg.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.types.callidarg.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.types.callidarg.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.types.callidarg.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.types.callidarg.CallIdArg


        Passes the thread id as an argument to a SlangPy function.




----

.. py:class:: slangpy.types.callidarg.CallIdArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.types.CallIdArg
    Alias class: :py:class:`slangpy.types.callidarg.CallIdArg`



----

.. py:class:: slangpy.types.valueref.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.types.valueref.ValueRef


        Minimal class to hold a reference to a scalar value, allowing user to get outputs
        from scalar inout/out arguments.




----

.. py:class:: slangpy.types.ValueRef
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`



----

.. py:class:: slangpy.types.WangHashArg
    Alias class: :py:class:`slangpy.types.wanghasharg.WangHashArg`



----

.. py:class:: slangpy.types.tensor.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.types.tensor.TensorDesc
    Alias class: :py:class:`slangpy.native_func.TensorDesc`



----

.. py:class:: slangpy.types.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.DiffPair
    Alias class: :py:class:`slangpy.types.diffpair.DiffPair`



----

.. py:class:: slangpy.RandFloatArg
    Alias class: :py:class:`slangpy.types.randfloatarg.RandFloatArg`



----

.. py:class:: slangpy.ThreadIdArg
    Alias class: :py:class:`slangpy.types.threadidarg.ThreadIdArg`



----

.. py:class:: slangpy.CallIdArg
    Alias class: :py:class:`slangpy.types.callidarg.CallIdArg`



----

.. py:class:: slangpy.ValueRef
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`



----

.. py:class:: slangpy.WangHashArg
    Alias class: :py:class:`slangpy.types.wanghasharg.WangHashArg`



----

.. py:class:: slangpy.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.builtin.value.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.value.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.value.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.value.NativeValueMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`



----

.. py:function:: slangpy.builtin.value.unpack_arg(arg: object) -> object

    N/A



----

.. py:class:: slangpy.builtin.value.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.builtin.value.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.value.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.value.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.value.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.value.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.value.ValueMarshall

    Base class: :py:class:`slangpy.slangpy.NativeValueMarshall`



----

.. py:class:: slangpy.builtin.value.ScalarMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.value.NoneMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.value.VectorMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.value.MatrixMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.value.vec_type
    Alias class: :py:class:`slangpy.math.float16_t4`



----

.. py:class:: slangpy.builtin.ValueMarshall
    Alias class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.valueref.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.valueref.BufferCursor
    Alias class: :py:class:`slangpy.BufferCursor`



----

.. py:class:: slangpy.builtin.valueref.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.valueref.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.valueref.Buffer
    Alias class: :py:class:`slangpy.Buffer`



----

.. py:class:: slangpy.builtin.valueref.BufferUsage
    Alias class: :py:class:`slangpy.BufferUsage`



----

.. py:class:: slangpy.builtin.valueref.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.valueref.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.valueref.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.valueref.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.valueref.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.valueref.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.builtin.valueref.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.valueref.ValueRef
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`



----

.. py:class:: slangpy.builtin.valueref.ValueRefMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.ValueRefMarshall
    Alias class: :py:class:`slangpy.builtin.valueref.ValueRefMarshall`



----

.. py:class:: slangpy.builtin.diffpair.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.diffpair.PrimType
    Alias class: :py:class:`slangpy.core.enums.PrimType`



----

.. py:class:: slangpy.builtin.diffpair.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.diffpair.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.diffpair.NativeMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`



----

.. py:class:: slangpy.builtin.diffpair.Buffer
    Alias class: :py:class:`slangpy.Buffer`



----

.. py:class:: slangpy.builtin.diffpair.BufferUsage
    Alias class: :py:class:`slangpy.BufferUsage`



----

.. py:class:: slangpy.builtin.diffpair.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.diffpair.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.diffpair.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.diffpair.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.diffpair.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.diffpair.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.diffpair.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.diffpair.DiffPair
    Alias class: :py:class:`slangpy.types.diffpair.DiffPair`



----

.. py:class:: slangpy.builtin.diffpair.DiffPairMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.DiffPairMarshall
    Alias class: :py:class:`slangpy.builtin.diffpair.DiffPairMarshall`



----

.. py:class:: slangpy.builtin.struct.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.struct.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.builtin.struct.NativeMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`



----

.. py:class:: slangpy.builtin.struct.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.struct.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.struct.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.struct.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.struct.UnknownType
    Alias class: :py:class:`slangpy.native_refl.UnknownType`



----

.. py:class:: slangpy.builtin.struct.StructType
    Alias class: :py:class:`slangpy.native_refl.StructType`



----

.. py:class:: slangpy.builtin.struct.InterfaceType
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`



----

.. py:class:: slangpy.builtin.struct.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.struct.ValueMarshall
    Alias class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.struct.StructMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.StructMarshall
    Alias class: :py:class:`slangpy.builtin.struct.StructMarshall`



----

.. py:class:: slangpy.builtin.structuredbuffer.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.structuredbuffer.NativeBufferMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeBufferMarshall`



----

.. py:class:: slangpy.builtin.structuredbuffer.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.structuredbuffer.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.structuredbuffer.StructuredBufferType
    Alias class: :py:class:`slangpy.native_refl.StructuredBufferType`



----

.. py:class:: slangpy.builtin.structuredbuffer.ByteAddressBufferType
    Alias class: :py:class:`slangpy.native_refl.ByteAddressBufferType`



----

.. py:class:: slangpy.builtin.structuredbuffer.PointerType
    Alias class: :py:class:`slangpy.native_refl.PointerType`



----

.. py:class:: slangpy.builtin.structuredbuffer.Buffer
    Alias class: :py:class:`slangpy.Buffer`



----

.. py:class:: slangpy.builtin.structuredbuffer.BufferUsage
    Alias class: :py:class:`slangpy.BufferUsage`



----

.. py:class:: slangpy.builtin.structuredbuffer.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.structuredbuffer.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.structuredbuffer.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.structuredbuffer.BufferMarshall

    Base class: :py:class:`slangpy.slangpy.NativeBufferMarshall`



----

.. py:class:: slangpy.builtin.BufferMarshall
    Alias class: :py:class:`slangpy.builtin.structuredbuffer.BufferMarshall`



----

.. py:class:: slangpy.builtin.descriptor.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.descriptor.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.descriptor.NativeDescriptorMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeDescriptorMarshall`



----

.. py:function:: slangpy.builtin.descriptor.unpack_arg(arg: object) -> object

    N/A



----

.. py:class:: slangpy.builtin.descriptor.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.builtin.descriptor.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.builtin.descriptor.ShaderCursor
    Alias class: :py:class:`slangpy.ShaderCursor`



----

.. py:class:: slangpy.builtin.descriptor.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.descriptor.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.descriptor.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.descriptor.DescriptorHandle
    Alias class: :py:class:`slangpy.DescriptorHandle`



----

.. py:class:: slangpy.builtin.descriptor.DescriptorHandleType
    Alias class: :py:class:`slangpy.DescriptorHandleType`



----

.. py:class:: slangpy.builtin.descriptor.DescriptorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeDescriptorMarshall`



----

.. py:class:: slangpy.builtin.DescriptorMarshall
    Alias class: :py:class:`slangpy.builtin.descriptor.DescriptorMarshall`



----

.. py:class:: slangpy.builtin.texture.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.texture.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.texture.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.texture.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.builtin.texture.NativeTextureMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeTextureMarshall`



----

.. py:class:: slangpy.builtin.texture.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.builtin.texture.FormatType
    Alias class: :py:class:`slangpy.FormatType`



----

.. py:class:: slangpy.builtin.texture.TextureType
    Alias class: :py:class:`slangpy.TextureType`



----

.. py:class:: slangpy.builtin.texture.TextureUsage
    Alias class: :py:class:`slangpy.TextureUsage`



----

.. py:class:: slangpy.builtin.texture.Sampler
    Alias class: :py:class:`slangpy.Sampler`



----

.. py:class:: slangpy.builtin.texture.Texture
    Alias class: :py:class:`slangpy.Texture`



----

.. py:class:: slangpy.builtin.texture.Format
    Alias class: :py:class:`slangpy.Format`



----

.. py:function:: slangpy.builtin.texture.get_format_info(arg: slangpy.Format, /) -> slangpy.FormatInfo



----

.. py:class:: slangpy.builtin.texture.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.texture.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.texture.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.texture.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.texture.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.texture.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.builtin.texture.TextureMarshall

    Base class: :py:class:`slangpy.slangpy.NativeTextureMarshall`



----

.. py:class:: slangpy.builtin.texture.SamplerMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.TextureMarshall
    Alias class: :py:class:`slangpy.builtin.texture.TextureMarshall`



----

.. py:class:: slangpy.builtin.array.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.array.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.builtin.array.ValueMarshall
    Alias class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.array.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.array.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.array.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.array.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.array.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.array.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.array.ShaderCursor
    Alias class: :py:class:`slangpy.ShaderCursor`



----

.. py:class:: slangpy.builtin.array.ShaderObject
    Alias class: :py:class:`slangpy.ShaderObject`



----

.. py:class:: slangpy.builtin.array.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.array.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.array.NativeValueMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`



----

.. py:function:: slangpy.builtin.array.unpack_arg(arg: object) -> object

    N/A



----

.. py:class:: slangpy.builtin.array.ArrayMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`



----

.. py:class:: slangpy.builtin.ArrayMarshall
    Alias class: :py:class:`slangpy.builtin.array.ArrayMarshall`



----

.. py:class:: slangpy.builtin.resourceview.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.resourceview.TextureView
    Alias class: :py:class:`slangpy.TextureView`



----

.. py:class:: slangpy.builtin.resourceview.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.TextureView
    Alias class: :py:class:`slangpy.TextureView`



----

.. py:class:: slangpy.builtin.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.accelerationstructure.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.accelerationstructure.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.accelerationstructure.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.accelerationstructure.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.builtin.accelerationstructure.AccelerationStructure
    Alias class: :py:class:`slangpy.AccelerationStructure`



----

.. py:class:: slangpy.builtin.accelerationstructure.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.accelerationstructure.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.accelerationstructure.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.accelerationstructure.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.accelerationstructure.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.accelerationstructure.AccelerationStructureMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.AccelerationStructureMarshall
    Alias class: :py:class:`slangpy.builtin.accelerationstructure.AccelerationStructureMarshall`



----

.. py:class:: slangpy.builtin.range.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.range.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.range.CallContext
    Alias class: :py:class:`slangpy.slangpy.CallContext`



----

.. py:class:: slangpy.builtin.range.Shape
    Alias class: :py:class:`slangpy.slangpy.Shape`



----

.. py:class:: slangpy.builtin.range.TypeReflection
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.builtin.range.Marshall
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.range.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.range.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.range.BoundVariableRuntime
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`



----

.. py:class:: slangpy.builtin.range.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.range.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.range.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.range.TR
    Alias class: :py:class:`slangpy.TypeReflection`



----

.. py:class:: slangpy.builtin.range.RangeMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`



----

.. py:class:: slangpy.builtin.RangeMarshall
    Alias class: :py:class:`slangpy.builtin.range.RangeMarshall`



----

.. py:class:: slangpy.builtin.tensorcommon.Protocol
    Alias class: :py:class:`typing.Protocol`



----

.. py:class:: slangpy.builtin.tensorcommon.DeviceType
    Alias class: :py:class:`slangpy.DeviceType`



----

.. py:class:: slangpy.builtin.tensorcommon.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.tensorcommon.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.tensorcommon.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.tensorcommon.CallMode
    Alias class: :py:class:`slangpy.slangpy.CallMode`



----

.. py:class:: slangpy.builtin.tensorcommon.AccessType
    Alias class: :py:class:`slangpy.slangpy.AccessType`



----

.. py:class:: slangpy.builtin.tensorcommon.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.tensorcommon.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.builtin.tensorcommon.ITensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType`



----

.. py:class:: slangpy.builtin.tensorcommon.TensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`



----

.. py:class:: slangpy.builtin.tensorcommon.TensorViewType
    Alias class: :py:class:`slangpy.native_refl.TensorViewType`



----

.. py:class:: slangpy.builtin.tensorcommon.DiffTensorViewType
    Alias class: :py:class:`slangpy.native_refl.DiffTensorViewType`



----

.. py:class:: slangpy.builtin.tensorcommon.ArrayType
    Alias class: :py:class:`slangpy.native_refl.ArrayType`



----

.. py:class:: slangpy.builtin.tensorcommon.InterfaceType
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`



----

.. py:class:: slangpy.builtin.tensorcommon.UnknownType
    Alias class: :py:class:`slangpy.native_refl.UnknownType`



----

.. py:class:: slangpy.builtin.tensorcommon.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.tensorcommon.ResourceType
    Alias class: :py:class:`slangpy.native_refl.ResourceType`



----

.. py:class:: slangpy.builtin.tensorcommon.TensorAccess
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`



----

.. py:class:: slangpy.builtin.tensorcommon.VectorType
    Alias class: :py:class:`slangpy.native_refl.VectorType`



----

.. py:class:: slangpy.builtin.tensorcommon.ITensorMarshall

    Base class: :py:class:`typing.Protocol`


        Protocol for type marshalling of any container that behaves as a tensor.




----

.. py:class:: slangpy.builtin.tensor.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.tensor.ShaderObject
    Alias class: :py:class:`slangpy.ShaderObject`



----

.. py:class:: slangpy.builtin.tensor.ShaderCursor
    Alias class: :py:class:`slangpy.ShaderCursor`



----

.. py:class:: slangpy.builtin.tensor.BufferUsage
    Alias class: :py:class:`slangpy.BufferUsage`



----

.. py:class:: slangpy.builtin.tensor.TensorMarshallBase
    Alias class: :py:class:`slangpy.slangpy.TensorMarshall`



----

.. py:class:: slangpy.builtin.tensor.Tensor
    Alias class: :py:class:`slangpy.native_func.Tensor`



----

.. py:class:: slangpy.builtin.tensor.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.tensor.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.tensor.ArrayType
    Alias class: :py:class:`slangpy.native_refl.ArrayType`



----

.. py:class:: slangpy.builtin.tensor.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.builtin.tensor.VectorType
    Alias class: :py:class:`slangpy.native_refl.VectorType`



----

.. py:class:: slangpy.builtin.tensor.MatrixType
    Alias class: :py:class:`slangpy.native_refl.MatrixType`



----

.. py:class:: slangpy.builtin.tensor.TensorType
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`



----

.. py:class:: slangpy.builtin.tensor.TensorAccess
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`



----

.. py:class:: slangpy.builtin.tensor.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.tensor.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.tensor.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.tensor.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.builtin.tensor.TensorMarshall

    Base class: :py:class:`slangpy.slangpy.TensorMarshall`



----

.. py:class:: slangpy.builtin.numpy.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.builtin.numpy.BoundVariable
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`



----

.. py:class:: slangpy.builtin.numpy.CodeGenBlock
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`



----

.. py:class:: slangpy.builtin.numpy.BindContext
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`



----

.. py:class:: slangpy.builtin.numpy.ReturnContext
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`



----

.. py:class:: slangpy.builtin.numpy.NativeNumpyMarshall
    Alias class: :py:class:`slangpy.slangpy.NativeNumpyMarshall`



----

.. py:class:: slangpy.builtin.numpy.SlangProgramLayout
    Alias class: :py:class:`slangpy.native_refl.Layout`



----

.. py:class:: slangpy.builtin.numpy.ScalarType
    Alias class: :py:class:`slangpy.native_refl.ScalarType`



----

.. py:class:: slangpy.builtin.numpy.SlangType
    Alias class: :py:class:`slangpy.native_refl.Type`



----

.. py:class:: slangpy.builtin.numpy.VectorType
    Alias class: :py:class:`slangpy.native_refl.VectorType`



----

.. py:class:: slangpy.builtin.numpy.MatrixType
    Alias class: :py:class:`slangpy.native_refl.MatrixType`



----

.. py:class:: slangpy.builtin.numpy.NumpyMarshall

    Base class: :py:class:`slangpy.slangpy.NativeNumpyMarshall`



----

.. py:class:: slangpy.builtin.NumpyMarshall
    Alias class: :py:class:`slangpy.builtin.numpy.NumpyMarshall`



----

.. py:class:: slangpy.builtin.TensorMarshall
    Alias class: :py:class:`slangpy.builtin.tensor.TensorMarshall`



----

.. py:class:: slangpy.Function
    Alias class: :py:class:`slangpy.core.function.Function`



----

.. py:class:: slangpy.Struct
    Alias class: :py:class:`slangpy.core.struct.Struct`



----

.. py:class:: slangpy.Module
    Alias class: :py:class:`slangpy.core.module.Module`



----

.. py:class:: slangpy.InstanceList
    Alias class: :py:class:`slangpy.core.instance.InstanceList`



----

.. py:class:: slangpy.InstanceTensor
    Alias class: :py:class:`slangpy.core.instance.InstanceTensor`



----

.. py:class:: slangpy.torchintegration.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.torchintegration.NativeTorchTensorDiffPair
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorDiffPair`



----

.. py:class:: slangpy.Any
    Alias class: :py:class:`typing.Any`



----

.. py:class:: slangpy.NativeTorchTensorDiffPair
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorDiffPair`
