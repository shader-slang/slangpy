Core
----

.. py:class:: slangpy.DataType

    Base class: :py:class:`enum.Enum`
    
    
    
    .. py:attribute:: slangpy.DataType.void
        :type: DataType
        :value: DataType.void
    
    .. py:attribute:: slangpy.DataType.bool
        :type: DataType
        :value: DataType.bool
    
    .. py:attribute:: slangpy.DataType.int8
        :type: DataType
        :value: DataType.int8
    
    .. py:attribute:: slangpy.DataType.int16
        :type: DataType
        :value: DataType.int16
    
    .. py:attribute:: slangpy.DataType.int32
        :type: DataType
        :value: DataType.int32
    
    .. py:attribute:: slangpy.DataType.int64
        :type: DataType
        :value: DataType.int64
    
    .. py:attribute:: slangpy.DataType.uint8
        :type: DataType
        :value: DataType.uint8
    
    .. py:attribute:: slangpy.DataType.uint16
        :type: DataType
        :value: DataType.uint16
    
    .. py:attribute:: slangpy.DataType.uint32
        :type: DataType
        :value: DataType.uint32
    
    .. py:attribute:: slangpy.DataType.uint64
        :type: DataType
        :value: DataType.uint64
    
    .. py:attribute:: slangpy.DataType.float16
        :type: DataType
        :value: DataType.float16
    
    .. py:attribute:: slangpy.DataType.float32
        :type: DataType
        :value: DataType.float32
    
    .. py:attribute:: slangpy.DataType.float64
        :type: DataType
        :value: DataType.float64
    
    .. py:attribute:: slangpy.DataType.float8_e4m3
        :type: DataType
        :value: DataType.float8_e4m3
    
    .. py:attribute:: slangpy.DataType.float8_e5m2
        :type: DataType
        :value: DataType.float8_e5m2
    


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
        
        
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.y
            :type: PixelFormat
            :value: PixelFormat.y
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.ya
            :type: PixelFormat
            :value: PixelFormat.ya
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.r
            :type: PixelFormat
            :value: PixelFormat.r
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.rg
            :type: PixelFormat
            :value: PixelFormat.rg
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.rgb
            :type: PixelFormat
            :value: PixelFormat.rgb
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.rgba
            :type: PixelFormat
            :value: PixelFormat.rgba
        
        .. py:attribute:: slangpy.Bitmap.PixelFormat.multi_channel
            :type: PixelFormat
            :value: PixelFormat.multi_channel
        
    .. py:class:: slangpy.Bitmap.ComponentType
        :canonical: slangpy.DataStruct.Type
        
        Alias class: :py:class:`slangpy.DataStruct.Type`
        
    .. py:class:: slangpy.Bitmap.FileFormat
    
        Base class: :py:class:`enum.Enum`
        
        
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.unknown
            :type: FileFormat
            :value: FileFormat.unknown
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.auto
            :type: FileFormat
            :value: FileFormat.auto
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.png
            :type: FileFormat
            :value: FileFormat.png
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.jpg
            :type: FileFormat
            :value: FileFormat.jpg
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.bmp
            :type: FileFormat
            :value: FileFormat.bmp
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.tga
            :type: FileFormat
            :value: FileFormat.tga
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.hdr
            :type: FileFormat
            :value: FileFormat.hdr
        
        .. py:attribute:: slangpy.Bitmap.FileFormat.exr
            :type: FileFormat
            :value: FileFormat.exr
        
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
    
    .. py:method:: resample(self, target: slangpy.Bitmap, filter: slangpy.BoxFilter | slangpy.TentFilter | slangpy.GaussianFilter | slangpy.MitchellFilter | slangpy.LanczosFilter = <slangpy.BoxFilter object at 0x0000021D5F434A50>, bc: tuple[slangpy.FilterBoundaryCondition, slangpy.FilterBoundaryCondition] = (FilterBoundaryCondition.clamp, FilterBoundaryCondition.clamp), clamp: tuple[float, float] = (-inf, inf)) -> None
    
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
        
    .. py:method:: resample(self, width: int, height: int, filter: slangpy.BoxFilter | slangpy.TentFilter | slangpy.GaussianFilter | slangpy.MitchellFilter | slangpy.LanczosFilter = <slangpy.BoxFilter object at 0x0000021D5F434AB0>, bc: tuple[slangpy.FilterBoundaryCondition, slangpy.FilterBoundaryCondition] = (FilterBoundaryCondition.clamp, FilterBoundaryCondition.clamp), clamp: tuple[float, float] = (-inf, inf)) -> slangpy.Bitmap
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
        
        .. py:attribute:: slangpy.DataStruct.Type.int8
            :type: Type
            :value: Type.int8
        
        .. py:attribute:: slangpy.DataStruct.Type.int16
            :type: Type
            :value: Type.int16
        
        .. py:attribute:: slangpy.DataStruct.Type.int32
            :type: Type
            :value: Type.int32
        
        .. py:attribute:: slangpy.DataStruct.Type.int64
            :type: Type
            :value: Type.int64
        
        .. py:attribute:: slangpy.DataStruct.Type.uint8
            :type: Type
            :value: Type.uint8
        
        .. py:attribute:: slangpy.DataStruct.Type.uint16
            :type: Type
            :value: Type.uint16
        
        .. py:attribute:: slangpy.DataStruct.Type.uint32
            :type: Type
            :value: Type.uint32
        
        .. py:attribute:: slangpy.DataStruct.Type.uint64
            :type: Type
            :value: Type.uint64
        
        .. py:attribute:: slangpy.DataStruct.Type.float16
            :type: Type
            :value: Type.float16
        
        .. py:attribute:: slangpy.DataStruct.Type.float32
            :type: Type
            :value: Type.float32
        
        .. py:attribute:: slangpy.DataStruct.Type.float64
            :type: Type
            :value: Type.float64
        
    .. py:class:: slangpy.DataStruct.Flags
    
        Base class: :py:class:`enum.IntFlag`
        
        Struct field flags.
        
        .. py:attribute:: slangpy.DataStruct.Flags.none
            :type: Flags
            :value: 0
        
        .. py:attribute:: slangpy.DataStruct.Flags.normalized
            :type: Flags
            :value: 1
        
        .. py:attribute:: slangpy.DataStruct.Flags.srgb_gamma
            :type: Flags
            :value: 2
        
        .. py:attribute:: slangpy.DataStruct.Flags.default
            :type: Flags
            :value: 4
        
    .. py:class:: slangpy.DataStruct.ByteOrder
    
        Base class: :py:class:`enum.Enum`
        
        Byte order.
        
        .. py:attribute:: slangpy.DataStruct.ByteOrder.little_endian
            :type: ByteOrder
            :value: ByteOrder.little_endian
        
        .. py:attribute:: slangpy.DataStruct.ByteOrder.big_endian
            :type: ByteOrder
            :value: ByteOrder.big_endian
        
        .. py:attribute:: slangpy.DataStruct.ByteOrder.host
            :type: ByteOrder
            :value: ByteOrder.host
        
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

.. py:data:: slangpy.SGL_BUILD_TYPE
    :type: str
    :value: "Debug"



----

.. py:data:: slangpy.SGL_GIT_VERSION
    :type: str
    :value: "commit: 09ed09e3 / branch: dev/skallweit/shader-profile (local changes)"



----

.. py:data:: slangpy.SGL_VERSION_MAJOR
    :type: int
    :value: 0



----

.. py:data:: slangpy.SGL_VERSION_MINOR
    :type: int
    :value: 41



----

.. py:data:: slangpy.SGL_VERSION_PATCH
    :type: int
    :value: 0



----

.. py:data:: slangpy.SGL_VERSION
    :type: str
    :value: "0.41.0"



----

Logging
-------

.. py:class:: slangpy.LogLevel

    Base class: :py:class:`enum.IntEnum`
    
    Log level.
    
    .. py:attribute:: slangpy.LogLevel.none
        :type: LogLevel
        :value: LogLevel.none
    
    .. py:attribute:: slangpy.LogLevel.debug
        :type: LogLevel
        :value: LogLevel.debug
    
    .. py:attribute:: slangpy.LogLevel.info
        :type: LogLevel
        :value: LogLevel.info
    
    .. py:attribute:: slangpy.LogLevel.warn
        :type: LogLevel
        :value: LogLevel.warn
    
    .. py:attribute:: slangpy.LogLevel.error
        :type: LogLevel
        :value: LogLevel.error
    
    .. py:attribute:: slangpy.LogLevel.fatal
        :type: LogLevel
        :value: LogLevel.fatal
    


----

.. py:class:: slangpy.LogFrequency

    Base class: :py:class:`enum.Enum`
    
    Log frequency.
    
    .. py:attribute:: slangpy.LogFrequency.always
        :type: LogFrequency
        :value: LogFrequency.always
    
    .. py:attribute:: slangpy.LogFrequency.once
        :type: LogFrequency
        :value: LogFrequency.once
    


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
    
    .. py:attribute:: slangpy.CursorMode.normal
        :type: CursorMode
        :value: CursorMode.normal
    
    .. py:attribute:: slangpy.CursorMode.hidden
        :type: CursorMode
        :value: CursorMode.hidden
    
    .. py:attribute:: slangpy.CursorMode.disabled
        :type: CursorMode
        :value: CursorMode.disabled
    


----

.. py:class:: slangpy.CursorShape

    Base class: :py:class:`enum.Enum`
    
    Mouse cursor shapes.
    
    .. py:attribute:: slangpy.CursorShape.arrow
        :type: CursorShape
        :value: CursorShape.arrow
    
    .. py:attribute:: slangpy.CursorShape.ibeam
        :type: CursorShape
        :value: CursorShape.ibeam
    
    .. py:attribute:: slangpy.CursorShape.crosshair
        :type: CursorShape
        :value: CursorShape.crosshair
    
    .. py:attribute:: slangpy.CursorShape.hand
        :type: CursorShape
        :value: CursorShape.hand
    
    .. py:attribute:: slangpy.CursorShape.hresize
        :type: CursorShape
        :value: CursorShape.hresize
    
    .. py:attribute:: slangpy.CursorShape.vresize
        :type: CursorShape
        :value: CursorShape.vresize
    


----

.. py:class:: slangpy.WindowMode

    Base class: :py:class:`enum.Enum`
    
    Window modes.
    
    .. py:attribute:: slangpy.WindowMode.normal
        :type: WindowMode
        :value: WindowMode.normal
    
    .. py:attribute:: slangpy.WindowMode.minimized
        :type: WindowMode
        :value: WindowMode.minimized
    
    .. py:attribute:: slangpy.WindowMode.fullscreen
        :type: WindowMode
        :value: WindowMode.fullscreen
    


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
    
    .. py:attribute:: slangpy.MouseButton.left
        :type: MouseButton
        :value: MouseButton.left
    
    .. py:attribute:: slangpy.MouseButton.middle
        :type: MouseButton
        :value: MouseButton.middle
    
    .. py:attribute:: slangpy.MouseButton.right
        :type: MouseButton
        :value: MouseButton.right
    
    .. py:attribute:: slangpy.MouseButton.unknown
        :type: MouseButton
        :value: MouseButton.unknown
    


----

.. py:class:: slangpy.KeyModifierFlags

    Base class: :py:class:`enum.IntFlag`
    
    Keyboard modifier flags.
    
    .. py:attribute:: slangpy.KeyModifierFlags.none
        :type: KeyModifierFlags
        :value: 0
    
    .. py:attribute:: slangpy.KeyModifierFlags.shift
        :type: KeyModifierFlags
        :value: 1
    
    .. py:attribute:: slangpy.KeyModifierFlags.ctrl
        :type: KeyModifierFlags
        :value: 2
    
    .. py:attribute:: slangpy.KeyModifierFlags.alt
        :type: KeyModifierFlags
        :value: 4
    


----

.. py:class:: slangpy.KeyModifier

    Base class: :py:class:`enum.Enum`
    
    Keyboard modifiers.
    
    .. py:attribute:: slangpy.KeyModifier.shift
        :type: KeyModifier
        :value: KeyModifier.shift
    
    .. py:attribute:: slangpy.KeyModifier.ctrl
        :type: KeyModifier
        :value: KeyModifier.ctrl
    
    .. py:attribute:: slangpy.KeyModifier.alt
        :type: KeyModifier
        :value: KeyModifier.alt
    


----

.. py:class:: slangpy.KeyCode

    Base class: :py:class:`enum.Enum`
    
    Keyboard key codes.
    
    .. py:attribute:: slangpy.KeyCode.space
        :type: KeyCode
        :value: KeyCode.space
    
    .. py:attribute:: slangpy.KeyCode.apostrophe
        :type: KeyCode
        :value: KeyCode.apostrophe
    
    .. py:attribute:: slangpy.KeyCode.comma
        :type: KeyCode
        :value: KeyCode.comma
    
    .. py:attribute:: slangpy.KeyCode.minus
        :type: KeyCode
        :value: KeyCode.minus
    
    .. py:attribute:: slangpy.KeyCode.period
        :type: KeyCode
        :value: KeyCode.period
    
    .. py:attribute:: slangpy.KeyCode.slash
        :type: KeyCode
        :value: KeyCode.slash
    
    .. py:attribute:: slangpy.KeyCode.key0
        :type: KeyCode
        :value: KeyCode.key0
    
    .. py:attribute:: slangpy.KeyCode.key1
        :type: KeyCode
        :value: KeyCode.key1
    
    .. py:attribute:: slangpy.KeyCode.key2
        :type: KeyCode
        :value: KeyCode.key2
    
    .. py:attribute:: slangpy.KeyCode.key3
        :type: KeyCode
        :value: KeyCode.key3
    
    .. py:attribute:: slangpy.KeyCode.key4
        :type: KeyCode
        :value: KeyCode.key4
    
    .. py:attribute:: slangpy.KeyCode.key5
        :type: KeyCode
        :value: KeyCode.key5
    
    .. py:attribute:: slangpy.KeyCode.key6
        :type: KeyCode
        :value: KeyCode.key6
    
    .. py:attribute:: slangpy.KeyCode.key7
        :type: KeyCode
        :value: KeyCode.key7
    
    .. py:attribute:: slangpy.KeyCode.key8
        :type: KeyCode
        :value: KeyCode.key8
    
    .. py:attribute:: slangpy.KeyCode.key9
        :type: KeyCode
        :value: KeyCode.key9
    
    .. py:attribute:: slangpy.KeyCode.semicolon
        :type: KeyCode
        :value: KeyCode.semicolon
    
    .. py:attribute:: slangpy.KeyCode.equal
        :type: KeyCode
        :value: KeyCode.equal
    
    .. py:attribute:: slangpy.KeyCode.a
        :type: KeyCode
        :value: KeyCode.a
    
    .. py:attribute:: slangpy.KeyCode.b
        :type: KeyCode
        :value: KeyCode.b
    
    .. py:attribute:: slangpy.KeyCode.c
        :type: KeyCode
        :value: KeyCode.c
    
    .. py:attribute:: slangpy.KeyCode.d
        :type: KeyCode
        :value: KeyCode.d
    
    .. py:attribute:: slangpy.KeyCode.e
        :type: KeyCode
        :value: KeyCode.e
    
    .. py:attribute:: slangpy.KeyCode.f
        :type: KeyCode
        :value: KeyCode.f
    
    .. py:attribute:: slangpy.KeyCode.g
        :type: KeyCode
        :value: KeyCode.g
    
    .. py:attribute:: slangpy.KeyCode.h
        :type: KeyCode
        :value: KeyCode.h
    
    .. py:attribute:: slangpy.KeyCode.i
        :type: KeyCode
        :value: KeyCode.i
    
    .. py:attribute:: slangpy.KeyCode.j
        :type: KeyCode
        :value: KeyCode.j
    
    .. py:attribute:: slangpy.KeyCode.k
        :type: KeyCode
        :value: KeyCode.k
    
    .. py:attribute:: slangpy.KeyCode.l
        :type: KeyCode
        :value: KeyCode.l
    
    .. py:attribute:: slangpy.KeyCode.m
        :type: KeyCode
        :value: KeyCode.m
    
    .. py:attribute:: slangpy.KeyCode.n
        :type: KeyCode
        :value: KeyCode.n
    
    .. py:attribute:: slangpy.KeyCode.o
        :type: KeyCode
        :value: KeyCode.o
    
    .. py:attribute:: slangpy.KeyCode.p
        :type: KeyCode
        :value: KeyCode.p
    
    .. py:attribute:: slangpy.KeyCode.q
        :type: KeyCode
        :value: KeyCode.q
    
    .. py:attribute:: slangpy.KeyCode.r
        :type: KeyCode
        :value: KeyCode.r
    
    .. py:attribute:: slangpy.KeyCode.s
        :type: KeyCode
        :value: KeyCode.s
    
    .. py:attribute:: slangpy.KeyCode.t
        :type: KeyCode
        :value: KeyCode.t
    
    .. py:attribute:: slangpy.KeyCode.u
        :type: KeyCode
        :value: KeyCode.u
    
    .. py:attribute:: slangpy.KeyCode.v
        :type: KeyCode
        :value: KeyCode.v
    
    .. py:attribute:: slangpy.KeyCode.w
        :type: KeyCode
        :value: KeyCode.w
    
    .. py:attribute:: slangpy.KeyCode.x
        :type: KeyCode
        :value: KeyCode.x
    
    .. py:attribute:: slangpy.KeyCode.y
        :type: KeyCode
        :value: KeyCode.y
    
    .. py:attribute:: slangpy.KeyCode.z
        :type: KeyCode
        :value: KeyCode.z
    
    .. py:attribute:: slangpy.KeyCode.left_bracket
        :type: KeyCode
        :value: KeyCode.left_bracket
    
    .. py:attribute:: slangpy.KeyCode.backslash
        :type: KeyCode
        :value: KeyCode.backslash
    
    .. py:attribute:: slangpy.KeyCode.right_bracket
        :type: KeyCode
        :value: KeyCode.right_bracket
    
    .. py:attribute:: slangpy.KeyCode.grave_accent
        :type: KeyCode
        :value: KeyCode.grave_accent
    
    .. py:attribute:: slangpy.KeyCode.world_1
        :type: KeyCode
        :value: KeyCode.world_1
    
    .. py:attribute:: slangpy.KeyCode.world_2
        :type: KeyCode
        :value: KeyCode.world_2
    
    .. py:attribute:: slangpy.KeyCode.escape
        :type: KeyCode
        :value: KeyCode.escape
    
    .. py:attribute:: slangpy.KeyCode.tab
        :type: KeyCode
        :value: KeyCode.tab
    
    .. py:attribute:: slangpy.KeyCode.enter
        :type: KeyCode
        :value: KeyCode.enter
    
    .. py:attribute:: slangpy.KeyCode.backspace
        :type: KeyCode
        :value: KeyCode.backspace
    
    .. py:attribute:: slangpy.KeyCode.insert
        :type: KeyCode
        :value: KeyCode.insert
    
    .. py:attribute:: slangpy.KeyCode.delete
        :type: KeyCode
        :value: KeyCode.delete
    
    .. py:attribute:: slangpy.KeyCode.right
        :type: KeyCode
        :value: KeyCode.right
    
    .. py:attribute:: slangpy.KeyCode.left
        :type: KeyCode
        :value: KeyCode.left
    
    .. py:attribute:: slangpy.KeyCode.down
        :type: KeyCode
        :value: KeyCode.down
    
    .. py:attribute:: slangpy.KeyCode.up
        :type: KeyCode
        :value: KeyCode.up
    
    .. py:attribute:: slangpy.KeyCode.page_up
        :type: KeyCode
        :value: KeyCode.page_up
    
    .. py:attribute:: slangpy.KeyCode.page_down
        :type: KeyCode
        :value: KeyCode.page_down
    
    .. py:attribute:: slangpy.KeyCode.home
        :type: KeyCode
        :value: KeyCode.home
    
    .. py:attribute:: slangpy.KeyCode.end
        :type: KeyCode
        :value: KeyCode.end
    
    .. py:attribute:: slangpy.KeyCode.caps_lock
        :type: KeyCode
        :value: KeyCode.caps_lock
    
    .. py:attribute:: slangpy.KeyCode.scroll_lock
        :type: KeyCode
        :value: KeyCode.scroll_lock
    
    .. py:attribute:: slangpy.KeyCode.num_lock
        :type: KeyCode
        :value: KeyCode.num_lock
    
    .. py:attribute:: slangpy.KeyCode.print_screen
        :type: KeyCode
        :value: KeyCode.print_screen
    
    .. py:attribute:: slangpy.KeyCode.pause
        :type: KeyCode
        :value: KeyCode.pause
    
    .. py:attribute:: slangpy.KeyCode.f1
        :type: KeyCode
        :value: KeyCode.f1
    
    .. py:attribute:: slangpy.KeyCode.f2
        :type: KeyCode
        :value: KeyCode.f2
    
    .. py:attribute:: slangpy.KeyCode.f3
        :type: KeyCode
        :value: KeyCode.f3
    
    .. py:attribute:: slangpy.KeyCode.f4
        :type: KeyCode
        :value: KeyCode.f4
    
    .. py:attribute:: slangpy.KeyCode.f5
        :type: KeyCode
        :value: KeyCode.f5
    
    .. py:attribute:: slangpy.KeyCode.f6
        :type: KeyCode
        :value: KeyCode.f6
    
    .. py:attribute:: slangpy.KeyCode.f7
        :type: KeyCode
        :value: KeyCode.f7
    
    .. py:attribute:: slangpy.KeyCode.f8
        :type: KeyCode
        :value: KeyCode.f8
    
    .. py:attribute:: slangpy.KeyCode.f9
        :type: KeyCode
        :value: KeyCode.f9
    
    .. py:attribute:: slangpy.KeyCode.f10
        :type: KeyCode
        :value: KeyCode.f10
    
    .. py:attribute:: slangpy.KeyCode.f11
        :type: KeyCode
        :value: KeyCode.f11
    
    .. py:attribute:: slangpy.KeyCode.f12
        :type: KeyCode
        :value: KeyCode.f12
    
    .. py:attribute:: slangpy.KeyCode.keypad0
        :type: KeyCode
        :value: KeyCode.keypad0
    
    .. py:attribute:: slangpy.KeyCode.keypad1
        :type: KeyCode
        :value: KeyCode.keypad1
    
    .. py:attribute:: slangpy.KeyCode.keypad2
        :type: KeyCode
        :value: KeyCode.keypad2
    
    .. py:attribute:: slangpy.KeyCode.keypad3
        :type: KeyCode
        :value: KeyCode.keypad3
    
    .. py:attribute:: slangpy.KeyCode.keypad4
        :type: KeyCode
        :value: KeyCode.keypad4
    
    .. py:attribute:: slangpy.KeyCode.keypad5
        :type: KeyCode
        :value: KeyCode.keypad5
    
    .. py:attribute:: slangpy.KeyCode.keypad6
        :type: KeyCode
        :value: KeyCode.keypad6
    
    .. py:attribute:: slangpy.KeyCode.keypad7
        :type: KeyCode
        :value: KeyCode.keypad7
    
    .. py:attribute:: slangpy.KeyCode.keypad8
        :type: KeyCode
        :value: KeyCode.keypad8
    
    .. py:attribute:: slangpy.KeyCode.keypad9
        :type: KeyCode
        :value: KeyCode.keypad9
    
    .. py:attribute:: slangpy.KeyCode.keypad_del
        :type: KeyCode
        :value: KeyCode.keypad_del
    
    .. py:attribute:: slangpy.KeyCode.keypad_divide
        :type: KeyCode
        :value: KeyCode.keypad_divide
    
    .. py:attribute:: slangpy.KeyCode.keypad_multiply
        :type: KeyCode
        :value: KeyCode.keypad_multiply
    
    .. py:attribute:: slangpy.KeyCode.keypad_subtract
        :type: KeyCode
        :value: KeyCode.keypad_subtract
    
    .. py:attribute:: slangpy.KeyCode.keypad_add
        :type: KeyCode
        :value: KeyCode.keypad_add
    
    .. py:attribute:: slangpy.KeyCode.keypad_enter
        :type: KeyCode
        :value: KeyCode.keypad_enter
    
    .. py:attribute:: slangpy.KeyCode.keypad_equal
        :type: KeyCode
        :value: KeyCode.keypad_equal
    
    .. py:attribute:: slangpy.KeyCode.left_shift
        :type: KeyCode
        :value: KeyCode.left_shift
    
    .. py:attribute:: slangpy.KeyCode.left_control
        :type: KeyCode
        :value: KeyCode.left_control
    
    .. py:attribute:: slangpy.KeyCode.left_alt
        :type: KeyCode
        :value: KeyCode.left_alt
    
    .. py:attribute:: slangpy.KeyCode.left_super
        :type: KeyCode
        :value: KeyCode.left_super
    
    .. py:attribute:: slangpy.KeyCode.right_shift
        :type: KeyCode
        :value: KeyCode.right_shift
    
    .. py:attribute:: slangpy.KeyCode.right_control
        :type: KeyCode
        :value: KeyCode.right_control
    
    .. py:attribute:: slangpy.KeyCode.right_alt
        :type: KeyCode
        :value: KeyCode.right_alt
    
    .. py:attribute:: slangpy.KeyCode.right_super
        :type: KeyCode
        :value: KeyCode.right_super
    
    .. py:attribute:: slangpy.KeyCode.menu
        :type: KeyCode
        :value: KeyCode.menu
    
    .. py:attribute:: slangpy.KeyCode.unknown
        :type: KeyCode
        :value: KeyCode.unknown
    


----

.. py:class:: slangpy.KeyboardEventType

    Base class: :py:class:`enum.Enum`
    
    Keyboard event types.
    
    .. py:attribute:: slangpy.KeyboardEventType.key_press
        :type: KeyboardEventType
        :value: KeyboardEventType.key_press
    
    .. py:attribute:: slangpy.KeyboardEventType.key_release
        :type: KeyboardEventType
        :value: KeyboardEventType.key_release
    
    .. py:attribute:: slangpy.KeyboardEventType.key_repeat
        :type: KeyboardEventType
        :value: KeyboardEventType.key_repeat
    
    .. py:attribute:: slangpy.KeyboardEventType.input
        :type: KeyboardEventType
        :value: KeyboardEventType.input
    


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
    
    .. py:attribute:: slangpy.MouseEventType.button_down
        :type: MouseEventType
        :value: MouseEventType.button_down
    
    .. py:attribute:: slangpy.MouseEventType.button_up
        :type: MouseEventType
        :value: MouseEventType.button_up
    
    .. py:attribute:: slangpy.MouseEventType.move
        :type: MouseEventType
        :value: MouseEventType.move
    
    .. py:attribute:: slangpy.MouseEventType.scroll
        :type: MouseEventType
        :value: MouseEventType.scroll
    


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
    
    .. py:attribute:: slangpy.GamepadEventType.button_down
        :type: GamepadEventType
        :value: GamepadEventType.button_down
    
    .. py:attribute:: slangpy.GamepadEventType.button_up
        :type: GamepadEventType
        :value: GamepadEventType.button_up
    
    .. py:attribute:: slangpy.GamepadEventType.connect
        :type: GamepadEventType
        :value: GamepadEventType.connect
    
    .. py:attribute:: slangpy.GamepadEventType.disconnect
        :type: GamepadEventType
        :value: GamepadEventType.disconnect
    


----

.. py:class:: slangpy.GamepadButton

    Base class: :py:class:`enum.Enum`
    
    Gamepad buttons.
    
    .. py:attribute:: slangpy.GamepadButton.unknown
        :type: GamepadButton
        :value: GamepadButton.unknown
    
    .. py:attribute:: slangpy.GamepadButton.a
        :type: GamepadButton
        :value: GamepadButton.a
    
    .. py:attribute:: slangpy.GamepadButton.b
        :type: GamepadButton
        :value: GamepadButton.b
    
    .. py:attribute:: slangpy.GamepadButton.x
        :type: GamepadButton
        :value: GamepadButton.x
    
    .. py:attribute:: slangpy.GamepadButton.y
        :type: GamepadButton
        :value: GamepadButton.y
    
    .. py:attribute:: slangpy.GamepadButton.left_bumper
        :type: GamepadButton
        :value: GamepadButton.left_bumper
    
    .. py:attribute:: slangpy.GamepadButton.right_bumper
        :type: GamepadButton
        :value: GamepadButton.right_bumper
    
    .. py:attribute:: slangpy.GamepadButton.back
        :type: GamepadButton
        :value: GamepadButton.back
    
    .. py:attribute:: slangpy.GamepadButton.start
        :type: GamepadButton
        :value: GamepadButton.start
    
    .. py:attribute:: slangpy.GamepadButton.guide
        :type: GamepadButton
        :value: GamepadButton.guide
    
    .. py:attribute:: slangpy.GamepadButton.left_thumb
        :type: GamepadButton
        :value: GamepadButton.left_thumb
    
    .. py:attribute:: slangpy.GamepadButton.right_thumb
        :type: GamepadButton
        :value: GamepadButton.right_thumb
    
    .. py:attribute:: slangpy.GamepadButton.up
        :type: GamepadButton
        :value: GamepadButton.up
    
    .. py:attribute:: slangpy.GamepadButton.right
        :type: GamepadButton
        :value: GamepadButton.right
    
    .. py:attribute:: slangpy.GamepadButton.down
        :type: GamepadButton
        :value: GamepadButton.down
    
    .. py:attribute:: slangpy.GamepadButton.left
        :type: GamepadButton
        :value: GamepadButton.left
    


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

.. py:data:: slangpy.platform.page_size
    :type: int
    :value: 65536



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
    
    .. py:attribute:: slangpy.AccelerationStructureBuildFlags.none
        :type: AccelerationStructureBuildFlags
        :value: 0
    
    .. py:attribute:: slangpy.AccelerationStructureBuildFlags.allow_update
        :type: AccelerationStructureBuildFlags
        :value: 1
    
    .. py:attribute:: slangpy.AccelerationStructureBuildFlags.allow_compaction
        :type: AccelerationStructureBuildFlags
        :value: 2
    
    .. py:attribute:: slangpy.AccelerationStructureBuildFlags.prefer_fast_trace
        :type: AccelerationStructureBuildFlags
        :value: 4
    
    .. py:attribute:: slangpy.AccelerationStructureBuildFlags.prefer_fast_build
        :type: AccelerationStructureBuildFlags
        :value: 8
    
    .. py:attribute:: slangpy.AccelerationStructureBuildFlags.minimize_memory
        :type: AccelerationStructureBuildFlags
        :value: 16
    


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
    


----

.. py:class:: slangpy.AccelerationStructureBuildMode

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.AccelerationStructureBuildMode.build
        :type: AccelerationStructureBuildMode
        :value: AccelerationStructureBuildMode.build
    
    .. py:attribute:: slangpy.AccelerationStructureBuildMode.update
        :type: AccelerationStructureBuildMode
        :value: AccelerationStructureBuildMode.update
    


----

.. py:class:: slangpy.AccelerationStructureCopyMode

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.AccelerationStructureCopyMode.clone
        :type: AccelerationStructureCopyMode
        :value: AccelerationStructureCopyMode.clone
    
    .. py:attribute:: slangpy.AccelerationStructureCopyMode.compact
        :type: AccelerationStructureCopyMode
        :value: AccelerationStructureCopyMode.compact
    


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
    
    .. py:attribute:: slangpy.AccelerationStructureGeometryFlags.none
        :type: AccelerationStructureGeometryFlags
        :value: 0
    
    .. py:attribute:: slangpy.AccelerationStructureGeometryFlags.opaque
        :type: AccelerationStructureGeometryFlags
        :value: 1
    
    .. py:attribute:: slangpy.AccelerationStructureGeometryFlags.no_duplicate_any_hit_invocation
        :type: AccelerationStructureGeometryFlags
        :value: 2
    


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
    
    .. py:attribute:: slangpy.AccelerationStructureInstanceFlags.none
        :type: AccelerationStructureInstanceFlags
        :value: 0
    
    .. py:attribute:: slangpy.AccelerationStructureInstanceFlags.triangle_facing_cull_disable
        :type: AccelerationStructureInstanceFlags
        :value: 1
    
    .. py:attribute:: slangpy.AccelerationStructureInstanceFlags.triangle_front_counter_clockwise
        :type: AccelerationStructureInstanceFlags
        :value: 2
    
    .. py:attribute:: slangpy.AccelerationStructureInstanceFlags.force_opaque
        :type: AccelerationStructureInstanceFlags
        :value: 4
    
    .. py:attribute:: slangpy.AccelerationStructureInstanceFlags.no_opaque
        :type: AccelerationStructureInstanceFlags
        :value: 8
    


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
    
    .. py:attribute:: slangpy.AccelerationStructureKind.unknown
        :type: AccelerationStructureKind
        :value: AccelerationStructureKind.unknown
    
    .. py:attribute:: slangpy.AccelerationStructureKind.bottom_level
        :type: AccelerationStructureKind
        :value: AccelerationStructureKind.bottom_level
    
    .. py:attribute:: slangpy.AccelerationStructureKind.top_level
        :type: AccelerationStructureKind
        :value: AccelerationStructureKind.top_level
    


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
    
    .. py:attribute:: slangpy.BlendFactor.zero
        :type: BlendFactor
        :value: BlendFactor.zero
    
    .. py:attribute:: slangpy.BlendFactor.one
        :type: BlendFactor
        :value: BlendFactor.one
    
    .. py:attribute:: slangpy.BlendFactor.src_color
        :type: BlendFactor
        :value: BlendFactor.src_color
    
    .. py:attribute:: slangpy.BlendFactor.inv_src_color
        :type: BlendFactor
        :value: BlendFactor.inv_src_color
    
    .. py:attribute:: slangpy.BlendFactor.src_alpha
        :type: BlendFactor
        :value: BlendFactor.src_alpha
    
    .. py:attribute:: slangpy.BlendFactor.inv_src_alpha
        :type: BlendFactor
        :value: BlendFactor.inv_src_alpha
    
    .. py:attribute:: slangpy.BlendFactor.dest_alpha
        :type: BlendFactor
        :value: BlendFactor.dest_alpha
    
    .. py:attribute:: slangpy.BlendFactor.inv_dest_alpha
        :type: BlendFactor
        :value: BlendFactor.inv_dest_alpha
    
    .. py:attribute:: slangpy.BlendFactor.dest_color
        :type: BlendFactor
        :value: BlendFactor.dest_color
    
    .. py:attribute:: slangpy.BlendFactor.inv_dest_color
        :type: BlendFactor
        :value: BlendFactor.inv_dest_color
    
    .. py:attribute:: slangpy.BlendFactor.src_alpha_saturate
        :type: BlendFactor
        :value: BlendFactor.src_alpha_saturate
    
    .. py:attribute:: slangpy.BlendFactor.blend_color
        :type: BlendFactor
        :value: BlendFactor.blend_color
    
    .. py:attribute:: slangpy.BlendFactor.inv_blend_color
        :type: BlendFactor
        :value: BlendFactor.inv_blend_color
    
    .. py:attribute:: slangpy.BlendFactor.secondary_src_color
        :type: BlendFactor
        :value: BlendFactor.secondary_src_color
    
    .. py:attribute:: slangpy.BlendFactor.inv_secondary_src_color
        :type: BlendFactor
        :value: BlendFactor.inv_secondary_src_color
    
    .. py:attribute:: slangpy.BlendFactor.secondary_src_alpha
        :type: BlendFactor
        :value: BlendFactor.secondary_src_alpha
    
    .. py:attribute:: slangpy.BlendFactor.inv_secondary_src_alpha
        :type: BlendFactor
        :value: BlendFactor.inv_secondary_src_alpha
    


----

.. py:class:: slangpy.BlendOp

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.BlendOp.add
        :type: BlendOp
        :value: BlendOp.add
    
    .. py:attribute:: slangpy.BlendOp.subtract
        :type: BlendOp
        :value: BlendOp.subtract
    
    .. py:attribute:: slangpy.BlendOp.reverse_subtract
        :type: BlendOp
        :value: BlendOp.reverse_subtract
    
    .. py:attribute:: slangpy.BlendOp.min
        :type: BlendOp
        :value: BlendOp.min
    
    .. py:attribute:: slangpy.BlendOp.max
        :type: BlendOp
        :value: BlendOp.max
    


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
    
    .. py:attribute:: slangpy.BufferUsage.none
        :type: BufferUsage
        :value: 0
    
    .. py:attribute:: slangpy.BufferUsage.vertex_buffer
        :type: BufferUsage
        :value: 1
    
    .. py:attribute:: slangpy.BufferUsage.index_buffer
        :type: BufferUsage
        :value: 2
    
    .. py:attribute:: slangpy.BufferUsage.constant_buffer
        :type: BufferUsage
        :value: 4
    
    .. py:attribute:: slangpy.BufferUsage.shader_resource
        :type: BufferUsage
        :value: 8
    
    .. py:attribute:: slangpy.BufferUsage.unordered_access
        :type: BufferUsage
        :value: 16
    
    .. py:attribute:: slangpy.BufferUsage.indirect_argument
        :type: BufferUsage
        :value: 32
    
    .. py:attribute:: slangpy.BufferUsage.copy_source
        :type: BufferUsage
        :value: 64
    
    .. py:attribute:: slangpy.BufferUsage.copy_destination
        :type: BufferUsage
        :value: 128
    
    .. py:attribute:: slangpy.BufferUsage.acceleration_structure
        :type: BufferUsage
        :value: 256
    
    .. py:attribute:: slangpy.BufferUsage.acceleration_structure_build_input
        :type: BufferUsage
        :value: 512
    
    .. py:attribute:: slangpy.BufferUsage.shader_table
        :type: BufferUsage
        :value: 1024
    
    .. py:attribute:: slangpy.BufferUsage.shared
        :type: BufferUsage
        :value: 2048
    


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
    
    .. py:attribute:: slangpy.CommandQueueType.graphics
        :type: CommandQueueType
        :value: CommandQueueType.graphics
    


----

.. py:class:: slangpy.ComparisonFunc

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.ComparisonFunc.never
        :type: ComparisonFunc
        :value: ComparisonFunc.never
    
    .. py:attribute:: slangpy.ComparisonFunc.less
        :type: ComparisonFunc
        :value: ComparisonFunc.less
    
    .. py:attribute:: slangpy.ComparisonFunc.equal
        :type: ComparisonFunc
        :value: ComparisonFunc.equal
    
    .. py:attribute:: slangpy.ComparisonFunc.less_equal
        :type: ComparisonFunc
        :value: ComparisonFunc.less_equal
    
    .. py:attribute:: slangpy.ComparisonFunc.greater
        :type: ComparisonFunc
        :value: ComparisonFunc.greater
    
    .. py:attribute:: slangpy.ComparisonFunc.not_equal
        :type: ComparisonFunc
        :value: ComparisonFunc.not_equal
    
    .. py:attribute:: slangpy.ComparisonFunc.greater_equal
        :type: ComparisonFunc
        :value: ComparisonFunc.greater_equal
    
    .. py:attribute:: slangpy.ComparisonFunc.always
        :type: ComparisonFunc
        :value: ComparisonFunc.always
    


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
    
    
    
    .. py:attribute:: slangpy.CoopVecMatrixLayout.row_major
        :type: CoopVecMatrixLayout
        :value: CoopVecMatrixLayout.row_major
    
    .. py:attribute:: slangpy.CoopVecMatrixLayout.column_major
        :type: CoopVecMatrixLayout
        :value: CoopVecMatrixLayout.column_major
    
    .. py:attribute:: slangpy.CoopVecMatrixLayout.inferencing_optimal
        :type: CoopVecMatrixLayout
        :value: CoopVecMatrixLayout.inferencing_optimal
    
    .. py:attribute:: slangpy.CoopVecMatrixLayout.training_optimal
        :type: CoopVecMatrixLayout
        :value: CoopVecMatrixLayout.training_optimal
    


----

.. py:class:: slangpy.CullMode

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.CullMode.none
        :type: CullMode
        :value: CullMode.none
    
    .. py:attribute:: slangpy.CullMode.front
        :type: CullMode
        :value: CullMode.front
    
    .. py:attribute:: slangpy.CullMode.back
        :type: CullMode
        :value: CullMode.back
    


----

.. py:class:: slangpy.DeclReflection

    Base class: :py:class:`slangpy.BaseReflectionObject`
    
    
    
    .. py:class:: slangpy.DeclReflection.Kind
    
        Base class: :py:class:`enum.Enum`
        
        Different kinds of decl slang can return.
        
        .. py:attribute:: slangpy.DeclReflection.Kind.unsupported
            :type: Kind
            :value: Kind.unsupported
        
        .. py:attribute:: slangpy.DeclReflection.Kind.struct
            :type: Kind
            :value: Kind.struct
        
        .. py:attribute:: slangpy.DeclReflection.Kind.func
            :type: Kind
            :value: Kind.func
        
        .. py:attribute:: slangpy.DeclReflection.Kind.module
            :type: Kind
            :value: Kind.module
        
        .. py:attribute:: slangpy.DeclReflection.Kind.generic
            :type: Kind
            :value: Kind.generic
        
        .. py:attribute:: slangpy.DeclReflection.Kind.variable
            :type: Kind
            :value: Kind.variable
        
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
    
    
    
    .. py:method:: __init__(self, type: slangpy.DeviceType = DeviceType.automatic, enable_debug_layers: bool = False, debug_layers_log_level: slangpy.LogLevel = LogLevel.warn, enable_rhi_validation: bool = True, rhi_validation_log_level: slangpy.LogLevel = LogLevel.warn, enable_ray_tracing_validation: bool = False, enable_aftermath: bool = False, enable_cuda_interop: bool = False, enable_print: bool = False, enable_hot_reload: bool = True, enable_compilation_reports: bool = False, adapter_luid: collections.abc.Sequence[int] | None = None, compiler_options: slangpy.SlangCompilerOptions | None = None, module_cache_path: str | os.PathLike | None = None, shader_cache_path: str | os.PathLike | None = None, shader_cache_size: int = 134217728, existing_device_handles: collections.abc.Sequence[slangpy.NativeHandle] | None = None, bindless_options: slangpy.BindlessDesc | None = None, additional_vulkan_instance_extensions: collections.abc.Sequence[str] | None = None, additional_vulkan_device_extensions: collections.abc.Sequence[str] | None = None, enable_cuda_launch_from_gfx: bool = True, enable_ray_tracing: bool = True, label: str = '') -> None
    
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
        
    .. py:property:: supported_profiles
        :type: list[str]
    
        List of supported profiles (e.g. "sm_6_6", "spirv_1_5").
        
    .. py:method:: has_profile(self, profile: str) -> bool
    
        Check if the device supports a given profile.
        
    .. py:property:: default_profile
        :type: str
    
        The default profile for the device.
        
    .. py:property:: features
        :type: list[slangpy.Feature]
    
        List of features supported by the device.
        
    .. py:property:: capabilities
        :type: list[str]
    
        List of slang capabilities supported by the device.
        
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
        
        Usage:
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
        
    .. py:method:: create_compute_pipeline(self, program: slangpy.ShaderProgram, defer_target_compilation: bool = False, label: str | None = None) -> slangpy.ComputePipeline
    
        Create a compute pipeline.
        
    .. py:method:: create_compute_pipeline(self, desc: slangpy.ComputePipelineDesc) -> slangpy.ComputePipeline
        :no-index:
    
    .. py:method:: create_render_pipeline(self, program: slangpy.ShaderProgram, input_layout: slangpy.InputLayout | None, primitive_topology: slangpy.PrimitiveTopology = PrimitiveTopology.triangle_list, targets: collections.abc.Sequence[slangpy.ColorTargetDesc] = [], depth_stencil: slangpy.DepthStencilDesc | None = None, rasterizer: slangpy.RasterizerDesc | None = None, multisample: slangpy.MultisampleDesc | None = None, defer_target_compilation: bool = False, label: str | None = None) -> slangpy.RenderPipeline
    
        Create a render pipeline.
        
    .. py:method:: create_render_pipeline(self, desc: slangpy.RenderPipelineDesc) -> slangpy.RenderPipeline
        :no-index:
    
    .. py:method:: create_ray_tracing_pipeline(self, program: slangpy.ShaderProgram, hit_groups: collections.abc.Sequence[slangpy.HitGroupDesc], max_recursion: int = 0, max_ray_payload_size: int = 0, max_attribute_size: int = 8, flags: slangpy.RayTracingPipelineFlags = 0, defer_target_compilation: bool = False, label: str | None = None) -> slangpy.RayTracingPipeline
    
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
    
    .. py:attribute:: slangpy.DeviceType.automatic
        :type: DeviceType
        :value: DeviceType.automatic
    
    .. py:attribute:: slangpy.DeviceType.d3d12
        :type: DeviceType
        :value: DeviceType.d3d12
    
    .. py:attribute:: slangpy.DeviceType.vulkan
        :type: DeviceType
        :value: DeviceType.vulkan
    
    .. py:attribute:: slangpy.DeviceType.metal
        :type: DeviceType
        :value: DeviceType.metal
    
    .. py:attribute:: slangpy.DeviceType.wgpu
        :type: DeviceType
        :value: DeviceType.wgpu
    
    .. py:attribute:: slangpy.DeviceType.cpu
        :type: DeviceType
        :value: DeviceType.cpu
    
    .. py:attribute:: slangpy.DeviceType.cuda
        :type: DeviceType
        :value: DeviceType.cuda
    


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
    
    .. py:attribute:: slangpy.Feature.hardware_device
        :type: Feature
        :value: Feature.hardware_device
    
    .. py:attribute:: slangpy.Feature.software_device
        :type: Feature
        :value: Feature.software_device
    
    .. py:attribute:: slangpy.Feature.parameter_block
        :type: Feature
        :value: Feature.parameter_block
    
    .. py:attribute:: slangpy.Feature.bindless
        :type: Feature
        :value: Feature.bindless
    
    .. py:attribute:: slangpy.Feature.surface
        :type: Feature
        :value: Feature.surface
    
    .. py:attribute:: slangpy.Feature.pipeline_cache
        :type: Feature
        :value: Feature.pipeline_cache
    
    .. py:attribute:: slangpy.Feature.rasterization
        :type: Feature
        :value: Feature.rasterization
    
    .. py:attribute:: slangpy.Feature.barycentrics
        :type: Feature
        :value: Feature.barycentrics
    
    .. py:attribute:: slangpy.Feature.multi_view
        :type: Feature
        :value: Feature.multi_view
    
    .. py:attribute:: slangpy.Feature.rasterizer_ordered_views
        :type: Feature
        :value: Feature.rasterizer_ordered_views
    
    .. py:attribute:: slangpy.Feature.conservative_rasterization
        :type: Feature
        :value: Feature.conservative_rasterization
    
    .. py:attribute:: slangpy.Feature.custom_border_color
        :type: Feature
        :value: Feature.custom_border_color
    
    .. py:attribute:: slangpy.Feature.fragment_shading_rate
        :type: Feature
        :value: Feature.fragment_shading_rate
    
    .. py:attribute:: slangpy.Feature.sampler_feedback
        :type: Feature
        :value: Feature.sampler_feedback
    
    .. py:attribute:: slangpy.Feature.combined_texture_sampler
        :type: Feature
        :value: Feature.combined_texture_sampler
    
    .. py:attribute:: slangpy.Feature.acceleration_structure
        :type: Feature
        :value: Feature.acceleration_structure
    
    .. py:attribute:: slangpy.Feature.acceleration_structure_spheres
        :type: Feature
        :value: Feature.acceleration_structure_spheres
    
    .. py:attribute:: slangpy.Feature.acceleration_structure_linear_swept_spheres
        :type: Feature
        :value: Feature.acceleration_structure_linear_swept_spheres
    
    .. py:attribute:: slangpy.Feature.ray_tracing
        :type: Feature
        :value: Feature.ray_tracing
    
    .. py:attribute:: slangpy.Feature.ray_query
        :type: Feature
        :value: Feature.ray_query
    
    .. py:attribute:: slangpy.Feature.shader_execution_reordering
        :type: Feature
        :value: Feature.shader_execution_reordering
    
    .. py:attribute:: slangpy.Feature.ray_tracing_motion_blur
        :type: Feature
        :value: Feature.ray_tracing_motion_blur
    
    .. py:attribute:: slangpy.Feature.ray_tracing_validation
        :type: Feature
        :value: Feature.ray_tracing_validation
    
    .. py:attribute:: slangpy.Feature.cluster_acceleration_structure
        :type: Feature
        :value: Feature.cluster_acceleration_structure
    
    .. py:attribute:: slangpy.Feature.timestamp_query
        :type: Feature
        :value: Feature.timestamp_query
    
    .. py:attribute:: slangpy.Feature.timestamp_calibration
        :type: Feature
        :value: Feature.timestamp_calibration
    
    .. py:attribute:: slangpy.Feature.realtime_clock
        :type: Feature
        :value: Feature.realtime_clock
    
    .. py:attribute:: slangpy.Feature.cooperative_vector
        :type: Feature
        :value: Feature.cooperative_vector
    
    .. py:attribute:: slangpy.Feature.cooperative_matrix
        :type: Feature
        :value: Feature.cooperative_matrix
    
    .. py:attribute:: slangpy.Feature.cooperative_matrix2
        :type: Feature
        :value: Feature.cooperative_matrix2
    
    .. py:attribute:: slangpy.Feature.sm_5_1
        :type: Feature
        :value: Feature.sm_5_1
    
    .. py:attribute:: slangpy.Feature.sm_6_0
        :type: Feature
        :value: Feature.sm_6_0
    
    .. py:attribute:: slangpy.Feature.sm_6_1
        :type: Feature
        :value: Feature.sm_6_1
    
    .. py:attribute:: slangpy.Feature.sm_6_2
        :type: Feature
        :value: Feature.sm_6_2
    
    .. py:attribute:: slangpy.Feature.sm_6_3
        :type: Feature
        :value: Feature.sm_6_3
    
    .. py:attribute:: slangpy.Feature.sm_6_4
        :type: Feature
        :value: Feature.sm_6_4
    
    .. py:attribute:: slangpy.Feature.sm_6_5
        :type: Feature
        :value: Feature.sm_6_5
    
    .. py:attribute:: slangpy.Feature.sm_6_6
        :type: Feature
        :value: Feature.sm_6_6
    
    .. py:attribute:: slangpy.Feature.sm_6_7
        :type: Feature
        :value: Feature.sm_6_7
    
    .. py:attribute:: slangpy.Feature.sm_6_8
        :type: Feature
        :value: Feature.sm_6_8
    
    .. py:attribute:: slangpy.Feature.sm_6_9
        :type: Feature
        :value: Feature.sm_6_9
    
    .. py:attribute:: slangpy.Feature.sm_6_10
        :type: Feature
        :value: Feature.sm_6_10
    
    .. py:attribute:: slangpy.Feature.half
        :type: Feature
        :value: Feature.half
    
    .. py:attribute:: slangpy.Feature.double
        :type: Feature
        :value: Feature.double
    
    .. py:attribute:: slangpy.Feature.int16
        :type: Feature
        :value: Feature.int16
    
    .. py:attribute:: slangpy.Feature.int64
        :type: Feature
        :value: Feature.int64
    
    .. py:attribute:: slangpy.Feature.atomic_float
        :type: Feature
        :value: Feature.atomic_float
    
    .. py:attribute:: slangpy.Feature.atomic_half
        :type: Feature
        :value: Feature.atomic_half
    
    .. py:attribute:: slangpy.Feature.atomic_int64
        :type: Feature
        :value: Feature.atomic_int64
    
    .. py:attribute:: slangpy.Feature.wave_ops
        :type: Feature
        :value: Feature.wave_ops
    
    .. py:attribute:: slangpy.Feature.mesh_shader
        :type: Feature
        :value: Feature.mesh_shader
    
    .. py:attribute:: slangpy.Feature.pointer
        :type: Feature
        :value: Feature.pointer
    
    .. py:attribute:: slangpy.Feature.float8
        :type: Feature
        :value: Feature.float8
    
    .. py:attribute:: slangpy.Feature.bfloat16
        :type: Feature
        :value: Feature.bfloat16
    
    .. py:attribute:: slangpy.Feature.conservative_rasterization1
        :type: Feature
        :value: Feature.conservative_rasterization1
    
    .. py:attribute:: slangpy.Feature.conservative_rasterization2
        :type: Feature
        :value: Feature.conservative_rasterization2
    
    .. py:attribute:: slangpy.Feature.conservative_rasterization3
        :type: Feature
        :value: Feature.conservative_rasterization3
    
    .. py:attribute:: slangpy.Feature.programmable_sample_positions1
        :type: Feature
        :value: Feature.programmable_sample_positions1
    
    .. py:attribute:: slangpy.Feature.programmable_sample_positions2
        :type: Feature
        :value: Feature.programmable_sample_positions2
    
    .. py:attribute:: slangpy.Feature.shader_resource_min_lod
        :type: Feature
        :value: Feature.shader_resource_min_lod
    
    .. py:attribute:: slangpy.Feature.shader_abort
        :type: Feature
        :value: Feature.shader_abort
    
    .. py:attribute:: slangpy.Feature.argument_buffer_tier2
        :type: Feature
        :value: Feature.argument_buffer_tier2
    
    .. py:attribute:: slangpy.Feature.residency_set
        :type: Feature
        :value: Feature.residency_set
    
    .. py:attribute:: slangpy.Feature.atomic_bfloat16
        :type: Feature
        :value: Feature.atomic_bfloat16
    


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
    
    .. py:attribute:: slangpy.FillMode.solid
        :type: FillMode
        :value: FillMode.solid
    
    .. py:attribute:: slangpy.FillMode.wireframe
        :type: FillMode
        :value: FillMode.wireframe
    


----

.. py:class:: slangpy.Format

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.Format.undefined
        :type: Format
        :value: Format.undefined
    
    .. py:attribute:: slangpy.Format.r8_uint
        :type: Format
        :value: Format.r8_uint
    
    .. py:attribute:: slangpy.Format.r8_sint
        :type: Format
        :value: Format.r8_sint
    
    .. py:attribute:: slangpy.Format.r8_unorm
        :type: Format
        :value: Format.r8_unorm
    
    .. py:attribute:: slangpy.Format.r8_snorm
        :type: Format
        :value: Format.r8_snorm
    
    .. py:attribute:: slangpy.Format.rg8_uint
        :type: Format
        :value: Format.rg8_uint
    
    .. py:attribute:: slangpy.Format.rg8_sint
        :type: Format
        :value: Format.rg8_sint
    
    .. py:attribute:: slangpy.Format.rg8_unorm
        :type: Format
        :value: Format.rg8_unorm
    
    .. py:attribute:: slangpy.Format.rg8_snorm
        :type: Format
        :value: Format.rg8_snorm
    
    .. py:attribute:: slangpy.Format.rgba8_uint
        :type: Format
        :value: Format.rgba8_uint
    
    .. py:attribute:: slangpy.Format.rgba8_sint
        :type: Format
        :value: Format.rgba8_sint
    
    .. py:attribute:: slangpy.Format.rgba8_unorm
        :type: Format
        :value: Format.rgba8_unorm
    
    .. py:attribute:: slangpy.Format.rgba8_unorm_srgb
        :type: Format
        :value: Format.rgba8_unorm_srgb
    
    .. py:attribute:: slangpy.Format.rgba8_snorm
        :type: Format
        :value: Format.rgba8_snorm
    
    .. py:attribute:: slangpy.Format.bgra8_unorm
        :type: Format
        :value: Format.bgra8_unorm
    
    .. py:attribute:: slangpy.Format.bgra8_unorm_srgb
        :type: Format
        :value: Format.bgra8_unorm_srgb
    
    .. py:attribute:: slangpy.Format.bgrx8_unorm
        :type: Format
        :value: Format.bgrx8_unorm
    
    .. py:attribute:: slangpy.Format.bgrx8_unorm_srgb
        :type: Format
        :value: Format.bgrx8_unorm_srgb
    
    .. py:attribute:: slangpy.Format.r16_uint
        :type: Format
        :value: Format.r16_uint
    
    .. py:attribute:: slangpy.Format.r16_sint
        :type: Format
        :value: Format.r16_sint
    
    .. py:attribute:: slangpy.Format.r16_unorm
        :type: Format
        :value: Format.r16_unorm
    
    .. py:attribute:: slangpy.Format.r16_snorm
        :type: Format
        :value: Format.r16_snorm
    
    .. py:attribute:: slangpy.Format.r16_float
        :type: Format
        :value: Format.r16_float
    
    .. py:attribute:: slangpy.Format.rg16_uint
        :type: Format
        :value: Format.rg16_uint
    
    .. py:attribute:: slangpy.Format.rg16_sint
        :type: Format
        :value: Format.rg16_sint
    
    .. py:attribute:: slangpy.Format.rg16_unorm
        :type: Format
        :value: Format.rg16_unorm
    
    .. py:attribute:: slangpy.Format.rg16_snorm
        :type: Format
        :value: Format.rg16_snorm
    
    .. py:attribute:: slangpy.Format.rg16_float
        :type: Format
        :value: Format.rg16_float
    
    .. py:attribute:: slangpy.Format.rgba16_uint
        :type: Format
        :value: Format.rgba16_uint
    
    .. py:attribute:: slangpy.Format.rgba16_sint
        :type: Format
        :value: Format.rgba16_sint
    
    .. py:attribute:: slangpy.Format.rgba16_unorm
        :type: Format
        :value: Format.rgba16_unorm
    
    .. py:attribute:: slangpy.Format.rgba16_snorm
        :type: Format
        :value: Format.rgba16_snorm
    
    .. py:attribute:: slangpy.Format.rgba16_float
        :type: Format
        :value: Format.rgba16_float
    
    .. py:attribute:: slangpy.Format.r32_uint
        :type: Format
        :value: Format.r32_uint
    
    .. py:attribute:: slangpy.Format.r32_sint
        :type: Format
        :value: Format.r32_sint
    
    .. py:attribute:: slangpy.Format.r32_float
        :type: Format
        :value: Format.r32_float
    
    .. py:attribute:: slangpy.Format.rg32_uint
        :type: Format
        :value: Format.rg32_uint
    
    .. py:attribute:: slangpy.Format.rg32_sint
        :type: Format
        :value: Format.rg32_sint
    
    .. py:attribute:: slangpy.Format.rg32_float
        :type: Format
        :value: Format.rg32_float
    
    .. py:attribute:: slangpy.Format.rgb32_uint
        :type: Format
        :value: Format.rgb32_uint
    
    .. py:attribute:: slangpy.Format.rgb32_sint
        :type: Format
        :value: Format.rgb32_sint
    
    .. py:attribute:: slangpy.Format.rgb32_float
        :type: Format
        :value: Format.rgb32_float
    
    .. py:attribute:: slangpy.Format.rgba32_uint
        :type: Format
        :value: Format.rgba32_uint
    
    .. py:attribute:: slangpy.Format.rgba32_sint
        :type: Format
        :value: Format.rgba32_sint
    
    .. py:attribute:: slangpy.Format.rgba32_float
        :type: Format
        :value: Format.rgba32_float
    
    .. py:attribute:: slangpy.Format.r64_uint
        :type: Format
        :value: Format.r64_uint
    
    .. py:attribute:: slangpy.Format.r64_sint
        :type: Format
        :value: Format.r64_sint
    
    .. py:attribute:: slangpy.Format.bgra4_unorm
        :type: Format
        :value: Format.bgra4_unorm
    
    .. py:attribute:: slangpy.Format.b5g6r5_unorm
        :type: Format
        :value: Format.b5g6r5_unorm
    
    .. py:attribute:: slangpy.Format.bgr5a1_unorm
        :type: Format
        :value: Format.bgr5a1_unorm
    
    .. py:attribute:: slangpy.Format.rgb9e5_ufloat
        :type: Format
        :value: Format.rgb9e5_ufloat
    
    .. py:attribute:: slangpy.Format.rgb10a2_uint
        :type: Format
        :value: Format.rgb10a2_uint
    
    .. py:attribute:: slangpy.Format.rgb10a2_unorm
        :type: Format
        :value: Format.rgb10a2_unorm
    
    .. py:attribute:: slangpy.Format.r11g11b10_float
        :type: Format
        :value: Format.r11g11b10_float
    
    .. py:attribute:: slangpy.Format.d32_float
        :type: Format
        :value: Format.d32_float
    
    .. py:attribute:: slangpy.Format.d16_unorm
        :type: Format
        :value: Format.d16_unorm
    
    .. py:attribute:: slangpy.Format.d32_float_s8_uint
        :type: Format
        :value: Format.d32_float_s8_uint
    
    .. py:attribute:: slangpy.Format.bc1_unorm
        :type: Format
        :value: Format.bc1_unorm
    
    .. py:attribute:: slangpy.Format.bc1_unorm_srgb
        :type: Format
        :value: Format.bc1_unorm_srgb
    
    .. py:attribute:: slangpy.Format.bc2_unorm
        :type: Format
        :value: Format.bc2_unorm
    
    .. py:attribute:: slangpy.Format.bc2_unorm_srgb
        :type: Format
        :value: Format.bc2_unorm_srgb
    
    .. py:attribute:: slangpy.Format.bc3_unorm
        :type: Format
        :value: Format.bc3_unorm
    
    .. py:attribute:: slangpy.Format.bc3_unorm_srgb
        :type: Format
        :value: Format.bc3_unorm_srgb
    
    .. py:attribute:: slangpy.Format.bc4_unorm
        :type: Format
        :value: Format.bc4_unorm
    
    .. py:attribute:: slangpy.Format.bc4_snorm
        :type: Format
        :value: Format.bc4_snorm
    
    .. py:attribute:: slangpy.Format.bc5_unorm
        :type: Format
        :value: Format.bc5_unorm
    
    .. py:attribute:: slangpy.Format.bc5_snorm
        :type: Format
        :value: Format.bc5_snorm
    
    .. py:attribute:: slangpy.Format.bc6h_ufloat
        :type: Format
        :value: Format.bc6h_ufloat
    
    .. py:attribute:: slangpy.Format.bc6h_sfloat
        :type: Format
        :value: Format.bc6h_sfloat
    
    .. py:attribute:: slangpy.Format.bc7_unorm
        :type: Format
        :value: Format.bc7_unorm
    
    .. py:attribute:: slangpy.Format.bc7_unorm_srgb
        :type: Format
        :value: Format.bc7_unorm_srgb
    
    .. py:attribute:: slangpy.Format.astc4x4_unorm
        :type: Format
        :value: Format.astc4x4_unorm
    
    .. py:attribute:: slangpy.Format.astc4x4_unorm_srgb
        :type: Format
        :value: Format.astc4x4_unorm_srgb
    
    .. py:attribute:: slangpy.Format.astc6x6_unorm
        :type: Format
        :value: Format.astc6x6_unorm
    
    .. py:attribute:: slangpy.Format.astc6x6_unorm_srgb
        :type: Format
        :value: Format.astc6x6_unorm_srgb
    
    .. py:attribute:: slangpy.Format.astc8x8_unorm
        :type: Format
        :value: Format.astc8x8_unorm
    
    .. py:attribute:: slangpy.Format.astc8x8_unorm_srgb
        :type: Format
        :value: Format.astc8x8_unorm_srgb
    


----

.. py:class:: slangpy.FormatChannels

    Base class: :py:class:`enum.IntFlag`
    
    .. py:attribute:: slangpy.FormatChannels.none
        :type: FormatChannels
        :value: 0
    
    .. py:attribute:: slangpy.FormatChannels.r
        :type: FormatChannels
        :value: 1
    
    .. py:attribute:: slangpy.FormatChannels.g
        :type: FormatChannels
        :value: 2
    
    .. py:attribute:: slangpy.FormatChannels.b
        :type: FormatChannels
        :value: 4
    
    .. py:attribute:: slangpy.FormatChannels.a
        :type: FormatChannels
        :value: 8
    
    .. py:attribute:: slangpy.FormatChannels.rg
        :type: FormatChannels
        :value: 3
    
    .. py:attribute:: slangpy.FormatChannels.rgb
        :type: FormatChannels
        :value: 7
    
    .. py:attribute:: slangpy.FormatChannels.rgba
        :type: FormatChannels
        :value: 15
    


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
    
    .. py:attribute:: slangpy.FormatSupport.none
        :type: FormatSupport
        :value: 0
    
    .. py:attribute:: slangpy.FormatSupport.copy_source
        :type: FormatSupport
        :value: 1
    
    .. py:attribute:: slangpy.FormatSupport.copy_destination
        :type: FormatSupport
        :value: 2
    
    .. py:attribute:: slangpy.FormatSupport.texture
        :type: FormatSupport
        :value: 4
    
    .. py:attribute:: slangpy.FormatSupport.depth_stencil
        :type: FormatSupport
        :value: 8
    
    .. py:attribute:: slangpy.FormatSupport.render_target
        :type: FormatSupport
        :value: 16
    
    .. py:attribute:: slangpy.FormatSupport.blendable
        :type: FormatSupport
        :value: 32
    
    .. py:attribute:: slangpy.FormatSupport.multisampling
        :type: FormatSupport
        :value: 64
    
    .. py:attribute:: slangpy.FormatSupport.resolvable
        :type: FormatSupport
        :value: 128
    
    .. py:attribute:: slangpy.FormatSupport.shader_load
        :type: FormatSupport
        :value: 256
    
    .. py:attribute:: slangpy.FormatSupport.shader_sample
        :type: FormatSupport
        :value: 512
    
    .. py:attribute:: slangpy.FormatSupport.shader_uav_load
        :type: FormatSupport
        :value: 1024
    
    .. py:attribute:: slangpy.FormatSupport.shader_uav_store
        :type: FormatSupport
        :value: 2048
    
    .. py:attribute:: slangpy.FormatSupport.shader_atomic
        :type: FormatSupport
        :value: 4096
    
    .. py:attribute:: slangpy.FormatSupport.buffer
        :type: FormatSupport
        :value: 8192
    
    .. py:attribute:: slangpy.FormatSupport.index_buffer
        :type: FormatSupport
        :value: 16384
    
    .. py:attribute:: slangpy.FormatSupport.vertex_buffer
        :type: FormatSupport
        :value: 32768
    


----

.. py:class:: slangpy.FormatType

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.FormatType.unknown
        :type: FormatType
        :value: FormatType.unknown
    
    .. py:attribute:: slangpy.FormatType.float
        :type: FormatType
        :value: FormatType.float
    
    .. py:attribute:: slangpy.FormatType.unorm
        :type: FormatType
        :value: FormatType.unorm
    
    .. py:attribute:: slangpy.FormatType.unorm_srgb
        :type: FormatType
        :value: FormatType.unorm_srgb
    
    .. py:attribute:: slangpy.FormatType.snorm
        :type: FormatType
        :value: FormatType.snorm
    
    .. py:attribute:: slangpy.FormatType.uint
        :type: FormatType
        :value: FormatType.uint
    
    .. py:attribute:: slangpy.FormatType.sint
        :type: FormatType
        :value: FormatType.sint
    


----

.. py:class:: slangpy.FrontFaceMode

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.FrontFaceMode.counter_clockwise
        :type: FrontFaceMode
        :value: FrontFaceMode.counter_clockwise
    
    .. py:attribute:: slangpy.FrontFaceMode.clockwise
        :type: FrontFaceMode
        :value: FrontFaceMode.clockwise
    


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
    
    .. py:attribute:: slangpy.IndexFormat.uint16
        :type: IndexFormat
        :value: IndexFormat.uint16
    
    .. py:attribute:: slangpy.IndexFormat.uint32
        :type: IndexFormat
        :value: IndexFormat.uint32
    


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
    
    .. py:attribute:: slangpy.InputSlotClass.per_vertex
        :type: InputSlotClass
        :value: InputSlotClass.per_vertex
    
    .. py:attribute:: slangpy.InputSlotClass.per_instance
        :type: InputSlotClass
        :value: InputSlotClass.per_instance
    


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
    
    .. py:attribute:: slangpy.LoadOp.load
        :type: LoadOp
        :value: LoadOp.load
    
    .. py:attribute:: slangpy.LoadOp.clear
        :type: LoadOp
        :value: LoadOp.clear
    
    .. py:attribute:: slangpy.LoadOp.dont_care
        :type: LoadOp
        :value: LoadOp.dont_care
    


----

.. py:class:: slangpy.LogicOp

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.LogicOp.no_op
        :type: LogicOp
        :value: LogicOp.no_op
    


----

.. py:class:: slangpy.MemoryType

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.MemoryType.device_local
        :type: MemoryType
        :value: MemoryType.device_local
    
    .. py:attribute:: slangpy.MemoryType.upload
        :type: MemoryType
        :value: MemoryType.upload
    
    .. py:attribute:: slangpy.MemoryType.read_back
        :type: MemoryType
        :value: MemoryType.read_back
    


----

.. py:class:: slangpy.ModifierID

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.ModifierID.shared
        :type: ModifierID
        :value: ModifierID.shared
    
    .. py:attribute:: slangpy.ModifierID.nodiff
        :type: ModifierID
        :value: ModifierID.nodiff
    
    .. py:attribute:: slangpy.ModifierID.static
        :type: ModifierID
        :value: ModifierID.static
    
    .. py:attribute:: slangpy.ModifierID.const
        :type: ModifierID
        :value: ModifierID.const
    
    .. py:attribute:: slangpy.ModifierID.export
        :type: ModifierID
        :value: ModifierID.export
    
    .. py:attribute:: slangpy.ModifierID.extern
        :type: ModifierID
        :value: ModifierID.extern
    
    .. py:attribute:: slangpy.ModifierID.differentiable
        :type: ModifierID
        :value: ModifierID.differentiable
    
    .. py:attribute:: slangpy.ModifierID.mutating
        :type: ModifierID
        :value: ModifierID.mutating
    
    .. py:attribute:: slangpy.ModifierID.inn
        :type: ModifierID
        :value: ModifierID.inn
    
    .. py:attribute:: slangpy.ModifierID.out
        :type: ModifierID
        :value: ModifierID.out
    
    .. py:attribute:: slangpy.ModifierID.inout
        :type: ModifierID
        :value: ModifierID.inout
    


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
    


----

.. py:class:: slangpy.NativeHandleType

    Base class: :py:class:`enum.Enum`
    
    
    
    .. py:attribute:: slangpy.NativeHandleType.undefined
        :type: NativeHandleType
        :value: NativeHandleType.undefined
    
    .. py:attribute:: slangpy.NativeHandleType.win32
        :type: NativeHandleType
        :value: NativeHandleType.win32
    
    .. py:attribute:: slangpy.NativeHandleType.file_descriptor
        :type: NativeHandleType
        :value: NativeHandleType.file_descriptor
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12Device
        :type: NativeHandleType
        :value: NativeHandleType.D3D12Device
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12CommandQueue
        :type: NativeHandleType
        :value: NativeHandleType.D3D12CommandQueue
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12GraphicsCommandList
        :type: NativeHandleType
        :value: NativeHandleType.D3D12GraphicsCommandList
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12Resource
        :type: NativeHandleType
        :value: NativeHandleType.D3D12Resource
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12PipelineState
        :type: NativeHandleType
        :value: NativeHandleType.D3D12PipelineState
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12StateObject
        :type: NativeHandleType
        :value: NativeHandleType.D3D12StateObject
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12CpuDescriptorHandle
        :type: NativeHandleType
        :value: NativeHandleType.D3D12CpuDescriptorHandle
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12Fence
        :type: NativeHandleType
        :value: NativeHandleType.D3D12Fence
    
    .. py:attribute:: slangpy.NativeHandleType.D3D12DeviceAddress
        :type: NativeHandleType
        :value: NativeHandleType.D3D12DeviceAddress
    
    .. py:attribute:: slangpy.NativeHandleType.VkDevice
        :type: NativeHandleType
        :value: NativeHandleType.VkDevice
    
    .. py:attribute:: slangpy.NativeHandleType.VkPhysicalDevice
        :type: NativeHandleType
        :value: NativeHandleType.VkPhysicalDevice
    
    .. py:attribute:: slangpy.NativeHandleType.VkInstance
        :type: NativeHandleType
        :value: NativeHandleType.VkInstance
    
    .. py:attribute:: slangpy.NativeHandleType.VkQueue
        :type: NativeHandleType
        :value: NativeHandleType.VkQueue
    
    .. py:attribute:: slangpy.NativeHandleType.VkCommandBuffer
        :type: NativeHandleType
        :value: NativeHandleType.VkCommandBuffer
    
    .. py:attribute:: slangpy.NativeHandleType.VkBuffer
        :type: NativeHandleType
        :value: NativeHandleType.VkBuffer
    
    .. py:attribute:: slangpy.NativeHandleType.VkImage
        :type: NativeHandleType
        :value: NativeHandleType.VkImage
    
    .. py:attribute:: slangpy.NativeHandleType.VkImageView
        :type: NativeHandleType
        :value: NativeHandleType.VkImageView
    
    .. py:attribute:: slangpy.NativeHandleType.VkAccelerationStructureKHR
        :type: NativeHandleType
        :value: NativeHandleType.VkAccelerationStructureKHR
    
    .. py:attribute:: slangpy.NativeHandleType.VkSampler
        :type: NativeHandleType
        :value: NativeHandleType.VkSampler
    
    .. py:attribute:: slangpy.NativeHandleType.VkPipeline
        :type: NativeHandleType
        :value: NativeHandleType.VkPipeline
    
    .. py:attribute:: slangpy.NativeHandleType.VkSemaphore
        :type: NativeHandleType
        :value: NativeHandleType.VkSemaphore
    
    .. py:attribute:: slangpy.NativeHandleType.MTLDevice
        :type: NativeHandleType
        :value: NativeHandleType.MTLDevice
    
    .. py:attribute:: slangpy.NativeHandleType.MTLCommandQueue
        :type: NativeHandleType
        :value: NativeHandleType.MTLCommandQueue
    
    .. py:attribute:: slangpy.NativeHandleType.MTLCommandBuffer
        :type: NativeHandleType
        :value: NativeHandleType.MTLCommandBuffer
    
    .. py:attribute:: slangpy.NativeHandleType.MTLTexture
        :type: NativeHandleType
        :value: NativeHandleType.MTLTexture
    
    .. py:attribute:: slangpy.NativeHandleType.MTLBuffer
        :type: NativeHandleType
        :value: NativeHandleType.MTLBuffer
    
    .. py:attribute:: slangpy.NativeHandleType.MTLComputePipelineState
        :type: NativeHandleType
        :value: NativeHandleType.MTLComputePipelineState
    
    .. py:attribute:: slangpy.NativeHandleType.MTLRenderPipelineState
        :type: NativeHandleType
        :value: NativeHandleType.MTLRenderPipelineState
    
    .. py:attribute:: slangpy.NativeHandleType.MTLSharedEvent
        :type: NativeHandleType
        :value: NativeHandleType.MTLSharedEvent
    
    .. py:attribute:: slangpy.NativeHandleType.MTLSamplerState
        :type: NativeHandleType
        :value: NativeHandleType.MTLSamplerState
    
    .. py:attribute:: slangpy.NativeHandleType.MTLAccelerationStructure
        :type: NativeHandleType
        :value: NativeHandleType.MTLAccelerationStructure
    
    .. py:attribute:: slangpy.NativeHandleType.CUdevice
        :type: NativeHandleType
        :value: NativeHandleType.CUdevice
    
    .. py:attribute:: slangpy.NativeHandleType.CUdeviceptr
        :type: NativeHandleType
        :value: NativeHandleType.CUdeviceptr
    
    .. py:attribute:: slangpy.NativeHandleType.CUtexObject
        :type: NativeHandleType
        :value: NativeHandleType.CUtexObject
    
    .. py:attribute:: slangpy.NativeHandleType.CUstream
        :type: NativeHandleType
        :value: NativeHandleType.CUstream
    
    .. py:attribute:: slangpy.NativeHandleType.CUcontext
        :type: NativeHandleType
        :value: NativeHandleType.CUcontext
    
    .. py:attribute:: slangpy.NativeHandleType.OptixDeviceContext
        :type: NativeHandleType
        :value: NativeHandleType.OptixDeviceContext
    
    .. py:attribute:: slangpy.NativeHandleType.OptixTraversableHandle
        :type: NativeHandleType
        :value: NativeHandleType.OptixTraversableHandle
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUDevice
        :type: NativeHandleType
        :value: NativeHandleType.WGPUDevice
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUBuffer
        :type: NativeHandleType
        :value: NativeHandleType.WGPUBuffer
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUTexture
        :type: NativeHandleType
        :value: NativeHandleType.WGPUTexture
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUSampler
        :type: NativeHandleType
        :value: NativeHandleType.WGPUSampler
    
    .. py:attribute:: slangpy.NativeHandleType.WGPURenderPipeline
        :type: NativeHandleType
        :value: NativeHandleType.WGPURenderPipeline
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUComputePipeline
        :type: NativeHandleType
        :value: NativeHandleType.WGPUComputePipeline
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUQueue
        :type: NativeHandleType
        :value: NativeHandleType.WGPUQueue
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUCommandBuffer
        :type: NativeHandleType
        :value: NativeHandleType.WGPUCommandBuffer
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUTextureView
        :type: NativeHandleType
        :value: NativeHandleType.WGPUTextureView
    
    .. py:attribute:: slangpy.NativeHandleType.WGPUCommandEncoder
        :type: NativeHandleType
        :value: NativeHandleType.WGPUCommandEncoder
    


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
    
    .. py:attribute:: slangpy.PrimitiveTopology.point_list
        :type: PrimitiveTopology
        :value: PrimitiveTopology.point_list
    
    .. py:attribute:: slangpy.PrimitiveTopology.line_list
        :type: PrimitiveTopology
        :value: PrimitiveTopology.line_list
    
    .. py:attribute:: slangpy.PrimitiveTopology.line_strip
        :type: PrimitiveTopology
        :value: PrimitiveTopology.line_strip
    
    .. py:attribute:: slangpy.PrimitiveTopology.triangle_list
        :type: PrimitiveTopology
        :value: PrimitiveTopology.triangle_list
    
    .. py:attribute:: slangpy.PrimitiveTopology.triangle_strip
        :type: PrimitiveTopology
        :value: PrimitiveTopology.triangle_strip
    
    .. py:attribute:: slangpy.PrimitiveTopology.patch_list
        :type: PrimitiveTopology
        :value: PrimitiveTopology.patch_list
    


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
    
    .. py:attribute:: slangpy.QueryType.timestamp
        :type: QueryType
        :value: QueryType.timestamp
    
    .. py:attribute:: slangpy.QueryType.acceleration_structure_compacted_size
        :type: QueryType
        :value: QueryType.acceleration_structure_compacted_size
    
    .. py:attribute:: slangpy.QueryType.acceleration_structure_current_size
        :type: QueryType
        :value: QueryType.acceleration_structure_current_size
    


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
    
    .. py:property:: label
        :type: str
    


----

.. py:class:: slangpy.RayTracingPipelineFlags

    Base class: :py:class:`enum.IntFlag`
    
    .. py:attribute:: slangpy.RayTracingPipelineFlags.none
        :type: RayTracingPipelineFlags
        :value: 0
    
    .. py:attribute:: slangpy.RayTracingPipelineFlags.skip_triangles
        :type: RayTracingPipelineFlags
        :value: 1
    
    .. py:attribute:: slangpy.RayTracingPipelineFlags.skip_procedurals
        :type: RayTracingPipelineFlags
        :value: 2
    
    .. py:attribute:: slangpy.RayTracingPipelineFlags.enable_spheres
        :type: RayTracingPipelineFlags
        :value: 4
    
    .. py:attribute:: slangpy.RayTracingPipelineFlags.enable_linear_swept_spheres
        :type: RayTracingPipelineFlags
        :value: 8
    


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
    
    .. py:method:: draw_indirect(self, max_draw_count: int, arg_buffer: slangpy.BufferOffsetPair, count_buffer: slangpy.BufferOffsetPair = <slangpy.BufferOffsetPair object at 0x0000021D5F51D8F0>) -> None
    
    .. py:method:: draw_indexed_indirect(self, max_draw_count: int, arg_buffer: slangpy.BufferOffsetPair, count_buffer: slangpy.BufferOffsetPair = <slangpy.BufferOffsetPair object at 0x0000021D5F51D920>) -> None
    
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
    
    .. py:attribute:: slangpy.RenderTargetWriteMask.none
        :type: RenderTargetWriteMask
        :value: 0
    
    .. py:attribute:: slangpy.RenderTargetWriteMask.red
        :type: RenderTargetWriteMask
        :value: 1
    
    .. py:attribute:: slangpy.RenderTargetWriteMask.green
        :type: RenderTargetWriteMask
        :value: 2
    
    .. py:attribute:: slangpy.RenderTargetWriteMask.blue
        :type: RenderTargetWriteMask
        :value: 4
    
    .. py:attribute:: slangpy.RenderTargetWriteMask.alpha
        :type: RenderTargetWriteMask
        :value: 8
    
    .. py:attribute:: slangpy.RenderTargetWriteMask.all
        :type: RenderTargetWriteMask
        :value: 15
    


----

.. py:class:: slangpy.Resource

    Base class: :py:class:`slangpy.DeviceChild`
    
    
    
    .. py:property:: native_handle
        :type: slangpy.NativeHandle
    
        Get the native resource handle.
        


----

.. py:class:: slangpy.ResourceState

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.ResourceState.undefined
        :type: ResourceState
        :value: ResourceState.undefined
    
    .. py:attribute:: slangpy.ResourceState.general
        :type: ResourceState
        :value: ResourceState.general
    
    .. py:attribute:: slangpy.ResourceState.vertex_buffer
        :type: ResourceState
        :value: ResourceState.vertex_buffer
    
    .. py:attribute:: slangpy.ResourceState.index_buffer
        :type: ResourceState
        :value: ResourceState.index_buffer
    
    .. py:attribute:: slangpy.ResourceState.constant_buffer
        :type: ResourceState
        :value: ResourceState.constant_buffer
    
    .. py:attribute:: slangpy.ResourceState.stream_output
        :type: ResourceState
        :value: ResourceState.stream_output
    
    .. py:attribute:: slangpy.ResourceState.shader_resource
        :type: ResourceState
        :value: ResourceState.shader_resource
    
    .. py:attribute:: slangpy.ResourceState.unordered_access
        :type: ResourceState
        :value: ResourceState.unordered_access
    
    .. py:attribute:: slangpy.ResourceState.render_target
        :type: ResourceState
        :value: ResourceState.render_target
    
    .. py:attribute:: slangpy.ResourceState.depth_read
        :type: ResourceState
        :value: ResourceState.depth_read
    
    .. py:attribute:: slangpy.ResourceState.depth_write
        :type: ResourceState
        :value: ResourceState.depth_write
    
    .. py:attribute:: slangpy.ResourceState.present
        :type: ResourceState
        :value: ResourceState.present
    
    .. py:attribute:: slangpy.ResourceState.indirect_argument
        :type: ResourceState
        :value: ResourceState.indirect_argument
    
    .. py:attribute:: slangpy.ResourceState.copy_source
        :type: ResourceState
        :value: ResourceState.copy_source
    
    .. py:attribute:: slangpy.ResourceState.copy_destination
        :type: ResourceState
        :value: ResourceState.copy_destination
    
    .. py:attribute:: slangpy.ResourceState.resolve_source
        :type: ResourceState
        :value: ResourceState.resolve_source
    
    .. py:attribute:: slangpy.ResourceState.resolve_destination
        :type: ResourceState
        :value: ResourceState.resolve_destination
    
    .. py:attribute:: slangpy.ResourceState.acceleration_structure_read
        :type: ResourceState
        :value: ResourceState.acceleration_structure_read
    
    .. py:attribute:: slangpy.ResourceState.acceleration_structure_write
        :type: ResourceState
        :value: ResourceState.acceleration_structure_write
    
    .. py:attribute:: slangpy.ResourceState.acceleration_structure_build_output
        :type: ResourceState
        :value: ResourceState.acceleration_structure_build_output
    


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
    
    .. py:attribute:: slangpy.ShaderStage.none
        :type: ShaderStage
        :value: ShaderStage.none
    
    .. py:attribute:: slangpy.ShaderStage.vertex
        :type: ShaderStage
        :value: ShaderStage.vertex
    
    .. py:attribute:: slangpy.ShaderStage.hull
        :type: ShaderStage
        :value: ShaderStage.hull
    
    .. py:attribute:: slangpy.ShaderStage.domain
        :type: ShaderStage
        :value: ShaderStage.domain
    
    .. py:attribute:: slangpy.ShaderStage.geometry
        :type: ShaderStage
        :value: ShaderStage.geometry
    
    .. py:attribute:: slangpy.ShaderStage.fragment
        :type: ShaderStage
        :value: ShaderStage.fragment
    
    .. py:attribute:: slangpy.ShaderStage.compute
        :type: ShaderStage
        :value: ShaderStage.compute
    
    .. py:attribute:: slangpy.ShaderStage.ray_generation
        :type: ShaderStage
        :value: ShaderStage.ray_generation
    
    .. py:attribute:: slangpy.ShaderStage.intersection
        :type: ShaderStage
        :value: ShaderStage.intersection
    
    .. py:attribute:: slangpy.ShaderStage.any_hit
        :type: ShaderStage
        :value: ShaderStage.any_hit
    
    .. py:attribute:: slangpy.ShaderStage.closest_hit
        :type: ShaderStage
        :value: ShaderStage.closest_hit
    
    .. py:attribute:: slangpy.ShaderStage.miss
        :type: ShaderStage
        :value: ShaderStage.miss
    
    .. py:attribute:: slangpy.ShaderStage.callable
        :type: ShaderStage
        :value: ShaderStage.callable
    
    .. py:attribute:: slangpy.ShaderStage.mesh
        :type: ShaderStage
        :value: ShaderStage.mesh
    
    .. py:attribute:: slangpy.ShaderStage.amplification
        :type: ShaderStage
        :value: ShaderStage.amplification
    


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
        
    .. py:property:: profile
        :type: str
    
        Specifies the target profile to use (e.g. "sm_6_6"). Defaults to the
        device's default profile when empty.
        
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
        downstream compiler.
        
    .. py:property:: dump_intermediates
        :type: bool
    
        When set will dump the intermediate source output.
        
    .. py:property:: dump_intermediates_prefix
        :type: str
    
        The file name prefix for the intermediate source output.
        


----

.. py:class:: slangpy.SlangDebugInfoLevel

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.SlangDebugInfoLevel.none
        :type: SlangDebugInfoLevel
        :value: SlangDebugInfoLevel.none
    
    .. py:attribute:: slangpy.SlangDebugInfoLevel.minimal
        :type: SlangDebugInfoLevel
        :value: SlangDebugInfoLevel.minimal
    
    .. py:attribute:: slangpy.SlangDebugInfoLevel.standard
        :type: SlangDebugInfoLevel
        :value: SlangDebugInfoLevel.standard
    
    .. py:attribute:: slangpy.SlangDebugInfoLevel.maximal
        :type: SlangDebugInfoLevel
        :value: SlangDebugInfoLevel.maximal
    


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
    
    .. py:attribute:: slangpy.SlangFloatingPointMode.default
        :type: SlangFloatingPointMode
        :value: SlangFloatingPointMode.default
    
    .. py:attribute:: slangpy.SlangFloatingPointMode.fast
        :type: SlangFloatingPointMode
        :value: SlangFloatingPointMode.fast
    
    .. py:attribute:: slangpy.SlangFloatingPointMode.precise
        :type: SlangFloatingPointMode
        :value: SlangFloatingPointMode.precise
    


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
        downstream compiler.
        
    .. py:property:: dump_intermediates
        :type: bool | None
    
        When set will dump the intermediate source output.
        
    .. py:property:: dump_intermediates_prefix
        :type: str | None
    
        The file name prefix for the intermediate source output.
        


----

.. py:class:: slangpy.SlangMatrixLayout

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.SlangMatrixLayout.row_major
        :type: SlangMatrixLayout
        :value: SlangMatrixLayout.row_major
    
    .. py:attribute:: slangpy.SlangMatrixLayout.column_major
        :type: SlangMatrixLayout
        :value: SlangMatrixLayout.column_major
    


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
    
    .. py:attribute:: slangpy.SlangOptimizationLevel.none
        :type: SlangOptimizationLevel
        :value: SlangOptimizationLevel.none
    
    .. py:attribute:: slangpy.SlangOptimizationLevel.default
        :type: SlangOptimizationLevel
        :value: SlangOptimizationLevel.default
    
    .. py:attribute:: slangpy.SlangOptimizationLevel.high
        :type: SlangOptimizationLevel
        :value: SlangOptimizationLevel.high
    
    .. py:attribute:: slangpy.SlangOptimizationLevel.maximal
        :type: SlangOptimizationLevel
        :value: SlangOptimizationLevel.maximal
    


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
    
    .. py:attribute:: slangpy.StencilOp.keep
        :type: StencilOp
        :value: StencilOp.keep
    
    .. py:attribute:: slangpy.StencilOp.zero
        :type: StencilOp
        :value: StencilOp.zero
    
    .. py:attribute:: slangpy.StencilOp.replace
        :type: StencilOp
        :value: StencilOp.replace
    
    .. py:attribute:: slangpy.StencilOp.increment_saturate
        :type: StencilOp
        :value: StencilOp.increment_saturate
    
    .. py:attribute:: slangpy.StencilOp.decrement_saturate
        :type: StencilOp
        :value: StencilOp.decrement_saturate
    
    .. py:attribute:: slangpy.StencilOp.invert
        :type: StencilOp
        :value: StencilOp.invert
    
    .. py:attribute:: slangpy.StencilOp.increment_wrap
        :type: StencilOp
        :value: StencilOp.increment_wrap
    
    .. py:attribute:: slangpy.StencilOp.decrement_wrap
        :type: StencilOp
        :value: StencilOp.decrement_wrap
    


----

.. py:class:: slangpy.StoreOp

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.StoreOp.store
        :type: StoreOp
        :value: StoreOp.store
    
    .. py:attribute:: slangpy.StoreOp.dont_care
        :type: StoreOp
        :value: StoreOp.dont_care
    


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
    
    .. py:attribute:: slangpy.TextureAddressingMode.wrap
        :type: TextureAddressingMode
        :value: TextureAddressingMode.wrap
    
    .. py:attribute:: slangpy.TextureAddressingMode.clamp_to_edge
        :type: TextureAddressingMode
        :value: TextureAddressingMode.clamp_to_edge
    
    .. py:attribute:: slangpy.TextureAddressingMode.clamp_to_border
        :type: TextureAddressingMode
        :value: TextureAddressingMode.clamp_to_border
    
    .. py:attribute:: slangpy.TextureAddressingMode.mirror_repeat
        :type: TextureAddressingMode
        :value: TextureAddressingMode.mirror_repeat
    
    .. py:attribute:: slangpy.TextureAddressingMode.mirror_once
        :type: TextureAddressingMode
        :value: TextureAddressingMode.mirror_once
    


----

.. py:class:: slangpy.TextureAspect

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.TextureAspect.all
        :type: TextureAspect
        :value: TextureAspect.all
    
    .. py:attribute:: slangpy.TextureAspect.depth_only
        :type: TextureAspect
        :value: TextureAspect.depth_only
    
    .. py:attribute:: slangpy.TextureAspect.stencil_only
        :type: TextureAspect
        :value: TextureAspect.stencil_only
    


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
    
    .. py:attribute:: slangpy.TextureFilteringMode.point
        :type: TextureFilteringMode
        :value: TextureFilteringMode.point
    
    .. py:attribute:: slangpy.TextureFilteringMode.linear
        :type: TextureFilteringMode
        :value: TextureFilteringMode.linear
    


----

.. py:class:: slangpy.TextureReductionOp

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.TextureReductionOp.average
        :type: TextureReductionOp
        :value: TextureReductionOp.average
    
    .. py:attribute:: slangpy.TextureReductionOp.comparison
        :type: TextureReductionOp
        :value: TextureReductionOp.comparison
    
    .. py:attribute:: slangpy.TextureReductionOp.minimum
        :type: TextureReductionOp
        :value: TextureReductionOp.minimum
    
    .. py:attribute:: slangpy.TextureReductionOp.maximum
        :type: TextureReductionOp
        :value: TextureReductionOp.maximum
    


----

.. py:class:: slangpy.TextureType

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.TextureType.texture_1d
        :type: TextureType
        :value: TextureType.texture_1d
    
    .. py:attribute:: slangpy.TextureType.texture_1d_array
        :type: TextureType
        :value: TextureType.texture_1d_array
    
    .. py:attribute:: slangpy.TextureType.texture_2d
        :type: TextureType
        :value: TextureType.texture_2d
    
    .. py:attribute:: slangpy.TextureType.texture_2d_array
        :type: TextureType
        :value: TextureType.texture_2d_array
    
    .. py:attribute:: slangpy.TextureType.texture_2d_ms
        :type: TextureType
        :value: TextureType.texture_2d_ms
    
    .. py:attribute:: slangpy.TextureType.texture_2d_ms_array
        :type: TextureType
        :value: TextureType.texture_2d_ms_array
    
    .. py:attribute:: slangpy.TextureType.texture_3d
        :type: TextureType
        :value: TextureType.texture_3d
    
    .. py:attribute:: slangpy.TextureType.texture_cube
        :type: TextureType
        :value: TextureType.texture_cube
    
    .. py:attribute:: slangpy.TextureType.texture_cube_array
        :type: TextureType
        :value: TextureType.texture_cube_array
    


----

.. py:class:: slangpy.TextureUsage

    Base class: :py:class:`enum.IntFlag`
    
    .. py:attribute:: slangpy.TextureUsage.none
        :type: TextureUsage
        :value: 0
    
    .. py:attribute:: slangpy.TextureUsage.shader_resource
        :type: TextureUsage
        :value: 1
    
    .. py:attribute:: slangpy.TextureUsage.unordered_access
        :type: TextureUsage
        :value: 2
    
    .. py:attribute:: slangpy.TextureUsage.render_target
        :type: TextureUsage
        :value: 4
    
    .. py:attribute:: slangpy.TextureUsage.depth_stencil
        :type: TextureUsage
        :value: 8
    
    .. py:attribute:: slangpy.TextureUsage.present
        :type: TextureUsage
        :value: 16
    
    .. py:attribute:: slangpy.TextureUsage.copy_source
        :type: TextureUsage
        :value: 32
    
    .. py:attribute:: slangpy.TextureUsage.copy_destination
        :type: TextureUsage
        :value: 64
    
    .. py:attribute:: slangpy.TextureUsage.resolve_source
        :type: TextureUsage
        :value: 128
    
    .. py:attribute:: slangpy.TextureUsage.resolve_destination
        :type: TextureUsage
        :value: 256
    
    .. py:attribute:: slangpy.TextureUsage.typeless
        :type: TextureUsage
        :value: 512
    
    .. py:attribute:: slangpy.TextureUsage.shared
        :type: TextureUsage
        :value: 1024
    


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
        
        
        
        .. py:attribute:: slangpy.TypeReflection.Kind.none
            :type: Kind
            :value: Kind.none
        
        .. py:attribute:: slangpy.TypeReflection.Kind.struct
            :type: Kind
            :value: Kind.struct
        
        .. py:attribute:: slangpy.TypeReflection.Kind.array
            :type: Kind
            :value: Kind.array
        
        .. py:attribute:: slangpy.TypeReflection.Kind.matrix
            :type: Kind
            :value: Kind.matrix
        
        .. py:attribute:: slangpy.TypeReflection.Kind.vector
            :type: Kind
            :value: Kind.vector
        
        .. py:attribute:: slangpy.TypeReflection.Kind.scalar
            :type: Kind
            :value: Kind.scalar
        
        .. py:attribute:: slangpy.TypeReflection.Kind.constant_buffer
            :type: Kind
            :value: Kind.constant_buffer
        
        .. py:attribute:: slangpy.TypeReflection.Kind.resource
            :type: Kind
            :value: Kind.resource
        
        .. py:attribute:: slangpy.TypeReflection.Kind.sampler_state
            :type: Kind
            :value: Kind.sampler_state
        
        .. py:attribute:: slangpy.TypeReflection.Kind.texture_buffer
            :type: Kind
            :value: Kind.texture_buffer
        
        .. py:attribute:: slangpy.TypeReflection.Kind.shader_storage_buffer
            :type: Kind
            :value: Kind.shader_storage_buffer
        
        .. py:attribute:: slangpy.TypeReflection.Kind.parameter_block
            :type: Kind
            :value: Kind.parameter_block
        
        .. py:attribute:: slangpy.TypeReflection.Kind.generic_type_parameter
            :type: Kind
            :value: Kind.generic_type_parameter
        
        .. py:attribute:: slangpy.TypeReflection.Kind.interface
            :type: Kind
            :value: Kind.interface
        
        .. py:attribute:: slangpy.TypeReflection.Kind.output_stream
            :type: Kind
            :value: Kind.output_stream
        
        .. py:attribute:: slangpy.TypeReflection.Kind.specialized
            :type: Kind
            :value: Kind.specialized
        
        .. py:attribute:: slangpy.TypeReflection.Kind.feedback
            :type: Kind
            :value: Kind.feedback
        
        .. py:attribute:: slangpy.TypeReflection.Kind.pointer
            :type: Kind
            :value: Kind.pointer
        
    .. py:class:: slangpy.TypeReflection.ScalarType
    
        Base class: :py:class:`enum.Enum`
        
        
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.none
            :type: ScalarType
            :value: ScalarType.none
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.void
            :type: ScalarType
            :value: ScalarType.void
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.bool
            :type: ScalarType
            :value: ScalarType.bool
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.int32
            :type: ScalarType
            :value: ScalarType.int32
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.uint32
            :type: ScalarType
            :value: ScalarType.uint32
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.int64
            :type: ScalarType
            :value: ScalarType.int64
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.uint64
            :type: ScalarType
            :value: ScalarType.uint64
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.float16
            :type: ScalarType
            :value: ScalarType.float16
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.float32
            :type: ScalarType
            :value: ScalarType.float32
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.float64
            :type: ScalarType
            :value: ScalarType.float64
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.int8
            :type: ScalarType
            :value: ScalarType.int8
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.uint8
            :type: ScalarType
            :value: ScalarType.uint8
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.int16
            :type: ScalarType
            :value: ScalarType.int16
        
        .. py:attribute:: slangpy.TypeReflection.ScalarType.uint16
            :type: ScalarType
            :value: ScalarType.uint16
        
    .. py:class:: slangpy.TypeReflection.ResourceShape
    
        Base class: :py:class:`enum.Enum`
        
        
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.none
            :type: ResourceShape
            :value: ResourceShape.none
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_1d
            :type: ResourceShape
            :value: ResourceShape.texture_1d
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_2d
            :type: ResourceShape
            :value: ResourceShape.texture_2d
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_3d
            :type: ResourceShape
            :value: ResourceShape.texture_3d
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_cube
            :type: ResourceShape
            :value: ResourceShape.texture_cube
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_buffer
            :type: ResourceShape
            :value: ResourceShape.texture_buffer
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.structured_buffer
            :type: ResourceShape
            :value: ResourceShape.structured_buffer
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.byte_address_buffer
            :type: ResourceShape
            :value: ResourceShape.byte_address_buffer
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.unknown
            :type: ResourceShape
            :value: ResourceShape.unknown
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.acceleration_structure
            :type: ResourceShape
            :value: ResourceShape.acceleration_structure
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_feedback_flag
            :type: ResourceShape
            :value: ResourceShape.texture_feedback_flag
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_array_flag
            :type: ResourceShape
            :value: ResourceShape.texture_array_flag
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_multisample_flag
            :type: ResourceShape
            :value: ResourceShape.texture_multisample_flag
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_1d_array
            :type: ResourceShape
            :value: ResourceShape.texture_1d_array
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_2d_array
            :type: ResourceShape
            :value: ResourceShape.texture_2d_array
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_cube_array
            :type: ResourceShape
            :value: ResourceShape.texture_cube_array
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_2d_multisample
            :type: ResourceShape
            :value: ResourceShape.texture_2d_multisample
        
        .. py:attribute:: slangpy.TypeReflection.ResourceShape.texture_2d_multisample_array
            :type: ResourceShape
            :value: ResourceShape.texture_2d_multisample_array
        
    .. py:class:: slangpy.TypeReflection.ResourceAccess
    
        Base class: :py:class:`enum.Enum`
        
        
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.none
            :type: ResourceAccess
            :value: ResourceAccess.none
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.read
            :type: ResourceAccess
            :value: ResourceAccess.read
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.read_write
            :type: ResourceAccess
            :value: ResourceAccess.read_write
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.raster_ordered
            :type: ResourceAccess
            :value: ResourceAccess.raster_ordered
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.access_append
            :type: ResourceAccess
            :value: ResourceAccess.access_append
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.access_consume
            :type: ResourceAccess
            :value: ResourceAccess.access_consume
        
        .. py:attribute:: slangpy.TypeReflection.ResourceAccess.access_write
            :type: ResourceAccess
            :value: ResourceAccess.access_write
        
    .. py:class:: slangpy.TypeReflection.ParameterCategory
    
        Base class: :py:class:`enum.Enum`
        
        
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.none
            :type: ParameterCategory
            :value: ParameterCategory.none
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.mixed
            :type: ParameterCategory
            :value: ParameterCategory.mixed
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.constant_buffer
            :type: ParameterCategory
            :value: ParameterCategory.constant_buffer
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.shader_resource
            :type: ParameterCategory
            :value: ParameterCategory.shader_resource
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.unordered_access
            :type: ParameterCategory
            :value: ParameterCategory.unordered_access
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.varying_input
            :type: ParameterCategory
            :value: ParameterCategory.varying_input
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.varying_output
            :type: ParameterCategory
            :value: ParameterCategory.varying_output
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.sampler_state
            :type: ParameterCategory
            :value: ParameterCategory.sampler_state
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.uniform
            :type: ParameterCategory
            :value: ParameterCategory.uniform
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.descriptor_table_slot
            :type: ParameterCategory
            :value: ParameterCategory.descriptor_table_slot
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.specialization_constant
            :type: ParameterCategory
            :value: ParameterCategory.specialization_constant
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.push_constant_buffer
            :type: ParameterCategory
            :value: ParameterCategory.push_constant_buffer
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.register_space
            :type: ParameterCategory
            :value: ParameterCategory.register_space
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.generic
            :type: ParameterCategory
            :value: ParameterCategory.generic
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.ray_payload
            :type: ParameterCategory
            :value: ParameterCategory.ray_payload
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.hit_attributes
            :type: ParameterCategory
            :value: ParameterCategory.hit_attributes
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.callable_payload
            :type: ParameterCategory
            :value: ParameterCategory.callable_payload
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.shader_record
            :type: ParameterCategory
            :value: ParameterCategory.shader_record
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.existential_type_param
            :type: ParameterCategory
            :value: ParameterCategory.existential_type_param
        
        .. py:attribute:: slangpy.TypeReflection.ParameterCategory.existential_object_param
            :type: ParameterCategory
            :value: ParameterCategory.existential_object_param
        
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
    :canonical: slangpy.math.float1
    
    Alias class: :py:class:`slangpy.math.float1`
    


----

.. py:class:: slangpy.float2
    :canonical: slangpy.math.float2
    
    Alias class: :py:class:`slangpy.math.float2`
    


----

.. py:class:: slangpy.float3
    :canonical: slangpy.math.float3
    
    Alias class: :py:class:`slangpy.math.float3`
    


----

.. py:class:: slangpy.float4
    :canonical: slangpy.math.float4
    
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
    :canonical: slangpy.math.int1
    
    Alias class: :py:class:`slangpy.math.int1`
    


----

.. py:class:: slangpy.int2
    :canonical: slangpy.math.int2
    
    Alias class: :py:class:`slangpy.math.int2`
    


----

.. py:class:: slangpy.int3
    :canonical: slangpy.math.int3
    
    Alias class: :py:class:`slangpy.math.int3`
    


----

.. py:class:: slangpy.int4
    :canonical: slangpy.math.int4
    
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
    :canonical: slangpy.math.uint1
    
    Alias class: :py:class:`slangpy.math.uint1`
    


----

.. py:class:: slangpy.uint2
    :canonical: slangpy.math.uint2
    
    Alias class: :py:class:`slangpy.math.uint2`
    


----

.. py:class:: slangpy.uint3
    :canonical: slangpy.math.uint3
    
    Alias class: :py:class:`slangpy.math.uint3`
    


----

.. py:class:: slangpy.uint4
    :canonical: slangpy.math.uint4
    
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
    :canonical: slangpy.math.bool1
    
    Alias class: :py:class:`slangpy.math.bool1`
    


----

.. py:class:: slangpy.bool2
    :canonical: slangpy.math.bool2
    
    Alias class: :py:class:`slangpy.math.bool2`
    


----

.. py:class:: slangpy.bool3
    :canonical: slangpy.math.bool3
    
    Alias class: :py:class:`slangpy.math.bool3`
    


----

.. py:class:: slangpy.bool4
    :canonical: slangpy.math.bool4
    
    Alias class: :py:class:`slangpy.math.bool4`
    


----

.. py:class:: slangpy.math.float16_t

    .. py:method:: __init__(self, value: float) -> None
    
    .. py:method:: __init__(self, value: float) -> None
        :no-index:
    


----

.. py:class:: slangpy.float16_t
    :canonical: slangpy.math.float16_t
    
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
    :canonical: slangpy.math.float16_t1
    
    Alias class: :py:class:`slangpy.math.float16_t1`
    


----

.. py:class:: slangpy.float16_t2
    :canonical: slangpy.math.float16_t2
    
    Alias class: :py:class:`slangpy.math.float16_t2`
    


----

.. py:class:: slangpy.float16_t3
    :canonical: slangpy.math.float16_t3
    
    Alias class: :py:class:`slangpy.math.float16_t3`
    


----

.. py:class:: slangpy.float16_t4
    :canonical: slangpy.math.float16_t4
    
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
    :canonical: slangpy.math.float2x2
    
    Alias class: :py:class:`slangpy.math.float2x2`
    


----

.. py:class:: slangpy.float2x3
    :canonical: slangpy.math.float2x3
    
    Alias class: :py:class:`slangpy.math.float2x3`
    


----

.. py:class:: slangpy.float2x4
    :canonical: slangpy.math.float2x4
    
    Alias class: :py:class:`slangpy.math.float2x4`
    


----

.. py:class:: slangpy.float3x2
    :canonical: slangpy.math.float3x2
    
    Alias class: :py:class:`slangpy.math.float3x2`
    


----

.. py:class:: slangpy.float3x3
    :canonical: slangpy.math.float3x3
    
    Alias class: :py:class:`slangpy.math.float3x3`
    


----

.. py:class:: slangpy.float3x4
    :canonical: slangpy.math.float3x4
    
    Alias class: :py:class:`slangpy.math.float3x4`
    


----

.. py:class:: slangpy.float4x2
    :canonical: slangpy.math.float4x2
    
    Alias class: :py:class:`slangpy.math.float4x2`
    


----

.. py:class:: slangpy.float4x3
    :canonical: slangpy.math.float4x3
    
    Alias class: :py:class:`slangpy.math.float4x3`
    


----

.. py:class:: slangpy.float4x4
    :canonical: slangpy.math.float4x4
    
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
    :canonical: slangpy.math.quatf
    
    Alias class: :py:class:`slangpy.math.quatf`
    


----

.. py:class:: slangpy.math.Handedness

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.math.Handedness.right_handed
        :type: Handedness
        :value: Handedness.right_handed
    
    .. py:attribute:: slangpy.math.Handedness.left_handed
        :type: Handedness
        :value: Handedness.left_handed
    


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
    
    
    
    .. py:attribute:: slangpy.ui.SliderFlags.none
        :type: SliderFlags
        :value: 0
    
    .. py:attribute:: slangpy.ui.SliderFlags.always_clamp
        :type: SliderFlags
        :value: 1536
    
    .. py:attribute:: slangpy.ui.SliderFlags.logarithmic
        :type: SliderFlags
        :value: 32
    
    .. py:attribute:: slangpy.ui.SliderFlags.no_round_to_format
        :type: SliderFlags
        :value: 64
    
    .. py:attribute:: slangpy.ui.SliderFlags.no_input
        :type: SliderFlags
        :value: 128
    


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
    
    
    
    .. py:attribute:: slangpy.ui.InputTextFlags.none
        :type: InputTextFlags
        :value: 0
    
    .. py:attribute:: slangpy.ui.InputTextFlags.chars_decimal
        :type: InputTextFlags
        :value: 1
    
    .. py:attribute:: slangpy.ui.InputTextFlags.chars_hexadecimal
        :type: InputTextFlags
        :value: 2
    
    .. py:attribute:: slangpy.ui.InputTextFlags.chars_uppercase
        :type: InputTextFlags
        :value: 8
    
    .. py:attribute:: slangpy.ui.InputTextFlags.chars_no_blank
        :type: InputTextFlags
        :value: 16
    
    .. py:attribute:: slangpy.ui.InputTextFlags.auto_select_all
        :type: InputTextFlags
        :value: 4096
    
    .. py:attribute:: slangpy.ui.InputTextFlags.enter_returns_true
        :type: InputTextFlags
        :value: 64
    
    .. py:attribute:: slangpy.ui.InputTextFlags.callback_completion
        :type: InputTextFlags
        :value: 262144
    
    .. py:attribute:: slangpy.ui.InputTextFlags.callback_history
        :type: InputTextFlags
        :value: 524288
    
    .. py:attribute:: slangpy.ui.InputTextFlags.callback_always
        :type: InputTextFlags
        :value: 1048576
    
    .. py:attribute:: slangpy.ui.InputTextFlags.callback_char_filter
        :type: InputTextFlags
        :value: 2097152
    
    .. py:attribute:: slangpy.ui.InputTextFlags.allow_tab_input
        :type: InputTextFlags
        :value: 32
    
    .. py:attribute:: slangpy.ui.InputTextFlags.ctrl_enter_for_new_line
        :type: InputTextFlags
        :value: 256
    
    .. py:attribute:: slangpy.ui.InputTextFlags.no_horizontal_scroll
        :type: InputTextFlags
        :value: 32768
    
    .. py:attribute:: slangpy.ui.InputTextFlags.always_overwrite
        :type: InputTextFlags
        :value: 2048
    
    .. py:attribute:: slangpy.ui.InputTextFlags.read_only
        :type: InputTextFlags
        :value: 512
    
    .. py:attribute:: slangpy.ui.InputTextFlags.password
        :type: InputTextFlags
        :value: 1024
    
    .. py:attribute:: slangpy.ui.InputTextFlags.no_undo_redo
        :type: InputTextFlags
        :value: 65536
    
    .. py:attribute:: slangpy.ui.InputTextFlags.chars_scientific
        :type: InputTextFlags
        :value: 4
    
    .. py:attribute:: slangpy.ui.InputTextFlags.escape_clears_all
        :type: InputTextFlags
        :value: 128
    


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
    
        Load a texture from an image file.
        
        Parameter ``path``:
            Image file path.
        
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
    
    To end the frame capture, call ``end_frame_capture``().
    
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
    
    .. py:attribute:: slangpy.slangpy.AccessType.none
        :type: AccessType
        :value: AccessType.none
    
    .. py:attribute:: slangpy.slangpy.AccessType.read
        :type: AccessType
        :value: AccessType.read
    
    .. py:attribute:: slangpy.slangpy.AccessType.write
        :type: AccessType
        :value: AccessType.write
    
    .. py:attribute:: slangpy.slangpy.AccessType.readwrite
        :type: AccessType
        :value: AccessType.readwrite
    


----

.. py:class:: slangpy.slangpy.CallMode

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.slangpy.CallMode.prim
        :type: CallMode
        :value: CallMode.prim
    
    .. py:attribute:: slangpy.slangpy.CallMode.bwds
        :type: CallMode
        :value: CallMode.bwds
    
    .. py:attribute:: slangpy.slangpy.CallMode.fwds
        :type: CallMode
        :value: CallMode.fwds
    


----

.. py:class:: slangpy.slangpy.AutogradAccess

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.slangpy.AutogradAccess.none
        :type: AutogradAccess
        :value: AutogradAccess.none
    
    .. py:attribute:: slangpy.slangpy.AutogradAccess.read
        :type: AutogradAccess
        :value: AutogradAccess.read
    
    .. py:attribute:: slangpy.slangpy.AutogradAccess.write
        :type: AutogradAccess
        :value: AutogradAccess.write
    
    .. py:attribute:: slangpy.slangpy.AutogradAccess.readwrite
        :type: AutogradAccess
        :value: AutogradAccess.readwrite
    


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
    
    .. py:attribute:: slangpy.slangpy.FunctionNodeType.unknown
        :type: FunctionNodeType
        :value: FunctionNodeType.unknown
    
    .. py:attribute:: slangpy.slangpy.FunctionNodeType.uniforms
        :type: FunctionNodeType
        :value: FunctionNodeType.uniforms
    
    .. py:attribute:: slangpy.slangpy.FunctionNodeType.kernelgen
        :type: FunctionNodeType
        :value: FunctionNodeType.kernelgen
    
    .. py:attribute:: slangpy.slangpy.FunctionNodeType.this
        :type: FunctionNodeType
        :value: FunctionNodeType.this
    
    .. py:attribute:: slangpy.slangpy.FunctionNodeType.cuda_stream
        :type: FunctionNodeType
        :value: FunctionNodeType.cuda_stream
    
    .. py:attribute:: slangpy.slangpy.FunctionNodeType.ray_tracing
        :type: FunctionNodeType
        :value: FunctionNodeType.ray_tracing
    


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



----

.. py:data:: slangpy.package_dir
    :type: str
    :value: "C:\src\slangpy\slangpy"



----

.. py:data:: slangpy.build_dir
    :type: str
    :value: "C:/src/slangpy/build/windows-msvc/Debug"



----

.. py:data:: slangpy.SLANG_BUILD_TAG
    :type: str
    :value: "2026.12"



----

.. py:class:: slangpy.FilterBoundaryCondition

    Base class: :py:class:`enum.Enum`
    
    Filter boundary condition used for resampling images.
    
    .. py:attribute:: slangpy.FilterBoundaryCondition.clamp
        :type: FilterBoundaryCondition
        :value: FilterBoundaryCondition.clamp
    
    .. py:attribute:: slangpy.FilterBoundaryCondition.repeat
        :type: FilterBoundaryCondition
        :value: FilterBoundaryCondition.repeat
    
    .. py:attribute:: slangpy.FilterBoundaryCondition.mirror
        :type: FilterBoundaryCondition
        :value: FilterBoundaryCondition.mirror
    
    .. py:attribute:: slangpy.FilterBoundaryCondition.zero
        :type: FilterBoundaryCondition
        :value: FilterBoundaryCondition.zero
    
    .. py:attribute:: slangpy.FilterBoundaryCondition.one
        :type: FilterBoundaryCondition
        :value: FilterBoundaryCondition.one
    


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
    
    .. py:attribute:: slangpy.DescriptorHandleType.undefined
        :type: DescriptorHandleType
        :value: DescriptorHandleType.undefined
    
    .. py:attribute:: slangpy.DescriptorHandleType.buffer
        :type: DescriptorHandleType
        :value: DescriptorHandleType.buffer
    
    .. py:attribute:: slangpy.DescriptorHandleType.rw_buffer
        :type: DescriptorHandleType
        :value: DescriptorHandleType.rw_buffer
    
    .. py:attribute:: slangpy.DescriptorHandleType.texture
        :type: DescriptorHandleType
        :value: DescriptorHandleType.texture
    
    .. py:attribute:: slangpy.DescriptorHandleType.rw_texture
        :type: DescriptorHandleType
        :value: DescriptorHandleType.rw_texture
    
    .. py:attribute:: slangpy.DescriptorHandleType.sampler
        :type: DescriptorHandleType
        :value: DescriptorHandleType.sampler
    
    .. py:attribute:: slangpy.DescriptorHandleType.combined_texture_sampler
        :type: DescriptorHandleType
        :value: DescriptorHandleType.combined_texture_sampler
    
    .. py:attribute:: slangpy.DescriptorHandleType.acceleration_structure
        :type: DescriptorHandleType
        :value: DescriptorHandleType.acceleration_structure
    


----

.. py:class:: slangpy.CpuTimestampDomain

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.CpuTimestampDomain.unknown
        :type: CpuTimestampDomain
        :value: CpuTimestampDomain.unknown
    
    .. py:attribute:: slangpy.CpuTimestampDomain.query_performance_counter
        :type: CpuTimestampDomain
        :value: CpuTimestampDomain.query_performance_counter
    
    .. py:attribute:: slangpy.CpuTimestampDomain.clock_monotonic
        :type: CpuTimestampDomain
        :value: CpuTimestampDomain.clock_monotonic
    
    .. py:attribute:: slangpy.CpuTimestampDomain.clock_monotonic_raw
        :type: CpuTimestampDomain
        :value: CpuTimestampDomain.clock_monotonic_raw
    
    .. py:attribute:: slangpy.CpuTimestampDomain.mach_absolute_time
        :type: CpuTimestampDomain
        :value: CpuTimestampDomain.mach_absolute_time
    


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
    
    .. py:attribute:: slangpy.SpecializationArgKind.unknown
        :type: SpecializationArgKind
        :value: SpecializationArgKind.unknown
    
    .. py:attribute:: slangpy.SpecializationArgKind.type
        :type: SpecializationArgKind
        :value: SpecializationArgKind.type
    
    .. py:attribute:: slangpy.SpecializationArgKind.expr
        :type: SpecializationArgKind
        :value: SpecializationArgKind.expr
    


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
    
    .. py:attribute:: slangpy.LinearSweptSpheresIndexingMode.list
        :type: LinearSweptSpheresIndexingMode
        :value: LinearSweptSpheresIndexingMode.list
    
    .. py:attribute:: slangpy.LinearSweptSpheresIndexingMode.successive
        :type: LinearSweptSpheresIndexingMode
        :value: LinearSweptSpheresIndexingMode.successive
    


----

.. py:class:: slangpy.LinearSweptSpheresEndCapsMode

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.LinearSweptSpheresEndCapsMode.none
        :type: LinearSweptSpheresEndCapsMode
        :value: LinearSweptSpheresEndCapsMode.none
    
    .. py:attribute:: slangpy.LinearSweptSpheresEndCapsMode.chained
        :type: LinearSweptSpheresEndCapsMode
        :value: LinearSweptSpheresEndCapsMode.chained
    


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

.. py:function:: slangpy.create_compute_pipeline(program: slangpy.ShaderProgram, defer_target_compilation: bool = False, label: str | None = None) -> slangpy.ComputePipeline

    Create a compute pipeline.
    
.. py:function:: slangpy.create_compute_pipeline(desc: slangpy.ComputePipelineDesc) -> slangpy.ComputePipeline
    :no-index:



----

.. py:function:: slangpy.create_render_pipeline(program: slangpy.ShaderProgram, input_layout: slangpy.InputLayout | None, primitive_topology: slangpy.PrimitiveTopology = PrimitiveTopology.triangle_list, targets: collections.abc.Sequence[slangpy.ColorTargetDesc] = [], depth_stencil: slangpy.DepthStencilDesc | None = None, rasterizer: slangpy.RasterizerDesc | None = None, multisample: slangpy.MultisampleDesc | None = None, defer_target_compilation: bool = False, label: str | None = None) -> slangpy.RenderPipeline

    Create a render pipeline.
    
.. py:function:: slangpy.create_render_pipeline(desc: slangpy.RenderPipelineDesc) -> slangpy.RenderPipeline
    :no-index:



----

.. py:function:: slangpy.create_ray_tracing_pipeline(program: slangpy.ShaderProgram, hit_groups: collections.abc.Sequence[slangpy.HitGroupDesc], max_recursion: int = 0, max_ray_payload_size: int = 0, max_attribute_size: int = 8, flags: slangpy.RayTracingPipelineFlags = 0, defer_target_compilation: bool = False, label: str | None = None) -> slangpy.RayTracingPipeline

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
    
    .. py:attribute:: slangpy.native_refl.IOType.none
        :type: IOType
        :value: IOType.none
    
    .. py:attribute:: slangpy.native_refl.IOType.inn
        :type: IOType
        :value: IOType.inn
    
    .. py:attribute:: slangpy.native_refl.IOType.out
        :type: IOType
        :value: IOType.out
    
    .. py:attribute:: slangpy.native_refl.IOType.inout
        :type: IOType
        :value: IOType.inout
    


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
        
        .. py:attribute:: slangpy.native_refl.TensorType.Kind.tensor
            :type: Kind
            :value: Kind.tensor
        
        .. py:attribute:: slangpy.native_refl.TensorType.Kind.itensor
            :type: Kind
            :value: Kind.itensor
        
        .. py:attribute:: slangpy.native_refl.TensorType.Kind.diff_tensor
            :type: Kind
            :value: Kind.diff_tensor
        
        .. py:attribute:: slangpy.native_refl.TensorType.Kind.idiff_tensor
            :type: Kind
            :value: Kind.idiff_tensor
        
        .. py:attribute:: slangpy.native_refl.TensorType.Kind.primal_tensor
            :type: Kind
            :value: Kind.primal_tensor
        
        .. py:attribute:: slangpy.native_refl.TensorType.Kind.atomic
            :type: Kind
            :value: Kind.atomic
        
    .. py:class:: slangpy.native_refl.TensorType.Access
    
        Base class: :py:class:`enum.Enum`
        
        N/A
        
        .. py:attribute:: slangpy.native_refl.TensorType.Access.read
            :type: Access
            :value: Access.read
        
        .. py:attribute:: slangpy.native_refl.TensorType.Access.write
            :type: Access
            :value: Access.write
        
        .. py:attribute:: slangpy.native_refl.TensorType.Access.read_write
            :type: Access
            :value: Access.read_write
        
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
    
    .. py:attribute:: slangpy.YAHandling.expand_to_rgba
        :type: YAHandling
        :value: YAHandling.expand_to_rgba
    
    .. py:attribute:: slangpy.YAHandling.preserve_as_rg
        :type: YAHandling
        :value: YAHandling.preserve_as_rg
    


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
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.core.native.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.core.native.AutogradAccess
    :canonical: slangpy.slangpy.AutogradAccess
    
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
    :canonical: slangpy.slangpy.SignatureBuilder
    
    Alias class: :py:class:`slangpy.slangpy.SignatureBuilder`
    


----

.. py:class:: slangpy.core.native.NativeObject
    :canonical: slangpy.slangpy.NativeObject
    
    Alias class: :py:class:`slangpy.slangpy.NativeObject`
    


----

.. py:class:: slangpy.core.native.NativeMarshall
    :canonical: slangpy.slangpy.NativeMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`
    


----

.. py:class:: slangpy.core.native.NativeBoundVariableRuntime
    :canonical: slangpy.slangpy.NativeBoundVariableRuntime
    
    Alias class: :py:class:`slangpy.slangpy.NativeBoundVariableRuntime`
    


----

.. py:class:: slangpy.core.native.NativeBoundCallRuntime
    :canonical: slangpy.slangpy.NativeBoundCallRuntime
    
    Alias class: :py:class:`slangpy.slangpy.NativeBoundCallRuntime`
    


----

.. py:class:: slangpy.core.native.NativeCallRuntimeOptions
    :canonical: slangpy.slangpy.NativeCallRuntimeOptions
    
    Alias class: :py:class:`slangpy.slangpy.NativeCallRuntimeOptions`
    


----

.. py:class:: slangpy.core.native.NativeCallData
    :canonical: slangpy.slangpy.NativeCallData
    
    Alias class: :py:class:`slangpy.slangpy.NativeCallData`
    


----

.. py:class:: slangpy.core.native.NativeCallDataCache
    :canonical: slangpy.slangpy.NativeCallDataCache
    
    Alias class: :py:class:`slangpy.slangpy.NativeCallDataCache`
    


----

.. py:class:: slangpy.core.native.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.core.native.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.core.native.FunctionNodeType
    :canonical: slangpy.slangpy.FunctionNodeType
    
    Alias class: :py:class:`slangpy.slangpy.FunctionNodeType`
    


----

.. py:class:: slangpy.core.native.NativeFunctionNode
    :canonical: slangpy.slangpy.NativeFunctionNode
    
    Alias class: :py:class:`slangpy.slangpy.NativeFunctionNode`
    


----

.. py:class:: slangpy.core.native.NativePackedArg
    :canonical: slangpy.slangpy.NativePackedArg
    
    Alias class: :py:class:`slangpy.slangpy.NativePackedArg`
    


----

.. py:function:: slangpy.core.native.get_texture_shape(texture: slangpy.Texture, mip: int = 0) -> slangpy.slangpy.Shape

    N/A
    


----

.. py:class:: slangpy.core.native.NativeBufferMarshall
    :canonical: slangpy.slangpy.NativeBufferMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeBufferMarshall`
    


----

.. py:class:: slangpy.core.native.NativeDescriptorMarshall
    :canonical: slangpy.slangpy.NativeDescriptorMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeDescriptorMarshall`
    


----

.. py:class:: slangpy.core.native.NativeTextureMarshall
    :canonical: slangpy.slangpy.NativeTextureMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeTextureMarshall`
    


----

.. py:class:: slangpy.core.native.TensorMarshall
    :canonical: slangpy.slangpy.TensorMarshall
    
    Alias class: :py:class:`slangpy.slangpy.TensorMarshall`
    


----

.. py:class:: slangpy.core.native.NativeNumpyMarshall
    :canonical: slangpy.slangpy.NativeNumpyMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeNumpyMarshall`
    


----

.. py:class:: slangpy.core.native.NativeTorchTensorMarshall
    :canonical: slangpy.slangpy.NativeTorchTensorMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorMarshall`
    


----

.. py:class:: slangpy.core.native.NativeTorchTensorDiffPair
    :canonical: slangpy.slangpy.NativeTorchTensorDiffPair
    
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorDiffPair`
    


----

.. py:class:: slangpy.core.native.NativeValueMarshall
    :canonical: slangpy.slangpy.NativeValueMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`
    


----

.. py:class:: slangpy.core.utils.PathLike
    :canonical: os.PathLike
    
    Alias class: :py:class:`os.PathLike`
    


----

.. py:class:: slangpy.core.utils.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.utils.DeclReflection
    :canonical: slangpy.DeclReflection
    
    Alias class: :py:class:`slangpy.DeclReflection`
    


----

.. py:class:: slangpy.core.utils.ProgramLayout
    :canonical: slangpy.ProgramLayout
    
    Alias class: :py:class:`slangpy.ProgramLayout`
    


----

.. py:class:: slangpy.core.utils.TypeLayoutReflection
    :canonical: slangpy.TypeLayoutReflection
    
    Alias class: :py:class:`slangpy.TypeLayoutReflection`
    


----

.. py:class:: slangpy.core.utils.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.core.utils.DeviceType
    :canonical: slangpy.DeviceType
    
    Alias class: :py:class:`slangpy.DeviceType`
    


----

.. py:class:: slangpy.core.utils.Device
    :canonical: slangpy.Device
    
    Alias class: :py:class:`slangpy.Device`
    


----

.. py:class:: slangpy.core.utils.NativeHandle
    :canonical: slangpy.NativeHandle
    
    Alias class: :py:class:`slangpy.NativeHandle`
    


----

.. py:function:: slangpy.core.utils.get_cuda_current_context_native_handles() -> list[slangpy.NativeHandle]

    Gets the device and context handles for the current CUDA context. Use
    to retrieve an existing context (eg from PyTorch) to pass as the
    existing_device_handles from which to create a device in the
    DeviceDesc.
    


----

.. py:class:: slangpy.core.utils.BindlessDesc
    :canonical: slangpy.BindlessDesc
    
    Alias class: :py:class:`slangpy.BindlessDesc`
    


----

.. py:class:: slangpy.core.utils.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.core.utils.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.core.enums.Enum
    :canonical: enum.Enum
    
    Alias class: :py:class:`enum.Enum`
    


----

.. py:class:: slangpy.core.enums.IOType
    :canonical: slangpy.native_refl.IOType
    
    Alias class: :py:class:`slangpy.native_refl.IOType`
    


----

.. py:class:: slangpy.core.enums.PrimType

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.core.enums.PrimType.primal
        :type: PrimType
        :value: PrimType.primal
    
    .. py:attribute:: slangpy.core.enums.PrimType.derivative
        :type: PrimType
        :value: PrimType.derivative
    


----

.. py:class:: slangpy.core.logging.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.logging.FunctionReflection
    :canonical: slangpy.FunctionReflection
    
    Alias class: :py:class:`slangpy.FunctionReflection`
    


----

.. py:class:: slangpy.core.logging.ModifierID
    :canonical: slangpy.ModifierID
    
    Alias class: :py:class:`slangpy.ModifierID`
    


----

.. py:class:: slangpy.core.logging.VariableReflection
    :canonical: slangpy.VariableReflection
    
    Alias class: :py:class:`slangpy.VariableReflection`
    


----

.. py:class:: slangpy.core.logging.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.core.logging.TableColumn



----

.. py:class:: slangpy.core.generator.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.generator.CodeGen
    :canonical: slangpy.bindings.codegen.CodeGen
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`
    


----

.. py:class:: slangpy.core.generator.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.core.generator.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.core.generator.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:data:: slangpy.core.generator.MAX_INLINE_TYPE_LEN
    :type: int
    :value: 60



----

.. py:data:: slangpy.core.generator.MAX_GENERATED_THREAD_GROUP_SIZE
    :type: int
    :value: 1024



----

.. py:data:: slangpy.core.generator.MAX_DISPATCH_THREAD_GROUPS_X
    :type: int
    :value: 2097151



----

.. py:class:: slangpy.core.generator.KernelGenException

    Base class: :py:class:`builtins.Exception`
    


----

.. py:class:: slangpy.core.shapes.TArgShapesResult

    Base class: :py:class:`builtins.dict`
    


----

.. py:class:: slangpy.core.callsignature.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.callsignature.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.core.callsignature.tr.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.callsignature.tr.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.core.callsignature.tr.NativeMarshall
    :canonical: slangpy.slangpy.NativeMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`
    


----

.. py:class:: slangpy.core.callsignature.ModifierID
    :canonical: slangpy.ModifierID
    
    Alias class: :py:class:`slangpy.ModifierID`
    


----

.. py:class:: slangpy.core.callsignature.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.core.callsignature.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.core.callsignature.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.core.callsignature.BoundCall
    :canonical: slangpy.bindings.boundvariable.BoundCall
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundCall`
    


----

.. py:class:: slangpy.core.callsignature.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.core.callsignature.NoneMarshall
    :canonical: slangpy.builtin.value.NoneMarshall
    
    Alias class: :py:class:`slangpy.builtin.value.NoneMarshall`
    


----

.. py:class:: slangpy.core.callsignature.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.core.callsignature.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.core.callsignature.ResolvedParam
    :canonical: slangpy.reflection.typeresolution.ResolvedParam
    
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolvedParam`
    


----

.. py:class:: slangpy.core.callsignature.ResolutionDiagnostic
    :canonical: slangpy.reflection.typeresolution.ResolutionDiagnostic
    
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolutionDiagnostic`
    


----

.. py:class:: slangpy.core.callsignature.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.core.callsignature.ValueRef
    :canonical: slangpy.types.valueref.ValueRef
    
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`
    


----

.. py:class:: slangpy.core.callsignature.MismatchReason



----

.. py:class:: slangpy.core.callsignature.ResolveException

    Base class: :py:class:`builtins.Exception`
    


----

.. py:class:: slangpy.core.callsignature.KernelGenException
    :canonical: slangpy.core.generator.KernelGenException
    
    Alias class: :py:class:`slangpy.core.generator.KernelGenException`
    


----

.. py:class:: slangpy.core.function.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.function.Protocol
    :canonical: typing.Protocol
    
    Alias class: :py:class:`typing.Protocol`
    


----

.. py:class:: slangpy.core.function.Enum
    :canonical: enum.Enum
    
    Alias class: :py:class:`enum.Enum`
    


----

.. py:class:: slangpy.core.function.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.core.function.SignatureBuilder
    :canonical: slangpy.slangpy.SignatureBuilder
    
    Alias class: :py:class:`slangpy.slangpy.SignatureBuilder`
    


----

.. py:class:: slangpy.core.function.NativeCallRuntimeOptions
    :canonical: slangpy.slangpy.NativeCallRuntimeOptions
    
    Alias class: :py:class:`slangpy.slangpy.NativeCallRuntimeOptions`
    


----

.. py:class:: slangpy.core.function.NativeFunctionNode
    :canonical: slangpy.slangpy.NativeFunctionNode
    
    Alias class: :py:class:`slangpy.slangpy.NativeFunctionNode`
    


----

.. py:class:: slangpy.core.function.FunctionNodeType
    :canonical: slangpy.slangpy.FunctionNodeType
    
    Alias class: :py:class:`slangpy.slangpy.FunctionNodeType`
    


----

.. py:class:: slangpy.core.function.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.core.function.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.core.function.CommandEncoder
    :canonical: slangpy.CommandEncoder
    
    Alias class: :py:class:`slangpy.CommandEncoder`
    


----

.. py:class:: slangpy.core.function.TypeConformance
    :canonical: slangpy.TypeConformance
    
    Alias class: :py:class:`slangpy.TypeConformance`
    


----

.. py:class:: slangpy.core.function.uint3
    :canonical: slangpy.math.uint3
    
    Alias class: :py:class:`slangpy.math.uint3`
    


----

.. py:class:: slangpy.core.function.Logger
    :canonical: slangpy.Logger
    
    Alias class: :py:class:`slangpy.Logger`
    


----

.. py:class:: slangpy.core.function.NativeHandle
    :canonical: slangpy.NativeHandle
    
    Alias class: :py:class:`slangpy.NativeHandle`
    


----

.. py:class:: slangpy.core.function.NativeHandleType
    :canonical: slangpy.NativeHandleType
    
    Alias class: :py:class:`slangpy.NativeHandleType`
    


----

.. py:class:: slangpy.core.function.RayTracingPipelineFlags
    :canonical: slangpy.RayTracingPipelineFlags
    
    Alias class: :py:class:`slangpy.RayTracingPipelineFlags`
    


----

.. py:class:: slangpy.core.function.HitGroupDesc
    :canonical: slangpy.HitGroupDesc
    
    Alias class: :py:class:`slangpy.HitGroupDesc`
    


----

.. py:class:: slangpy.core.function.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.core.function.IThis

    Base class: :py:class:`typing.Protocol`
    
    .. py:attribute:: slangpy.core.function.IThis.get_this
        :type: function
        :value: <function IThis.get_this at 0x0000021D5F66B560>
    
    .. py:attribute:: slangpy.core.function.IThis.update_this
        :type: function
        :value: <function IThis.update_this at 0x0000021D5F66B420>
    


----

.. py:class:: slangpy.core.function.PipelineType

    Base class: :py:class:`enum.Enum`
    
    .. py:attribute:: slangpy.core.function.PipelineType.compute
        :type: PipelineType
        :value: PipelineType.compute
    
    .. py:attribute:: slangpy.core.function.PipelineType.ray_tracing
        :type: PipelineType
        :value: PipelineType.ray_tracing
    


----

.. py:class:: slangpy.core.function.FunctionBuildInfo



----

.. py:class:: slangpy.core.function.FunctionNode

    Base class: :py:class:`slangpy.slangpy.NativeFunctionNode`
    
    .. py:attribute:: slangpy.core.function.FunctionNode.torch
        :type: function
        :value: <function FunctionNode.torch at 0x0000021D5F66BB00>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.bind
        :type: function
        :value: <function FunctionNode.bind at 0x0000021D5F66BBA0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.map
        :type: function
        :value: <function FunctionNode.map at 0x0000021D5F66BC40>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.set
        :type: function
        :value: <function FunctionNode.set at 0x0000021D5F66BCE0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.write
        :type: function
        :value: <function FunctionNode.write at 0x0000021D5F66BD80>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.cuda_stream
        :type: function
        :value: <function FunctionNode.cuda_stream at 0x0000021D5F66BE20>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.constants
        :type: function
        :value: <function FunctionNode.constants at 0x0000021D5F66BEC0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.prelude
        :type: function
        :value: <function FunctionNode.prelude at 0x0000021D5F66BF60>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.type_conformances
        :type: function
        :value: <function FunctionNode.type_conformances at 0x0000021D5F69C040>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.ray_tracing
        :type: function
        :value: <function FunctionNode.ray_tracing at 0x0000021D5F69C0E0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.return_type
        :type: function
        :value: <function FunctionNode.return_type at 0x0000021D5F69C180>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.thread_group_size
        :type: function
        :value: <function FunctionNode.thread_group_size at 0x0000021D5F69C2C0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.as_func
        :type: function
        :value: <function FunctionNode.as_func at 0x0000021D5F69C360>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.as_struct
        :type: function
        :value: <function FunctionNode.as_struct at 0x0000021D5F69C400>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.debug_build_call_data
        :type: function
        :value: <function FunctionNode.debug_build_call_data at 0x0000021D5F69C4A0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.call
        :type: function
        :value: <function FunctionNode.call at 0x0000021D5F69C540>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.append_to
        :type: function
        :value: <function FunctionNode.append_to at 0x0000021D5F69C5E0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.dispatch
        :type: function
        :value: <function FunctionNode.dispatch at 0x0000021D5F69C680>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.calc_build_info
        :type: function
        :value: <function FunctionNode.calc_build_info at 0x0000021D5F69C720>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.generate_call_data
        :type: function
        :value: <function FunctionNode.generate_call_data at 0x0000021D5F69C900>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.generate_bwds_call_data
        :type: function
        :value: <function FunctionNode.generate_bwds_call_data at 0x0000021D5F69C9A0>
    
    .. py:attribute:: slangpy.core.function.FunctionNode.call_group_shape
        :type: function
        :value: <function FunctionNode.call_group_shape at 0x0000021D5F69CA40>
    


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
    :canonical: pathlib._local.Path
    
    Alias class: :py:class:`pathlib._local.Path`
    


----

.. py:class:: slangpy.core.calldata.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.calldata.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.core.calldata.ModifierID
    :canonical: slangpy.ModifierID
    
    Alias class: :py:class:`slangpy.ModifierID`
    


----

.. py:class:: slangpy.core.calldata.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.core.calldata.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.core.calldata.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.core.calldata.BoundCall
    :canonical: slangpy.bindings.boundvariable.BoundCall
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundCall`
    


----

.. py:class:: slangpy.core.calldata.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.core.calldata.NoneMarshall
    :canonical: slangpy.builtin.value.NoneMarshall
    
    Alias class: :py:class:`slangpy.builtin.value.NoneMarshall`
    


----

.. py:class:: slangpy.core.calldata.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.core.calldata.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.core.calldata.ResolvedParam
    :canonical: slangpy.reflection.typeresolution.ResolvedParam
    
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolvedParam`
    


----

.. py:class:: slangpy.core.calldata.ResolutionDiagnostic
    :canonical: slangpy.reflection.typeresolution.ResolutionDiagnostic
    
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolutionDiagnostic`
    


----

.. py:class:: slangpy.core.calldata.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.core.calldata.ValueRef
    :canonical: slangpy.types.valueref.ValueRef
    
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`
    


----

.. py:class:: slangpy.core.calldata.MismatchReason
    :canonical: slangpy.core.callsignature.MismatchReason
    
    Alias class: :py:class:`slangpy.core.callsignature.MismatchReason`
    


----

.. py:class:: slangpy.core.calldata.ResolveException
    :canonical: slangpy.core.callsignature.ResolveException
    
    Alias class: :py:class:`slangpy.core.callsignature.ResolveException`
    


----

.. py:class:: slangpy.core.calldata.KernelGenException
    :canonical: slangpy.core.generator.KernelGenException
    
    Alias class: :py:class:`slangpy.core.generator.KernelGenException`
    


----

.. py:class:: slangpy.core.calldata.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.core.calldata.NativeCallData
    :canonical: slangpy.slangpy.NativeCallData
    
    Alias class: :py:class:`slangpy.slangpy.NativeCallData`
    


----

.. py:function:: slangpy.core.calldata.unpack_args(*args) -> tuple

    N/A
    


----

.. py:function:: slangpy.core.calldata.unpack_kwargs(**kwargs) -> tuple

    N/A
    


----

.. py:class:: slangpy.core.calldata.PipelineType
    :canonical: slangpy.core.function.PipelineType
    
    Alias class: :py:class:`slangpy.core.function.PipelineType`
    


----

.. py:class:: slangpy.core.calldata.SlangCompileError
    :canonical: slangpy.SlangCompileError
    
    Alias class: :py:class:`slangpy.SlangCompileError`
    


----

.. py:class:: slangpy.core.calldata.SlangLinkOptions
    :canonical: slangpy.SlangLinkOptions
    
    Alias class: :py:class:`slangpy.SlangLinkOptions`
    


----

.. py:class:: slangpy.core.calldata.NativeHandle
    :canonical: slangpy.NativeHandle
    
    Alias class: :py:class:`slangpy.NativeHandle`
    


----

.. py:class:: slangpy.core.calldata.DeviceType
    :canonical: slangpy.DeviceType
    
    Alias class: :py:class:`slangpy.DeviceType`
    


----

.. py:class:: slangpy.core.calldata.TypeConformance
    :canonical: slangpy.TypeConformance
    
    Alias class: :py:class:`slangpy.TypeConformance`
    


----

.. py:function:: slangpy.core.calldata.is_torch_bridge_using_fallback() -> bool

    N/A
    


----

.. py:function:: slangpy.core.calldata.get_torch_bridge_fallback_reason() -> str

    N/A
    


----

.. py:class:: slangpy.core.calldata.BoundCallRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundCallRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundCallRuntime`
    


----

.. py:class:: slangpy.core.calldata.BoundVariableException
    :canonical: slangpy.bindings.boundvariable.BoundVariableException
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariableException`
    


----

.. py:class:: slangpy.core.calldata.CodeGen
    :canonical: slangpy.bindings.codegen.CodeGen
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`
    


----

.. py:class:: slangpy.core.calldata.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.core.calldata.ITensorType
    :canonical: slangpy.native_refl.TensorType
    
    Alias class: :py:class:`slangpy.native_refl.TensorType`
    


----

.. py:class:: slangpy.core.calldata.TensorAccess
    :canonical: slangpy.native_refl.TensorType.Access
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`
    


----

.. py:class:: slangpy.core.calldata.CallData

    Base class: :py:class:`slangpy.slangpy.NativeCallData`
    
    .. py:attribute:: slangpy.core.calldata.CallData.build
        :type: function
        :value: <function CallData.build at 0x0000021D5F69E700>
    


----

.. py:class:: slangpy.core.struct.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.struct.Function
    :canonical: slangpy.core.function.Function
    
    Alias class: :py:class:`slangpy.core.function.Function`
    


----

.. py:class:: slangpy.core.struct.BaseStruct
    :canonical: slangpy.native_func.BaseStruct
    
    Alias class: :py:class:`slangpy.native_func.BaseStruct`
    


----

.. py:class:: slangpy.core.struct.Struct

    Base class: :py:class:`slangpy.native_func.BaseStruct`
    
    
    A Slang struct, typically created by accessing it via a module or parent struct. i.e. mymodule.Foo,
    or mymodule.Foo.Bar.
    
    
    .. py:attribute:: slangpy.core.struct.Struct.torch
        :type: function
        :value: <function Struct.torch at 0x0000021D5F69F600>
    
    .. py:attribute:: slangpy.core.struct.Struct.try_get_child
        :type: function
        :value: <function Struct.try_get_child at 0x0000021D5F69F6A0>
    
    .. py:attribute:: slangpy.core.struct.Struct.as_func
        :type: function
        :value: <function Struct.as_func at 0x0000021D5F69F920>
    
    .. py:attribute:: slangpy.core.struct.Struct.as_struct
        :type: function
        :value: <function Struct.as_struct at 0x0000021D5F69F9C0>
    


----

.. py:class:: slangpy.core.module.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.module.Function
    :canonical: slangpy.core.function.Function
    
    Alias class: :py:class:`slangpy.core.function.Function`
    


----

.. py:class:: slangpy.core.module.Struct
    :canonical: slangpy.core.struct.Struct
    
    Alias class: :py:class:`slangpy.core.struct.Struct`
    


----

.. py:class:: slangpy.core.module.Pipeline
    :canonical: slangpy.Pipeline
    
    Alias class: :py:class:`slangpy.Pipeline`
    


----

.. py:class:: slangpy.core.module.ShaderTable
    :canonical: slangpy.ShaderTable
    
    Alias class: :py:class:`slangpy.ShaderTable`
    


----

.. py:class:: slangpy.core.module.SlangModule
    :canonical: slangpy.SlangModule
    
    Alias class: :py:class:`slangpy.SlangModule`
    


----

.. py:class:: slangpy.core.module.Device
    :canonical: slangpy.Device
    
    Alias class: :py:class:`slangpy.Device`
    


----

.. py:class:: slangpy.core.module.Logger
    :canonical: slangpy.Logger
    
    Alias class: :py:class:`slangpy.Logger`
    


----

.. py:class:: slangpy.core.module.NativeCallDataCache
    :canonical: slangpy.slangpy.NativeCallDataCache
    
    Alias class: :py:class:`slangpy.slangpy.NativeCallDataCache`
    


----

.. py:class:: slangpy.core.module.BaseModule
    :canonical: slangpy.native_func.BaseModule
    
    Alias class: :py:class:`slangpy.native_func.BaseModule`
    


----

.. py:class:: slangpy.core.module.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.core.module.CallDataCache

    Base class: :py:class:`slangpy.slangpy.NativeCallDataCache`
    
    .. py:attribute:: slangpy.core.module.CallDataCache.lookup_value_signature
        :type: function
        :value: <function CallDataCache.lookup_value_signature at 0x0000021D5F69FF60>
    


----

.. py:class:: slangpy.core.module.Module

    Base class: :py:class:`slangpy.native_func.BaseModule`
    
    
    A Slang module, created either by loading a slang file or providing a loaded SGL module.
    
    
    .. py:attribute:: slangpy.core.module.Module.load_from_source
        :type: function
        :value: <function Module.load_from_source at 0x0000021D5F6AC220>
    
    .. py:attribute:: slangpy.core.module.Module.load_from_file
        :type: function
        :value: <function Module.load_from_file at 0x0000021D5F6AC180>
    
    .. py:attribute:: slangpy.core.module.Module.load_from_module
        :type: function
        :value: <function Module.load_from_module at 0x0000021D5F6AC2C0>
    
    .. py:attribute:: slangpy.core.module.Module.find_struct
        :type: function
        :value: <function Module.find_struct at 0x0000021D5F6AC360>
    
    .. py:attribute:: slangpy.core.module.Module.require_struct
        :type: function
        :value: <function Module.require_struct at 0x0000021D5F6AC400>
    
    .. py:attribute:: slangpy.core.module.Module.find_function
        :type: function
        :value: <function Module.find_function at 0x0000021D5F6AC4A0>
    
    .. py:attribute:: slangpy.core.module.Module.require_function
        :type: function
        :value: <function Module.require_function at 0x0000021D5F6AC540>
    
    .. py:attribute:: slangpy.core.module.Module.find_function_in_struct
        :type: function
        :value: <function Module.find_function_in_struct at 0x0000021D5F6AC5E0>
    
    .. py:attribute:: slangpy.core.module.Module.on_hot_reload
        :type: function
        :value: <function Module.on_hot_reload at 0x0000021D5F6AC680>
    


----

.. py:class:: slangpy.core.instance.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.instance.FunctionNode
    :canonical: slangpy.core.function.FunctionNode
    
    Alias class: :py:class:`slangpy.core.function.FunctionNode`
    


----

.. py:class:: slangpy.core.instance.Struct
    :canonical: slangpy.core.struct.Struct
    
    Alias class: :py:class:`slangpy.core.struct.Struct`
    


----

.. py:class:: slangpy.core.instance.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.core.instance.InstanceList

    
    Represents a list of instances of a struct, either as a single buffer
    or an SOA style set of buffers for each field. data can either
    be a dictionary of field names to buffers, or a single buffer.
    
    
    .. py:attribute:: slangpy.core.instance.InstanceList.set_data
        :type: function
        :value: <function InstanceList.set_data at 0x0000021D5F6ACCC0>
    
    .. py:attribute:: slangpy.core.instance.InstanceList.get_this
        :type: function
        :value: <function InstanceList.get_this at 0x0000021D5F6ACD60>
    
    .. py:attribute:: slangpy.core.instance.InstanceList.update_this
        :type: function
        :value: <function InstanceList.update_this at 0x0000021D5F6ACE00>
    
    .. py:attribute:: slangpy.core.instance.InstanceList.construct
        :type: function
        :value: <function InstanceList.construct at 0x0000021D5F6ACEA0>
    
    .. py:attribute:: slangpy.core.instance.InstanceList.pack
        :type: function
        :value: <function InstanceList.pack at 0x0000021D5F6ACF40>
    


----

.. py:class:: slangpy.core.instance.InstanceTensor

    Base class: :py:class:`slangpy.core.instance.InstanceList`
    
    
    Simplified implementation of InstanceList that uses a single buffer for all instances and
    provides buffer convenience functions for accessing its data.
    
    
    .. py:attribute:: slangpy.core.instance.InstanceTensor.to_numpy
        :type: function
        :value: <function InstanceTensor.to_numpy at 0x0000021D5F6AD3A0>
    
    .. py:attribute:: slangpy.core.instance.InstanceTensor.copy_from_numpy
        :type: function
        :value: <function InstanceTensor.copy_from_numpy at 0x0000021D5F6AD440>
    


----

.. py:class:: slangpy.core.packedarg.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.core.packedarg.Module
    :canonical: slangpy.core.module.Module
    
    Alias class: :py:class:`slangpy.core.module.Module`
    


----

.. py:function:: slangpy.core.packedarg.get_value_signature(o: object) -> str

    N/A
    


----

.. py:class:: slangpy.core.packedarg.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.core.packedarg.NativePackedArg
    :canonical: slangpy.slangpy.NativePackedArg
    
    Alias class: :py:class:`slangpy.slangpy.NativePackedArg`
    


----

.. py:function:: slangpy.core.packedarg.unpack_arg(arg: object) -> object

    N/A
    


----

.. py:class:: slangpy.core.packedarg.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.core.packedarg.PackedArg

    Base class: :py:class:`slangpy.slangpy.NativePackedArg`
    
    
    Represents an argument that has been efficiently packed into
    a shader object for use in later functionc alls.
    
    


----

.. py:class:: slangpy.reflection.reflectiontypes.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.TR
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.ArrayType
    :canonical: slangpy.native_refl.ArrayType
    
    Alias class: :py:class:`slangpy.native_refl.ArrayType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.ByteAddressBufferType
    :canonical: slangpy.native_refl.ByteAddressBufferType
    
    Alias class: :py:class:`slangpy.native_refl.ByteAddressBufferType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.DiffTensorViewType
    :canonical: slangpy.native_refl.DiffTensorViewType
    
    Alias class: :py:class:`slangpy.native_refl.DiffTensorViewType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.DifferentialPairType
    :canonical: slangpy.native_refl.DifferentialPairType
    
    Alias class: :py:class:`slangpy.native_refl.DifferentialPairType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SlangField
    :canonical: slangpy.native_refl.Field
    
    Alias class: :py:class:`slangpy.native_refl.Field`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.InterfaceType
    :canonical: slangpy.native_refl.InterfaceType
    
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.MatrixType
    :canonical: slangpy.native_refl.MatrixType
    
    Alias class: :py:class:`slangpy.native_refl.MatrixType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SlangParameter
    :canonical: slangpy.native_refl.Parameter
    
    Alias class: :py:class:`slangpy.native_refl.Parameter`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.PointerType
    :canonical: slangpy.native_refl.PointerType
    
    Alias class: :py:class:`slangpy.native_refl.PointerType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.RaytracingAccelerationStructureType
    :canonical: slangpy.native_refl.RaytracingAccelerationStructureType
    
    Alias class: :py:class:`slangpy.native_refl.RaytracingAccelerationStructureType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.ResourceType
    :canonical: slangpy.native_refl.ResourceType
    
    Alias class: :py:class:`slangpy.native_refl.ResourceType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SamplerStateType
    :canonical: slangpy.native_refl.SamplerStateType
    
    Alias class: :py:class:`slangpy.native_refl.SamplerStateType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.ScalarType
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.StructuredBufferType
    :canonical: slangpy.native_refl.StructuredBufferType
    
    Alias class: :py:class:`slangpy.native_refl.StructuredBufferType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.StructType
    :canonical: slangpy.native_refl.StructType
    
    Alias class: :py:class:`slangpy.native_refl.StructType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.ITensorType
    :canonical: slangpy.native_refl.TensorType
    
    Alias class: :py:class:`slangpy.native_refl.TensorType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.TensorViewType
    :canonical: slangpy.native_refl.TensorViewType
    
    Alias class: :py:class:`slangpy.native_refl.TensorViewType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.TextureType
    :canonical: slangpy.native_refl.TextureType
    
    Alias class: :py:class:`slangpy.native_refl.TextureType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.SlangLayout
    :canonical: slangpy.native_refl.TypeLayout
    
    Alias class: :py:class:`slangpy.native_refl.TypeLayout`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.UnhandledType
    :canonical: slangpy.native_refl.UnhandledType
    
    Alias class: :py:class:`slangpy.native_refl.UnhandledType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.UnknownType
    :canonical: slangpy.native_refl.UnknownType
    
    Alias class: :py:class:`slangpy.native_refl.UnknownType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.VectorType
    :canonical: slangpy.native_refl.VectorType
    
    Alias class: :py:class:`slangpy.native_refl.VectorType`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.VoidType
    :canonical: slangpy.native_refl.VoidType
    
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
    :canonical: slangpy.native_refl.TensorType.Kind
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`
    


----

.. py:class:: slangpy.reflection.reflectiontypes.TensorAccess
    :canonical: slangpy.native_refl.TensorType.Access
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`
    


----

.. py:class:: slangpy.reflection.SlangLayout
    :canonical: slangpy.native_refl.TypeLayout
    
    Alias class: :py:class:`slangpy.native_refl.TypeLayout`
    


----

.. py:class:: slangpy.reflection.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.reflection.VoidType
    :canonical: slangpy.native_refl.VoidType
    
    Alias class: :py:class:`slangpy.native_refl.VoidType`
    


----

.. py:class:: slangpy.reflection.ScalarType
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.reflection.VectorType
    :canonical: slangpy.native_refl.VectorType
    
    Alias class: :py:class:`slangpy.native_refl.VectorType`
    


----

.. py:class:: slangpy.reflection.MatrixType
    :canonical: slangpy.native_refl.MatrixType
    
    Alias class: :py:class:`slangpy.native_refl.MatrixType`
    


----

.. py:class:: slangpy.reflection.ArrayType
    :canonical: slangpy.native_refl.ArrayType
    
    Alias class: :py:class:`slangpy.native_refl.ArrayType`
    


----

.. py:class:: slangpy.reflection.StructType
    :canonical: slangpy.native_refl.StructType
    
    Alias class: :py:class:`slangpy.native_refl.StructType`
    


----

.. py:class:: slangpy.reflection.InterfaceType
    :canonical: slangpy.native_refl.InterfaceType
    
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`
    


----

.. py:class:: slangpy.reflection.TextureType
    :canonical: slangpy.native_refl.TextureType
    
    Alias class: :py:class:`slangpy.native_refl.TextureType`
    


----

.. py:class:: slangpy.reflection.PointerType
    :canonical: slangpy.native_refl.PointerType
    
    Alias class: :py:class:`slangpy.native_refl.PointerType`
    


----

.. py:class:: slangpy.reflection.UnknownType
    :canonical: slangpy.native_refl.UnknownType
    
    Alias class: :py:class:`slangpy.native_refl.UnknownType`
    


----

.. py:class:: slangpy.reflection.ResourceType
    :canonical: slangpy.native_refl.ResourceType
    
    Alias class: :py:class:`slangpy.native_refl.ResourceType`
    


----

.. py:class:: slangpy.reflection.StructuredBufferType
    :canonical: slangpy.native_refl.StructuredBufferType
    
    Alias class: :py:class:`slangpy.native_refl.StructuredBufferType`
    


----

.. py:class:: slangpy.reflection.ByteAddressBufferType
    :canonical: slangpy.native_refl.ByteAddressBufferType
    
    Alias class: :py:class:`slangpy.native_refl.ByteAddressBufferType`
    


----

.. py:class:: slangpy.reflection.DifferentialPairType
    :canonical: slangpy.native_refl.DifferentialPairType
    
    Alias class: :py:class:`slangpy.native_refl.DifferentialPairType`
    


----

.. py:class:: slangpy.reflection.RaytracingAccelerationStructureType
    :canonical: slangpy.native_refl.RaytracingAccelerationStructureType
    
    Alias class: :py:class:`slangpy.native_refl.RaytracingAccelerationStructureType`
    


----

.. py:class:: slangpy.reflection.SamplerStateType
    :canonical: slangpy.native_refl.SamplerStateType
    
    Alias class: :py:class:`slangpy.native_refl.SamplerStateType`
    


----

.. py:class:: slangpy.reflection.UnhandledType
    :canonical: slangpy.native_refl.UnhandledType
    
    Alias class: :py:class:`slangpy.native_refl.UnhandledType`
    


----

.. py:class:: slangpy.reflection.ITensorType
    :canonical: slangpy.native_refl.TensorType
    
    Alias class: :py:class:`slangpy.native_refl.TensorType`
    


----

.. py:class:: slangpy.reflection.TensorType
    :canonical: slangpy.native_refl.TensorType.Kind
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`
    


----

.. py:class:: slangpy.reflection.TensorAccess
    :canonical: slangpy.native_refl.TensorType.Access
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`
    


----

.. py:class:: slangpy.reflection.TensorViewType
    :canonical: slangpy.native_refl.TensorViewType
    
    Alias class: :py:class:`slangpy.native_refl.TensorViewType`
    


----

.. py:class:: slangpy.reflection.DiffTensorViewType
    :canonical: slangpy.native_refl.DiffTensorViewType
    
    Alias class: :py:class:`slangpy.native_refl.DiffTensorViewType`
    


----

.. py:class:: slangpy.reflection.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.reflection.SlangField
    :canonical: slangpy.native_refl.Field
    
    Alias class: :py:class:`slangpy.native_refl.Field`
    


----

.. py:class:: slangpy.reflection.SlangParameter
    :canonical: slangpy.native_refl.Parameter
    
    Alias class: :py:class:`slangpy.native_refl.Parameter`
    


----

.. py:class:: slangpy.reflection.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
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
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.reflection.typeresolution.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.reflection.typeresolution.ResolutionArg

    
    Holds a single argument for type resolution. Starts with the python marshall and slang type
    being resolved, and ends with the resolved vector type (if any). In the case of functions
    with optional parameters, the python variable can be None and the parameter type will be used
    
    


----

.. py:class:: slangpy.reflection.typeresolution.ResolutionDiagnostic

    .. py:attribute:: slangpy.reflection.typeresolution.ResolutionDiagnostic.summary
        :type: function
        :value: <function ResolutionDiagnostic.summary at 0x0000021D5F608D60>
    
    .. py:attribute:: slangpy.reflection.typeresolution.ResolutionDiagnostic.detail
        :type: function
        :value: <function ResolutionDiagnostic.detail at 0x0000021D5F608E00>
    


----

.. py:class:: slangpy.reflection.typeresolution.ResolvedParam



----

.. py:class:: slangpy.reflection.typeresolution.ResolveResult



----

.. py:class:: slangpy.reflection.lookup.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.reflection.lookup.Device
    :canonical: slangpy.Device
    
    Alias class: :py:class:`slangpy.Device`
    


----

.. py:class:: slangpy.reflection.lookup.TypeLayoutReflection
    :canonical: slangpy.TypeLayoutReflection
    
    Alias class: :py:class:`slangpy.TypeLayoutReflection`
    


----

.. py:class:: slangpy.reflection.lookup.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.reflection.lookup.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.reflection.lookup.BaseStruct
    :canonical: slangpy.native_func.BaseStruct
    
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
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.reflection.lookup.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.reflection.lookup.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.reflection.lookup.ST
    :canonical: slangpy.TypeReflection.ScalarType
    
    Alias class: :py:class:`slangpy.TypeReflection.ScalarType`
    


----

.. py:class:: slangpy.bindings.codegen.CodeGenBlock

    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.add_import
        :type: function
        :value: <function CodeGenBlock.add_import at 0x0000021D5F5EE200>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.inc_indent
        :type: function
        :value: <function CodeGenBlock.inc_indent at 0x0000021D5F5EE2A0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.dec_indent
        :type: function
        :value: <function CodeGenBlock.dec_indent at 0x0000021D5F5EE340>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.append_indent
        :type: function
        :value: <function CodeGenBlock.append_indent at 0x0000021D5F5EE3E0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.append_code
        :type: function
        :value: <function CodeGenBlock.append_code at 0x0000021D5F5EE480>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.append_code_indented
        :type: function
        :value: <function CodeGenBlock.append_code_indented at 0x0000021D5F5EE520>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.empty_line
        :type: function
        :value: <function CodeGenBlock.empty_line at 0x0000021D5F5EE5C0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.append_line
        :type: function
        :value: <function CodeGenBlock.append_line at 0x0000021D5F5EE660>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.append_statement
        :type: function
        :value: <function CodeGenBlock.append_statement at 0x0000021D5F5EE700>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.begin_block
        :type: function
        :value: <function CodeGenBlock.begin_block at 0x0000021D5F5EE7A0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.end_block
        :type: function
        :value: <function CodeGenBlock.end_block at 0x0000021D5F5EE840>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.begin_struct
        :type: function
        :value: <function CodeGenBlock.begin_struct at 0x0000021D5F5EE8E0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.end_struct
        :type: function
        :value: <function CodeGenBlock.end_struct at 0x0000021D5F5EE980>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.type_alias
        :type: function
        :value: <function CodeGenBlock.type_alias at 0x0000021D5F5EEA20>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.diff_pair
        :type: function
        :value: <function CodeGenBlock.diff_pair at 0x0000021D5F5EEAC0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.declare
        :type: function
        :value: <function CodeGenBlock.declare at 0x0000021D5F5EEB60>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.assign
        :type: function
        :value: <function CodeGenBlock.assign at 0x0000021D5F5EEC00>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.declarevar
        :type: function
        :value: <function CodeGenBlock.declarevar at 0x0000021D5F5EECA0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.statement
        :type: function
        :value: <function CodeGenBlock.statement at 0x0000021D5F5EED40>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.add_snippet
        :type: function
        :value: <function CodeGenBlock.add_snippet at 0x0000021D5F5EEDE0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGenBlock.finish
        :type: function
        :value: <function CodeGenBlock.finish at 0x0000021D5F5EEE80>
    


----

.. py:class:: slangpy.bindings.codegen.CodeGen

    
    Tool for generating the code for a SlangPy kernel. Contains a set of
    different code blocks that can be filled in and then combined to
    generate the final code.
    
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGen.add_snippet
        :type: function
        :value: <function CodeGen.add_snippet at 0x0000021D5F5EEFC0>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGen.add_import
        :type: function
        :value: <function CodeGen.add_import at 0x0000021D5F5EF060>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGen.add_parameter_block
        :type: function
        :value: <function CodeGen.add_parameter_block at 0x0000021D5F5EF100>
    
    .. py:attribute:: slangpy.bindings.codegen.CodeGen.finish
        :type: function
        :value: <function CodeGen.finish at 0x0000021D5F5EF1A0>
    


----

.. py:class:: slangpy.bindings.marshall.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.bindings.marshall.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.bindings.marshall.NativeMarshall
    :canonical: slangpy.slangpy.NativeMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`
    


----

.. py:class:: slangpy.bindings.marshall.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
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
    
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.gen_calldata
        :type: function
        :value: <function Marshall.gen_calldata at 0x0000021D5F5EF560>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.gen_trampoline_load
        :type: function
        :value: <function Marshall.gen_trampoline_load at 0x0000021D5F5EF600>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.gen_trampoline_store
        :type: function
        :value: <function Marshall.gen_trampoline_store at 0x0000021D5F5EF6A0>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.can_direct_bind
        :type: function
        :value: <function Marshall.can_direct_bind at 0x0000021D5F5EF740>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.reduce_type
        :type: function
        :value: <function Marshall.reduce_type at 0x0000021D5F5EF7E0>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.resolve_type
        :type: function
        :value: <function Marshall.resolve_type at 0x0000021D5F5EF880>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.resolve_types
        :type: function
        :value: <function Marshall.resolve_types at 0x0000021D5F5EF920>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.resolve_dimensionality
        :type: function
        :value: <function Marshall.resolve_dimensionality at 0x0000021D5F5EF9C0>
    
    .. py:attribute:: slangpy.bindings.marshall.Marshall.build_shader_object
        :type: function
        :value: <function Marshall.build_shader_object at 0x0000021D5F5EFA60>
    


----

.. py:class:: slangpy.bindings.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.bindings.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.bindings.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.bindings.boundvariable.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.bindings.boundvariable.IOType
    :canonical: slangpy.native_refl.IOType
    
    Alias class: :py:class:`slangpy.native_refl.IOType`
    


----

.. py:class:: slangpy.bindings.boundvariable.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.bindings.boundvariable.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.bindings.boundvariable.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.bindings.boundvariable.NativeMarshall
    :canonical: slangpy.slangpy.NativeMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`
    


----

.. py:class:: slangpy.bindings.boundvariable.ModifierID
    :canonical: slangpy.ModifierID
    
    Alias class: :py:class:`slangpy.ModifierID`
    


----

.. py:class:: slangpy.bindings.boundvariable.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.bindings.boundvariable.CodeGen
    :canonical: slangpy.bindings.codegen.CodeGen
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`
    


----

.. py:class:: slangpy.bindings.boundvariable.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.bindings.boundvariable.SlangField
    :canonical: slangpy.native_refl.Field
    
    Alias class: :py:class:`slangpy.native_refl.Field`
    


----

.. py:class:: slangpy.bindings.boundvariable.SlangFunction
    :canonical: slangpy.native_refl.Function
    
    Alias class: :py:class:`slangpy.native_refl.Function`
    


----

.. py:class:: slangpy.bindings.boundvariable.SlangParameter
    :canonical: slangpy.native_refl.Parameter
    
    Alias class: :py:class:`slangpy.native_refl.Parameter`
    


----

.. py:class:: slangpy.bindings.boundvariable.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.bindings.boundvariable.ResolvedParam
    :canonical: slangpy.reflection.typeresolution.ResolvedParam
    
    Alias class: :py:class:`slangpy.reflection.typeresolution.ResolvedParam`
    


----

.. py:data:: slangpy.bindings.boundvariable.MAX_INLINE_TYPE_LEN
    :type: int
    :value: 60



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
    
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundCall.bind
        :type: function
        :value: <function BoundCall.bind at 0x0000021D5F60ACA0>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundCall.apply_explicit_vectorization
        :type: function
        :value: <function BoundCall.apply_explicit_vectorization at 0x0000021D5F60B060>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundCall.values
        :type: function
        :value: <function BoundCall.values at 0x0000021D5F60B100>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundCall.apply_implicit_vectorization
        :type: function
        :value: <function BoundCall.apply_implicit_vectorization at 0x0000021D5F60B1A0>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundCall.finalize_mappings
        :type: function
        :value: <function BoundCall.finalize_mappings at 0x0000021D5F60B240>
    


----

.. py:class:: slangpy.bindings.boundvariable.BoundVariable

    
    Node in a built signature tree, maintains a pairing of python+slang marshall,
    and a potential set of child nodes for use during kernel generation.
    
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.gen_calldata_type_name
        :type: function
        :value: <function BoundVariable.gen_calldata_type_name at 0x0000021D5F60B4C0>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.gen_call_data_code
        :type: function
        :value: <function BoundVariable.gen_call_data_code at 0x0000021D5F60B560>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.bind
        :type: function
        :value: <function BoundVariable.bind at 0x0000021D5F60B600>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.apply_explicit_vectorization
        :type: function
        :value: <function BoundVariable.apply_explicit_vectorization at 0x0000021D5F60B7E0>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.apply_implicit_vectorization
        :type: function
        :value: <function BoundVariable.apply_implicit_vectorization at 0x0000021D5F60B920>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.finalize_mappings
        :type: function
        :value: <function BoundVariable.finalize_mappings at 0x0000021D5F60BA60>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.calculate_differentiability
        :type: function
        :value: <function BoundVariable.calculate_differentiability at 0x0000021D5F60BBA0>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.calculate_direct_bind
        :type: function
        :value: <function BoundVariable.calculate_direct_bind at 0x0000021D5F60BC40>
    
    .. py:attribute:: slangpy.bindings.boundvariable.BoundVariable.get_input_list
        :type: function
        :value: <function BoundVariable.get_input_list at 0x0000021D5F60BCE0>
    


----

.. py:class:: slangpy.bindings.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.bindings.BoundCall
    :canonical: slangpy.bindings.boundvariable.BoundCall
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundCall`
    


----

.. py:class:: slangpy.bindings.BoundVariableException
    :canonical: slangpy.bindings.boundvariable.BoundVariableException
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariableException`
    


----

.. py:class:: slangpy.bindings.boundvariableruntime.NativeBoundCallRuntime
    :canonical: slangpy.slangpy.NativeBoundCallRuntime
    
    Alias class: :py:class:`slangpy.slangpy.NativeBoundCallRuntime`
    


----

.. py:class:: slangpy.bindings.boundvariableruntime.NativeBoundVariableRuntime
    :canonical: slangpy.slangpy.NativeBoundVariableRuntime
    
    Alias class: :py:class:`slangpy.slangpy.NativeBoundVariableRuntime`
    


----

.. py:class:: slangpy.bindings.boundvariableruntime.Shape
    :canonical: slangpy.slangpy.Shape
    
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
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.bindings.BoundCallRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundCallRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundCallRuntime`
    


----

.. py:class:: slangpy.bindings.CodeGen
    :canonical: slangpy.bindings.codegen.CodeGen
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGen`
    


----

.. py:class:: slangpy.bindings.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.bindings.cursor.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.bindings.cursor.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.bindings.cursor.BoundVariableException
    :canonical: slangpy.bindings.boundvariable.BoundVariableException
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariableException`
    


----

.. py:class:: slangpy.bindings.cursor.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.bindings.cursor.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.bindings.cursor.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.bindings.cursor.NativeValueMarshall
    :canonical: slangpy.slangpy.NativeValueMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`
    


----

.. py:class:: slangpy.bindings.cursor.WriteToCursorMarshallInfo

    Metadata needed to marshal values through native cursor-writer registration.
    
    .. py:attribute:: slangpy.bindings.cursor.WriteToCursorMarshallInfo.accepts_type
        :type: function
        :value: <function WriteToCursorMarshallInfo.accepts_type at 0x0000021D5F620720>
    


----

.. py:class:: slangpy.bindings.cursor.WriteToCursorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeValueMarshall`
    
    Marshall for scalar values that are written through the native cursor fast path.
    
    .. py:attribute:: slangpy.bindings.cursor.WriteToCursorMarshall.resolve_types
        :type: function
        :value: <function WriteToCursorMarshall.resolve_types at 0x0000021D5F620900>
    
    .. py:attribute:: slangpy.bindings.cursor.WriteToCursorMarshall.resolve_dimensionality
        :type: function
        :value: <function WriteToCursorMarshall.resolve_dimensionality at 0x0000021D5F6207C0>
    
    .. py:attribute:: slangpy.bindings.cursor.WriteToCursorMarshall.gen_calldata
        :type: function
        :value: <function WriteToCursorMarshall.gen_calldata at 0x0000021D5F620D60>
    
    .. py:attribute:: slangpy.bindings.cursor.WriteToCursorMarshall.can_direct_bind
        :type: function
        :value: <function WriteToCursorMarshall.can_direct_bind at 0x0000021D5F620E00>
    
    .. py:attribute:: slangpy.bindings.cursor.WriteToCursorMarshall.gen_trampoline_load
        :type: function
        :value: <function WriteToCursorMarshall.gen_trampoline_load at 0x0000021D5F620EA0>
    


----

.. py:class:: slangpy.bindings.WriteToCursorMarshall
    :canonical: slangpy.bindings.cursor.WriteToCursorMarshall
    
    Alias class: :py:class:`slangpy.bindings.cursor.WriteToCursorMarshall`
    


----

.. py:class:: slangpy.bindings.WriteToCursorMarshallInfo
    :canonical: slangpy.bindings.cursor.WriteToCursorMarshallInfo
    
    Alias class: :py:class:`slangpy.bindings.cursor.WriteToCursorMarshallInfo`
    


----

.. py:class:: slangpy.bindings.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.bindings.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.bindings.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.experimental.gridarg.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.experimental.gridarg.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.experimental.gridarg.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.experimental.gridarg.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.experimental.gridarg.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.experimental.gridarg.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.experimental.gridarg.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.experimental.gridarg.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.experimental.gridarg.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.experimental.gridarg.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.experimental.gridarg.NativeObject
    :canonical: slangpy.slangpy.NativeObject
    
    Alias class: :py:class:`slangpy.slangpy.NativeObject`
    


----

.. py:class:: slangpy.experimental.gridarg.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.experimental.gridarg.GridArg

    Base class: :py:class:`slangpy.slangpy.NativeObject`
    
    
    Passes the thread id as an argument to a SlangPy function.
    
    


----

.. py:class:: slangpy.experimental.gridarg.GridArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.experimental.gridarg.GridArgMarshall.gen_calldata
        :type: function
        :value: <function GridArgMarshall.gen_calldata at 0x0000021D5F623240>
    
    .. py:attribute:: slangpy.experimental.gridarg.GridArgMarshall.create_calldata
        :type: function
        :value: <function GridArgMarshall.create_calldata at 0x0000021D5F6232E0>
    
    .. py:attribute:: slangpy.experimental.gridarg.GridArgMarshall.get_shape
        :type: function
        :value: <function GridArgMarshall.get_shape at 0x0000021D5F623380>
    
    .. py:attribute:: slangpy.experimental.gridarg.GridArgMarshall.resolve_type
        :type: function
        :value: <function GridArgMarshall.resolve_type at 0x0000021D5F623420>
    
    .. py:attribute:: slangpy.experimental.gridarg.GridArgMarshall.resolve_dimensionality
        :type: function
        :value: <function GridArgMarshall.resolve_dimensionality at 0x0000021D5F6234C0>
    


----

.. py:class:: slangpy.types.diffpair.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.types.diffpair.PrimType
    :canonical: slangpy.core.enums.PrimType
    
    Alias class: :py:class:`slangpy.core.enums.PrimType`
    


----

.. py:class:: slangpy.types.diffpair.DiffPair

    
    A pair of values, one representing the primal value and the other representing the gradient value.
    Typically only required when wanting to output gradients from scalar calls to a function.
    
    
    .. py:attribute:: slangpy.types.diffpair.DiffPair.get
        :type: function
        :value: <function DiffPair.get at 0x0000021D5F5ECF40>
    
    .. py:attribute:: slangpy.types.diffpair.DiffPair.set
        :type: function
        :value: <function DiffPair.set at 0x0000021D5F5ED080>
    


----

.. py:class:: slangpy.types.DiffPair
    :canonical: slangpy.types.diffpair.DiffPair
    
    Alias class: :py:class:`slangpy.types.diffpair.DiffPair`
    


----

.. py:class:: slangpy.types.helpers.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.types.helpers.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.types.helpers.ArrayType
    :canonical: slangpy.native_refl.ArrayType
    
    Alias class: :py:class:`slangpy.native_refl.ArrayType`
    


----

.. py:class:: slangpy.types.helpers.ScalarType
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.types.helpers.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.types.helpers.VectorType
    :canonical: slangpy.native_refl.VectorType
    
    Alias class: :py:class:`slangpy.native_refl.VectorType`
    


----

.. py:class:: slangpy.types.wanghasharg.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.types.wanghasharg.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.types.wanghasharg.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.types.wanghasharg.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.types.wanghasharg.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.types.wanghasharg.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.types.wanghasharg.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.types.wanghasharg.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.types.wanghasharg.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.types.wanghasharg.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.types.wanghasharg.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.types.wanghasharg.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.types.wanghasharg.WangHashArg

    
    Generates a random int/vector per thread when passed as an argument using a wang
    hash of the thread id.
    
    


----

.. py:class:: slangpy.types.wanghasharg.WangHashArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.types.wanghasharg.WangHashArgMarshall.gen_calldata
        :type: function
        :value: <function WangHashArgMarshall.gen_calldata at 0x0000021D5F621BC0>
    
    .. py:attribute:: slangpy.types.wanghasharg.WangHashArgMarshall.create_calldata
        :type: function
        :value: <function WangHashArgMarshall.create_calldata at 0x0000021D5F621C60>
    
    .. py:attribute:: slangpy.types.wanghasharg.WangHashArgMarshall.resolve_type
        :type: function
        :value: <function WangHashArgMarshall.resolve_type at 0x0000021D5F621D00>
    
    .. py:attribute:: slangpy.types.wanghasharg.WangHashArgMarshall.resolve_dimensionality
        :type: function
        :value: <function WangHashArgMarshall.resolve_dimensionality at 0x0000021D5F621DA0>
    


----

.. py:class:: slangpy.types.randfloatarg.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.types.randfloatarg.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.types.randfloatarg.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.types.randfloatarg.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.types.randfloatarg.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.types.randfloatarg.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.types.randfloatarg.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.types.randfloatarg.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.types.randfloatarg.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.types.randfloatarg.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.types.randfloatarg.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.types.randfloatarg.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.types.randfloatarg.RandFloatArg

    
    Generates a random float/vector per thread when passed as an argument
    to a SlangPy function. The min and max values are inclusive.
    
    


----

.. py:class:: slangpy.types.randfloatarg.RandFloatArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.types.randfloatarg.RandFloatArgMarshall.gen_calldata
        :type: function
        :value: <function RandFloatArgMarshall.gen_calldata at 0x0000021D5F622160>
    
    .. py:attribute:: slangpy.types.randfloatarg.RandFloatArgMarshall.create_calldata
        :type: function
        :value: <function RandFloatArgMarshall.create_calldata at 0x0000021D5F622200>
    
    .. py:attribute:: slangpy.types.randfloatarg.RandFloatArgMarshall.resolve_type
        :type: function
        :value: <function RandFloatArgMarshall.resolve_type at 0x0000021D5F6222A0>
    
    .. py:attribute:: slangpy.types.randfloatarg.RandFloatArgMarshall.resolve_dimensionality
        :type: function
        :value: <function RandFloatArgMarshall.resolve_dimensionality at 0x0000021D5F622340>
    


----

.. py:class:: slangpy.types.RandFloatArg
    :canonical: slangpy.types.randfloatarg.RandFloatArg
    
    Alias class: :py:class:`slangpy.types.randfloatarg.RandFloatArg`
    


----

.. py:class:: slangpy.types.threadidarg.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.types.threadidarg.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.types.threadidarg.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.types.threadidarg.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.types.threadidarg.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.types.threadidarg.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.types.threadidarg.NativeObject
    :canonical: slangpy.slangpy.NativeObject
    
    Alias class: :py:class:`slangpy.slangpy.NativeObject`
    


----

.. py:class:: slangpy.types.threadidarg.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.types.threadidarg.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.types.threadidarg.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.types.threadidarg.ThreadIdArg

    Base class: :py:class:`slangpy.slangpy.NativeObject`
    
    
    Passes the thread id as an argument to a SlangPy function.
    
    


----

.. py:class:: slangpy.types.threadidarg.ThreadIdArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.types.threadidarg.ThreadIdArgMarshall.gen_calldata
        :type: function
        :value: <function ThreadIdArgMarshall.gen_calldata at 0x0000021D5F622840>
    
    .. py:attribute:: slangpy.types.threadidarg.ThreadIdArgMarshall.resolve_type
        :type: function
        :value: <function ThreadIdArgMarshall.resolve_type at 0x0000021D5F6228E0>
    
    .. py:attribute:: slangpy.types.threadidarg.ThreadIdArgMarshall.resolve_dimensionality
        :type: function
        :value: <function ThreadIdArgMarshall.resolve_dimensionality at 0x0000021D5F622980>
    


----

.. py:class:: slangpy.types.ThreadIdArg
    :canonical: slangpy.types.threadidarg.ThreadIdArg
    
    Alias class: :py:class:`slangpy.types.threadidarg.ThreadIdArg`
    


----

.. py:class:: slangpy.types.callidarg.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.types.callidarg.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.types.callidarg.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.types.callidarg.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.types.callidarg.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.types.callidarg.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.types.callidarg.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.types.callidarg.CallIdArg

    
    Passes the thread id as an argument to a SlangPy function.
    
    


----

.. py:class:: slangpy.types.callidarg.CallIdArgMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.types.callidarg.CallIdArgMarshall.gen_calldata
        :type: function
        :value: <function CallIdArgMarshall.gen_calldata at 0x0000021D5F623740>
    
    .. py:attribute:: slangpy.types.callidarg.CallIdArgMarshall.resolve_type
        :type: function
        :value: <function CallIdArgMarshall.resolve_type at 0x0000021D5F6237E0>
    
    .. py:attribute:: slangpy.types.callidarg.CallIdArgMarshall.resolve_dimensionality
        :type: function
        :value: <function CallIdArgMarshall.resolve_dimensionality at 0x0000021D5F623880>
    


----

.. py:class:: slangpy.types.CallIdArg
    :canonical: slangpy.types.callidarg.CallIdArg
    
    Alias class: :py:class:`slangpy.types.callidarg.CallIdArg`
    


----

.. py:class:: slangpy.types.valueref.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.types.valueref.ValueRef

    
    Minimal class to hold a reference to a scalar value, allowing user to get outputs
    from scalar inout/out arguments.
    
    


----

.. py:class:: slangpy.types.ValueRef
    :canonical: slangpy.types.valueref.ValueRef
    
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`
    


----

.. py:class:: slangpy.types.WangHashArg
    :canonical: slangpy.types.wanghasharg.WangHashArg
    
    Alias class: :py:class:`slangpy.types.wanghasharg.WangHashArg`
    


----

.. py:class:: slangpy.types.tensor.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.types.tensor.TensorDesc
    :canonical: slangpy.native_func.TensorDesc
    
    Alias class: :py:class:`slangpy.native_func.TensorDesc`
    


----

.. py:class:: slangpy.types.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.DiffPair
    :canonical: slangpy.types.diffpair.DiffPair
    
    Alias class: :py:class:`slangpy.types.diffpair.DiffPair`
    


----

.. py:class:: slangpy.RandFloatArg
    :canonical: slangpy.types.randfloatarg.RandFloatArg
    
    Alias class: :py:class:`slangpy.types.randfloatarg.RandFloatArg`
    


----

.. py:class:: slangpy.ThreadIdArg
    :canonical: slangpy.types.threadidarg.ThreadIdArg
    
    Alias class: :py:class:`slangpy.types.threadidarg.ThreadIdArg`
    


----

.. py:class:: slangpy.CallIdArg
    :canonical: slangpy.types.callidarg.CallIdArg
    
    Alias class: :py:class:`slangpy.types.callidarg.CallIdArg`
    


----

.. py:class:: slangpy.ValueRef
    :canonical: slangpy.types.valueref.ValueRef
    
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`
    


----

.. py:class:: slangpy.WangHashArg
    :canonical: slangpy.types.wanghasharg.WangHashArg
    
    Alias class: :py:class:`slangpy.types.wanghasharg.WangHashArg`
    


----

.. py:class:: slangpy.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.builtin.value.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.value.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.value.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.value.NativeValueMarshall
    :canonical: slangpy.slangpy.NativeValueMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`
    


----

.. py:function:: slangpy.builtin.value.unpack_arg(arg: object) -> object

    N/A
    


----

.. py:class:: slangpy.builtin.value.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.builtin.value.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.value.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.value.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.value.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.value.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.value.ValueMarshall

    Base class: :py:class:`slangpy.slangpy.NativeValueMarshall`
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.can_direct_bind
        :type: function
        :value: <function ValueMarshall.can_direct_bind at 0x0000021D5F6513A0>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.gen_calldata
        :type: function
        :value: <function ValueMarshall.gen_calldata at 0x0000021D5F651440>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.gen_trampoline_load
        :type: function
        :value: <function ValueMarshall.gen_trampoline_load at 0x0000021D5F6514E0>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.gen_trampoline_store
        :type: function
        :value: <function ValueMarshall.gen_trampoline_store at 0x0000021D5F651580>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.create_dispatchdata
        :type: function
        :value: <function ValueMarshall.create_dispatchdata at 0x0000021D5F651620>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.create_output
        :type: function
        :value: <function ValueMarshall.create_output at 0x0000021D5F6516C0>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.read_output
        :type: function
        :value: <function ValueMarshall.read_output at 0x0000021D5F651760>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.resolve_types
        :type: function
        :value: <function ValueMarshall.resolve_types at 0x0000021D5F651800>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.reduce_type
        :type: function
        :value: <function ValueMarshall.reduce_type at 0x0000021D5F6518A0>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.resolve_dimensionality
        :type: function
        :value: <function ValueMarshall.resolve_dimensionality at 0x0000021D5F651940>
    
    .. py:attribute:: slangpy.builtin.value.ValueMarshall.build_shader_object
        :type: function
        :value: <function ValueMarshall.build_shader_object at 0x0000021D5F6519E0>
    


----

.. py:class:: slangpy.builtin.value.ScalarMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`
    
    .. py:attribute:: slangpy.builtin.value.ScalarMarshall.reduce_type
        :type: function
        :value: <function ScalarMarshall.reduce_type at 0x0000021D5F651B20>
    
    .. py:attribute:: slangpy.builtin.value.ScalarMarshall.resolve_types
        :type: function
        :value: <function ScalarMarshall.resolve_types at 0x0000021D5F651BC0>
    


----

.. py:class:: slangpy.builtin.value.NoneMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`
    
    .. py:attribute:: slangpy.builtin.value.NoneMarshall.resolve_dimensionality
        :type: function
        :value: <function NoneMarshall.resolve_dimensionality at 0x0000021D5F651DA0>
    


----

.. py:class:: slangpy.builtin.value.VectorMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`
    
    .. py:attribute:: slangpy.builtin.value.VectorMarshall.reduce_type
        :type: function
        :value: <function VectorMarshall.reduce_type at 0x0000021D5F651EE0>
    
    .. py:attribute:: slangpy.builtin.value.VectorMarshall.resolve_types
        :type: function
        :value: <function VectorMarshall.resolve_types at 0x0000021D5F651F80>
    
    .. py:attribute:: slangpy.builtin.value.VectorMarshall.gen_calldata
        :type: function
        :value: <function VectorMarshall.gen_calldata at 0x0000021D5F652020>
    
    .. py:attribute:: slangpy.builtin.value.VectorMarshall.build_shader_object
        :type: function
        :value: <function VectorMarshall.build_shader_object at 0x0000021D5F6520C0>
    


----

.. py:class:: slangpy.builtin.value.MatrixMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`
    
    .. py:attribute:: slangpy.builtin.value.MatrixMarshall.reduce_type
        :type: function
        :value: <function MatrixMarshall.reduce_type at 0x0000021D5F652200>
    
    .. py:attribute:: slangpy.builtin.value.MatrixMarshall.resolve_types
        :type: function
        :value: <function MatrixMarshall.resolve_types at 0x0000021D5F6522A0>
    


----

.. py:data:: slangpy.builtin.value.base_name
    :type: str
    :value: "float16_t"



----

.. py:data:: slangpy.builtin.value.dim
    :type: int
    :value: 4



----

.. py:class:: slangpy.builtin.value.vec_type
    :canonical: slangpy.math.float16_t4
    
    Alias class: :py:class:`slangpy.math.float16_t4`
    


----

.. py:data:: slangpy.builtin.value.row
    :type: int
    :value: 4



----

.. py:data:: slangpy.builtin.value.col
    :type: int
    :value: 4



----

.. py:class:: slangpy.builtin.ValueMarshall
    :canonical: slangpy.builtin.value.ValueMarshall
    
    Alias class: :py:class:`slangpy.builtin.value.ValueMarshall`
    


----

.. py:class:: slangpy.builtin.valueref.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.valueref.BufferCursor
    :canonical: slangpy.BufferCursor
    
    Alias class: :py:class:`slangpy.BufferCursor`
    


----

.. py:class:: slangpy.builtin.valueref.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.valueref.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.valueref.Buffer
    :canonical: slangpy.Buffer
    
    Alias class: :py:class:`slangpy.Buffer`
    


----

.. py:class:: slangpy.builtin.valueref.BufferUsage
    :canonical: slangpy.BufferUsage
    
    Alias class: :py:class:`slangpy.BufferUsage`
    


----

.. py:class:: slangpy.builtin.valueref.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.builtin.valueref.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.valueref.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.valueref.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.valueref.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.valueref.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.builtin.valueref.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.valueref.ValueRef
    :canonical: slangpy.types.valueref.ValueRef
    
    Alias class: :py:class:`slangpy.types.valueref.ValueRef`
    


----

.. py:class:: slangpy.builtin.valueref.ValueRefMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.resolve_type
        :type: function
        :value: <function ValueRefMarshall.resolve_type at 0x0000021D5F653E20>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.resolve_types
        :type: function
        :value: <function ValueRefMarshall.resolve_types at 0x0000021D5F653EC0>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.resolve_dimensionality
        :type: function
        :value: <function ValueRefMarshall.resolve_dimensionality at 0x0000021D5F653F60>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.can_direct_bind
        :type: function
        :value: <function ValueRefMarshall.can_direct_bind at 0x0000021D5F654040>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.gen_calldata
        :type: function
        :value: <function ValueRefMarshall.gen_calldata at 0x0000021D5F6540E0>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.gen_trampoline_load
        :type: function
        :value: <function ValueRefMarshall.gen_trampoline_load at 0x0000021D5F654180>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.gen_trampoline_store
        :type: function
        :value: <function ValueRefMarshall.gen_trampoline_store at 0x0000021D5F654220>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.create_calldata
        :type: function
        :value: <function ValueRefMarshall.create_calldata at 0x0000021D5F6542C0>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.create_dispatchdata
        :type: function
        :value: <function ValueRefMarshall.create_dispatchdata at 0x0000021D5F654360>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.read_calldata
        :type: function
        :value: <function ValueRefMarshall.read_calldata at 0x0000021D5F654400>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.create_output
        :type: function
        :value: <function ValueRefMarshall.create_output at 0x0000021D5F6544A0>
    
    .. py:attribute:: slangpy.builtin.valueref.ValueRefMarshall.read_output
        :type: function
        :value: <function ValueRefMarshall.read_output at 0x0000021D5F654540>
    


----

.. py:class:: slangpy.builtin.ValueRefMarshall
    :canonical: slangpy.builtin.valueref.ValueRefMarshall
    
    Alias class: :py:class:`slangpy.builtin.valueref.ValueRefMarshall`
    


----

.. py:class:: slangpy.builtin.diffpair.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.diffpair.PrimType
    :canonical: slangpy.core.enums.PrimType
    
    Alias class: :py:class:`slangpy.core.enums.PrimType`
    


----

.. py:class:: slangpy.builtin.diffpair.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.diffpair.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.diffpair.NativeMarshall
    :canonical: slangpy.slangpy.NativeMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`
    


----

.. py:class:: slangpy.builtin.diffpair.Buffer
    :canonical: slangpy.Buffer
    
    Alias class: :py:class:`slangpy.Buffer`
    


----

.. py:class:: slangpy.builtin.diffpair.BufferUsage
    :canonical: slangpy.BufferUsage
    
    Alias class: :py:class:`slangpy.BufferUsage`
    


----

.. py:class:: slangpy.builtin.diffpair.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.builtin.diffpair.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.diffpair.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.diffpair.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.diffpair.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.diffpair.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.diffpair.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.diffpair.DiffPair
    :canonical: slangpy.types.diffpair.DiffPair
    
    Alias class: :py:class:`slangpy.types.diffpair.DiffPair`
    


----

.. py:class:: slangpy.builtin.diffpair.DiffPairMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.resolve_type
        :type: function
        :value: <function DiffPairMarshall.resolve_type at 0x0000021D5F654AE0>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.resolve_dimensionality
        :type: function
        :value: <function DiffPairMarshall.resolve_dimensionality at 0x0000021D5F654B80>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.gen_calldata
        :type: function
        :value: <function DiffPairMarshall.gen_calldata at 0x0000021D5F654C20>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.get_type
        :type: function
        :value: <function DiffPairMarshall.get_type at 0x0000021D5F654CC0>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.create_calldata
        :type: function
        :value: <function DiffPairMarshall.create_calldata at 0x0000021D5F654D60>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.read_calldata
        :type: function
        :value: <function DiffPairMarshall.read_calldata at 0x0000021D5F654E00>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.create_output
        :type: function
        :value: <function DiffPairMarshall.create_output at 0x0000021D5F654EA0>
    
    .. py:attribute:: slangpy.builtin.diffpair.DiffPairMarshall.read_output
        :type: function
        :value: <function DiffPairMarshall.read_output at 0x0000021D5F654F40>
    


----

.. py:class:: slangpy.builtin.DiffPairMarshall
    :canonical: slangpy.builtin.diffpair.DiffPairMarshall
    
    Alias class: :py:class:`slangpy.builtin.diffpair.DiffPairMarshall`
    


----

.. py:class:: slangpy.builtin.struct.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.struct.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.builtin.struct.NativeMarshall
    :canonical: slangpy.slangpy.NativeMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeMarshall`
    


----

.. py:class:: slangpy.builtin.struct.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.struct.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.struct.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.struct.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.struct.UnknownType
    :canonical: slangpy.native_refl.UnknownType
    
    Alias class: :py:class:`slangpy.native_refl.UnknownType`
    


----

.. py:class:: slangpy.builtin.struct.StructType
    :canonical: slangpy.native_refl.StructType
    
    Alias class: :py:class:`slangpy.native_refl.StructType`
    


----

.. py:class:: slangpy.builtin.struct.InterfaceType
    :canonical: slangpy.native_refl.InterfaceType
    
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`
    


----

.. py:class:: slangpy.builtin.struct.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.struct.ValueMarshall
    :canonical: slangpy.builtin.value.ValueMarshall
    
    Alias class: :py:class:`slangpy.builtin.value.ValueMarshall`
    


----

.. py:class:: slangpy.builtin.struct.StructMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`
    
    .. py:attribute:: slangpy.builtin.struct.StructMarshall.resolve_types
        :type: function
        :value: <function StructMarshall.resolve_types at 0x0000021D5F6553A0>
    
    .. py:attribute:: slangpy.builtin.struct.StructMarshall.resolve_dimensionality
        :type: function
        :value: <function StructMarshall.resolve_dimensionality at 0x0000021D5F655440>
    
    .. py:attribute:: slangpy.builtin.struct.StructMarshall.can_direct_bind
        :type: function
        :value: <function StructMarshall.can_direct_bind at 0x0000021D5F6554E0>
    
    .. py:attribute:: slangpy.builtin.struct.StructMarshall.gen_trampoline_load
        :type: function
        :value: <function StructMarshall.gen_trampoline_load at 0x0000021D5F655580>
    
    .. py:attribute:: slangpy.builtin.struct.StructMarshall.gen_trampoline_store
        :type: function
        :value: <function StructMarshall.gen_trampoline_store at 0x0000021D5F655620>
    
    .. py:attribute:: slangpy.builtin.struct.StructMarshall.create_dispatchdata
        :type: function
        :value: <function StructMarshall.create_dispatchdata at 0x0000021D5F6556C0>
    


----

.. py:class:: slangpy.builtin.StructMarshall
    :canonical: slangpy.builtin.struct.StructMarshall
    
    Alias class: :py:class:`slangpy.builtin.struct.StructMarshall`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.NativeBufferMarshall
    :canonical: slangpy.slangpy.NativeBufferMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeBufferMarshall`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.StructuredBufferType
    :canonical: slangpy.native_refl.StructuredBufferType
    
    Alias class: :py:class:`slangpy.native_refl.StructuredBufferType`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.ByteAddressBufferType
    :canonical: slangpy.native_refl.ByteAddressBufferType
    
    Alias class: :py:class:`slangpy.native_refl.ByteAddressBufferType`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.PointerType
    :canonical: slangpy.native_refl.PointerType
    
    Alias class: :py:class:`slangpy.native_refl.PointerType`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.Buffer
    :canonical: slangpy.Buffer
    
    Alias class: :py:class:`slangpy.Buffer`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.BufferUsage
    :canonical: slangpy.BufferUsage
    
    Alias class: :py:class:`slangpy.BufferUsage`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.structuredbuffer.BufferMarshall

    Base class: :py:class:`slangpy.slangpy.NativeBufferMarshall`
    
    .. py:attribute:: slangpy.builtin.structuredbuffer.BufferMarshall.resolve_dimensionality
        :type: function
        :value: <function BufferMarshall.resolve_dimensionality at 0x0000021D5F655A80>
    
    .. py:attribute:: slangpy.builtin.structuredbuffer.BufferMarshall.resolve_types
        :type: function
        :value: <function BufferMarshall.resolve_types at 0x0000021D5F655B20>
    
    .. py:attribute:: slangpy.builtin.structuredbuffer.BufferMarshall.gen_calldata
        :type: function
        :value: <function BufferMarshall.gen_calldata at 0x0000021D5F655BC0>
    
    .. py:attribute:: slangpy.builtin.structuredbuffer.BufferMarshall.reduce_type
        :type: function
        :value: <function BufferMarshall.reduce_type at 0x0000021D5F655D00>
    


----

.. py:class:: slangpy.builtin.BufferMarshall
    :canonical: slangpy.builtin.structuredbuffer.BufferMarshall
    
    Alias class: :py:class:`slangpy.builtin.structuredbuffer.BufferMarshall`
    


----

.. py:class:: slangpy.builtin.descriptor.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.descriptor.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.descriptor.NativeDescriptorMarshall
    :canonical: slangpy.slangpy.NativeDescriptorMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeDescriptorMarshall`
    


----

.. py:function:: slangpy.builtin.descriptor.unpack_arg(arg: object) -> object

    N/A
    


----

.. py:class:: slangpy.builtin.descriptor.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.builtin.descriptor.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.builtin.descriptor.ShaderCursor
    :canonical: slangpy.ShaderCursor
    
    Alias class: :py:class:`slangpy.ShaderCursor`
    


----

.. py:class:: slangpy.builtin.descriptor.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.descriptor.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.descriptor.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.descriptor.DescriptorHandle
    :canonical: slangpy.DescriptorHandle
    
    Alias class: :py:class:`slangpy.DescriptorHandle`
    


----

.. py:class:: slangpy.builtin.descriptor.DescriptorHandleType
    :canonical: slangpy.DescriptorHandleType
    
    Alias class: :py:class:`slangpy.DescriptorHandleType`
    


----

.. py:class:: slangpy.builtin.descriptor.DescriptorMarshall

    Base class: :py:class:`slangpy.slangpy.NativeDescriptorMarshall`
    
    .. py:attribute:: slangpy.builtin.descriptor.DescriptorMarshall.resolve_type
        :type: function
        :value: <function DescriptorMarshall.resolve_type at 0x0000021D5F6560C0>
    
    .. py:attribute:: slangpy.builtin.descriptor.DescriptorMarshall.resolve_dimensionality
        :type: function
        :value: <function DescriptorMarshall.resolve_dimensionality at 0x0000021D5F656160>
    
    .. py:attribute:: slangpy.builtin.descriptor.DescriptorMarshall.gen_calldata
        :type: function
        :value: <function DescriptorMarshall.gen_calldata at 0x0000021D5F656200>
    
    .. py:attribute:: slangpy.builtin.descriptor.DescriptorMarshall.reduce_type
        :type: function
        :value: <function DescriptorMarshall.reduce_type at 0x0000021D5F6562A0>
    
    .. py:attribute:: slangpy.builtin.descriptor.DescriptorMarshall.build_shader_object
        :type: function
        :value: <function DescriptorMarshall.build_shader_object at 0x0000021D5F656340>
    


----

.. py:class:: slangpy.builtin.DescriptorMarshall
    :canonical: slangpy.builtin.descriptor.DescriptorMarshall
    
    Alias class: :py:class:`slangpy.builtin.descriptor.DescriptorMarshall`
    


----

.. py:class:: slangpy.builtin.texture.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.texture.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.texture.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.texture.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.builtin.texture.NativeTextureMarshall
    :canonical: slangpy.slangpy.NativeTextureMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeTextureMarshall`
    


----

.. py:class:: slangpy.builtin.texture.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.builtin.texture.FormatType
    :canonical: slangpy.FormatType
    
    Alias class: :py:class:`slangpy.FormatType`
    


----

.. py:class:: slangpy.builtin.texture.TextureType
    :canonical: slangpy.TextureType
    
    Alias class: :py:class:`slangpy.TextureType`
    


----

.. py:class:: slangpy.builtin.texture.TextureUsage
    :canonical: slangpy.TextureUsage
    
    Alias class: :py:class:`slangpy.TextureUsage`
    


----

.. py:class:: slangpy.builtin.texture.Sampler
    :canonical: slangpy.Sampler
    
    Alias class: :py:class:`slangpy.Sampler`
    


----

.. py:class:: slangpy.builtin.texture.Texture
    :canonical: slangpy.Texture
    
    Alias class: :py:class:`slangpy.Texture`
    


----

.. py:class:: slangpy.builtin.texture.Format
    :canonical: slangpy.Format
    
    Alias class: :py:class:`slangpy.Format`
    


----

.. py:function:: slangpy.builtin.texture.get_format_info(arg: slangpy.Format, /) -> slangpy.FormatInfo



----

.. py:class:: slangpy.builtin.texture.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.builtin.texture.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.texture.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.texture.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.texture.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.texture.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.builtin.texture.TextureMarshall

    Base class: :py:class:`slangpy.slangpy.NativeTextureMarshall`
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.reduce_type
        :type: function
        :value: <function TextureMarshall.reduce_type at 0x0000021D5F656980>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.resolve_type
        :type: function
        :value: <function TextureMarshall.resolve_type at 0x0000021D5F656A20>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.resolve_types
        :type: function
        :value: <function TextureMarshall.resolve_types at 0x0000021D5F656AC0>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.resolve_dimensionality
        :type: function
        :value: <function TextureMarshall.resolve_dimensionality at 0x0000021D5F656B60>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.build_type_name
        :type: function
        :value: <function TextureMarshall.build_type_name at 0x0000021D5F656CA0>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.build_type
        :type: function
        :value: <function TextureMarshall.build_type at 0x0000021D5F656D40>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.build_accessor_name
        :type: function
        :value: <function TextureMarshall.build_accessor_name at 0x0000021D5F656DE0>
    
    .. py:attribute:: slangpy.builtin.texture.TextureMarshall.gen_calldata
        :type: function
        :value: <function TextureMarshall.gen_calldata at 0x0000021D5F656E80>
    


----

.. py:class:: slangpy.builtin.texture.SamplerMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.builtin.texture.SamplerMarshall.gen_calldata
        :type: function
        :value: <function SamplerMarshall.gen_calldata at 0x0000021D5F6572E0>
    
    .. py:attribute:: slangpy.builtin.texture.SamplerMarshall.create_calldata
        :type: function
        :value: <function SamplerMarshall.create_calldata at 0x0000021D5F657380>
    
    .. py:attribute:: slangpy.builtin.texture.SamplerMarshall.create_dispatchdata
        :type: function
        :value: <function SamplerMarshall.create_dispatchdata at 0x0000021D5F657420>
    


----

.. py:class:: slangpy.builtin.TextureMarshall
    :canonical: slangpy.builtin.texture.TextureMarshall
    
    Alias class: :py:class:`slangpy.builtin.texture.TextureMarshall`
    


----

.. py:class:: slangpy.builtin.array.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.array.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.builtin.array.ValueMarshall
    :canonical: slangpy.builtin.value.ValueMarshall
    
    Alias class: :py:class:`slangpy.builtin.value.ValueMarshall`
    


----

.. py:class:: slangpy.builtin.array.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.array.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.array.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.array.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.array.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.array.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.array.ShaderCursor
    :canonical: slangpy.ShaderCursor
    
    Alias class: :py:class:`slangpy.ShaderCursor`
    


----

.. py:class:: slangpy.builtin.array.ShaderObject
    :canonical: slangpy.ShaderObject
    
    Alias class: :py:class:`slangpy.ShaderObject`
    


----

.. py:class:: slangpy.builtin.array.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.array.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.array.NativeValueMarshall
    :canonical: slangpy.slangpy.NativeValueMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeValueMarshall`
    


----

.. py:function:: slangpy.builtin.array.unpack_arg(arg: object) -> object

    N/A
    


----

.. py:class:: slangpy.builtin.array.ArrayMarshall

    Base class: :py:class:`slangpy.builtin.value.ValueMarshall`
    
    .. py:attribute:: slangpy.builtin.array.ArrayMarshall.reduce_type
        :type: function
        :value: <function ArrayMarshall.reduce_type at 0x0000021D5F6576A0>
    
    .. py:attribute:: slangpy.builtin.array.ArrayMarshall.resolve_types
        :type: function
        :value: <function ArrayMarshall.resolve_types at 0x0000021D5F657740>
    
    .. py:attribute:: slangpy.builtin.array.ArrayMarshall.gen_calldata
        :type: function
        :value: <function ArrayMarshall.gen_calldata at 0x0000021D5F6577E0>
    
    .. py:attribute:: slangpy.builtin.array.ArrayMarshall.build_shader_object
        :type: function
        :value: <function ArrayMarshall.build_shader_object at 0x0000021D5F657880>
    


----

.. py:class:: slangpy.builtin.ArrayMarshall
    :canonical: slangpy.builtin.array.ArrayMarshall
    
    Alias class: :py:class:`slangpy.builtin.array.ArrayMarshall`
    


----

.. py:class:: slangpy.builtin.resourceview.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.resourceview.TextureView
    :canonical: slangpy.TextureView
    
    Alias class: :py:class:`slangpy.TextureView`
    


----

.. py:class:: slangpy.builtin.resourceview.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.TextureView
    :canonical: slangpy.TextureView
    
    Alias class: :py:class:`slangpy.TextureView`
    


----

.. py:class:: slangpy.builtin.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.AccelerationStructure
    :canonical: slangpy.AccelerationStructure
    
    Alias class: :py:class:`slangpy.AccelerationStructure`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.accelerationstructure.AccelerationStructureMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.builtin.accelerationstructure.AccelerationStructureMarshall.gen_calldata
        :type: function
        :value: <function AccelerationStructureMarshall.gen_calldata at 0x0000021D5F657D80>
    
    .. py:attribute:: slangpy.builtin.accelerationstructure.AccelerationStructureMarshall.create_calldata
        :type: function
        :value: <function AccelerationStructureMarshall.create_calldata at 0x0000021D5F657E20>
    
    .. py:attribute:: slangpy.builtin.accelerationstructure.AccelerationStructureMarshall.create_dispatchdata
        :type: function
        :value: <function AccelerationStructureMarshall.create_dispatchdata at 0x0000021D5F657EC0>
    


----

.. py:class:: slangpy.builtin.AccelerationStructureMarshall
    :canonical: slangpy.builtin.accelerationstructure.AccelerationStructureMarshall
    
    Alias class: :py:class:`slangpy.builtin.accelerationstructure.AccelerationStructureMarshall`
    


----

.. py:class:: slangpy.builtin.range.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.range.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.range.CallContext
    :canonical: slangpy.slangpy.CallContext
    
    Alias class: :py:class:`slangpy.slangpy.CallContext`
    


----

.. py:class:: slangpy.builtin.range.Shape
    :canonical: slangpy.slangpy.Shape
    
    Alias class: :py:class:`slangpy.slangpy.Shape`
    


----

.. py:class:: slangpy.builtin.range.TypeReflection
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.builtin.range.Marshall
    :canonical: slangpy.bindings.marshall.Marshall
    
    Alias class: :py:class:`slangpy.bindings.marshall.Marshall`
    


----

.. py:class:: slangpy.builtin.range.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.range.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.range.BoundVariableRuntime
    :canonical: slangpy.bindings.boundvariableruntime.BoundVariableRuntime
    
    Alias class: :py:class:`slangpy.bindings.boundvariableruntime.BoundVariableRuntime`
    


----

.. py:class:: slangpy.builtin.range.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.range.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.range.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.range.TR
    :canonical: slangpy.TypeReflection
    
    Alias class: :py:class:`slangpy.TypeReflection`
    


----

.. py:class:: slangpy.builtin.range.RangeMarshall

    Base class: :py:class:`slangpy.bindings.marshall.Marshall`
    
    .. py:attribute:: slangpy.builtin.range.RangeMarshall.gen_calldata
        :type: function
        :value: <function RangeMarshall.gen_calldata at 0x0000021D5F6680E0>
    
    .. py:attribute:: slangpy.builtin.range.RangeMarshall.create_calldata
        :type: function
        :value: <function RangeMarshall.create_calldata at 0x0000021D5F668180>
    
    .. py:attribute:: slangpy.builtin.range.RangeMarshall.get_shape
        :type: function
        :value: <function RangeMarshall.get_shape at 0x0000021D5F668220>
    
    .. py:attribute:: slangpy.builtin.range.RangeMarshall.resolve_types
        :type: function
        :value: <function RangeMarshall.resolve_types at 0x0000021D5F6682C0>
    
    .. py:attribute:: slangpy.builtin.range.RangeMarshall.resolve_dimensionality
        :type: function
        :value: <function RangeMarshall.resolve_dimensionality at 0x0000021D5F668360>
    


----

.. py:class:: slangpy.builtin.RangeMarshall
    :canonical: slangpy.builtin.range.RangeMarshall
    
    Alias class: :py:class:`slangpy.builtin.range.RangeMarshall`
    


----

.. py:class:: slangpy.builtin.tensorcommon.Protocol
    :canonical: typing.Protocol
    
    Alias class: :py:class:`typing.Protocol`
    


----

.. py:class:: slangpy.builtin.tensorcommon.DeviceType
    :canonical: slangpy.DeviceType
    
    Alias class: :py:class:`slangpy.DeviceType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.tensorcommon.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.tensorcommon.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.tensorcommon.CallMode
    :canonical: slangpy.slangpy.CallMode
    
    Alias class: :py:class:`slangpy.slangpy.CallMode`
    


----

.. py:class:: slangpy.builtin.tensorcommon.AccessType
    :canonical: slangpy.slangpy.AccessType
    
    Alias class: :py:class:`slangpy.slangpy.AccessType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.tensorcommon.ScalarType
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.ITensorType
    :canonical: slangpy.native_refl.TensorType
    
    Alias class: :py:class:`slangpy.native_refl.TensorType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.TensorType
    :canonical: slangpy.native_refl.TensorType.Kind
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`
    


----

.. py:class:: slangpy.builtin.tensorcommon.TensorViewType
    :canonical: slangpy.native_refl.TensorViewType
    
    Alias class: :py:class:`slangpy.native_refl.TensorViewType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.DiffTensorViewType
    :canonical: slangpy.native_refl.DiffTensorViewType
    
    Alias class: :py:class:`slangpy.native_refl.DiffTensorViewType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.ArrayType
    :canonical: slangpy.native_refl.ArrayType
    
    Alias class: :py:class:`slangpy.native_refl.ArrayType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.InterfaceType
    :canonical: slangpy.native_refl.InterfaceType
    
    Alias class: :py:class:`slangpy.native_refl.InterfaceType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.UnknownType
    :canonical: slangpy.native_refl.UnknownType
    
    Alias class: :py:class:`slangpy.native_refl.UnknownType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.tensorcommon.ResourceType
    :canonical: slangpy.native_refl.ResourceType
    
    Alias class: :py:class:`slangpy.native_refl.ResourceType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.TensorAccess
    :canonical: slangpy.native_refl.TensorType.Access
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`
    


----

.. py:class:: slangpy.builtin.tensorcommon.VectorType
    :canonical: slangpy.native_refl.VectorType
    
    Alias class: :py:class:`slangpy.native_refl.VectorType`
    


----

.. py:class:: slangpy.builtin.tensorcommon.ITensorMarshall

    Base class: :py:class:`typing.Protocol`
    
    
    Protocol for type marshalling of any container that behaves as a tensor.
    
    


----

.. py:class:: slangpy.builtin.tensor.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.tensor.ShaderObject
    :canonical: slangpy.ShaderObject
    
    Alias class: :py:class:`slangpy.ShaderObject`
    


----

.. py:class:: slangpy.builtin.tensor.ShaderCursor
    :canonical: slangpy.ShaderCursor
    
    Alias class: :py:class:`slangpy.ShaderCursor`
    


----

.. py:class:: slangpy.builtin.tensor.BufferUsage
    :canonical: slangpy.BufferUsage
    
    Alias class: :py:class:`slangpy.BufferUsage`
    


----

.. py:class:: slangpy.builtin.tensor.TensorMarshallBase
    :canonical: slangpy.slangpy.TensorMarshall
    
    Alias class: :py:class:`slangpy.slangpy.TensorMarshall`
    


----

.. py:class:: slangpy.builtin.tensor.Tensor
    :canonical: slangpy.native_func.Tensor
    
    Alias class: :py:class:`slangpy.native_func.Tensor`
    


----

.. py:class:: slangpy.builtin.tensor.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.tensor.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.tensor.ArrayType
    :canonical: slangpy.native_refl.ArrayType
    
    Alias class: :py:class:`slangpy.native_refl.ArrayType`
    


----

.. py:class:: slangpy.builtin.tensor.ScalarType
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.builtin.tensor.VectorType
    :canonical: slangpy.native_refl.VectorType
    
    Alias class: :py:class:`slangpy.native_refl.VectorType`
    


----

.. py:class:: slangpy.builtin.tensor.MatrixType
    :canonical: slangpy.native_refl.MatrixType
    
    Alias class: :py:class:`slangpy.native_refl.MatrixType`
    


----

.. py:class:: slangpy.builtin.tensor.TensorType
    :canonical: slangpy.native_refl.TensorType.Kind
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Kind`
    


----

.. py:class:: slangpy.builtin.tensor.TensorAccess
    :canonical: slangpy.native_refl.TensorType.Access
    
    Alias class: :py:class:`slangpy.native_refl.TensorType.Access`
    


----

.. py:class:: slangpy.builtin.tensor.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.tensor.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.tensor.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.tensor.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.builtin.tensor.TensorMarshall

    Base class: :py:class:`slangpy.slangpy.TensorMarshall`
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.resolve_types
        :type: function
        :value: <function TensorMarshall.resolve_types at 0x0000021D5F669A80>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.reduce_type
        :type: function
        :value: <function TensorMarshall.reduce_type at 0x0000021D5F669B20>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.resolve_dimensionality
        :type: function
        :value: <function TensorMarshall.resolve_dimensionality at 0x0000021D5F669BC0>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.can_direct_bind
        :type: function
        :value: <function TensorMarshall.can_direct_bind at 0x0000021D5F669C60>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.gen_calldata
        :type: function
        :value: <function TensorMarshall.gen_calldata at 0x0000021D5F669D00>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.gen_trampoline_load
        :type: function
        :value: <function TensorMarshall.gen_trampoline_load at 0x0000021D5F669DA0>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.gen_trampoline_store
        :type: function
        :value: <function TensorMarshall.gen_trampoline_store at 0x0000021D5F669E40>
    
    .. py:attribute:: slangpy.builtin.tensor.TensorMarshall.build_shader_object
        :type: function
        :value: <function TensorMarshall.build_shader_object at 0x0000021D5F669EE0>
    


----

.. py:class:: slangpy.builtin.numpy.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.builtin.numpy.BoundVariable
    :canonical: slangpy.bindings.boundvariable.BoundVariable
    
    Alias class: :py:class:`slangpy.bindings.boundvariable.BoundVariable`
    


----

.. py:class:: slangpy.builtin.numpy.CodeGenBlock
    :canonical: slangpy.bindings.codegen.CodeGenBlock
    
    Alias class: :py:class:`slangpy.bindings.codegen.CodeGenBlock`
    


----

.. py:class:: slangpy.builtin.numpy.BindContext
    :canonical: slangpy.bindings.marshall.BindContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.BindContext`
    


----

.. py:class:: slangpy.builtin.numpy.ReturnContext
    :canonical: slangpy.bindings.marshall.ReturnContext
    
    Alias class: :py:class:`slangpy.bindings.marshall.ReturnContext`
    


----

.. py:class:: slangpy.builtin.numpy.NativeNumpyMarshall
    :canonical: slangpy.slangpy.NativeNumpyMarshall
    
    Alias class: :py:class:`slangpy.slangpy.NativeNumpyMarshall`
    


----

.. py:class:: slangpy.builtin.numpy.SlangProgramLayout
    :canonical: slangpy.native_refl.Layout
    
    Alias class: :py:class:`slangpy.native_refl.Layout`
    


----

.. py:class:: slangpy.builtin.numpy.ScalarType
    :canonical: slangpy.native_refl.ScalarType
    
    Alias class: :py:class:`slangpy.native_refl.ScalarType`
    


----

.. py:class:: slangpy.builtin.numpy.SlangType
    :canonical: slangpy.native_refl.Type
    
    Alias class: :py:class:`slangpy.native_refl.Type`
    


----

.. py:class:: slangpy.builtin.numpy.VectorType
    :canonical: slangpy.native_refl.VectorType
    
    Alias class: :py:class:`slangpy.native_refl.VectorType`
    


----

.. py:class:: slangpy.builtin.numpy.MatrixType
    :canonical: slangpy.native_refl.MatrixType
    
    Alias class: :py:class:`slangpy.native_refl.MatrixType`
    


----

.. py:class:: slangpy.builtin.numpy.NumpyMarshall

    Base class: :py:class:`slangpy.slangpy.NativeNumpyMarshall`
    
    .. py:attribute:: slangpy.builtin.numpy.NumpyMarshall.resolve_types
        :type: function
        :value: <function NumpyMarshall.resolve_types at 0x0000021D5F66A200>
    
    .. py:attribute:: slangpy.builtin.numpy.NumpyMarshall.reduce_type
        :type: function
        :value: <function NumpyMarshall.reduce_type at 0x0000021D5F66A2A0>
    
    .. py:attribute:: slangpy.builtin.numpy.NumpyMarshall.resolve_dimensionality
        :type: function
        :value: <function NumpyMarshall.resolve_dimensionality at 0x0000021D5F66A340>
    
    .. py:attribute:: slangpy.builtin.numpy.NumpyMarshall.gen_calldata
        :type: function
        :value: <function NumpyMarshall.gen_calldata at 0x0000021D5F66A3E0>
    


----

.. py:class:: slangpy.builtin.NumpyMarshall
    :canonical: slangpy.builtin.numpy.NumpyMarshall
    
    Alias class: :py:class:`slangpy.builtin.numpy.NumpyMarshall`
    


----

.. py:class:: slangpy.builtin.TensorMarshall
    :canonical: slangpy.builtin.tensor.TensorMarshall
    
    Alias class: :py:class:`slangpy.builtin.tensor.TensorMarshall`
    


----

.. py:class:: slangpy.Function
    :canonical: slangpy.core.function.Function
    
    Alias class: :py:class:`slangpy.core.function.Function`
    


----

.. py:class:: slangpy.Struct
    :canonical: slangpy.core.struct.Struct
    
    Alias class: :py:class:`slangpy.core.struct.Struct`
    


----

.. py:class:: slangpy.Module
    :canonical: slangpy.core.module.Module
    
    Alias class: :py:class:`slangpy.core.module.Module`
    


----

.. py:class:: slangpy.InstanceList
    :canonical: slangpy.core.instance.InstanceList
    
    Alias class: :py:class:`slangpy.core.instance.InstanceList`
    


----

.. py:class:: slangpy.InstanceTensor
    :canonical: slangpy.core.instance.InstanceTensor
    
    Alias class: :py:class:`slangpy.core.instance.InstanceTensor`
    


----

.. py:class:: slangpy.torchintegration.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.torchintegration.NativeTorchTensorDiffPair
    :canonical: slangpy.slangpy.NativeTorchTensorDiffPair
    
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorDiffPair`
    


----

.. py:class:: slangpy.Any
    :canonical: typing.Any
    
    Alias class: :py:class:`typing.Any`
    


----

.. py:class:: slangpy.NativeTorchTensorDiffPair
    :canonical: slangpy.slangpy.NativeTorchTensorDiffPair
    
    Alias class: :py:class:`slangpy.slangpy.NativeTorchTensorDiffPair`
    


----

.. py:data:: slangpy.SHADER_PATH
    :type: str
    :value: "C:\src\slangpy\slangpy\slang"



----

