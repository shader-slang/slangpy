// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdlib>

namespace sgl {

// bitmap.h

class Bitmap;

// blob.h

class Blob;

// file_stream.h

class EOFException;
class FileStream;

// file_system_watcher.h

struct FileSystemWatchDesc;
struct FileSystemWatchEvent;
class FileSystemWatcher;

// input.h

struct KeyboardEvent;
struct MouseEvent;
struct GamepadEvent;
struct GamepadState;

// logger.h

class LoggerOutput;
class ConsoleLoggerOutput;
class FileLoggerOutput;
class DebugConsoleLoggerOutput;
class Logger;

// memory_mapped_file_stream.h

class MemoryMappedFileStream;

// memory_mapped_file.h

class MemoryMappedFile;

// memory_stream.h

class MemoryStream;

// object.h

class Object;
template<typename>
class ref;
template<typename>
class breakable_ref;

// short_vector.h

template<typename T, size_t N>
class short_vector;

// static_vector.h

template<typename T, size_t N>
class static_vector;

// stream.h

class Stream;

// data_struct.h

class DataStruct;
class DataStructConverter;

// timer.h

class Timer;

// window.h

class Window;

} // namespace sgl
