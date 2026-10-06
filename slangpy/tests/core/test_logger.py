# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Any
import pytest
import os
import slangpy as spy
from slangpy import (
    Logger,
    LoggerOutput,
    ConsoleLoggerOutput,
    FileLoggerOutput,
    LogLevel,
    LogFrequency,
)


class CustomLoggerOutput(LoggerOutput):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.messages = []

    def write(self, level: LogLevel, name: str, msg: str):
        self.messages.append((level, name, msg))


def test_logger():
    logger = Logger(level=LogLevel.debug, name="test", use_default_outputs=False)
    output = CustomLoggerOutput()
    logger.add_output(output)

    messages = [
        (LogLevel.none, "plain text"),
        (LogLevel.debug, "debug message"),
        (LogLevel.info, "info message"),
        (LogLevel.warn, "warn message"),
        (LogLevel.error, "error message"),
        (LogLevel.fatal, "fatal message"),
    ]

    for level in [
        LogLevel.debug,
        LogLevel.info,
        LogLevel.warn,
        LogLevel.error,
        LogLevel.fatal,
    ]:
        logger.level = level
        logger.name = f"test_{level}"

        for msg in messages:
            output.clear()
            logger.log(msg[0], msg[1])
            if msg[0] == LogLevel.none or msg[0] >= level:
                assert len(output.messages) == 1
                assert output.messages[0][0] == msg[0]
                assert output.messages[0][1] == f"test_{level}"
                assert output.messages[0][2] == msg[1]
            else:
                assert len(output.messages) == 0


def test_logger_frequency():
    logger = Logger(level=LogLevel.info, name="test", use_default_outputs=False)
    output = CustomLoggerOutput()
    logger.add_output(output)

    logger.log(LogLevel.info, "repeated", LogFrequency.once)
    assert len(output.messages) == 1

    logger.log(LogLevel.info, "repeated", LogFrequency.once)
    assert len(output.messages) == 1

    logger.log(LogLevel.info, "repeated 2", LogFrequency.once)
    assert len(output.messages) == 2

    logger.log(LogLevel.info, "repeated 2", LogFrequency.once)
    assert len(output.messages) == 2


def _test_console_output():
    output = ConsoleLoggerOutput(colored=False)
    logger = Logger(level=LogLevel.debug, name="test", use_default_outputs=False)
    logger.add_output(output)
    logger.log(LogLevel.none, "plain message")
    logger.debug("debug message")
    logger.info("info message")
    logger.warn("warn message")
    logger.error("error message")
    logger.fatal("fatal message")


@pytest.mark.skip("Test not working reliably")
def test_console_output(capfd: Any):
    _test_console_output()
    out, err = capfd.readouterr()
    out_lines = out.splitlines()
    err_lines = err.splitlines()
    assert len(out_lines) == 4
    assert out_lines[0].startswith("plain message")
    assert out_lines[1].startswith("[DEBUG] (test) debug message")
    assert out_lines[2].startswith("[INFO] (test) info message")
    assert out_lines[3].startswith("[WARN] (test) warn message")
    assert len(err_lines) == 2
    assert err_lines[0].startswith("[ERROR] (test) error message")
    assert err_lines[1].startswith("[FATAL] (test) fatal message")


def test_file_output(tmpdir: Path):
    path = os.path.join(tmpdir, "test.log")
    output = FileLoggerOutput(path)
    logger = Logger(level=LogLevel.debug, name="test", use_default_outputs=False)
    logger.add_output(output)
    logger.log(LogLevel.none, "plain message")
    logger.debug("debug message")
    logger.info("info message")
    logger.warn("warn message")
    logger.error("error message")
    logger.fatal("fatal message")
    del output
    del logger
    lines = open(path, "r").readlines()
    assert len(lines) == 6
    assert lines[0].startswith("plain message")
    assert lines[1].startswith("[DEBUG] (test) debug message")
    assert lines[2].startswith("[INFO] (test) info message")
    assert lines[3].startswith("[WARN] (test) warn message")
    assert lines[4].startswith("[ERROR] (test) error message")
    assert lines[5].startswith("[FATAL] (test) fatal message")


class RaisingLoggerOutput(LoggerOutput):
    def __init__(self, exception_type: type[Exception]) -> None:
        super().__init__()
        self.exception_type = exception_type
        self.calls = 0

    def write(self, level: LogLevel, name: str, msg: str) -> None:
        self.calls += 1
        raise self.exception_type("output failed")


@pytest.mark.parametrize("use_global", [False, True])
@pytest.mark.parametrize(
    "method",
    ["log", "debug", "info", "warn", "error", "fatal"]
    + [f"{level}_once" for level in ["debug", "info", "warn", "error", "fatal"]],
)
def test_output_failure_isolation(use_global: bool, method: str) -> None:
    logger = Logger.get() if use_global else Logger(use_default_outputs=False)
    saved = Logger(level=logger.level, use_default_outputs=False)
    saved.use_same_outputs(logger)
    logger.remove_all_outputs()
    logger.level = LogLevel.debug
    first = RaisingLoggerOutput(ValueError)
    second = RaisingLoggerOutput(RuntimeError)
    healthy = CustomLoggerOutput()
    try:
        logger.add_output(first)
        logger.add_output(second)
        logger.add_output(healthy)
        emit = (
            getattr(spy, "log" if method == "log" else f"log_{method}")
            if use_global
            else getattr(logger, method)
        )
        message = f"output failure isolation {use_global} {method}"
        for _ in range(2):
            if method == "log":
                emit(LogLevel.info, message)
            else:
                emit(message)
        expected_calls = 1 if method.endswith("_once") else 2
        # Two failing outputs establish that dispatch continues regardless of output ordering.
        assert first.calls == expected_calls
        assert second.calls == expected_calls
        assert len(healthy.messages) == expected_calls
        assert all(entry[2] == message for entry in healthy.messages)

        logger.remove_output(first)
        logger.remove_output(second)
        logger.info("still usable")
        assert healthy.messages[-1][2] == "still usable"

        # Direct output calls retain their throwing contract.
        with pytest.raises(ValueError, match="output failed"):
            first.write(LogLevel.info, "", "direct")
    finally:
        logger.use_same_outputs(saved)
        logger.level = saved.level


def test_calldata_logging_isolates_output_failures() -> None:
    from slangpy.slangpy import NativeCallData

    logger = Logger(level=LogLevel.debug, use_default_outputs=False)
    first = RaisingLoggerOutput(ValueError)
    second = RaisingLoggerOutput(RuntimeError)
    healthy = CustomLoggerOutput()
    logger.add_output(first)
    logger.add_output(second)
    logger.add_output(healthy)
    call = NativeCallData()
    call.logger = logger
    call.log(LogLevel.info, "plain")
    for level in ["debug", "info", "warn", "error", "fatal"]:
        getattr(call, f"log_{level}")(level)
    assert first.calls == 6
    assert second.calls == 6
    assert len(healthy.messages) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
