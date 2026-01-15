# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Optional
import os
import platform
import shutil
import glob
import subprocess
import json
import slangpy as spy
from _pytest._io import TerminalWriter

PROJECT_DIR = Path(__file__).parent.parent.parent

CRASHPAD_DATABASE_DIR = PROJECT_DIR / ".crashpad"
TOOLS_DIR = PROJECT_DIR / ".tools"

# rust-minidump release version and download configuration
RUST_MINIDUMP_VERSION = "0.26.1"
RUST_MINIDUMP_BASE_URL = (
    f"https://github.com/rust-minidump/rust-minidump/releases/download/v{RUST_MINIDUMP_VERSION}"
)

# Map (system, machine) to release asset name
# system: Windows, Linux, Darwin
# machine: x86_64, AMD64, arm64, aarch64
RUST_MINIDUMP_ASSETS = {
    ("Windows", "AMD64"): "minidump-stackwalk-x86_64-pc-windows-msvc.zip",
    ("Windows", "x86_64"): "minidump-stackwalk-x86_64-pc-windows-msvc.zip",
    ("Linux", "x86_64"): "minidump-stackwalk-x86_64-unknown-linux-gnu.tar.xz",
    ("Darwin", "x86_64"): "minidump-stackwalk-x86_64-apple-darwin.tar.xz",
    ("Darwin", "arm64"): "minidump-stackwalk-aarch64-apple-darwin.tar.xz",
    ("Darwin", "aarch64"): "minidump-stackwalk-aarch64-apple-darwin.tar.xz",
}


def setup():
    # Remove database directory from previous runs.
    shutil.rmtree(CRASHPAD_DATABASE_DIR, ignore_errors=True)


def report(writer: TerminalWriter):
    reports = glob.glob(str(CRASHPAD_DATABASE_DIR / "reports/*.dmp"))
    if len(reports) == 0:
        return

    writer.sep("=", "CRASHPAD REPORTS")

    _postprocess_reports()

    for index, report in enumerate(reports):
        report_path = Path(report)
        writer.write(f"Crash report {index}: {report_path.name}\n")
        # Try to figure out what test triggered the crash.
        report_json_path = Path(report_path).with_suffix(".json")
        if report_json_path.exists():
            report_json = json.load(open(report_json_path))
            pid = report_json["pid"]
            test_name = _find_test_name(pid)
            if test_name:
                writer.write(f"Crash most probably originated in {test_name}\n")
        # Dump crash report.
        report_txt_path = Path(report_path).with_suffix(".txt")
        if report_txt_path.exists():
            writer.write(open(report_txt_path).read())
            writer.sep("-")


def start_handler():
    print("Starting Crashpad handler ...")
    try:
        spy.crashpad.start_handler(database=CRASHPAD_DATABASE_DIR)
    except Exception as e:
        print(f"Failed to start Crashpad handler ({e})")


def notify_current_test(name: str):
    pid = os.getpid()
    path = CRASHPAD_DATABASE_DIR / f"{pid}.txt"
    with open(path, "w") as f:
        f.write(f"{name}\n")


def _find_test_name(pid: int) -> Optional[str]:
    path = CRASHPAD_DATABASE_DIR / f"{pid}.txt"
    if path.exists():
        lines = open(path).readlines()
        if len(lines) > 0:
            return lines[-1].strip()

    return None


def _postprocess_reports():
    minidump_stackwalk = _get_minidump_stackwalk_path()
    if minidump_stackwalk is None:
        system = platform.system()
        machine = platform.machine()
        print(
            f"minidump-stackwalk is not available for {system}/{machine}. "
            "Skipping crash report postprocessing."
        )
        return

    reports = glob.glob(str(CRASHPAD_DATABASE_DIR / "reports/*.dmp"))
    for report in reports:
        report = Path(report)
        report_txt = report.with_suffix(".txt")
        report_json = report.with_suffix(".json")
        subprocess.run(
            [
                str(minidump_stackwalk),
                str(report),
                "--brief",
                "--use-local-debuginfo",
                "--output-file",
                str(report_txt),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            [
                str(minidump_stackwalk),
                str(report),
                "--json",
                "--pretty",
                "--use-local-debuginfo",
                "--output-file",
                str(report_json),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def _get_minidump_stackwalk_path() -> Optional[Path]:
    """Get the path to the minidump-stackwalk executable, downloading if necessary."""
    system = platform.system()
    machine = platform.machine()

    asset_name = RUST_MINIDUMP_ASSETS.get((system, machine))
    if asset_name is None:
        return None

    # Determine executable name based on platform
    exe_name = "minidump-stackwalk.exe" if system == "Windows" else "minidump-stackwalk"
    exe_path = TOOLS_DIR / "rust-minidump" / exe_name

    # Return path if already downloaded
    if exe_path.exists():
        return exe_path

    # Download and extract
    if not _download_minidump_stackwalk(asset_name, exe_path.parent):
        return None

    return exe_path if exe_path.exists() else None


def _download_minidump_stackwalk(asset_name: str, extract_dir: Path) -> bool:
    """Download and extract the minidump-stackwalk binary."""

    import tarfile
    import zipfile
    import urllib.request

    url = f"{RUST_MINIDUMP_BASE_URL}/{asset_name}"

    print(f"Downloading minidump-stackwalk from {url} ...")

    try:
        # Create extract directory
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Download to a temporary file
        archive_path = extract_dir / asset_name
        urllib.request.urlretrieve(url, archive_path)

        # Extract based on file extension
        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
        elif asset_name.endswith(".tar.xz"):
            with tarfile.open(archive_path, "r:xz") as tf:
                tf.extractall(extract_dir)
        else:
            print(f"Unknown archive format: {asset_name}")
            return False

        # Clean up archive
        archive_path.unlink()

        # Make executable on Unix systems
        if platform.system() != "Windows":
            exe_path = extract_dir / "minidump-stackwalk"
            if exe_path.exists():
                exe_path.chmod(exe_path.stat().st_mode | 0o755)

        print(f"Successfully installed minidump-stackwalk to {extract_dir}")
        return True

    except Exception as e:
        print(f"Failed to download minidump-stackwalk: {e}")
        return False
