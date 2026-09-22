from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if len(sys.argv) == 1:
    subprocess.run(['nvidia-smi'], check=False)
    for case in ('90_standard', 'auto_standard'):
        result = subprocess.run([sys.executable, __file__, case], check=False)
        print('DIAGNOSTIC RESULT', case, result.returncode, flush=True)
    raise SystemExit(0)

import pytest
import slangpy as spy
from slangpy.testing import helpers

case = sys.argv[1]
target, debug = case.split('_')
dump_dir = ROOT / '.tmp' / 'cuda-diagnostics' / case
dump_dir.mkdir(parents=True, exist_ok=True)

class DiagnosticDevice(spy.Device):
    def __init__(self, *args, **kwargs):
        if kwargs.get('type') == spy.DeviceType.cuda:
            options = kwargs.get('compiler_options', spy.SlangCompilerOptions())
            options.cuda_architecture = None if target == 'auto' else int(target)
            options.debug_info = getattr(spy.SlangDebugInfoLevel, debug)
            options.dump_intermediates = True
            options.dump_intermediates_prefix = str(dump_dir / 'shader-')
            kwargs['compiler_options'] = options
        super().__init__(*args, **kwargs)

helpers.Device = DiagnosticDevice
device = spy.Device(type=spy.DeviceType.cuda)
compiler = device.cuda_compiler_info
print('CASE', case, 'NVRTC', compiler.path, compiler.version_major, compiler.version_minor,
      compiler.supported_architectures, 'OPTIX', device.info.optix_version, flush=True)
device.close()
result = pytest.main([
    'slangpy/tests/device/test_cluster_acceleration_structure.py::test_cluster_acceleration_structure_trace[DeviceType.cuda]',
    'slangpy/tests/device/test_opacity_micromap.py::test_opacity_micromap_trace[DeviceType.cuda]',
    'slangpy/tests/device/test_pipeline.py::test_raytrace_simple[ray-DeviceType.cuda]',
    'slangpy/tests/device/test_pipeline.py::test_raytrace_two_instance[ray-DeviceType.cuda]',
    'slangpy/tests/device/test_pipeline.py::test_raytrace_closest_instance[ray-DeviceType.cuda]',
    'slangpy/tests/slangpy_tests/test_raytracing.py::test_raytracing[DeviceType.cuda]',
    '-q', '-s',
])
for path in sorted(dump_dir.glob('*.ptx')):
    lines = path.read_text(errors='replace').splitlines()
    for index, line in enumerate(lines):
        if '_optix_trace_typed_32' in line:
            print('PTX', path.name, '\n' + '\n'.join(lines[:12]), flush=True)
            print('\n'.join(lines[max(0, index - 30):index + 3]), flush=True)
raise SystemExit(result)
