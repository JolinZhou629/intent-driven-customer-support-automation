"""Inspect local compute without downloading model weights or reading tickets."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch


def sysctl(key):
    p = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def inspect():
    available = torch.backends.mps.is_available()
    check = {'available': available, 'built': torch.backends.mps.is_built()}
    if available:
        try:
            x = torch.tensor([1., 2.], device='mps')
            check['tensor_smoke_passed'] = (x * 2).cpu().tolist() == [2., 4.]
        except Exception as exc:
            check['tensor_smoke_passed'] = False
            check['error_type'] = type(exc).__name__
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'architecture': platform.machine(),
            'cpu': sysctl('machdep.cpu.brand_string'),
            'logical_cpu_count': sysctl('hw.ncpu'), 'memory_bytes': sysctl('hw.memsize'),
            'mps': check, 'cuda_available': torch.cuda.is_available(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ['pydantic', 'PyYAML', 'numpy', 'scipy', 'scikit-learn', 'pytest', 'torch']}}


if __name__ == '__main__':
    report = inspect()
    Path('reports/environment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
