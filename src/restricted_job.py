"""macOS runner: restrict user-directory reads to source/runtime and one ticket partition.

Runs offline with a cleared environment. Extending inputs for later stages
requires an explicit reviewed allowlist change, not an arbitrary path option.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .data_access import ALLOWED,load_tickets


def profile(project, ticket, manifest, work):
    roots=['/dev','/System','/usr','/bin','/sbin','/opt/homebrew','/Library','/private/preboot','/private/var/db/dyld',
           str(project/'.venv'),str(project/'src')]
    read=' '.join('(subpath '+json.dumps(p)+')' for p in roots)
    read+=' '+ ' '.join('(literal '+json.dumps(str(p))+')' for p in [project,ticket,manifest,Path('/dev/null'),Path('/dev/urandom'),Path('/dev/random')])
    return f'''(version 1)
(allow default)
(deny network*)
(deny file-read-data (require-all (subpath "/Users") (require-not (require-any {read} (subpath {json.dumps(str(work))})))))
(deny file-write* (require-not (require-any (subpath {json.dumps(str(work))}) (literal "/dev/null"))))
'''



def launch(role,partition,probe=False):
    project=Path.cwd().resolve();root=project/'data/splits/bitext-v1'
    load_tickets(root,role,partition) # Validate before launch and again in child.
    work=Path(tempfile.mkdtemp(prefix='restricted-',dir=project/'reports/runs')).resolve()
    ticket=(root/f'blind/{partition}/tickets.jsonl').resolve()
    sandbox=work/'profile.sb';sandbox.write_text(profile(project,ticket,root/'manifest.json',work))
    command=['/usr/bin/sandbox-exec','-f',str(sandbox),sys.executable,'-B','-m','src.split_probe',
             '--role',role,'--partition',partition,'--output',str(work/'result.json')]
    if probe:command.append('--check-denials')
    env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(project),'PYTHONDONTWRITEBYTECODE':'1',
         'HOME':str(work),'TMPDIR':str(work),'PYTHONNOUSERSITE':'1'}
    subprocess.run(command,env=env,check=True)
    return json.loads((work/'result.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=ALLOWED,default='discovery')
    p.add_argument('--partition',default='train');p.add_argument('--check-denials',action='store_true')
    a=p.parse_args();print(json.dumps(launch(a.role,a.partition,a.check_denials),indent=2))
