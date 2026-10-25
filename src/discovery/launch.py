"""Run discovery under the Step 3 data restrictions with pinned extra inputs."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
from ..restricted_job import profile

if __name__=='__main__':
    project=Path.cwd().resolve();root=project/'data/splits/bitext-v1'
    work=Path(tempfile.mkdtemp(prefix='discovery-',dir=project/'reports/runs')).resolve()
    em=json.loads((project/'config/encoder-manifest.json').read_text())
    model=(project/em['path']).resolve()
    if not model.is_relative_to(project/'artifacts/encoders'):raise ValueError('encoder outside approved root')
    sb=profile(project,root/'blind/train/tickets.jsonl',root/'manifest.json',work)
    # Additional exact configuration files and pinned model directory only.
    extra=' '.join(['(literal '+json.dumps(str(project/p))+')' for p in ['config/encoder-manifest.json','config/experiments/discovery-v1.yaml']])
    extra+=' (subpath '+json.dumps(str(model))+')'
    sb=sb.replace('(require-any ', '(require-any '+extra+' ',1)
    (work/'profile.sb').write_text(sb)
    for p in ['config/encoder-manifest.json','config/experiments/discovery-v1.yaml']:
        (work/Path(p).name).write_bytes((project/p).read_bytes())
    env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(project),'HOME':str(work),'TMPDIR':str(work),
         'HF_HOME':str(work/'hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
         'NUMBA_CACHE_DIR':str(work/'numba'),'MPLCONFIGDIR':str(work/'mpl'),'PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','VECLIB_MAXIMUM_THREADS':'4','PYTHONNOUSERSITE':'1'}
    base=['/usr/bin/sandbox-exec','-f',str(work/'profile.sb'),sys.executable,'-B','-m']
    subprocess.run(base+['src.split_probe','--role','discovery','--partition','train','--output',str(work/'access-check.json'),'--check-denials'],env=env,check=True)
    print('Restricted discovery output:',work,flush=True)
    subprocess.run(base+['src.discovery.pipeline','--work',str(work)],env=env,check=True)
