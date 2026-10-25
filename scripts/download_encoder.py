"""Download pinned public model files, never dataset data."""
from pathlib import Path
import hashlib,json
from huggingface_hub import snapshot_download
REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
root=Path('artifacts/encoders/all-MiniLM-L6-v2')/REV
snapshot_download('sentence-transformers/all-MiniLM-L6-v2',revision=REV,local_dir=root,
    allow_patterns=['*.json','*.txt','model.safetensors','1_Pooling/config.json','README.md'],
    ignore_patterns=['onnx/*','openvino/*'],token=False)
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts}
Path('config/encoder-manifest.json').write_text(json.dumps({'model':'sentence-transformers/all-MiniLM-L6-v2','revision':REV,'path':str(root),'files':files},indent=2)+'\n')
print('Pinned encoder downloaded and checksummed:',len(files),'files')
