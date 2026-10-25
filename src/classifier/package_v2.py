"""Build a reloadable v2 classifier handoff bundle with integrity metadata."""
import hashlib, json, shutil
from pathlib import Path

def sha256(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    root=Path('.'); out=root/'artifacts/classifier-v2'; out.mkdir(parents=True,exist_ok=True)
    files={
        'model.joblib': root/'artifacts/students-v2-catalog/tfidf_linear_svm.joblib',
        'catalog.json': root/'config/catalogs/bitext-v2.json',
        'catalog-provenance.json': root/'config/catalogs/bitext-v2-provenance.json',
        'routing.yaml': root/'config/routing/routing-v1.yaml',
        'calibration.json': root/'reports/svm-calibration-v1.json',
        'merge-map.yaml': root/'config/catalogs/intent-merge-v1-to-v2.yaml',
    }
    copied=[]
    for name,source in files.items():
        if not source.exists(): raise FileNotFoundError(source)
        target=out/name; shutil.copy2(source,target); copied.append({'path':name,'sha256':sha256(target),'bytes':target.stat().st_size})
    manifest={'bundle_version':'classifier-v2','catalog_version':'bitext-v2','model_type':'tfidf_linear_svm','routing_policy_version':'routing-v1','files':copied}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
