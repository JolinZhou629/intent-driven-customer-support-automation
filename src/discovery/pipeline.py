"""Offline train-only discovery, incrementally saved for review."""
import argparse
from itertools import combinations
from pathlib import Path
import json,time
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from ..data_access import load_tickets
from ..data_prep import digest


def main(work):
    start=time.time();cfg=yaml.safe_load(Path('config/experiments/discovery-v1.yaml').read_text())
    em=json.loads(Path(cfg['encoder_manifest']).read_text());modelpath=Path(em['path'])
    for name,h in em['files'].items():
        if digest((modelpath/name).read_bytes())!=h:raise ValueError('encoder checksum mismatch')
    rows=load_tickets(Path(cfg['split_root']),'discovery',cfg['partition'])
    texts=[r.instruction for r in rows]
    model=SentenceTransformer(str(modelpath),local_files_only=True,trust_remote_code=False,device=cfg['device'])
    embeddings=model.encode(texts,batch_size=cfg['batch_size'],normalize_embeddings=True,show_progress_bar=False)
    np.save(work/'embeddings.npy',embeddings)
    (work/'ticket-ids.json').write_text(json.dumps([r.ticket_id for r in rows]))
    (work/'embedding-manifest.json').write_text(json.dumps({'encoder':em['revision'],'shape':list(embeddings.shape),
        'sha256':digest((work/'embeddings.npy').read_bytes()),'device':cfg['device'],
        'training_sha256':digest((Path(cfg['split_root'])/'blind/train/tickets.jsonl').read_bytes()),
        'config_sha256':digest(Path('config/experiments/discovery-v1.yaml').read_bytes()),
        'code_sha256':digest(Path(__file__).read_bytes())},indent=2))
    print('Embedded',len(rows),'training tickets',flush=True)
    results=[];labels_by={};probs_by={}
    variants=[(None,42)]+[(d,s) for d in cfg['umap_dimensions'] for s in cfg['seeds']]
    for dim,seed in variants:
        if dim is None:z=embeddings
        else:
            reducer=umap.UMAP(n_components=dim,n_neighbors=cfg['umap_neighbors'],min_dist=cfg['umap_min_dist'],
                metric=cfg['umap_metric'],random_state=seed,n_jobs=1)
            z=reducer.fit_transform(embeddings)
            np.save(work/f'projection-{dim}-{seed}.npy',z)
        for size in cfg['min_cluster_sizes']:
            name=f'd{dim}-s{seed}-m{size}'
            cluster=hdbscan.HDBSCAN(min_cluster_size=size,min_samples=cfg['min_samples'],metric='euclidean',core_dist_n_jobs=1)
            labels=cluster.fit_predict(z);mask=labels>=0
            n=len(set(labels[mask]));score=None
            if n>1 and mask.sum()>n:
                score=float(silhouette_score(embeddings[mask],labels[mask],metric='cosine',sample_size=min(cfg['silhouette_sample_size'],int(mask.sum())),random_state=42))
            info={'run':name,'dimensions':dim,'seed':seed,'min_cluster_size':size,'clusters':n,
                  'noise_fraction':float((~mask).mean()),'original_cosine_silhouette':score,
                  'selection_score':float(mask.mean())*max(0,score or 0),
                  'mean_cluster_persistence':float(np.mean(cluster.cluster_persistence_)) if n else 0}
            labels_by[name]=labels;probs_by[name]=cluster.probabilities_
            np.savez(work/f'{name}.npz',labels=labels,probabilities=cluster.probabilities_)
            results.append(info);(work/'comparison.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(info),flush=True)
    stability=[]
    for dim in cfg['umap_dimensions']:
        for size in cfg['min_cluster_sizes']:
            scores=[]
            for a,b in combinations(cfg['seeds'],2):
                scores.append(float(adjusted_mutual_info_score(labels_by[f'd{dim}-s{a}-m{size}'],labels_by[f'd{dim}-s{b}-m{size}'])))
            stability.append({'dimensions':dim,'min_cluster_size':size,'mean_pairwise_ami_including_noise':float(np.mean(scores))})
    (work/'stability.json').write_text(json.dumps(stability,indent=2))
    chosen=max(results,key=lambda r:r['selection_score']);name=chosen['run'];labels=labels_by[name];probs=probs_by[name]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english',min_df=2)
    tfidf=vectorizer.fit_transform(texts);terms=vectorizer.get_feature_names_out()
    clusters=[]
    for label in sorted(set(labels)):
        idx=np.flatnonzero(labels==label)
        if label>=0:
            # For unit vectors, maximizing dot product with the mean minimizes
            # summed cosine distance: an exact cosine medoid among members.
            center=embeddings[idx].mean(axis=0)
            ranked=idx[np.argsort(-(embeddings[idx]@center))]
            top=terms[np.argsort(-np.asarray(tfidf[idx].mean(axis=0)).ravel())[:5]].tolist()
        else:ranked=idx;top=[]
        unique=[];seen=set()
        for i in ranked:
            if texts[i] not in seen:unique.append({'ticket_id':rows[i].ticket_id,'instruction':texts[i]});seen.add(texts[i])
            if len(unique)==5:break
        boundary=idx[np.argsort(probs[idx])[:3]]
        clusters.append({'cluster_id':int(label),'count':len(idx),'candidate_name':' / '.join(top[:3]) if label>=0 else 'noise review pool',
            'keywords':top,'representatives':unique,
            'low_membership_examples':[{'ticket_id':rows[i].ticket_id,'instruction':texts[i]} for i in boundary]})
    (work/'cluster-review.json').write_text(json.dumps(clusters,indent=2))
    with (work/'assignments.jsonl').open('w') as f:
        for row,label,p in zip(rows,labels,probs):f.write(json.dumps({'ticket_id':row.ticket_id,'cluster_id':int(label),'membership_strength':float(p)})+'\n')
    result={'status':'discovery_complete_catalog_review_pending','selected':chosen,'rows':len(rows),'elapsed_seconds':round(time.time()-start,1),
            'note':'Keyword names are proposals, not human-adjudicated intents. Noise is not a trained unknown class.'}
    (work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();main(a.work)
