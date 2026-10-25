from pathlib import Path
models=['Logistic regression','Linear SVM','DistilBERT']; dev=[.98633,.98432,.97992]; cal=[.99754,.99889,.99583]
W,H=800,500; x=[140,360,580]; svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="400" y="40" text-anchor="middle" font-size="24" font-weight="bold">Multiclass one-vs-rest ROC-AUC</text><text x="400" y="65" text-anchor="middle" font-size="12">Only the 30 classes present in each split are included</text>']
for i,n in enumerate(models):
  for j,v in enumerate((dev[i],cal[i])):
    xx=x[i]-28+j*32; base=410; h=(v-.95)*300; yy=base-h; color=['#5b8db8','#d97745'][j]; svg += [f'<rect x="{xx}" y="{yy:.1f}" width="24" height="{h:.1f}" fill="{color}"/>',f'<text x="{xx+12}" y="{yy-7:.1f}" text-anchor="middle" font-size="11">{v:.3f}</text>']
  svg.append(f'<text x="{x[i]}" y="445" text-anchor="middle" font-size="12">{n}</text>')
svg += ['<line x1="80" y1="410" x2="700" y2="410" stroke="#555"/>','<rect x="260" y="470" width="14" height="14" fill="#5b8db8"/><text x="280" y="482" font-size="12">Development</text><rect x="390" y="470" width="14" height="14" fill="#d97745"/><text x="410" y="482" font-size="12">Calibration</text>','</svg>']; Path('reports/model-auc-v1.svg').write_text('\n'.join(svg)); print('reports/model-auc-v1.svg')
