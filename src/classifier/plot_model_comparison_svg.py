"""Render the student comparison as a dependency-free SVG."""
from pathlib import Path
W,H=1200,520; colors=['#5b8db8','#d97745']; names=['Logistic regression','Linear SVM','DistilBERT']
dev=[.9076,.9348,.9457]; cal=[.9333,.9556,.9667]; f1d=[.8847,.8930,.9263]; f1c=[.8975,.9187,.9334]
def esc(s): return s.replace('&','&amp;')
def panel(x,title,a,b,lo=.8):
 out=[f'<text x="{x+180}" y="42" text-anchor="middle" font-size="20" font-weight="bold">{title}</text>']
 base=410; height=300; top=85; scale=height/(1-lo)
 for j,vals in enumerate((a,b)):
  for i,v in enumerate(vals):
   xx=x+35+i*105+j*28; hh=(v-lo)*scale; yy=base-hh
   out.append(f'<rect x="{xx}" y="{yy:.1f}" width="24" height="{hh:.1f}" fill="{colors[j]}"/>')
   out.append(f'<text x="{xx+12}" y="{yy-6:.1f}" text-anchor="middle" font-size="11">{v:.3f}</text>')
 for i,n in enumerate(names): out.append(f'<text x="{x+58+i*105}" y="440" text-anchor="middle" font-size="11">{esc(n)}</text>')
 out.append(f'<line x1="{x+20}" y1="{base}" x2="{x+365}" y2="{base}" stroke="#555"/>'); return '\n'.join(out)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="600" y="25" text-anchor="middle" font-size="22" font-weight="bold">bitext-v2 student model comparison</text>',panel(25,'Accuracy',dev,cal,.85),panel(425,'Macro-F1',f1d,f1c,.8)]
svg.append('<text x="820" y="42" text-anchor="middle" font-size="20" font-weight="bold">Mean latency (log scale)</text>')
lat=[.000020,.000023,.01706]; base=410
for i,(n,v) in enumerate(zip(names,lat)):
 x=780+i*105; hh=(__import__('math').log10(v)-(-5))/3*300; y=base-hh; svg += [f'<rect x="{x}" y="{y:.1f}" width="50" height="{hh:.1f}" fill="#6a9f58"/>',f'<text x="{x+25}" y="{y-6:.1f}" text-anchor="middle" font-size="11">{v:.2g}s</text>',f'<text x="{x+25}" y="440" text-anchor="middle" font-size="11">{esc(n)}</text>']
svg += ['<text x="600" y="490" text-anchor="middle" font-size="12">Blue = development   Orange = calibration   DistilBERT ≈256 MB; TF-IDF artifacts are much smaller</text>','</svg>']
Path('reports/model-comparison-v1.svg').write_text('\n'.join(svg)); print('reports/model-comparison-v1.svg')
