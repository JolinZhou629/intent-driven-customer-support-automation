"""Create a reproducible comparison plot for the three student models."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

models=['Logistic regression','Linear SVM','DistilBERT']
dev_acc=[.9076,.9348,.9457]; cal_acc=[.9333,.9556,.9667]
dev_f1=[.8847,.8930,.9263]; cal_f1=[.8975,.9187,.9334]
latency=[0.000020,0.000023,0.01706]; size=[1.25,1.25,256]
colors=['#7aa6c2','#3f7f93','#d97745']; x=np.arange(3); w=.18
fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
axs[0].bar(x-w/2,dev_acc,w,label='Development',color='#5b8db8'); axs[0].bar(x+w/2,cal_acc,w,label='Calibration',color='#d97745'); axs[0].set_title('Quality: accuracy'); axs[0].set_ylim(.85,1.0); axs[0].set_ylabel('Score'); axs[0].set_xticks(x,models,rotation=20,ha='right'); axs[0].legend(fontsize=8)
axs[1].bar(x-w/2,dev_f1,w,label='Development',color='#5b8db8'); axs[1].bar(x+w/2,cal_f1,w,label='Calibration',color='#d97745'); axs[1].set_title('Quality: macro-F1'); axs[1].set_ylim(.8,1.0); axs[1].set_xticks(x,models,rotation=20,ha='right'); axs[1].legend(fontsize=8)
bars=axs[2].bar(x,latency,color=colors); axs[2].set_yscale('log'); axs[2].set_title('Operational cost'); axs[2].set_ylabel('Mean inference seconds (log scale)'); axs[2].set_xticks(x,models,rotation=20,ha='right')
for b,v in zip(bars,latency): axs[2].text(b.get_x()+b.get_width()/2,v*1.4,f'{v:.2g}s',ha='center',fontsize=8)
fig.suptitle('bitext-v2 student model comparison (same-input evaluation)',fontsize=14,fontweight='bold')
fig.text(.5,-.02,'Model artifact size: TF-IDF students ≈1–2 MB each; DistilBERT ≈256 MB. Scores are reviewed-label results.',ha='center',fontsize=9)
out=Path('reports/model-comparison-v1.png'); fig.savefig(out,dpi=180,bbox_inches='tight'); print(out)
