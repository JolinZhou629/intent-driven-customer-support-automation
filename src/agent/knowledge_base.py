"""Project-owned versioned knowledge-base article registry."""
from pathlib import Path
import re
from pydantic import BaseModel, ConfigDict

class KnowledgeArticle(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str; version: str; supported_intents: list[str]; title: str; required_information: list[str] = []
    body: str

class KnowledgeBaseRegistry:
    def __init__(self, root: Path = Path('config/knowledge_base')):
        self.articles=[]
        for path in sorted(root.glob('*.md')):
            text=path.read_text(); parts=text.split('---',2)
            if len(parts)<3: continue
            meta={}
            for line in parts[1].splitlines():
                if ':' in line:
                    k,v=line.split(':',1); v=v.strip(); meta[k.strip()]=v
            intents=re.findall(r'[\w.]+',meta.get('supported_intents',''))
            self.articles.append(KnowledgeArticle(id=meta['id'],version=meta['version'],supported_intents=intents,title=meta['title'],required_information=re.findall(r'[\w_]+',meta.get('required_information','')),body=parts[2].strip()))
    def retrieve(self, intent_id: str) -> KnowledgeArticle | None:
        return next((a for a in self.articles if intent_id in a.supported_intents),None)
