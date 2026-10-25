"""OpenAI Responses API teacher with strict, cached catalog classification."""
import hashlib, json, os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ..contracts import Catalog, Annotation

class TeacherDecision(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status: str = Field(pattern='^(resolved|abstained)$')
    intent_id: str | None = None
    confidence: float = Field(ge=0,le=1)
    reason: str = Field(min_length=1,max_length=500)
    reason_code: str = Field(default='accepted_intent', pattern='^(accepted_intent|unknown_request)$')
    abstention_cause: str | None = Field(default=None, pattern='^(ambiguous_intent|missing_information|unsupported_request)$')
    clarification_question: str | None = None

    @model_validator(mode='after')
    def consistent(self):
        if self.status == 'resolved':
            if self.intent_id is None or self.reason_code != 'accepted_intent':
                raise ValueError('resolved decisions require an accepted intent')
            if self.clarification_question is not None:
                raise ValueError('resolved decisions must not ask for clarification')
        else:
            if self.intent_id is not None:
                raise ValueError('abstentions must not select an intent')
            if self.reason_code == 'accepted_intent' or not self.clarification_question:
                raise ValueError('abstentions require a clarification question')
            if self.abstention_cause is None:
                raise ValueError('abstentions require an abstention cause')
        return self

class OpenAITeacher:
    def __init__(self, catalog_path: Path, cache_dir: Path, model='gpt-5-mini'):
        load_dotenv(dotenv_path=Path('.env'))
        if not os.getenv('OPENAI_API_KEY'): raise RuntimeError('OPENAI_API_KEY is not configured')
        self.client=OpenAI();self.model=model
        catalog_data=json.loads(catalog_path.read_text());catalog_data.pop('status',None)
        self.catalog=Catalog.model_validate(catalog_data)
        self.catalog_hash=hashlib.sha256(catalog_path.read_bytes()).hexdigest();self.cache=cache_dir;self.cache.mkdir(parents=True,exist_ok=True)
        self.schema={"type":"object","additionalProperties":False,"properties":{"status":{"type":"string","enum":["resolved","abstained"]},"intent_id":{"type":["string","null"]},"confidence":{"type":"number","minimum":0,"maximum":1},"reason":{"type":"string"},"reason_code":{"type":"string","enum":["accepted_intent","unknown_request"]},"abstention_cause":{"type":["string","null"],"enum":["ambiguous_intent","missing_information","unsupported_request",None]},"clarification_question":{"type":["string","null"]}},"required":["status","intent_id","confidence","reason","reason_code","abstention_cause","clarification_question"]}
    def classify(self,ticket_id,instruction):
        # Include the opaque ticket ID so identical instructions cannot reuse a
        # cached response with another ticket's provenance.
        key=hashlib.sha256((self.model+'\nteacher-v2\n'+self.catalog_hash+'\n'+ticket_id+'\n'+instruction).encode()).hexdigest();path=self.cache/(key+'.json')
        if path.exists(): return json.loads(path.read_text()),True
        catalog='\n'.join(f"- {e.id}: {e.name}. {e.definition} Include: {'; '.join(e.include_when)} Exclude: {'; '.join(e.exclude_when)}" for e in self.catalog.entries)
        prompt=f"Classify this support ticket against the catalog. Choose exactly one intent only when it clearly fits; otherwise abstain. Set reason_code to accepted_intent for resolved tickets, or unknown_request for abstentions. For abstentions, also set abstention_cause to ambiguous_intent, missing_information, or unsupported_request, and ask one precise clarification_question. Do not invent IDs. Return JSON only.\n\nCATALOG:\n{catalog}\n\nTICKET:\n{instruction}"
        last=None
        for attempt in range(3):
            try:
                response=self.client.responses.create(model=self.model,input=prompt,text={"format":{"type":"json_schema","name":"teacher_decision","strict":True,"schema":self.schema}})
                raw=response.output_text;decision=TeacherDecision.model_validate_json(raw)
                if decision.status=='resolved' and decision.intent_id not in {e.id for e in self.catalog.entries}: raise ValueError('unknown intent ID')
                result={'ticket_id':ticket_id,'catalog_version':self.catalog.version,'model_version':self.model,'prompt_version':'teacher-v2','annotation_version':'pilot-v2',**decision.model_dump()}
                path.write_text(json.dumps(result,indent=2)+'\n');return result,False
            except Exception as exc:
                last=exc;time.sleep(2**attempt)
        raise RuntimeError(f'teacher request failed after retries: {type(last).__name__}: {str(last)[:240]}')
