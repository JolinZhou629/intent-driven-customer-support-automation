"""Typed, allowlisted simulated tools for the first agent vertical slice."""
from typing import Literal
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated
from .knowledge_base import KnowledgeBaseRegistry

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class ToolResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    status: Literal['success', 'not_found', 'invalid_input', 'not_allowed', 'error']
    tool_name: Text
    evidence_id: Text
    data: dict[str, object] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False

class ToolRegistry:
    """Application-owned tool registry; no external side effects."""
    def __init__(self):
        self._tools = {
            'lookup_order': self.lookup_order,
            'lookup_refund': self.lookup_refund,
            'retrieve_invoice': self.retrieve_invoice,
            'validate_delivery_address': self.validate_delivery_address,
            'check_refund_eligibility': self.check_refund_eligibility,
            'create_simulated_refund_request': self.create_simulated_refund_request,
            'retrieve_kb_article': self.retrieve_kb_article,
        }
        self.knowledge_base=KnowledgeBaseRegistry()
        root=Path('data/fixtures')
        self.orders = {r['order_number']:{k:v for k,v in r.items() if k!='order_number'} for r in json.loads((root/'orders.json').read_text())}
        self.refunds = {r['refund_reference']:{k:v for k,v in r.items() if k!='refund_reference'} for r in json.loads((root/'refunds.json').read_text())}
        self.invoices = {r['invoice_reference']:{k:v for k,v in r.items() if k!='invoice_reference'} for r in json.loads((root/'invoices.json').read_text())}

    def names(self) -> list[str]: return sorted(self._tools)
    def call(self, name: str, **kwargs) -> ToolResult:
        if name not in self._tools:
            return ToolResult(status='not_allowed', tool_name=name, evidence_id=f'{name}-denied', error_code='tool_not_allowed')
        return self._tools[name](**kwargs)

    def lookup_order(self, order_number: str) -> ToolResult:
        if not order_number.strip(): return ToolResult(status='invalid_input', tool_name='lookup_order', evidence_id='order-invalid', error_code='missing_order_number')
        data=self.orders.get(order_number)
        return ToolResult(status='success' if data else 'not_found', tool_name='lookup_order', evidence_id=f'order-{order_number}', data={'order_number':order_number,**(data or {})}, error_code=None if data else 'order_not_found')

    def lookup_refund(self, refund_reference: str) -> ToolResult:
        if not refund_reference.strip(): return ToolResult(status='invalid_input', tool_name='lookup_refund', evidence_id='refund-invalid', error_code='missing_refund_reference')
        data=self.refunds.get(refund_reference)
        return ToolResult(status='success' if data else 'not_found', tool_name='lookup_refund', evidence_id=f'refund-{refund_reference}', data={'refund_reference':refund_reference,**(data or {})}, error_code=None if data else 'refund_not_found')

    def retrieve_invoice(self, invoice_reference: str) -> ToolResult:
        if not invoice_reference.strip(): return ToolResult(status='invalid_input', tool_name='retrieve_invoice', evidence_id='invoice-invalid', error_code='missing_invoice_reference')
        data=self.invoices.get(invoice_reference)
        return ToolResult(status='success' if data else 'not_found', tool_name='retrieve_invoice', evidence_id=f'invoice-{invoice_reference}', data={'invoice_reference':invoice_reference,**(data or {})}, error_code=None if data else 'invoice_not_found')

    def validate_delivery_address(self, order_number: str, new_address: str) -> ToolResult:
        if not order_number.strip() or not new_address.strip(): return ToolResult(status='invalid_input', tool_name='validate_delivery_address', evidence_id='address-invalid', error_code='missing_address_details')
        order=self.orders.get(order_number)
        if not order: return ToolResult(status='not_found', tool_name='validate_delivery_address', evidence_id=f'address-{order_number}', error_code='order_not_found')
        if order['order_status'] == 'shipped': return ToolResult(status='not_allowed', tool_name='validate_delivery_address', evidence_id=f'address-{order_number}', error_code='address_change_not_allowed')
        return ToolResult(status='success', tool_name='validate_delivery_address', evidence_id=f'address-{order_number}', data={'order_number':order_number,'new_address':new_address,'change_eligible':True})

    def check_refund_eligibility(self, order_number: str) -> ToolResult:
        if not order_number.strip(): return ToolResult(status='invalid_input',tool_name='check_refund_eligibility',evidence_id='refund-eligibility-invalid',error_code='missing_order_number')
        order=self.orders.get(order_number)
        if not order: return ToolResult(status='not_found',tool_name='check_refund_eligibility',evidence_id=f'refund-eligibility-{order_number}',error_code='order_not_found')
        eligible=bool(order.get('eligible_for_refund'))
        return ToolResult(status='success' if eligible else 'not_allowed',tool_name='check_refund_eligibility',evidence_id=f'refund-eligibility-{order_number}',data={'order_number':order_number,'eligible_for_refund':eligible},error_code=None if eligible else 'refund_not_eligible')

    def create_simulated_refund_request(self, order_number: str) -> ToolResult:
        check=self.check_refund_eligibility(order_number)
        if check.status!='success': return ToolResult(status=check.status,tool_name='create_simulated_refund_request',evidence_id=check.evidence_id,error_code=check.error_code)
        ref=f'REF-200{len(self.refunds)+1}'
        self.refunds[ref]={'refund_status':'submitted','amount':'simulated','order_number':order_number}
        return ToolResult(status='success',tool_name='create_simulated_refund_request',evidence_id=f'refund-request-{ref}',data={'order_number':order_number,'refund_reference':ref,'refund_status':'submitted'})

    def retrieve_kb_article(self, intent_id: str) -> ToolResult:
        article=self.knowledge_base.retrieve(intent_id)
        if not article: return ToolResult(status='not_found',tool_name='retrieve_kb_article',evidence_id=f'kb-{intent_id}',error_code='article_not_found')
        return ToolResult(status='success',tool_name='retrieve_kb_article',evidence_id=article.id,data={'article_id':article.id,'version':article.version,'title':article.title,'body':article.body})
