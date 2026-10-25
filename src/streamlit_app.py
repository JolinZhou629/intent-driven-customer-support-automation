"""Streamlit shell for the automated support ticket simulation."""
import uuid
import sys
import streamlit as st
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.agent.service import SupportAgentService

st.set_page_config(page_title='Automated Support Ticketing', page_icon='🎫', layout='wide')
st.title('Automated Support Ticketing')
st.caption('Classifier and deep-agent simulation using project-owned fixture data.')

if 'ticket_text' not in st.session_state: st.session_state.ticket_text=''
if 'session_id' not in st.session_state: st.session_state.session_id=f'session-{uuid.uuid4().hex[:10]}'
if 'last_result' not in st.session_state: st.session_state.last_result=None
if 'last_handoff' not in st.session_state: st.session_state.last_handoff=None
if 'last_response' not in st.session_state: st.session_state.last_response=None

@st.cache_resource
def service():
    return SupportAgentService()

with st.sidebar:
    st.header('Session'); st.code(st.session_state.session_id)
    if st.button('New session'):
        st.session_state.session_id=f'session-{uuid.uuid4().hex[:10]}'; st.session_state.last_result=None; st.session_state.last_handoff=None; st.session_state.last_response=None; st.rerun()
    st.info('All tools and ticket actions are simulated.')
    st.subheader('Demo tickets')
    presets={
        'Order status':'Where is my order ORD-1001?',
        'Refund request':'How can I apply for a refund for order ORD-1002?',
        'Invoice retrieval':'Please retrieve invoice INV-1001.',
        'Address change':'Change the address for order ORD-1002 to 10 Demo Street.',
    }
    choice=st.selectbox('Load example', ['None']+list(presets))
    if choice!='None' and st.button('Use example'):
        st.session_state.ticket_text=presets[choice]; st.rerun()

st.subheader('1. Submit a support ticket')
text=st.text_area('Ticket text', value=st.session_state.ticket_text, height=130, placeholder='Example: Where is my order ORD-1001?')
col1,col2=st.columns([1,5])
with col1: submitted=st.button('Submit ticket', type='primary')
with col2:
    if st.button('Clear'):
        st.session_state.ticket_text=''; st.session_state.last_result=None; st.rerun()
if submitted:
    if not text.strip(): st.error('Enter a support request first.')
    else:
        ticket_id=st.session_state.last_result['ticket_id'] if st.session_state.last_result and st.session_state.last_result.get('status')=='waiting_for_user' else f'ticket-{uuid.uuid4().hex[:12]}'
        response=service().classify_and_handle(st.session_state.session_id,ticket_id,text)
        st.session_state.ticket_text=text; st.session_state.last_result={'ticket_id':ticket_id,'status':response.status,'text':text}; st.session_state.last_response=response
        st.session_state.last_handoff=service().classifier.predict(st.session_state.session_id,ticket_id,text)

result=st.session_state.last_result
if result:
    st.divider()
    handoff=st.session_state.last_handoff; response=st.session_state.last_response
    state=service().runtime.state_store.get(st.session_state.session_id)
    st.subheader('2. Classifier result and routing')
    if handoff:
        a,b,c,d=st.columns(4)
        a.metric('Intent', handoff.intent_id or 'Unclear')
        b.metric('Domain', handoff.domain_id or '—')
        c.metric('Decision', handoff.decision)
        d.metric('Margin', f'{handoff.margin:.3f}' if handoff.margin is not None else '—')
        if handoff.decision=='clarify': st.info('The classifier is not confident enough to select a final intent.')
        with st.expander('Classifier contract'):
            st.json(handoff.model_dump())

    st.subheader('3. Agent plan and execution')
    p,q=st.columns(2)
    with p:
        st.markdown(f"**Selected skill:** `{response.skill_id or 'none'}`")
        st.markdown(f"**Workflow status:** `{response.status}`")
        if response.missing_information: st.markdown('**Required information:** ' + ', '.join(f'`{x}`' for x in response.missing_information))
    with q:
        st.markdown(f"**Tool calls:** `{response.tool_call_count}`")
        st.markdown(f"**Evidence IDs:** {', '.join(response.evidence_ids) if response.evidence_ids else 'None'}")
        st.caption('Simulation only — no external systems are modified.')

    st.subheader('4. Customer-facing response')
    if response.status=='completed': st.success(response.message)
    elif response.status=='waiting_for_user': st.warning(response.message)
    else: st.error(response.message)
    if response.status=='waiting_for_user':
        followup=st.text_input('Provide the missing information', key='clarification_reply')
        if st.button('Continue workflow', type='primary'):
            if not followup.strip(): st.error('Enter the requested information first.')
            else:
                continued=service().continue_workflow(st.session_state.session_id,followup)
                st.session_state.last_response=continued; st.session_state.last_result['status']=continued.status; st.rerun()

    st.subheader('5. Evidence and audit')
    if state and state.tool_results:
        for item in state.tool_results:
            st.json(item)
    audit={'event':'agent_execution','simulated_tools':True,'session_id':st.session_state.session_id,'skill_id':response.skill_id,'tool_results':state.tool_results if state else []}
    with st.expander('Technical audit details'):
        st.json(audit)
