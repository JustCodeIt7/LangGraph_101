import os, logging, json, asyncio
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
SUPPORT_LANGUAGE = os.getenv('SUPPORT_LANGUAGE', 'en')
GDPR_COMPLIANCE = os.getenv('GDPR_COMPLIANCE', 'true').lower() == 'true'


class SupportState(TypedDict):
    messages: Annotated[List, 'Conversation history']
    customer_data: Dict
    intent: str
    entities: Dict
    pending_action: Optional[str]
    ticket_id: Optional[str]
    response: str
    language: str
    timestamp: str


class IntentClassification(BaseModel):
    intent: str = Field(description='Customer intent')
    confidence: float = Field(description='Confidence score')
    entities: Dict = Field(description='Extracted entities')


class KnowledgeResult(BaseModel):
    found: bool = Field(description='Whether knowledge was found')
    answer: Optional[str] = Field(description='Knowledge base answer')
    confidence: float = Field(description='Confidence in match')


class TicketDetails(BaseModel):
    title: str = Field(description='Brief title')
    description: str = Field(description='Detailed description')
    priority: str = Field(description='Priority level')
    category: str = Field(description='Issue category')


class CustomerSupportAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4', temperature=0.1)
        self.setup_chains()

    def setup_chains(self):
        intent_template = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    'Classify customer intent: greeting, ask_question, report_issue, check_ticket_status, request_escalation, unknown. Return JSON with intent, confidence, entities.',
                ),
                ('human', '{user_message}'),
            ]
        )
        self.intent_chain = intent_template | self.llm.with_structured_output(IntentClassification)

        kb_template = ChatPromptTemplate.from_messages(
            [
                ('system', 'Search knowledge base for: {topic}. Return JSON with found, answer, confidence.'),
                ('human', 'Search for: {topic}'),
            ]
        )
        self.kb_chain = kb_template | self.llm.with_structured_output(KnowledgeResult)

        response_template = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    'Generate professional customer support response. Intent: {intent}, Context: {context}, Language: {language}',
                ),
                ('human', 'Generate response'),
            ]
        )
        self.response_chain = response_template | self.llm | StrOutputParser()

    def identify_intent(self, user_message: str, conversation_history: List) -> IntentClassification:
        try:
            context = '\n'.join([f'{msg.type}: {msg.content}' for msg in conversation_history[-5:]])
            return self.intent_chain.invoke({'user_message': user_message, 'context': context})
        except Exception as e:
            logger.error(f'Intent classification failed: {e}')
            return IntentClassification(intent='unknown', confidence=0.0, entities={})

    def lookup_knowledge_base(self, topic: str, context: str) -> KnowledgeResult:
        try:
            return self.kb_chain.invoke({'topic': topic, 'context': context})
        except Exception as e:
            logger.error(f'KB lookup failed: {e}')
            return KnowledgeResult(found=False, answer=None, confidence=0.0)

    def create_ticket(self, details: TicketDetails, customer_data: Dict) -> str:
        try:
            ticket_id = f'TK{datetime.now().strftime("%Y%m%d%H%M%S")}'
            logger.info(f'Created ticket {ticket_id}')
            return ticket_id
        except Exception as e:
            logger.error(f'Ticket creation failed: {e}')
            return None

    def route_ticket(self, ticket_id: str, issue_type: str) -> bool:
        try:
            departments = {'billing': 'Billing', 'technical': 'Technical', 'account': 'Account', 'general': 'General'}
            logger.info(f'Routed ticket {ticket_id} to {departments.get(issue_type, "General")}')
            return True
        except Exception as e:
            logger.error(f'Routing failed: {e}')
            return False

    def escalate_ticket(self, ticket_id: str, customer_data: Dict, reason: str = None) -> bool:
        try:
            level = 'HIGH' if customer_data.get('is_vip') else 'NORMAL'
            logger.info(f'Escalated ticket {ticket_id} to level {level}')
            return True
        except Exception as e:
            logger.error(f'Escalation failed: {e}')
            return False

    def get_ticket_status(self, ticket_id: str) -> str:
        try:
            import random

            return random.choice(['Open', 'In Progress', 'Resolved'])
        except Exception as e:
            logger.error(f'Status lookup failed: {e}')
            return 'Unknown'

    def generate_response(self, intent: str, context: Dict) -> str:
        try:
            return self.response_chain.invoke(
                {'intent': intent, 'context': json.dumps(context), 'language': SUPPORT_LANGUAGE}
            )
        except Exception as e:
            logger.error(f'Response generation failed: {e}')
            return "I apologize, but I'm having trouble processing your request."


class SupportWorkflow:
    def __init__(self):
        self.agent = CustomerSupportAgent()
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(SupportState)

        def classify_intent(state: SupportState) -> SupportState:
            if not state['messages']:
                return state
            last_message = state['messages'][-1]
            if isinstance(last_message, HumanMessage):
                result = self.agent.identify_intent(last_message.content, state['messages'])
                state['intent'] = result.intent
                state['entities'] = result.entities
            return state

        def route_by_intent_condition(state: SupportState) -> str:
            return state.get('pending_action') or state.get('intent', 'unknown')

        def handle_greeting(state: SupportState) -> SupportState:
            response = self.agent.generate_response(
                'greeting', {'customer_name': state['customer_data'].get('name', 'Customer')}
            )
            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        def handle_question(state: SupportState) -> SupportState:
            topic = state['entities'].get('topic', 'general')
            context = '\n'.join([msg.content for msg in state['messages'][-3:]])
            kb_result = self.agent.lookup_knowledge_base(topic, context)

            if kb_result.found and kb_result.confidence > 0.7:
                response = kb_result.answer
            else:
                response = "I couldn't find an answer. Would you like me to create a ticket?"
                state['pending_action'] = 'create_ticket_confirmation'

            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        def handle_issue(state: SupportState) -> SupportState:
            user_message = state['messages'][-1].content if state['messages'] else ''
            issue_type = state['entities'].get('issue_type', 'general')

            ticket_details = TicketDetails(
                title=f'Issue: {user_message[:50]}...',
                description=user_message,
                priority=state['entities'].get('severity', 'medium'),
                category=issue_type,
            )

            ticket_id = self.agent.create_ticket(ticket_details, state['customer_data'])
            if ticket_id:
                self.agent.route_ticket(ticket_id, issue_type)
                response = f"I've created ticket {ticket_id}. Someone will contact you soon."
                state['ticket_id'] = ticket_id
            else:
                response = "I couldn't create a ticket. Please try again."

            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        def handle_status_check(state: SupportState) -> SupportState:
            ticket_id = state['entities'].get('ticket_id')
            if ticket_id:
                status = self.agent.get_ticket_status(ticket_id)
                response = f'Ticket {ticket_id} status: {status}'
            else:
                response = 'Please provide a ticket ID.'

            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        def handle_escalation(state: SupportState) -> SupportState:
            is_vip = state['customer_data'].get('is_vip', False)
            severity = state['entities'].get('severity', 'medium')

            if is_vip or severity == 'high':
                ticket_id = state.get('ticket_id') or 'ESCALATED'
                if self.agent.escalate_ticket(ticket_id, state['customer_data']):
                    response = "I've escalated your issue. A senior member will contact you."
                else:
                    response = "I'm having trouble escalating. Please contact support."
            else:
                response = 'Please provide more details about why this is urgent.'
                state['pending_action'] = 'escalation_reason'

            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        def handle_pending_action(state: SupportState) -> SupportState:
            pending_action = state.get('pending_action')
            user_message = state['messages'][-1].content if state['messages'] else ''

            if pending_action == 'create_ticket_confirmation':
                if 'yes' in user_message.lower():
                    prev_question = next(
                        (
                            msg.content
                            for msg in reversed(state['messages'])
                            if isinstance(msg, HumanMessage) and msg != state['messages'][-1]
                        ),
                        '',
                    )
                    ticket_details = TicketDetails(
                        title=f'Question: {prev_question[:50]}...',
                        description=prev_question,
                        priority='medium',
                        category='general',
                    )
                    ticket_id = self.agent.create_ticket(ticket_details, state['customer_data'])
                    if ticket_id:
                        self.agent.route_ticket(ticket_id, 'general')
                        response = f'Created ticket {ticket_id} for your question.'
                        state['ticket_id'] = ticket_id
                    else:
                        response = "I couldn't create a ticket."
                else:
                    response = "Let me know if there's anything else I can help with."
                state['pending_action'] = None

            elif pending_action == 'escalation_reason':
                ticket_id = state.get('ticket_id') or 'ESCALATED'
                if self.agent.escalate_ticket(ticket_id, state['customer_data'], user_message):
                    response = "I've escalated your issue."
                else:
                    response = "I'm having trouble escalating your issue."
                state['pending_action'] = None

            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        def generate_fallback(state: SupportState) -> SupportState:
            response = "I'm not sure how to help with that. Can you please rephrase?"
            state['response'] = response
            state['messages'].append(AIMessage(content=response))
            return state

        workflow.add_node('classify_intent', classify_intent)
        workflow.add_node('handle_greeting', handle_greeting)
        workflow.add_node('handle_question', handle_question)
        workflow.add_node('handle_issue', handle_issue)
        workflow.add_node('handle_status_check', handle_status_check)
        workflow.add_node('handle_escalation', handle_escalation)
        workflow.add_node('handle_pending_action', handle_pending_action)
        workflow.add_node('generate_fallback', generate_fallback)

        workflow.add_edge('classify_intent', 'route_by_intent')
        workflow.add_conditional_edges(
            'route_by_intent',
            route_by_intent_condition,
            {
                'greeting': 'handle_greeting',
                'ask_question': 'handle_question',
                'report_issue': 'handle_issue',
                'check_ticket_status': 'handle_status_check',
                'request_escalation': 'handle_escalation',
                'pending_action': 'handle_pending_action',
                'unknown': 'generate_fallback',
            },
        )

        for node in [
            'handle_greeting',
            'handle_question',
            'handle_issue',
            'handle_status_check',
            'handle_escalation',
            'handle_pending_action',
            'generate_fallback',
        ]:
            workflow.add_edge(node, END)

        workflow.set_entry_point('classify_intent')
        return workflow.compile()


async def main():
    workflow = SupportWorkflow()

    customer_data = {'id': 'CUST001', 'name': 'John Doe', 'is_vip': False, 'email': 'john@example.com'}

    initial_state = SupportState(
        messages=[HumanMessage(content='Hello, I have a question about billing')],
        customer_data=customer_data,
        intent='',
        entities={},
        pending_action=None,
        ticket_id=None,
        response='',
        language=SUPPORT_LANGUAGE,
        timestamp=datetime.now().isoformat(),
    )

    result = await workflow.graph.ainvoke(initial_state)
    print(f'Agent Response: {result["response"]}')
    print(f'Intent: {result["intent"]}')
    print(f'Ticket ID: {result.get("ticket_id", "None")}')


if __name__ == '__main__':
    asyncio.run(main())
