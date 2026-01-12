import streamlit as st
import tempfile
import os
from typing import TypedDict, List, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader

# --- Configuration & Setup ---
llm_model = 'llama3.2'  # Using a lightweight model
st.set_page_config(page_title='Legal Doc Analyzer', layout='wide')

# --- UI Styling ---
st.markdown(
    """
    <style>
    .stApp { background-color: #f0f2f6; }
    h1 { color: #1f77b4; }
    .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; }
    .report-box { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- State Definition ---
class AgentState(TypedDict):
    original_text: str
    summary: str
    risks: str
    suggestions: str
    final_report: str


# --- Node Functions ---
def summarize_node(state: AgentState):
    """Summarizes the legal document."""
    text = state['original_text']
    prompt = ChatPromptTemplate.from_template(
        'You are an expert legal assistant. Summarize the following legal document concisely:\n\n{text}'
    )
    chain = prompt | ChatOllama(model=llm_model) | StrOutputParser()
    return {'summary': chain.invoke({'text': text[:10000]})}  # Truncate for context window safety


def analyze_risks_node(state: AgentState):
    """Identifies potential risks in the document."""
    text = state['original_text']
    prompt = ChatPromptTemplate.from_template(
        'You are an expert legal assistant. Identify key legal risks and liabilities in this document:\n\n{text}'
    )
    chain = prompt | ChatOllama(model=llm_model) | StrOutputParser()
    return {'risks': chain.invoke({'text': text[:10000]})}


def suggest_improvements_node(state: AgentState):
    """Suggests improvements for the document."""
    text = state['original_text']
    prompt = ChatPromptTemplate.from_template(
        'You are an expert legal assistant. Suggest clause improvements or missing protections for this document:\n\n{text}'
    )
    chain = prompt | ChatOllama(model=llm_model) | StrOutputParser()
    return {'suggestions': chain.invoke({'text': text[:10000]})}


def compile_report_node(state: AgentState):
    """Compiles the final markdown report."""
    report = f"""
### 📝 Document Summary
{state['summary']}

### ⚠️ Identified Risks
{state['risks']}

### 💡 Suggestions for Improvement
{state['suggestions']}
    """
    return {'final_report': report}


# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node('summarize', summarize_node)
workflow.add_node('analyze_risks', analyze_risks_node)
workflow.add_node('suggest_improvements', suggest_improvements_node)
workflow.add_node('compile_report', compile_report_node)

# Run parallel analysis
workflow.set_entry_point('summarize')
workflow.add_edge('summarize', 'analyze_risks')
workflow.add_edge('analyze_risks', 'suggest_improvements')
workflow.add_edge('suggest_improvements', 'compile_report')
workflow.add_edge('compile_report', END)

app = workflow.compile()


# --- Helper Functions ---
def load_doc(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    text = ''
    try:
        if uploaded_file.name.endswith('.pdf'):
            loader = PDFPlumberLoader(tmp_path)
            docs = loader.load()
            text = '\n'.join([d.page_content for d in docs])
        else:  # Text file
            with open(tmp_path, 'r') as f:
                text = f.read()
    finally:
        os.remove(tmp_path)

    return text


# --- Main App Interface ---
st.title('⚖️ AI Legal Document Analyzer')
st.markdown('Upload a contract or legal document to get an instant AI-powered analysis.')

uploaded_file = st.file_uploader('Upload Legal Document (PDF or TXT)', type=['pdf', 'txt'])

if uploaded_file:
    if st.button('Analyze Document'):
        with st.spinner('Reading document...'):
            doc_text = load_doc(uploaded_file)

        if not doc_text.strip():
            st.error('Could not extract text from document.')
        else:
            with st.spinner('Analyzing with AI Agents...'):
                initial_state = {'original_text': doc_text}
                result = app.invoke(initial_state)

                st.markdown('## 📊 Analysis Report')

                with st.container():
                    st.markdown(f"<div class='report-box'>{result['final_report']}</div>", unsafe_allow_html=True)

                st.download_button('Download Report', result['final_report'], file_name='legal_analysis.md')

# Sidebar
with st.sidebar:
    st.header('About')
    st.info('This tool uses **LangGraph** to orchestrate AI agents for legal analysis.')
    st.markdown("""
    **Pipeline:**
    1. **Summarizer Agent**: Extracts key points.
    2. **Risk Agent**: Finds liabilities.
    3. **Advisor Agent**: Suggests improvements.
    """)
    st.text(f'Model: {llm_model}')
