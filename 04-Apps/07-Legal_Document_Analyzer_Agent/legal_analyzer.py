from typing import Dict, List, Optional, TypedDict, Annotated
import re
from enum import Enum
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import pytesseract
from PIL import Image
import io
import base64


class AnalysisType(Enum):
    EXTRACT_CLAUSES = "extract_clauses"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_SCORING = "risk_scoring"
    SUMMARIZE_KEY_TERMS = "summarize_key_terms"
    FULL_ANALYSIS = "full_analysis"


@dataclass
class ClauseResult:
    clause_type: str
    content: str
    location: str
    confidence: float


@dataclass
class ComplianceIssue:
    regulation: str
    issue_type: str
    description: str
    severity: str
    recommendation: str


@dataclass
class RiskScore:
    category: str
    score: float
    factors: List[str]
    description: str


class AnalysisState(TypedDict):
    document_content: str
    analysis_type: str
    parameters: Dict
    clauses: List[ClauseResult]
    compliance_issues: List[ComplianceIssue]
    risk_scores: List[RiskScore]
    key_terms_summary: str
    final_result: str
    messages: Annotated[list, add_messages]


class LegalDocumentAnalyzer:
    def __init__(self, api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AnalysisState)
        
        workflow.add_node("preprocess", self._preprocess_document)
        workflow.add_node("extract_clauses", self._extract_clauses)
        workflow.add_node("compliance_check", self._compliance_check)
        workflow.add_node("risk_scoring", self._risk_scoring)
        workflow.add_node("summarize_terms", self._summarize_key_terms)
        workflow.add_node("format_results", self._format_results)
        
        workflow.set_entry_point("preprocess")
        
        workflow.add_conditional_edges(
            "preprocess",
            self._route_analysis,
            {
                "extract_clauses": "extract_clauses",
                "compliance_check": "compliance_check", 
                "risk_scoring": "risk_scoring",
                "summarize_key_terms": "summarize_terms",
                "full_analysis": "extract_clauses"
            }
        )
        
        workflow.add_edge("extract_clauses", "format_results")
        workflow.add_edge("compliance_check", "format_results")
        workflow.add_edge("risk_scoring", "format_results")
        workflow.add_edge("summarize_terms", "format_results")
        workflow.add_edge("format_results", END)
        
        return workflow.compile()

    def _route_analysis(self, state: AnalysisState) -> str:
        return state["analysis_type"]

    def _preprocess_document(self, state: AnalysisState) -> AnalysisState:
        content = state["document_content"]
        
        if content.startswith("data:image"):
            content = self._extract_text_from_image(content)
        
        cleaned_content = self._clean_text(content)
        state["document_content"] = cleaned_content
        return state

    def _extract_text_from_image(self, base64_image: str) -> str:
        try:
            image_data = base64.b64decode(base64_image.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\%\$\&]', '', text)
        return text.strip()

    def _extract_clauses(self, state: AnalysisState) -> AnalysisState:
        content = state["document_content"]
        clause_type = state["parameters"].get("clause_type", "all")
        
        prompt = f"""
        Extract {clause_type} clauses from the following legal document.
        
        Document:
        {content}
        
        Please identify and extract relevant clauses, providing:
        1. The clause type
        2. The full text of the clause
        3. The approximate location in the document
        4. A confidence score (0-1)
        
        Format as JSON with clause_type, content, location, confidence fields.
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            import json
            clauses_data = json.loads(response.content)
            clauses = [ClauseResult(**clause) for clause in clauses_data]
        except:
            clauses = [ClauseResult(
                clause_type=clause_type,
                content=response.content,
                location="Document body",
                confidence=0.8
            )]
        
        state["clauses"] = clauses
        return state

    def _compliance_check(self, state: AnalysisState) -> AnalysisState:
        content = state["document_content"]
        regulation_set = state["parameters"].get("regulation_set", "GDPR")
        
        prompt = f"""
        Check the following legal document for compliance with {regulation_set} regulations.
        
        Document:
        {content}
        
        Identify any compliance issues, providing:
        1. The specific regulation violated
        2. Type of issue (missing clause, inadequate provision, etc.)
        3. Description of the issue
        4. Severity (High, Medium, Low)
        5. Recommendation for remediation
        
        Format as JSON with regulation, issue_type, description, severity, recommendation fields.
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            import json
            issues_data = json.loads(response.content)
            issues = [ComplianceIssue(**issue) for issue in issues_data]
        except:
            issues = [ComplianceIssue(
                regulation=regulation_set,
                issue_type="General Review Needed",
                description=response.content,
                severity="Medium",
                recommendation="Manual review recommended"
            )]
        
        state["compliance_issues"] = issues
        return state

    def _risk_scoring(self, state: AnalysisState) -> AnalysisState:
        content = state["document_content"]
        risk_categories = state["parameters"].get("risk_categories", ["financial", "operational", "legal"])
        
        prompt = f"""
        Analyze the following legal document for risks in these categories: {', '.join(risk_categories)}.
        
        Document:
        {content}
        
        For each risk category, provide:
        1. Category name
        2. Risk score (0-10, where 10 is highest risk)
        3. List of risk factors identified
        4. Description of the risk assessment
        
        Format as JSON with category, score, factors, description fields.
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            import json
            scores_data = json.loads(response.content)
            scores = [RiskScore(**score) for score in scores_data]
        except:
            scores = [RiskScore(
                category="Overall",
                score=5.0,
                factors=["General analysis performed"],
                description=response.content
            )]
        
        state["risk_scores"] = scores
        return state

    def _summarize_key_terms(self, state: AnalysisState) -> AnalysisState:
        content = state["document_content"]
        
        prompt = f"""
        Summarize the key terms and provisions of the following legal document.
        Focus on:
        - Key obligations and rights
        - Important dates and deadlines
        - Financial terms and amounts
        - Termination conditions
        - Liability and indemnification provisions
        
        Document:
        {content}
        
        Provide a concise bullet-point summary.
        """
        
        response = self.llm.invoke(prompt)
        state["key_terms_summary"] = response.content
        return state

    def _format_results(self, state: AnalysisState) -> AnalysisState:
        analysis_type = state["analysis_type"]
        
        if analysis_type == AnalysisType.EXTRACT_CLAUSES.value:
            result = self._format_clauses(state["clauses"])
        elif analysis_type == AnalysisType.COMPLIANCE_CHECK.value:
            result = self._format_compliance(state["compliance_issues"])
        elif analysis_type == AnalysisType.RISK_SCORING.value:
            result = self._format_risks(state["risk_scores"])
        elif analysis_type == AnalysisType.SUMMARIZE_KEY_TERMS.value:
            result = f"Key Terms Summary:\n{state['key_terms_summary']}"
        elif analysis_type == AnalysisType.FULL_ANALYSIS.value:
            result = self._format_full_analysis(state)
        else:
            result = "Please specify the type of legal document analysis you need."
        
        state["final_result"] = result
        return state

    def _format_clauses(self, clauses: List[ClauseResult]) -> str:
        if not clauses:
            return "No clauses extracted."
        
        result = "## Extracted Clauses\n\n"
        for clause in clauses:
            result += f"### {clause.clause_type}\n"
            result += f"**Location:** {clause.location}\n"
            result += f"**Confidence:** {clause.confidence:.2f}\n"
            result += f"**Content:** {clause.content}\n\n"
        
        return result

    def _format_compliance(self, issues: List[ComplianceIssue]) -> str:
        if not issues:
            return "No compliance issues identified."
        
        result = "## Compliance Analysis\n\n"
        for issue in issues:
            result += f"### {issue.regulation} - {issue.severity} Priority\n"
            result += f"**Issue Type:** {issue.issue_type}\n"
            result += f"**Description:** {issue.description}\n"
            result += f"**Recommendation:** {issue.recommendation}\n\n"
        
        return result

    def _format_risks(self, scores: List[RiskScore]) -> str:
        if not scores:
            return "No risk assessment completed."
        
        result = "## Risk Assessment\n\n"
        for score in scores:
            result += f"### {score.category} Risk\n"
            result += f"**Score:** {score.score}/10\n"
            result += f"**Risk Factors:** {', '.join(score.factors)}\n"
            result += f"**Assessment:** {score.description}\n\n"
        
        return result

    def _format_full_analysis(self, state: AnalysisState) -> str:
        result = "# Comprehensive Legal Document Analysis\n\n"
        
        if state.get("clauses"):
            result += self._format_clauses(state["clauses"]) + "\n"
        
        if state.get("compliance_issues"):
            result += self._format_compliance(state["compliance_issues"]) + "\n"
        
        if state.get("risk_scores"):
            result += self._format_risks(state["risk_scores"]) + "\n"
        
        if state.get("key_terms_summary"):
            result += f"## Key Terms Summary\n{state['key_terms_summary']}\n\n"
        
        return result

    def analyze_document(self, document_content: str, analysis_type: str, parameters: Dict = None) -> str:
        if parameters is None:
            parameters = {}
        
        initial_state = AnalysisState(
            document_content=document_content,
            analysis_type=analysis_type,
            parameters=parameters,
            clauses=[],
            compliance_issues=[],
            risk_scores=[],
            key_terms_summary="",
            final_result="",
            messages=[]
        )
        
        result = self.workflow.invoke(initial_state)
        return result["final_result"]

    def load_document_from_file(self, file_path: str) -> str:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                return "\n".join([doc.page_content for doc in docs])
            else:
                loader = TextLoader(file_path)
                docs = loader.load()
                return docs[0].page_content
        except Exception as e:
            return f"Error loading document: {str(e)}"