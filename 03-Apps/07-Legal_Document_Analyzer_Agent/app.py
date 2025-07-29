import streamlit as st
import tempfile
import os
from typing import Dict, Any
from legal_analyzer import LegalDocumentAnalyzer, AnalysisType
import base64
from PIL import Image
import io


def encode_image_to_base64(uploaded_file) -> str:
    image = Image.open(uploaded_file)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def main():
    st.set_page_config(
        page_title="Legal Document Analyzer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document Analyzer")
    st.markdown("Analyze legal documents with AI-powered clause extraction, compliance checking, risk scoring, and key terms summarization.")
    
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    
    with st.sidebar:
        st.header("🔧 Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the analyzer"
        )
        
        if api_key:
            try:
                if st.session_state.analyzer is None:
                    st.session_state.analyzer = LegalDocumentAnalyzer(api_key=api_key)
                st.success("✅ API Key configured successfully!")
            except Exception as e:
                st.error(f"❌ Error configuring API key: {str(e)}")
        
        st.markdown("---")
        
        st.header("📋 Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            options=[
                ("Extract Clauses", AnalysisType.EXTRACT_CLAUSES.value),
                ("Compliance Check", AnalysisType.COMPLIANCE_CHECK.value),
                ("Risk Scoring", AnalysisType.RISK_SCORING.value),
                ("Summarize Key Terms", AnalysisType.SUMMARIZE_KEY_TERMS.value),
                ("Full Analysis", AnalysisType.FULL_ANALYSIS.value)
            ],
            format_func=lambda x: x[0]
        )[1]
        
        parameters = {}
        
        if analysis_type == AnalysisType.EXTRACT_CLAUSES.value:
            clause_type = st.selectbox(
                "Clause Type",
                ["all", "indemnification", "termination", "liability", "confidentiality", "payment", "intellectual_property"]
            )
            parameters["clause_type"] = clause_type
        
        elif analysis_type == AnalysisType.COMPLIANCE_CHECK.value:
            regulation_set = st.selectbox(
                "Regulation Set",
                ["GDPR", "HIPAA", "SOX", "CCPA", "PCI-DSS", "FERPA"]
            )
            parameters["regulation_set"] = regulation_set
        
        elif analysis_type == AnalysisType.RISK_SCORING.value:
            risk_categories = st.multiselect(
                "Risk Categories",
                ["financial", "operational", "legal", "compliance", "reputational", "cybersecurity"],
                default=["financial", "operational", "legal"]
            )
            parameters["risk_categories"] = risk_categories
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text", "Upload Image"]
        )
        
        document_content = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf', 'docx'],
                help="Upload a legal document for analysis"
            )
            
            if uploaded_file is not None:
                with st.spinner("Loading document..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_file_path = tmp_file.name
                        
                        if st.session_state.analyzer:
                            document_content = st.session_state.analyzer.load_document_from_file(tmp_file_path)
                            st.success("✅ Document loaded successfully!")
                        else:
                            st.error("❌ Please configure your API key first")
                        
                        os.unlink(tmp_file_path)
                    except Exception as e:
                        st.error(f"❌ Error loading document: {str(e)}")
        
        elif input_method == "Paste Text":
            document_content = st.text_area(
                "Paste your legal document text here:",
                height=300,
                placeholder="Enter the text of your legal document..."
            )
        
        elif input_method == "Upload Image":
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image of a legal document (OCR will be applied)"
            )
            
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                with st.spinner("Extracting text from image..."):
                    try:
                        document_content = encode_image_to_base64(uploaded_image)
                        st.success("✅ Image uploaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
        
        if document_content and st.session_state.analyzer:
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                with st.spinner("Analyzing document... This may take a few moments."):
                    try:
                        result = st.session_state.analyzer.analyze_document(
                            document_content=document_content,
                            analysis_type=analysis_type,
                            parameters=parameters
                        )
                        st.session_state.analysis_result = result
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if "analysis_result" in st.session_state:
            st.markdown(st.session_state.analysis_result)
            
            st.markdown("---")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                st.download_button(
                    label="📥 Download as Text",
                    data=st.session_state.analysis_result,
                    file_name="legal_analysis.txt",
                    mime="text/plain"
                )
            
            with col_download2:
                st.download_button(
                    label="📥 Download as Markdown",
                    data=st.session_state.analysis_result,
                    file_name="legal_analysis.md",
                    mime="text/markdown"
                )
        
        else:
            st.info("👈 Upload a document and click 'Analyze Document' to see results here.")
            
            with st.expander("ℹ️ About the Legal Document Analyzer"):
                st.markdown("""
                This tool provides comprehensive analysis of legal documents using AI:
                
                **🔍 Extract Clauses**
                - Identify and extract specific types of clauses
                - Support for common clause types like indemnification, termination, liability
                
                **⚖️ Compliance Check**
                - Check documents against regulatory requirements
                - Support for GDPR, HIPAA, SOX, CCPA, and more
                
                **⚠️ Risk Scoring**
                - Assess risks across multiple categories
                - Financial, operational, legal, and compliance risks
                
                **📋 Key Terms Summary**
                - Extract and summarize important terms and provisions
                - Bullet-point format for easy review
                
                **🔬 Full Analysis**
                - Comprehensive analysis combining all features
                - Complete document assessment in one report
                """)
    
    with st.expander("🛠️ Advanced Settings"):
        st.markdown("### Model Configuration")
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            temperature = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Lower values make the model more deterministic"
            )
        
        with col_adv2:
            chunk_size = st.number_input(
                "Text Chunk Size",
                min_value=500,
                max_value=4000,
                value=2000,
                step=100,
                help="Size of text chunks for processing large documents"
            )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Legal Document Analyzer | Powered by LangGraph and OpenAI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()