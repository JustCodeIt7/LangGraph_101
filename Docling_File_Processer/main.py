import streamlit as st
import tempfile
import zipfile
from pathlib import Path
from docling.document_converter import DocumentConverter
import shutil
import io

# Supported file types
SUPPORTED_TYPES = ['.pdf', '.docx', '.doc', '.html', '.htm', '.pptx', '.ppt',
                   '.xlsx', '.xls', '.txt', '.md', '.asciidoc', '.adoc']

def process_file(file_path: Path, converter: DocumentConverter) -> tuple[str, str]:
    """Process a single file and return markdown content and filename."""
    try:
        result = converter.convert(str(file_path))
        markdown_content = result.document.export_to_markdown()
        output_filename = file_path.stem + '.md'
        return markdown_content, output_filename
    except Exception as e:
        raise Exception(f"Error processing {file_path.name}: {str(e)}")

def create_zip(processed_files: list[tuple[str, str]]) -> bytes:
    """Create a ZIP file containing all processed markdown files."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for content, filename in processed_files:
            zip_file.writestr(filename, content)
    return zip_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="Docling File Processor",
        page_icon="=Ä",
        layout="centered"
    )

    st.title("=Ä Docling File Processor")
    st.markdown("""
    Convert your documents to Markdown format using Docling.

    **Supported formats:** PDF, DOCX, DOC, HTML, HTM, PPTX, PPT, XLSX, XLS, TXT, MD, AsciiDoc

    **Instructions:**
    1. Choose between single file or folder upload
    2. Upload your file(s)
    3. Click 'Process' to convert to Markdown
    4. Download the result(s)
    """)

    # Upload mode selection
    upload_mode = st.radio("Upload Mode:", ["Single File", "Folder (Multiple Files)"])

    uploaded_files = []

    if upload_mode == "Single File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[ext.lstrip('.') for ext in SUPPORTED_TYPES]
        )
        if uploaded_file:
            uploaded_files = [uploaded_file]
    else:
        uploaded_files = st.file_uploader(
            "Choose files from a folder",
            type=[ext.lstrip('.') for ext in SUPPORTED_TYPES],
            accept_multiple_files=True
        )

    if uploaded_files:
        st.info(f"=Á {len(uploaded_files)} file(s) uploaded")

        if st.button("=€ Process Files", type="primary"):
            converter = DocumentConverter()
            processed_files = []
            errors = []

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")

                # Create temp file
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / uploaded_file.name

                    # Write uploaded file to temp location
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Process file
                    try:
                        markdown_content, output_filename = process_file(temp_path, converter)
                        processed_files.append((markdown_content, output_filename))
                        st.success(f" {uploaded_file.name} ’ {output_filename}")
                    except Exception as e:
                        errors.append(f"L {uploaded_file.name}: {str(e)}")
                        st.error(errors[-1])

                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Processing complete!")

            # Display results
            if processed_files:
                st.success(f"<‰ Successfully processed {len(processed_files)} file(s)")

                if len(processed_files) == 1:
                    # Single file - direct download
                    markdown_content, filename = processed_files[0]
                    st.download_button(
                        label="=å Download Markdown File",
                        data=markdown_content,
                        file_name=filename,
                        mime="text/markdown"
                    )

                    # Preview
                    with st.expander("=Ö Preview Markdown"):
                        st.code(markdown_content, language="markdown")
                else:
                    # Multiple files - ZIP download
                    zip_data = create_zip(processed_files)
                    st.download_button(
                        label="=å Download ZIP File",
                        data=zip_data,
                        file_name="processed_documents.zip",
                        mime="application/zip"
                    )

                    # Show file list
                    with st.expander("=Ë Processed Files"):
                        for _, filename in processed_files:
                            st.text(f" {filename}")

            if errors:
                st.warning(f"  {len(errors)} file(s) failed to process")
                with st.expander("L View Errors"):
                    for error in errors:
                        st.text(error)
    else:
        st.info("=F Please upload file(s) to begin")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Powered by <a href='https://github.com/DS4SD/docling'>Docling</a> |
        Built with <a href='https://streamlit.io'>Streamlit</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
