import streamlit as st
import os
from zipfile import ZipFile, ZIP_DEFLATED
import io
import tempfile

from pathlib import Path
from docling.document_converter import DocumentConverter

# --- Docling Integration ---
# Using the Docling library to process uploaded files.

def process_file_to_markdown(file_path: str) -> str:
    """
    Processes a file with Docling and returns its Markdown representation.
    """
    try:
        doc_converter = DocumentConverter()  # Using default configuration
        input_path = Path(file_path)
        
        # `convert_all` expects a list of paths and returns a list of results
        conversion_results = doc_converter.convert_all([input_path])
        
        if not conversion_results:
            raise ValueError("Docling returned no results for the input file.")

        result = conversion_results[0]
        
        if result.error:
            # If Docling reports a specific error for the file (e.g., unsupported format)
            raise TypeError(f"Unsupported or corrupted file: {result.error}")
            
        if not result.document:
            raise ValueError("The converted document is empty.")

        return result.document.export_to_markdown()

    except Exception as e:
        # Catch any other exceptions during processing and report them
        raise IOError(f"Failed to process {os.path.basename(file_path)}: {e}") from e

# --- Streamlit App UI ---

st.set_page_config(layout="centered", page_title="Docling File Processor")

st.title("📄 Docling File Processor")
st.markdown(
    "Upload one or more files (e.g., PDF, DOCX, HTML) to convert them into Markdown format. "
    "If you upload a single file, you'll get a `.md` file. "
    "If you upload multiple files, you'll get a `.zip` archive."
)

uploaded_files = st.file_uploader(
    "Choose your files",
    accept_multiple_files=True,
    help="You can select multiple files at once."
)

# --- Processing Logic ---

if uploaded_files:
    if st.button(f"Process {len(uploaded_files)} File(s)"):
        
        # Use a temporary directory to avoid cluttering the main directory
        with tempfile.TemporaryDirectory() as temp_dir:
            
            if len(uploaded_files) == 1:
                # --- SINGLE FILE PROCESSING ---
                uploaded_file = uploaded_files[0]
                st.info(f"Processing `{uploaded_file.name}`...")
                progress_bar = st.progress(0, text="Processing...")

                try:
                    # Write the uploaded file to a temporary path
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file to Markdown
                    markdown_content = process_file_to_markdown(temp_file_path)
                    progress_bar.progress(100, text="Completed!")
                    st.success(f"Successfully processed `{uploaded_file.name}`!")

                    # Prepare for download
                    download_filename = f"{os.path.splitext(uploaded_file.name)[0]}.md"
                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=markdown_content.encode(),
                        file_name=download_filename,
                        mime="text/markdown",
                    )

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ An error occurred: {e}")

            else:
                # --- MULTIPLE FILE (FOLDER) PROCESSING ---
                st.info("Processing files and creating a ZIP archive...")
                progress_bar = st.progress(0, text="Starting...")
                zip_buffer = io.BytesIO()

                try:
                    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
                        total_files = len(uploaded_files)
                        for i, uploaded_file in enumerate(uploaded_files):
                            progress_text = f"Processing `{uploaded_file.name}` ({i+1}/{total_files})"
                            progress_bar.progress((i) / total_files, text=progress_text)

                            # Write to a temporary path
                            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Process the file
                            markdown_content = process_file_to_markdown(temp_file_path)
                            
                            # Add the processed file to the ZIP
                            markdown_filename = f"{os.path.splitext(uploaded_file.name)[0]}.md"
                            zip_file.writestr(markdown_filename, markdown_content)
                    
                    progress_bar.progress(100, text="Completed!")
                    st.success("✅ All files processed and zipped successfully!")

                    # Prepare ZIP for download
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="processed_markdown_files.zip",
                        mime="application/zip",
                    )

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ An error occurred during batch processing: {e}")

st.markdown("---")
st.write("Powered by Docling and Streamlit.")
