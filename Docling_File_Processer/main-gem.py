import streamlit as st
from io import BytesIO
import zipfile
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.conversion import ConversionStatus

st.title('Docling File Processor')

st.info("""
Upload a single file or multiple files/ZIP for folder processing. 
Supported formats: PDF, DOCX, HTML, MD, TXT, etc. 
Docling will convert to Markdown. Progress shown during processing.
""")

upload_type = st.radio('Choose upload type:', ('Single file', 'Multiple files or ZIP'))

if upload_type == 'Single file':
    uploaded_file = st.file_uploader(
        'Choose a file', type=['pdf', 'docx', 'html', 'md', 'txt', 'doc', 'pptx', 'png', 'jpg', 'jpeg']
    )
            if uploaded_file is not None:
                try:
                    with st.spinner('Processing file...'):
                        converter = DocumentConverter()
                        buf = BytesIO(uploaded_file.read())                source = DocumentStream(name=uploaded_file.name, stream=buf)
                result = converter.convert(source)
                if result.status == ConversionStatus.Success:
                    markdown_content = result.document.export_to_markdown()
                    st.success('Processing complete!')
                    st.download_button(
                        label='Download Markdown',
                        data=markdown_content,
                        file_name=f'{Path(uploaded_file.name).stem}.md',
                        mime='text/markdown',
                    )
                else:
                    st.error(f'Conversion failed: {result.status}')
        except Exception as e:
            st.error(f'Error processing file: {str(e)}')
else:
    uploaded_files = st.file_uploader(
        'Choose files or ZIP',
        accept_multiple_files=True,
        type=['pdf', 'docx', 'html', 'md', 'txt', 'doc', 'pptx', 'png', 'jpg', 'jpeg', 'zip'],
    )
    if uploaded_files:
        converter = DocumentConverter()
        markdown_files = {}
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, file in enumerate(uploaded_files):
            try:
                if file.type == 'application/zip':
                    # Handle ZIP separately
                    zip_buf = BytesIO(file.read())
                    with zipfile.ZipFile(zip_buf) as zf:
                        zip_files = [
                            f
                            for f in zf.namelist()
                            if not f.endswith('/')
                            and Path(f).suffix.lower()
                            in ['.pdf', '.docx', '.html', '.md', '.txt', '.doc', '.pptx', '.png', '.jpg', '.jpeg']
                        ]
                        for j, zfile in enumerate(zip_files):
                            with zf.open(zfile) as zf_file:
                                zbuf = BytesIO(zf_file.read())
                                zsource = DocumentStream(name=zfile, stream=zbuf)
                                zresult = converter.convert(zsource)
                                if zresult.status == ConversionStatus.Success:
                                    markdown_files[zfile] = zresult.document.export_to_markdown()
                else:
                    # Single file in multiple upload
                    buf = BytesIO(file.read())
                    source = DocumentStream(name=file.name, stream=buf)
                    result = converter.convert(source)
                    if result.status == ConversionStatus.Success:
                        markdown_files[file.name] = result.document.export_to_markdown()
            except Exception as e:
                st.warning(f'Skipped {file.name}: {str(e)}')
            progress_bar.progress((i + 1) / total_files)

        if markdown_files:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for filename, content in markdown_files.items():
                    md_filename = f'{Path(filename).stem}.md'
                    zf.writestr(md_filename, content)
            zip_buffer.seek(0)
            st.success(f'Processed {len(markdown_files)} files successfully!')
            st.download_button(
                label='Download ZIP of Markdown Files',
                data=zip_buffer.getvalue(),
                file_name='processed_markdowns.zip',
                mime='application/zip',
            )
        else:
            st.error('No files were processed successfully. Check supported formats and try again.')
