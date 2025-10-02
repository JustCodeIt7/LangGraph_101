import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

import streamlit as st
from docling.document_converter import DocumentConverter

# Supported suffixes are based on Docling's core formats and common text files.
SUPPORTED_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".eml",
    ".htm",
    ".html",
    ".md",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".txt",
    ".xls",
    ".xlsx",
}


class UnsupportedFileTypeError(Exception):
    """Raised when a file type cannot be processed."""


class ConversionFailureError(Exception):
    """Raised when Docling fails to convert the file."""


@st.cache_resource(show_spinner=False)
def get_converter() -> DocumentConverter:
    """Create and cache a single DocumentConverter instance per session."""
    return DocumentConverter()


def write_temp_file(uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile", root: str) -> Path:
    """Persist an uploaded file inside the temporary working directory."""
    destination = Path(root) / Path(uploaded_file.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return destination


def convert_path_to_markdown(path: Path) -> str:
    """Convert a single file to Markdown using Docling."""
    suffix = path.suffix.lower()
    if suffix and suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(f"Files of type `{suffix}` are not supported.")

    converter = get_converter()

    try:
        results = converter.convert_all([path])
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ConversionFailureError(f"Docling could not read `{path.name}`: {exc}") from exc

    if not results:
        raise ConversionFailureError("Docling returned no results.")

    result = results[0]
    if result.error:
        raise ConversionFailureError(str(result.error))
    if not result.document:
        raise ConversionFailureError("Docling produced an empty document.")

    return result.document.export_to_markdown()


def prepare_markdown_filename(source: Path) -> str:
    """Map the original file name to an `.md` name, preserving folders."""
    if source.suffix:
        return str(source.with_suffix(".md"))
    return f"{source}.md"


def convert_folder(
    uploaded_files: Iterable["st.runtime.uploaded_file_manager.UploadedFile"],
    progress_bar,
) -> Tuple[io.BytesIO, List[str], int]:
    """Convert multiple files and package the results into a ZIP archive."""
    errors: List[str] = []
    zip_buffer = io.BytesIO()
    files = list(uploaded_files)
    converted_count = 0

    if not files:
        return zip_buffer, ["No files were uploaded."], converted_count

    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
            total = len(files)
            for index, uploaded_file in enumerate(files, start=1):
                progress_text = f"Processing {uploaded_file.name} ({index}/{total})"
                progress_bar.progress(int(((index - 1) / total) * 100), text=progress_text)

                try:
                    local_path = write_temp_file(uploaded_file, temp_dir)
                    markdown = convert_path_to_markdown(local_path)
                    markdown_name = prepare_markdown_filename(Path(uploaded_file.name))
                    archive.writestr(markdown_name, markdown.encode("utf-8"))
                    converted_count += 1
                except UnsupportedFileTypeError as exc:
                    errors.append(f"{uploaded_file.name}: {exc}")
                except ConversionFailureError as exc:
                    errors.append(f"{uploaded_file.name}: {exc}")

                progress_bar.progress(int((index / total) * 100), text=progress_text)

    zip_buffer.seek(0)
    return zip_buffer, errors, converted_count


def convert_single_file(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile", progress_bar
) -> Tuple[str, str]:
    """Convert a single uploaded file and return Markdown content and name."""
    progress_bar.progress(10, text="Saving file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = write_temp_file(uploaded_file, temp_dir)
        progress_bar.progress(45, text="Running Docling...")
        markdown = convert_path_to_markdown(local_path)
        progress_bar.progress(100, text="Done")
        markdown_name = prepare_markdown_filename(Path(uploaded_file.name))
    return markdown, Path(markdown_name).name


# --- Streamlit UI -----------------------------------------------------------------

st.set_page_config(page_title="Docling File Processor", layout="centered")

st.title("📄 Docling File Processor")
st.write(
    "Convert documents into clean Markdown directly in the browser. Upload a single file "
    "or drag a folder to batch process everything at once."
)

with st.expander("Supported file types", expanded=False):
    st.markdown(", ".join(sorted(SUPPORTED_SUFFIXES)))

upload_mode = st.radio(
    "Select how you want to upload",
    ("Single file", "Folder of files"),
    help="Drag an entire folder if your browser supports it.",
)

multiple = upload_mode == "Folder of files"
upload_help = (
    "Choose one file to convert to Markdown."
    if not multiple
    else "Drag or select all files from the folder you want to convert."
)
uploaded_items = st.file_uploader(
    "Upload files",
    accept_multiple_files=multiple,
    help=upload_help,
)

st.caption("Nothing leaves your browser session—files are processed in memory using Docling.")

if upload_mode == "Single file" and uploaded_items:
    file_obj = uploaded_items if not isinstance(uploaded_items, list) else uploaded_items[0]
    convert_button = st.button("Convert to Markdown", type="primary")

    if convert_button:
        progress = st.progress(0, text="Starting...")
        try:
            markdown, markdown_name = convert_single_file(file_obj, progress)
            st.success(f"Converted `{file_obj.name}` successfully.")
            st.download_button(
                "Download Markdown",
                data=markdown.encode("utf-8"),
                file_name=markdown_name,
                mime="text/markdown",
            )
        except UnsupportedFileTypeError as exc:
            progress.empty()
            st.error(str(exc))
        except ConversionFailureError as exc:
            progress.empty()
            st.error(f"Conversion failed: {exc}")
        except Exception as exc:  # pragma: no cover
            progress.empty()
            st.error(f"Unexpected error: {exc}")

elif upload_mode == "Folder of files" and uploaded_items:
    convert_button = st.button(
        f"Convert {len(uploaded_items)} file{'s' if len(uploaded_items) != 1 else ''}",
        type="primary",
    )

    if convert_button:
        progress = st.progress(0, text="Preparing...")
        zip_buffer, errors, converted_count = convert_folder(uploaded_items, progress)

        if converted_count == 0:
            progress.empty()
            st.error("No files were converted. Check the error messages below.")
        else:
            progress.progress(100, text="Completed!")
            st.success(f"Converted {converted_count} file(s) successfully.")
            st.download_button(
                "Download ZIP of Markdown files",
                data=zip_buffer.getvalue(),
                file_name="docling_markdown_outputs.zip",
                mime="application/zip",
            )

        if errors:
            st.warning("\n".join(errors))

st.markdown("---")
st.caption("Powered by Streamlit and Docling ⚙️")
