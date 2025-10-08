import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from rich import print

# --- Setup and Initialization ---

# NOTE: The Docling library must be installed before running this script:
# Run: pip install 'docling[vlm]'

# 1. Define the output directory and ensure it exists, as requested.
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' ensured to exist.")


# 2. Define Pydantic models for structured data extraction.
class Contact(BaseModel):
    """Represents contact details for the sender or receiver."""

    name: Optional[str] = Field(default=None, examples=['Smith'])
    address: str = Field(default='123 Main St', examples=['456 Elm St'])
    postal_code: str = Field(default='12345', examples=['67890'])
    city: str = Field(default='Anytown', examples=['Othertown'])
    country: Optional[str] = Field(default=None, examples=['Canada'])


class Invoice(BaseModel):
    """A simple model matching the initial extraction template."""

    bill_no: str
    total: float


class ExtendedInvoice(BaseModel):
    """A complex model for deeper, nested extraction."""

    bill_no: str = Field(examples=['A123', '5414'])  # provides examples to guide the model
    total: float = Field(default=10, examples=[20])  # provides default and examples
    garden_work_hours: int = Field(default=1, examples=[2])
    sender: Contact = Field(default=Contact(), examples=[Contact()])
    receiver: Contact = Field(default=Contact(), examples=[Contact()])


# --- Docling Core Logic ---

# We use the remote URL provided in the original request as the source document.
# (To use a local file from "./input", you would change this variable.)
file_path = 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg'

from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor

print(f'\n--- Starting Document Extraction from: {file_path} ---')

# Initialize the DocumentExtractor
extractor = DocumentExtractor(allowed_formats=[InputFormat.IMAGE, InputFormat.PDF])


# --- 1. Extraction using JSON string template ---
print('\n[bold yellow]--- 1. Extracting with JSON string template ---[/bold yellow]')
result_json_str = extractor.extract(
    source=file_path,
    template='{"bill_no": "string", "total": "float"}',
)
print('Extracted Pages Data:')
print(result_json_str.pages[0].extracted_data)


# --- 2. Extraction using Python dict template ---
print('\n[bold yellow]--- 2. Extracting with Python dict template ---[/bold yellow]')
result_dict = extractor.extract(
    source=file_path,
    template={
        'bill_no': 'string',
        'total': 'float',
    },
)
print('Extracted Pages Data:')
print(result_dict.pages[0].extracted_data)


# --- 3. Extraction using simple Pydantic model (Invoice) ---
print('\n[bold yellow]--- 3. Extracting with simple Pydantic model (Invoice) ---[/bold yellow]')
result_invoice = extractor.extract(
    source=file_path,
    template=Invoice,
)
print('Extracted Pages Data:')
print(result_invoice.pages[0].extracted_data)


# --- 4. Extraction using complex Pydantic model (ExtendedInvoice) ---
print('\n[bold yellow]--- 4. Extracting with complex Pydantic model (ExtendedInvoice) ---[/bold yellow]')
result_extended = extractor.extract(
    source=file_path,
    template=ExtendedInvoice,
)
# Print the extracted data (raw dictionary from the Docling result)
print('Extracted Pages Data:')
print(result_extended.pages[0].extracted_data)


# --- 5. Pydantic Validation and Final Output ---
# Validate the extracted data against the Pydantic model for type checking and structure
invoice = ExtendedInvoice.model_validate(result_extended.pages[0].extracted_data)

print('\n[bold green]--- Pydantic Validation Result (Model Instance) ---[/bold green]')
print(invoice)

# Print specific fields using the validated model
print('\n[bold cyan]--- Formatted Output from Model ---[/bold cyan]')
print(
    f'Invoice [bold magenta]#{invoice.bill_no}[/bold magenta] was sent by [bold magenta]{invoice.sender.name}[/bold magenta] '
    f'to [bold magenta]{invoice.receiver.name}[/bold magenta] at {invoice.sender.address}.'
)

# --- 6. Write Result to Output Folder ---
output_file = os.path.join(output_dir, 'extracted_invoice.json')

# Convert the Pydantic model instance to a JSON dictionary
# We use .model_dump_json() and then load/dump to ensure a clean, formatted JSON file
json_data = json.loads(invoice.model_dump_json(indent=2))
with open(output_file, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f'\n[bold green]Successfully saved structured output to:[/bold green] [yellow]{output_file}[/yellow]')
