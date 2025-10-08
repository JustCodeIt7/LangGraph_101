import langextract as lx
import textwrap
from collections import Counter, defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define comprehensive prompt and examples for complex literary text
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships from the given text.

    Provide meaningful attributes for every entity to add context and depth.

    Important: Use exact text from the input for extraction_text. Do not paraphrase.
    Extract entities in order of appearance with no overlapping text spans.

    Note: In play scripts, speaker names appear in ALL-CAPS followed by a period.""")

examples = [
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            ROMEO. But soft! What light through yonder window breaks?
            It is the east, and Juliet is the sun.
            JULIET. O Romeo, Romeo! Wherefore art thou Romeo?"""),
        extractions=[
            lx.data.Extraction(
                extraction_class='character', extraction_text='ROMEO', attributes={'emotional_state': 'wonder'}
            ),
            lx.data.Extraction(
                extraction_class='emotion',
                extraction_text='But soft!',
                attributes={'feeling': 'gentle awe', 'character': 'Romeo'},
            ),
            lx.data.Extraction(
                extraction_class='relationship',
                extraction_text='Juliet is the sun',
                attributes={'type': 'metaphor', 'character_1': 'Romeo', 'character_2': 'Juliet'},
            ),
            lx.data.Extraction(
                extraction_class='character', extraction_text='JULIET', attributes={'emotional_state': 'yearning'}
            ),
            lx.data.Extraction(
                extraction_class='emotion',
                extraction_text='Wherefore art thou Romeo?',
                attributes={'feeling': 'longing question', 'character': 'Juliet'},
            ),
        ],
    )
]

# Process Romeo & Juliet directly from Project Gutenberg
print('Downloading and processing Romeo and Juliet from Project Gutenberg...')

result = lx.extract(
    text_or_documents='https://www.gutenberg.org/files/1513/1513-0.txt',
    prompt_description=prompt,
    examples=examples,
    model_id='gemini-2.5-flash-lite',
    extraction_passes=3,  # Multiple passes for improved recall
    max_workers=20,  # Parallel processing for speed
    max_char_buffer=1000,  # Smaller contexts for better accuracy
)

print(f'Extracted {len(result.extractions)} entities from {len(result.text):,} characters')

# Save and visualize the results
lx.io.save_annotated_documents([result], output_name='romeo_juliet_extractions.jsonl', output_dir='.')

# Generate the interactive visualization
html_content = lx.visualize('romeo_juliet_extractions.jsonl')
with open('romeo_juliet_visualization.html', 'w') as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)

print('Interactive visualization saved to romeo_juliet_visualization.html')

# Analyze character mentions
characters = {}
for e in result.extractions:
    if e.extraction_class == 'character':
        char_name = e.extraction_text
        if char_name not in characters:
            characters[char_name] = {'count': 0, 'attributes': set()}
        characters[char_name]['count'] += 1
        if e.attributes:
            for attr_key, attr_val in e.attributes.items():
                characters[char_name]['attributes'].add(f'{attr_key}: {attr_val}')

# Print character summary
print(f'\nCHARACTER SUMMARY ({len(characters)} unique characters)')
print('=' * 60)

sorted_chars = sorted(characters.items(), key=lambda x: x[1]['count'], reverse=True)
for char_name, char_data in sorted_chars[:10]:  # Top 10 characters
    attrs_preview = list(char_data['attributes'])[:3]
    attrs_str = f' ({", ".join(attrs_preview)})' if attrs_preview else ''
    print(f'{char_name}: {char_data["count"]} mentions{attrs_str}')

# Entity type breakdown
entity_counts = Counter(e.extraction_class for e in result.extractions)
print(f'\nENTITY TYPE BREAKDOWN')
print('=' * 60)
for entity_type, count in entity_counts.most_common():
    percentage = (count / len(result.extractions)) * 100
    print(f'{entity_type}: {count} ({percentage:.1f}%)')
