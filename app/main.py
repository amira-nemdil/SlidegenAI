from os import name
from app.pdf_reader import extract_text_from_pdf
from app.utils import clean_text
from transformers import pipeline

print("Loading grammar correction model...")
grammar_correction = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

def correct_grammar(text):
    # Optional: break long text into smaller chunks if needed
    max_chunk_length = 512
    sentences = text.split('. ')
    chunks = []
    chunk = ''

    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chunk_length:
            chunk += sentence + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '. '
    if chunk:
        chunks.append(chunk.strip())

    corrected = []
    for chunk in chunks:
        result = grammar_correction(chunk, max_length=512)[0]['generated_text']
        corrected.append(result)

    return ' '.join(corrected)

def main():
    source = input("Enter 'pdf' to upload PDF or 'prompt' to enter a prompt: ").strip().lower()

    if source == 'pdf':
        path = input("Enter path to PDF file: ").strip()
        raw_text = extract_text_from_pdf(path)
    elif source == 'prompt':
        raw_text = input("Enter your prompt: ").strip()
    else:
        print("Invalid input.")
        return

    cleaned = clean_text(raw_text)
    improved = correct_grammar(cleaned)

    print("\nImproved Text Preview:\n", improved[:1000])

if name == "main":
    main()
from app.pdf_reader import extract_text_from_pdf
from app.utils import clean_text
from transformers import pipeline
