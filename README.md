# SlidegenAI
Develop a system that generates presentation slides from a user-provided prompt or PDF document. The user may optionally specify the number of slides; if not, the system determines an appropriate structure automatically. The focus is on content quality and contextual coherence, not visual design.

System Components & Flow

Input Handling

Accepts either a text prompt or a PDF file.

Optionally accepts a specified number of slides.

Extracts text from PDFs using a text extraction module.

Natural Language Processing (NLP)

Processes and understands the input using a pre-trained NLP model.

Applies summarization and segmentation logic to break the content into coherent slide sections.

Performs semantic understanding to ensure content fidelity and logical flow.

Slide Content Generation

Generates structured content for each slide, including titles, bullet points, and short descriptions.

Respects the slide count if provided, otherwise dynamically determines the number of slides.

Output Formatting

For prototyping, outputs text-based slides as JSON or plain text.

Later stages may involve converting to formats such as .pptx using libraries like python-pptx.

Model Usage and Training

The system will initially use pre-trained NLP models for summarization and text generation.

A custom fine-tuned model may be trained later to improve domain-specific performance or better adapt to presentation structures.

A dataset of documents and human-written slide content will be collected and cleaned before any training is conducted.

Technologies and Tools

Python as the main programming language.

VS Code as the development environment.

External libraries: Transformers (Hugging Face), PyMuPDF or pdfminer (for PDF reading), Scikit-learn, and possibly OpenAI or Cohere APIs.

Folder Structure
Clearly separated into components such as app logic, models, data, notebooks for experimentation, and output directories.
https://www.researchgate.net/publication/380553365_Presentify_Automated_Presentation_Slide_Generation_from_Research_Papers_using_NLP_and_Deep_Learning_May_2024
