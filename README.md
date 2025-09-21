# SlidegenAI - AI-Powered Presentation Generator

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

SlidegenAI is an intelligent system that automatically generates structured presentation content from text prompts or PDF documents, focusing on content quality and contextual coherence.

## ✨ Key Features
- **Multi-Input Support**: Accepts both text prompts and PDF documents
- **Context-Aware Generation**: Maintains logical flow through semantic understanding
- **Flexible Output**:
  - Automatic slide count determination
  - Custom slide number specification
  - JSON/text outputs (PPTX support coming soon)
- **Advanced Processing**:
  - PDF text extraction with PyMuPDF/pdfminer
  - NLP-powered content analysis using Transformers
  - Semantic segmentation and summarization

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
git clone https://github.com/amira-nemdil/SlidegenAI.git
cd SlidegenAI
python -m venv venv
source venv/bin/activate  # Linux/MacOS
pip install -r requirements.txt
```

## 🛠️ Usage

### Basic CLI Usage
```bash
# From text prompt
python slidegen.py --prompt "Quantum computing fundamentals" --slides 5

# From PDF document
python slidegen.py --pdf input.pdf --output presentation.json
```

### API Integration
```python
from slidegen import SlideGenerator

generator = SlideGenerator()
presentation = generator.generate(
    source="prompt", 
    content="Renewable energy trends 2025",
    slides=7
)
```

## 📦 Output Structure
```json
{
  "metadata": {
    "total_slides": 5,
    "generated_at": "2025-09-21T02:48:02Z"
  },
  "slides": [
    {
      "slide_number": 1,
      "title": "Introduction to Quantum Computing",
      "content": {
        "bullet_points": [
          "Definition and basic principles",
          "Historical development timeline"
        ],
        "summary": "Fundamental concepts..."
      }
    }
  ]
}
```

## 🧠 Technology Stack
- **Core Language**: Python 3.8+
- **NLP Engine**: Hugging Face Transformers
- **PDF Processing**: PyMuPDF/pdfminer
- **ML Support**: Scikit-learn
- **Optional Integration**: OpenAI/Cohere APIs

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License
[License details to be added - check back soon]

## 🙏 Acknowledgments
- Hugging Face for Transformers library
- PyMuPDF developers
- OpenAI/Cohere API teams logic, models, data, notebooks for experimentation, and output directories.
https://www.researchgate.net/publication/380553365_Presentify_Automated_Presentation_Slide_Generation_from_Research_Papers_using_NLP_and_Deep_Learning_May_2024
