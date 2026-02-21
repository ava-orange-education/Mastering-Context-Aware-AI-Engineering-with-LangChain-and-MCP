"""
Document processing utilities.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import PyPDF2
from PIL import Image
import pytesseract


class DocumentProcessor:
    """Process various document formats"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        
        return '\n\n'.join(text)
    
    @staticmethod
    def extract_text_from_image(image_path: str, lang: str = 'eng') -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image
            lang: Language code
            
        Returns:
            Extracted text
        """
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text
    
    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        from docx import Document
        
        doc = Document(docx_path)
        text = []
        
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        
        return '\n'.join(text)
    
    @staticmethod
    def process_document(file_path: str) -> Dict[str, Any]:
        """
        Process document and extract information
        
        Args:
            file_path: Path to document
            
        Returns:
            Document information
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        result = {
            'filename': path.name,
            'extension': extension,
            'text': None,
            'error': None
        }
        
        try:
            if extension == '.pdf':
                result['text'] = DocumentProcessor.extract_text_from_pdf(file_path)
            elif extension == '.docx':
                result['text'] = DocumentProcessor.extract_text_from_docx(file_path)
            elif extension in ['.png', '.jpg', '.jpeg']:
                result['text'] = DocumentProcessor.extract_text_from_image(file_path)
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    result['text'] = f.read()
            else:
                result['error'] = f"Unsupported file type: {extension}"
        
        except Exception as e:
            result['error'] = str(e)
        
        return result