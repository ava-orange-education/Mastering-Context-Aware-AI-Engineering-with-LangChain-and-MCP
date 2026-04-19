"""
Setup configuration for Chapter 11 case studies
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chapter-11-case-studies",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-world case studies of context-aware AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chapter-11-case-studies",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "anthropic>=0.18.1",
        "langchain>=0.1.6",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.3",
        "pinecone-client>=3.0.3",
        "weaviate-client>=4.4.0",
    ],
    extras_require={
        "healthcare": [
            "hl7apy>=1.3.4",
            "fhir.resources>=7.1.0",
        ],
        "enterprise": [
            "python-docx>=1.1.0",
            "python-pptx>=0.6.23",
            "openpyxl>=3.1.2",
            "PyPDF2>=3.0.1",
        ],
        "devops": [
            "prometheus-client>=0.19.0",
            "kubernetes>=28.1.0",
            "elasticsearch>=8.11.1",
        ],
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "flake8>=7.0.0",
        ],
    },
)