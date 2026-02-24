"""
Setup script for Chapter 6: Data Engineering for Context-Aware AI
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh 
        if line.strip() and not line.startswith("#") and not line.startswith("--")
    ]

# Version
VERSION = "1.0.0"

setup(
    name="data-engineering-context-ai",
    version=VERSION,
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready data engineering framework for context-aware AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ava-orange-education/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP",
    project_urls={
        "Bug Tracker": "https://github.com/ava-orange-education/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/issues",
        "Documentation": "https://github.com/ava-orange-education/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/tree/main/chapter-6-data-engineering/docs",
        "Source Code": "https://github.com/ava-orange-education/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/tree/main/chapter-6-data-engineering",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "ipython>=8.14.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "airflow": [
            "apache-airflow>=2.7.0",
            "apache-airflow-providers-postgres>=5.6.0",
            "apache-airflow-providers-amazon>=8.4.0",
        ],
        "advanced": [
            "faiss-cpu>=1.7.4",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-pipeline=pipeline.orchestrator:main",
            "quality-check=monitoring.quality_monitor:main",
            "health-check=monitoring.pipeline_monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
        "configs": ["*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "data-engineering",
        "context-aware-ai",
        "vector-database",
        "knowledge-graph",
        "embedding",
        "rag",
        "retrieval",
        "nlp",
        "machine-learning",
    ],
)