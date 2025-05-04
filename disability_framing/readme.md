# IEP Disability Framing Analysis Tool

This repository contains Python tools for analyzing how disabilities are framed in Individualized Education Programs (IEPs). The analysis focuses on identifying patterns in strengths-based versus deficit-based language across different disability categories and demographic factors.

## Overview

The project provides two implementation options:
1. A standard Python implementation using NLP libraries
2. An enhanced version leveraging the Claude API for deeper semantic analysis

Both versions analyze disability framing patterns, language differences across demographic groups, and generate visualizations to highlight key findings.

## Features

### Document Processing
- Load and parse IEP documents from a folder
- Extract structured sections (present levels, student needs, goals, accommodations)
- Identify disability categories and demographics mentioned in documents

### Linguistic Analysis
- Compare strengths-based vs. deficit-based language patterns
- Identify distinctive language by disability type
- Analyze how accommodations are framed (mitigation vs. enhancement)

### Comparative Analysis
- Compare language patterns across disability categories
- Analyze patterns by demographic factors (gender, race, religion)
- Generate visualizations for clear pattern identification

### Report Generation
- Create comprehensive reports of findings
- Generate visualizations of language patterns
- Provide actionable insights for improving IEP language

## Implementation Options

### Standard Python Version
- Uses NLP libraries (spaCy, NLTK) for linguistic analysis
- Counts strengths-based and deficit-based language using predefined keyword lists
- Extracts key themes using TF-IDF analysis
- Generates visualizations of language patterns
- Performs detailed demographic comparisons
- No external API dependencies, but more limited semantic understanding

### Claude API Version
- Uses Claude for deeper semantic analysis
- Asks Claude to identify strengths-based and deficit-based language contextually
- Performs conceptual model analysis (medical vs. social models of disability)
- Analyzes agency attribution and goal orientation
- Generates comprehensive reports with recommendations
- Requires API key and has cost implications, but provides deeper insights

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages:
  ```
  pip install pandas nltk spacy wordcloud matplotlib seaborn scikit-learn
  ```
- For the spaCy model:
  ```
  python -m spacy download en_core_web_md
  ```
- For the Claude API version:
  ```
  pip install anthropic
  ```

### Usage

1. Organize your IEP documents in a folder
2. Update the path in the main section of the script
3. For the Claude API version, add your API key
4. Run the script to perform the analysis:
   ```
   python iep_analyzer.py
   ```
5. Review the generated visualizations and reports

## Example Output

The analysis generates:
- Visualizations comparing strengths vs. deficits across disability categories
- Word clouds showing distinctive language for each disability type
- Comparative charts showing language patterns by demographic factors
- A comprehensive report with key findings and recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool was developed to help educators and policymakers identify and address potential biases in how disabilities are framed in educational documents.
- The analysis methodology draws from disability studies, critical discourse analysis, and educational equity research.