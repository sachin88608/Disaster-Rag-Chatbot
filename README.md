# Disaster Management RAG System: Thesis Experiments Repository

## Overview
This repository is part of a thesis project focused on building and evaluating a Retrieval-Augmented Generation (RAG) system for disaster management. The project leverages Large Language Models (LLMs) and vector databases to answer disaster-related questions using real-world datasets. The goal is to compare different model combinations and techniques to identify the most effective approach for accurate and relevant information retrieval in emergency scenarios.

## Project Idea
The core idea of this thesis is to use advanced AI models to automate the process of answering disaster management questions by combining document retrieval with generative language models. By processing diverse datasets (earthquake, flood, tsunami, and covid-19 Pandemic), converting them into vector representations, and integrating them with LLMs, the system can provide grounded, context-aware answers. The project systematically tests various embedding and LLM model combinations to find the best setup for real-world disaster response applications.

## Repository Structure
This repository is organized to provide a clear and detailed view of the codebase and experiment results. Below is an explanation of the main files and folders:

- **config.py**
  - Stores all configuration settings, including model names, API keys, data sources, and vector database parameters.

- **setup_data.py**
  - Prepares and processes raw disaster data. Converts documents into embeddings and stores them in the vector database (ChromaDB).

- **data_ingestion.py**
  - Handles reading and extracting content from various data formats (PDF, CSV, Excel, text, and web pages). Cleans and standardizes the data for further processing.

- **vector_store.py**
  - Manages the storage and retrieval of document embeddings using ChromaDB. Handles chunking, embedding creation, and similarity search.

- **rag_system.py**
  - The core logic for the Retrieval-Augmented Generation system. Orchestrates data ingestion, retrieval, and LLM-based answer generation.

- **llm_interface.py**
  - Provides an interface to interact with Large Language Models (LLMs) via the GROQ API. Handles prompt creation and response parsing.

- **ragas_evaluation_fixed.py**
  - Implements the evaluation logic using the RAGAS framework. Calculates metrics like faithfulness, answer relevancy, and context recall for system outputs.

- **run_single_experiment.py**
  - Allows you to run a single experiment with a chosen combination of embedding and LLM models. Saves evaluation results for each run.

- **streamlit_app.py**
  - Launches the Streamlit web interface for interactive Q&A. Users can ask disaster management questions and get AI-powered answers with source citations.

- **data/**
  - Directory containing all raw disaster datasets (earthquake, flood, tsunami, etc.) in various formats.

- **batch_experiment_results/**
  - Folder where experiment results are saved as JSON files for later analysis and comparison.

- **README.md**
  - This file. Provides instructions, project overview, and a detailed file structure explanation.

---

## Experimental Context and Objectives
The experiments in this repository are designed to achieve several key objectives:

- **Comparative Analysis:** Evaluate and compare the effectiveness of different embedding and LLM model combinations for disaster management Q&A.
- **Real-World Data:** Use actual disaster datasets (earthquake, flood, tsunami, and Covid-19) to ensure practical relevance.
- **Performance Metrics:** Assess system performance using metrics such as faithfulness, answer relevancy, and context recall (via RAGAS).
- **Scalability Testing:** Test the system's ability to handle large and diverse datasets.
- **Best Practices:** Identify optimal configurations and techniques for integrating RAG systems into disaster response workflows.

## How to Use This Repository
- **Prepare the Data:** Place your disaster datasets in the `data/` folder as described in the configuration.
- **Set Configuration:** Edit `config.py` to specify data sources and model settings.
- **Run Data Setup:** Execute `python setup_data.py` to process and store vectors in the database.
- **Run Experiments:** Use `python run_single_experiment.py` to evaluate different model combinations. Results are saved in `batch_experiment_results/`.
- **Launch the UI:** Start the Streamlit app with `streamlit run streamlit_app.py` for interactive Q&A and testing.
- **Review Results:** Analyze the JSON files in `batch_experiment_results/` to compare experiment outcomes.

## Disclaimer
This repository is provided for academic and research purposes only. The results and conclusions are based on specific models, datasets, and techniques as described in the thesis. While every effort has been made to ensure accuracy, results may vary depending on data and model updates. Users should apply the information in this repository at their own discretion and risk.

## Copyright
© 2025 Sachin Gupta. All rights reserved. Unauthorized use of the materials in this repository without permission is strictly prohibited.