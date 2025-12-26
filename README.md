# langaug

**langaug** is a Python library designed for text augmentation using Large Language Models (LLMs), with a specific focus on optimizing workflows for low-resource languages. It provides a modular, object-oriented framework to build intelligent sampling and transformation pipelines for dataset enhancement.

## Key Features

- **LLM-Powered Augmentation:** Leverage state-of-the-art LLMs to generate high-quality synthetic text.
- **Low-Resource Language Optimization:** Tailored strategies for languages with limited training data.
- **Modular Architecture:** Extensible components for services, datasets, transforms, and pipelines.
- **Strict Data Validation:** Built on Pydantic v2 for robust schema enforcement.
- **Observability:** Integrated with Langfuse for tracing and monitoring LLM interactions.
- **Modern Python Tooling:** Powered by `uv` for fast, reliable dependency management.

## Tech Stack

- **Language:** Python 3.13.11
- **Package Manager:** [uv](https://github.com/astral-sh/uv)
- **LLM Orchestration:** [LangChain](https://github.com/langchain-ai/langchain)
- **Data Validation:** [Pydantic v2](https://docs.pydantic.dev/)
- **Observability:** [Langfuse](https://langfuse.com/)

## Architecture Overview

The library is structured around several core concepts:

- **LLM Services:** Singleton wrappers for LangChain clients with built-in observability.
- **Datasets:** Unified containers for data ingestion, validation, and serialization.
- **Transforms:** Atomic units of augmentation that render prompts and process LLM outputs.
- **Pipelines:** Chains of transforms that execute sequential augmentation steps.
- **Samplers:** Logic for selecting specific subsets of data for augmentation.
- **Augmentors:** High-level orchestrators that combine datasets, samplers, and pipelines.

## Getting Started

### Prerequisites

- Python 3.13.11
- `uv` installed on your system

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd teaug
   ```

2. Initialize the environment and install dependencies:
   ```bash
   uv sync
   ```

3. Install the package in editable mode for development:
   ```bash
   uv pip install -e .
   ```

## Development Workflow

- **Environment Management:** Use `uv` for all package and environment operations.
- **Configuration:** Manage settings via `.env` files and Pydantic models.
- **Testing:** Use the `lab/` directory for experimentation and playground scripts.
- **Observability:** Ensure Langfuse is configured to track all LLM calls during development.

---

*Note: This project follows strict coding standards including modern type hinting and structured logging.*
