# Project Context: langaug (Text Augmentation Library)

**langaug** is a Python package for facilitating text augmentation using Large Language Models (LLMs), specifically optimized for low-resource languages. It employs a modular, object-oriented architecture to augment datasets via intelligent sampling and transformation pipelines.

## Tech Stack & Dependencies

- **Language:** Python 3.13.11 (Pin via `uv python pin 3.13.11`).
- **Package Manager:** `uv`.
- **LLM Orchestration:** LangChain.
- **Validation:** Pydantic v2 (Strict usage for data modeling).
- **Logging:** Loguru (No `print` statements).
- **Observability:** Langfuse (Integration required for all LLM services).

## Project Structure

- **Core Source:** `src/langaug`
- **Prompts:** `src/langaug/data/prompts` (Format: `{prompt_name}.jinja2`).
- **Playground/Notebooks:** `lab/`
- **Base Data:** `data/base`
- **Generated Outputs:** `data/output`

---

## Architecture & Implementation Patterns

The project follows a strict Object-Oriented design. Components are designed for high extensibility.

### 1. LLMService
* **Role:** Singleton wrapper for LangChain LLM clients.
* **Behavior:**
    * Define once, invoke multiple times.
    * Must support `kwargs` configuration (e.g., `temperature`, `structured_output`).
    * Must integrate Langfuse callbacks automatically.
* **Interface:** specific generic interface for creating concrete LLM services (e.g., OpenAI, Anthropic).

### 2. Datasets
* **Role:** Unified container for managing data ingestion and structure.
* **Core Components:**
    * **Schema:** Every dataset must have a strict `Pydantic` model definition for its records.
    * **Loader:** Abstract interface for ingesting raw data (CSV, JSONL, HuggingFace) into the schema.
* **Behavior:**
    * **Immutability:** Base datasets should remain immutable. Augmentations produce derived dataset objects.
    * **Serialization:** Built-in support for exporting records to JSONL or arrow formats.
    * **Validation:** Enforces schema compliance upon loading.

### 3. Transforms (Atomic Units)
* **Role:** Performs a single augmentation task.
* **Inputs:**
    * `Input Schema`: Pydantic model representing the dataset record.
    * `Output Schema`: Pydantic model enforcing the LLM response structure (updates/adds fields to input).
    * `LLMService`: Optional. If `None`, perform deterministic/hard-coded logic.
    * `Prompt`: Jinja2 template. Must support raw string, relative file path, or package prompt ID.
* **Logic:** Render Prompt (using Input) -> Invoke LLM -> Parse to Output Schema -> Merge with Input.

### 4. Pipelines
* **Role:** Chain of Responsibility.
* **Logic:** Execute a sequence of `Transforms`.
* **Validation:** Ensure `Output Schema` of Transform $N$ matches `Input Schema` of Transform $N+1$.
* **Visualization:** Include Graphviz utilities to visualize the pipeline DAG.

### 5. Samplers
* **Role:** Selects data subsets for augmentation.
* **Inputs:** Dataset Object, Mapping (Dataset fields -> Sampling params), Count, Exclusions.
* **Flow:** `Pre-filter` -> `Sample` (Random/Lambda/Ordering) -> `Post-filter`.

### 6. Augmentors (Orchestrators)
* **Role:** High-level controller combining: Dataset + Pipeline + Sampler.
* **Workflow:**
    1.  **Sample:** Select records using the `Sampler`.
    2.  **Iterate:** Run the `Pipeline` on selected records (supports multiple iterations).
    3.  **Merge:** Add or replace records in the original dataset.
* **Features:**
    * **Preview:** Dry-run capability for a subset of records.
    * **Tracking:** Append metadata columns: `is_synthetic` (bool), `pipeline_id`, `sampler_id`.
    * **Reporting:** Generate comprehensive structured logs for benchmarking.

---

## Coding Standards

- **Comments:** DO NOT add docstrings or comments unless logic is highly complex and non-obvious.
- **Type Hinting:** Use modern Python 3.13+ syntax (e.g., `list[str]` instead of `List[str]`, `str | None` instead of `Optional[str]`). Annotate all inputs and outputs.
- **Output:** DO NOT use `print()`. Use `loguru` for all terminal outputs.
- **Logic:** Prefer standard library or installed package features over custom re-implementations. Explore documentation via MCP tools if unsure of latest syntax.
- **Configuration:** Use Pydantic models to manage settings. Load from `.env` (global) inspired from `.env.example`.

---

## Dev Environment & Workflow

### Setup & Installation
- **Init:** `uv init`.
- **Install Frozen:** `uv sync` (uses `pyproject.toml`).
- **Upgrade/Add:**
    1.  Add package name to `requirements.txt`.
    2.  Run `uv add -r requirements.txt`.
- **Dev Install:** Run `uv pip install -e .` in root to reflect changes in `lab/` scripts (allows `import langaug`).

### Testing & Iteration
- **Playground:** Use scripts in `lab/` to test logic. Import the library using `import langaug`.
- **Refactoring:** After changing source code, always re-run `uv pip install -e .` to update the environment reference.

---

## MCP Tooling & Documentation Strategy

This environment is equipped with specific Model Context Protocol (MCP) servers to fetch real-time documentation. **You MUST prioritize these tools over your internal training data** to avoid deprecated syntax.

### Tool Usage Rules

- **`uv-docs` (Usage: Mandatory)**
  - **Trigger:** Before running any package management command or adding dependencies.
  - **Goal:** Verify the correct `uv` CLI flags (e.g., usage of `uv add`, `uv sync`, `uv pip install`).
  - **Restriction:** Do not assume `pip` standard commands work 1:1 with `uv`.

- **`pydantic-docs` (Usage: High Priority)**
  - **Trigger:** When defining any `BaseModel`, `Field`, or validator.
  - **Goal:** Strict enforcement of **Pydantic V2** syntax.
  - **Anti-Pattern:** creating Pydantic V1 generic models or using `@validator` (deprecated) instead of `@field_validator`.

- **`langchain-docs` (Usage: High Priority)**
  - **Trigger:** When implementing `LLMService` or `Pipelines`.
  - **Goal:** Ensure usage of the latest **LCEL (LangChain Expression Language)** patterns.
  - **Anti-Pattern:** Using deprecated chains (e.g., `LLMChain`) instead of pipe syntax (`|`).

- **`langfuse-docs` (Usage: Mandatory)**
  - **Trigger:** When integrating observability or callbacks.
  - **Goal:** Fetch the specific integration pattern for the LangChain version being used.

### Workflow for Implementation
1. **Search First:** Before writing code for these libraries, query the respective tool.
   *Example: "Using `pydantic-docs`, check the syntax for computed fields in V2."*
2. **Verify:** If your internal knowledge conflicts with the tool output, trust the tool output.
