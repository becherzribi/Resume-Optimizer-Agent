# 🧾 Resume Optimizer Agent - Enhanced

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Replace with your actual license -->

An intelligent, Streamlit-based application designed to analyze and optimize resumes against job descriptions using local Language Models (LLMs) via Ollama, advanced semantic search, and comprehensive text analysis techniques. This enhanced version incorporates robust features for improved accuracy, user experience, security, and deployment.

---

## ✨ Key Features

*   **📄 PDF Parsing**: Extracts text accurately from PDF resumes and job descriptions using PyMuPDF.
*   **🔍 Multi-Faceted Analysis**:
    *   **Keyword Matching**: Identifies critical keywords from the job description missing in the resume.
    *   **Token Similarity**: Calculates a basic similarity score based on word overlap.
    *   **Date Formatting**: Detects inconsistencies and potential issues in date formats.
    *   **Semantic Search**: Uses Sentence Transformers and FAISS (GPU-accelerated if available) to find semantically similar sentences between the resume and job description, understanding context beyond keywords.
*   **🤖 AI-Powered Feedback (via Local LLMs)**:
    *   Integrates with multiple Ollama models (Mistral, Llama 3.1, DeepSeek, Gemma, CodeLlama).
    *   Provides structured feedback based on an enhanced prompt, covering:
        *   ATS Compatibility & Keyword Alignment
        *   Content Relevance & Impact (Quantifiable Results)
        *   Action Verbs & Language Clarity
        *   Formatting & Consistency
        *   Overall Summary & Key Recommendation
*   **🎯 Targeted Optimization**: Allows specifying a target job title/industry and language for tailored analysis.
*   **🔄 Iterative Refinement**: Users can edit the AI-generated feedback before proceeding.
*   **✍️ Automated Resume Generation**: Rewrites the resume incorporating the (edited) feedback using an LLM, preserving original structure, and generates a downloadable PDF via ReportLab.
*   **🔒 Secure Authentication**: User registration and login system with secure password hashing (bcrypt via passlib).
*   **💾 Analysis History**: Saves past analyses linked to user accounts in an SQLite database.
*   **⚡ Performance & Caching**: Caches semantic indexes for faster subsequent analyses of the same job description. Optimized for GPU usage (FAISS & Ollama).
*   **⚙️ Configuration**: Uses a `.env` file for managing database paths, cache locations, logging levels, and other settings.
*   **📝 Structured Logging**: Implements detailed logging using `structlog` for better monitoring and debugging.
*   **🐳 Dockerized Deployment**: Includes a comprehensive Dockerfile for easy, reproducible deployment with GPU support.

---

## 🛠️ Tech Stack

| Category                | Tool / Library                                      |
| ----------------------- | --------------------------------------------------- |
| **Web Framework**       | Streamlit                                           |
| **LLM Backend**         | Ollama                                              |
| **LLM Orchestration**   | Langchain Community                                 |
| **PDF Processing**      | PyMuPDF (`fitz`), ReportLab                         |
| **Semantic Search**     | `sentence-transformers`, FAISS (CPU/GPU)            |
| **Text Analysis**       | `scikit-learn`, Python `re`                         |
| **Password Hashing**    | `passlib[bcrypt]`                                   |
| **Configuration**       | `python-dotenv`                                     |
| **Logging**             | `structlog`, Python `logging`                       |
| **Database**            | SQLite3                                             |
| **Containerization**    | Docker                                              |
| **Core Language**       | Python 3.10+                                        |

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python**: Version 3.10 or higher.
2.  **Pip**: Python package installer (usually comes with Python).
3.  **Git**: For cloning the repository.
4.  **Ollama**: Install Ollama from [ollama.com](https://ollama.com/) and ensure the Ollama service is running.
5.  **Docker**: (Required for Docker deployment) Install Docker Engine or Docker Desktop.
6.  **(Optional - For GPU Acceleration)**:
    *   NVIDIA GPU with CUDA support.
    *   Appropriate NVIDIA drivers.
    *   NVIDIA Container Toolkit (for Docker GPU support).

---

## 🚀 Local Installation & Launch

Ideal for development and testing on your personal machine.

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd resume-optimizer-agent # Or your repository name
    ```

2.  **Rename Key Files** (if using the enhanced versions directly):
    ```bash
    mv enhanced_app.py app.py
    mv enhanced_resume_review.txt prompts/enhanced_resume_review.txt
    # Rename other enhanced files if necessary
    ```

3.  **Create and Activate Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note*: This installs `faiss-cpu` by default. For local GPU acceleration with FAISS, you might need to manually install `faiss-gpu` after meeting CUDA prerequisites.

5.  **Configure Environment**:
    *   Copy or create a `.env` file in the project root.
    *   Update `DATABASE_NAME`, `CACHE_DIR`, `LOG_LEVEL`, and **generate a strong `SECRET_KEY`**.
    ```bash
    # Example .env content:
    DATABASE_NAME="data/resume_optimizer.db"
    CACHE_DIR="cache/semantic_indexes"
    DEFAULT_LLM_MODEL="mistral:latest"
    SECRET_KEY="your_strong_random_32_byte_hex_secret"
    LOG_LEVEL="INFO"
    ```
    *   Create the data directory if specified: `mkdir data`

6.  **Download Ollama Models**:
    Ensure the Ollama service is running.
    ```bash
    ollama pull mistral:latest
    ollama pull llama3.1:latest
    ollama pull deepseek-r1:1.5b
    ollama pull gemma:7b
    ollama pull codellama:7b
    # Add any other models you intend to use
    ```

7.  **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

8.  **Access the App**: Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

---

## 🐳 Docker Deployment

Deploying with Docker provides a consistent and isolated environment. See the detailed guide for comprehensive instructions:

➡️ **[Docker Deployment Guide](./DOCKER_DEPLOYMENT.md)**

**Quick Start:**

1.  Ensure prerequisites (Docker, NVIDIA Container Toolkit for GPU) are met.
2.  Prepare project files (rename `enhanced_app.py` to `app.py`, configure `.env`).
3.  Build the image: `docker build -t resume-optimizer:latest .`
4.  Run the container (example with GPU and persistent data):
    ```bash
    docker run --gpus all \
      -p 8501:8501 \
      -p 11434:11434 \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/cache:/app/cache \
      -v ollama_models:/root/.ollama \
      -v $(pwd)/.env:/app/.env \
      --name resume-optimizer-app \
      --restart unless-stopped \
      -d resume-optimizer:latest
    ```
5.  Access the app at `http://localhost:8501`.

---

## 📖 Usage Guide

1.  **Authentication**: Access the application URL. You will be prompted to either **Login** or **Register** a new account. User accounts are required to save and view analysis history.
2.  **Navigation**: Use the sidebar to switch between "🚀 New Analysis" and "📊 Past Analyses".
3.  **New Analysis Page**:
    *   **Upload Files**: Upload your resume (PDF) and the target job description (PDF or TXT).
    *   **Select LLM Model**: Choose the AI model for analysis (e.g., Mistral, Llama 3.1). Recommendations based on the target role are provided.
    *   **Target Role (Optional)**: Specify the job title or industry (e.g., "Software Engineer", "Marketing Manager") to get more focused feedback.
    *   **Language**: Select the primary language of your documents.
    *   **Analyze**: Click the "✨ Analyze Resume" button.
4.  **Review Results**: The analysis results are displayed in tabs:
    *   **📊 Overview**: High-level metrics (similarity score, keyword count, date issues) and action verb suggestions.
    *   **🔤 Keywords**: List of missing keywords and a highlighted view of the resume text.
    *   **📅 Formatting**: Details on potential date formatting inconsistencies.
    *   **🎯 Semantic Matches**: Shows resume sentences semantically similar to job description requirements.
    *   **🤖 AI Feedback**: Detailed, structured feedback generated by the selected LLM based on the enhanced prompt.
5.  **Refine Feedback**: In the "AI Feedback" tab, you can edit the AI's suggestions in the text area before proceeding.
6.  **Save Analysis**: Click "💾 Save Analysis Results" to store the current analysis (including any edited feedback) in your history.
7.  **Generate Updated Resume**: Click "🚀 Generate PDF" under the Actions section. The application will:
    *   Use the selected LLM to rewrite the resume based on the (potentially edited) feedback.
    *   Generate a PDF version of the rewritten resume.
    *   Provide a "⬇️ Download Updated Resume PDF" button.
8.  **Past Analyses**: Navigate to this page to view a list of your previously saved analyses.

---

## ⚙️ Configuration

The application uses a `.env` file in the project root for configuration:

*   `DATABASE_NAME`: Path to the SQLite database file (e.g., `data/resume_optimizer.db`).
*   `CACHE_DIR`: Directory for storing semantic index caches.
*   `DEFAULT_LLM_MODEL`: Fallback LLM model.
*   `SECRET_KEY`: **Crucial for security.** A strong, random hexadecimal string (e.g., generate with `python -c 'import secrets; print(secrets.token_hex(32))'`). Used for password hashing context, potentially other features later.
*   `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

Remember to restart the application (or Docker container) after modifying the `.env` file.

---

## 🔒 Security Considerations

*   **Password Hashing**: Uses `bcrypt` via `passlib` for secure password storage.
*   **Secret Key**: Ensure the `SECRET_KEY` in `.env` is kept confidential and is sufficiently random.
*   **Input Sanitization**: While PDF parsing is generally safe, be mindful of potential risks if handling untrusted files.
*   **Dependencies**: Keep dependencies updated to patch potential vulnerabilities (`pip list --outdated`, `pip install -U <package>`).
*   **Docker Security**: Follow Docker security best practices, especially regarding volume mounts and network exposure in production environments.

---

## 🧪 Testing

A basic testing guide is provided:

➡️ **[Testing Guide](./TESTING.md)**

*(Currently focuses on manual end-to-end testing. Automated tests using `pytest` are recommended for future development.)*

---

## 🤝 Contribution

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure code follows best practices (linting, formatting - e.g., using Black).
5.  Add tests for new features if possible.
6.  Update documentation (README, guides) if necessary.
7.  Commit your changes (`git commit -m 'Add some feature'`).
8.  Push to the branch (`git push origin feature/your-feature-name`).
9.  Open a Pull Request.

Please open an issue first to discuss significant changes.

---

## 📄 License

This project is licensed under the [YOUR LICENSE HERE] License - see the `LICENSE` file for details. (e.g., MIT License)

---

*This README provides a comprehensive overview. For specific deployment steps, refer to the [Docker Deployment Guide](./DOCKER_DEPLOYMENT.md).*

