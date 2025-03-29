# TalentScout Hiring Assistant

## Description
AI-powered initial screening tool for technical candidates. Developed for TalentScout, a technology recruitment agency. Conducts automated interviews, gathers candidate information, and evaluates technical skills.

## Tech Stack
- Python
- Streamlit (for the user interface)
- Groq API (for language model interactions)

## Installation Instructions

### Prerequisites
- Python 3.8+
- Pip package installer

### Steps
1. Clone the repository (if applicable)
2. Install Python packages:
```bash
pip install streamlit groq python-dotenv
```
3. Set up Groq API Key:
   - Obtain a GROQ_API_KEY from Groq
   - Set the `GROQ_API_KEY` as an environment variable or in Streamlit secrets (`.streamlit/secrets.toml`)
   - For `.env` file, create a `.env` file in the project root and add:
     ```
     GROQ_API_KEY=your_groq_api_key
     ```

## Usage Instructions
1. Run the Streamlit application:
```bash
streamlit run app.py
```
2. Open the Streamlit app in your browser (usually at `http://localhost:8501`)
3. Interact with the TalentBot Pro as a candidate would in an initial screening interview

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
