import streamlit as st
from groq import Groq
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import re # Import regex library for better extraction

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="TalentScout Hiring Assistant",
    page_icon="🎯", # You can replace with a URL to a custom favicon: "https://your-domain.com/favicon.ico"
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="collapsed",
)

# --- Custom CSS Injection ---
# This CSS provides the "awesome" look and feel
# Replace the existing custom_css with this updated version:
# custom_css = """
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

#     :root {
#         --primary: #4361ee;
#         --primary-dark: #3a56d4;
#         --secondary: #3f37c9;
#         --accent: #4895ef;
#         --light: #f8f9fa;
#         --dark: #212529;
#         --success: #4cc9f0;
#         --warning: #f72585;
#         --text: #2b2d42;
#         --text-light: #8d99ae;
#         --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
#         --message-user: linear-gradient(135deg, var(--primary), var(--secondary));
#         --message-assistant: rgba(248, 249, 250, 0.95);
#     }

#     * {
#         transition: var(--transition);
#     }

#     body {
#         font-family: 'Inter', sans-serif;
#         background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
#         color: var(--text);
#         line-height: 1.6;
#     }

#     /* Main Container */
#     [data-testid="stAppViewContainer"] > .main .block-container {
#         max-width: 900px;
#         padding: 3rem 3rem 6rem;
#         margin: 2rem auto;
#         background: white;
#         border-radius: 16px;
#         box-shadow: var(--shadow);
#         position: relative;
#         overflow: hidden;
#     }

#     [data-testid="stAppViewContainer"] > .main .block-container::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 8px;
#         background: linear-gradient(90deg, var(--primary), var(--accent));
#     }

#     /* Header */
#     .title-container {
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         margin-bottom: 1rem;
#         flex-direction: column;
#         text-align: center;
#         gap: 0.5rem;
#     }

#     .title-container h1 {
#         font-family: 'Space Grotesk', sans-serif;
#         font-weight: 600;
#         font-size: 2.5rem;
#         margin: 0;
#         background: linear-gradient(90deg, var(--primary), var(--accent));
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         letter-spacing: -0.5px;
#     }

#     .stCaption {
#         color: var(--text-light);
#         font-size: 1rem;
#         margin-bottom: 2.5rem;
#         position: relative;
#     }

#     .stCaption::after {
#         content: '';
#         position: absolute;
#         bottom: -1rem;
#         left: 50%;
#         transform: translateX(-50%);
#         width: 100px;
#         height: 3px;
#         background: linear-gradient(90deg, var(--primary), var(--accent));
#         border-radius: 3px;
#     }

#     /* Chat Messages */
#     [data-testid="stChatMessage"] {
#         border-radius: 18px;
#         padding: 1.25rem 1.5rem;
#         margin-bottom: 1.25rem;
#         box-shadow: var(--shadow);
#         max-width: 85%;
#         position: relative;
#         opacity: 0;
#         animation: fadeIn 0.3s ease-out forwards;
#         transform-origin: center;
#         border: none;
#         backdrop-filter: blur(4px);
#     }

#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(10px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     /* Assistant Messages */
#     div.stChatMessage:has(span[data-testid="chatAvatarIcon-assistant"]) {
#         background: var(--message-assistant);
#         color: var(--text);
#         margin-right: auto;
#         border-bottom-left-radius: 4px;
#         border: 1px solid rgba(0, 0, 0, 0.05);
#     }

#     /* User Messages */
#     div.stChatMessage:has(span[data-testid="chatAvatarIcon-user"]) {
#         background: var(--message-user);
#         color: white;
#         margin-left: auto;
#         border-bottom-right-radius: 4px;
#     }

#     div.stChatMessage:has(span[data-testid="chatAvatarIcon-user"]) .stMarkdown p,
#     div.stChatMessage:has(span[data-testid="chatAvatarIcon-user"]) .stMarkdown li,
#     div.stChatMessage:has(span[data-testid="chatAvatarIcon-user"]) .stMarkdown {
#         color: white !important;
#     }

#     /* Message Container */
#     [data-testid="stChatMessageContainer"] {
#         background: transparent !important;
#         padding: 0 !important;
#         margin: 0 !important;
#     }

#     /* Chat Input Area */
#     [data-testid="stChatInputContainer"] {
#         background: rgba(255, 255, 255, 0.95) !important;
#         backdrop-filter: blur(10px);
#         border-top: 1px solid rgba(0, 0, 0, 0.08) !important;
#         padding: 1.5rem !important;
#         margin-top: 1rem;
#     }

#     [data-testid="stChatInput"] textarea {
#         background: white !important;
#         border: 2px solid rgba(67, 97, 238, 0.2) !important;
#         border-radius: 14px !important;
#         padding: 1rem 1.25rem !important;
#         color: var(--text) !important;
#         font-size: 0.95rem !important;
#         min-height: 60px !important;
#         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
#     }

#     [data-testid="stChatInput"] textarea:focus {
#         border-color: var(--primary) !important;
#         box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15) !important;
#         outline: none !important;
#     }

#     [data-testid="stChatInput"] button {
#         border-radius: 14px !important;
#         background: var(--primary) !important;
#         color: white !important;
#         width: 50px !important;
#         height: 50px !important;
#         margin-left: 12px !important;
#         box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3) !important;
#     }

#     [data-testid="stChatInput"] button:hover {
#         background: var(--primary-dark) !important;
#         transform: translateY(-2px) !important;
#         box-shadow: 0 6px 15px rgba(67, 97, 238, 0.4) !important;
#     }

#     [data-testid="stChatInput"] button:active {
#         transform: translateY(0) !important;
#     }

#     [data-testid="stChatInput"] button::before {
#         width: 24px !important;
#         height: 24px !important;
#         background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white' width='24px' height='24px'%3E%3Cpath d='M2.01 21L23 12 2.01 3 2 10l15 2-15 2z'/%3E%3C/svg%3E") !important;
#     }

#     /* Avatars */
#     [data-testid="chatAvatarIcon-assistant"] {
#         background: var(--primary) !important;
#         color: white !important;
#         font-size: 1rem !important;
#         box-shadow: 0 2px 5px rgba(67, 97, 238, 0.3) !important;
#     }

#     [data-testid="chatAvatarIcon-user"] {
#         background: var(--dark) !important;
#         color: white !important;
#         font-size: 1rem !important;
#         box-shadow: 0 2px 5px rgba(33, 37, 41, 0.3) !important;
#     }

#     /* Alerts */
#     [data-testid="stAlert"] {
#         border-radius: 12px !important;
#         padding: 1rem 1.25rem !important;
#         border-left: 4px solid !important;
#     }

#     [data-testid="stAlert"][data-baseweb="notification"][kind="positive"] {
#         background: rgba(76, 201, 240, 0.1) !important;
#         border-color: var(--success) !important;
#     }

#     [data-testid="stAlert"][data-baseweb="notification"][kind="negative"] {
#         background: rgba(247, 37, 133, 0.1) !important;
#         border-color: var(--warning) !important;
#     }

#     [data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
#         background: rgba(67, 97, 238, 0.1) !important;
#         border-color: var(--primary) !important;
#     }

#     /* Spinner */
#     .stSpinner > div {
#         border-top-color: var(--primary) !important;
#         border-left-color: var(--primary) !important;
#         width: 3rem !important;
#         height: 3rem !important;
#         animation: spin 1s linear infinite;
#     }

#     @keyframes spin {
#         0% { transform: rotate(0deg); }
#         100% { transform: rotate(360deg); }
#     }

#     /* Evaluation Section */
#     .evaluation-section {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin: 1.5rem 0;
#         box-shadow: var(--shadow);
#         border-left: 4px solid var(--primary);
#     }

#     .evaluation-section h3 {
#         color: var(--primary);
#         margin-bottom: 1rem;
#         font-family: 'Space Grotesk', sans-serif;
#         font-size: 1.25rem;
#     }

#     .stExpander {
#         border: none !important;
#         background: transparent !important;
#     }

#     .stExpander > summary {
#         background: white !important;
#         border-radius: 12px !important;
#         padding: 1rem 1.5rem !important;
#         font-weight: 600 !important;
#         color: var(--primary) !important;
#         box-shadow: var(--shadow) !important;
#         border-left: 4px solid var(--primary) !important;
#     }

#     .stExpanderDetails {
#         background: white !important;
#         border-radius: 0 0 12px 12px !important;
#         padding: 1.5rem !important;
#         margin-top: -8px !important;
#         box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1) !important;
#     }

#     .stExpanderDetails pre {
#         background: #f8f9fa !important;
#         padding: 1.25rem !important;
#         border-radius: 8px !important;
#         border: 1px solid #e9ecef !important;
#         font-family: 'Space Mono', monospace !important;
#         font-size: 0.85rem !important;
#     }

#     /* Responsive */
#     @media (max-width: 768px) {
#         [data-testid="stAppViewContainer"] > .main .block-container {
#             padding: 2rem 1.5rem 5rem;
#             margin: 1rem auto;
#             border-radius: 0;
#         }

#         [data-testid="stChatMessage"] {
#             max-width: 90%;
#         }

#         .title-container h1 {
#             font-size: 2rem;
#         }
#     }
# </style>
# """

# st.markdown(custom_css, unsafe_allow_html=True)

# --- Configuration & Initialization (Groq Client) ---

# Attempt to load API key from environment variables or Streamlit secrets
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
        st.stop()

    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# System prompt for the Groq API (Unchanged from your version)
SYSTEM_PROMPT = """
You are TalentBot Pro, an advanced AI hiring assistant for TalentScout, a technology recruitment agency. Your primary goal is to conduct an initial screening interview with potential candidates.

Your workflow is as follows:

1.  **Greeting:** Start with a professional greeting, introduce yourself (TalentBot Pro), and briefly explain your purpose (initial screening for TalentScout).
2.  **Information Gathering:** Systematically collect the following essential candidate information. Ask for one piece of information at a time unless the candidate provides multiple pieces in one response. Be polite and conversational.
    *   Full Name
    *   Email Address (validate format briefly if possible)
    *   Phone Number
    *   Years of Relevant Experience (specify technical/relevant experience)
    *   Desired Position(s) (e.g., Software Engineer, Data Scientist)
    *   Current Location (City, Country)
    *   Tech Stack (explicitly ask for programming languages, frameworks, databases, tools, etc.)
3.  **Confirmation:** Once all information is gathered, briefly summarize it back to the candidate for confirmation.
4.  **Technical Question Generation:** Based *only* on the confirmed 'Tech Stack' provided by the candidate, generate exactly 3-5 relevant technical questions. Number the questions clearly. Ask the questions one by one, waiting for the candidate's answer before proceeding to the next.
    *   Example: If stack is Python, Django: Ask 1-2 Python questions, 1-2 Django questions.
    *   Do *not* ask questions about technologies not mentioned by the candidate.
5.  **Answer Evaluation (Internal Task - Prepare for Summary):** As the candidate answers each technical question, internally note the question and answer. You will be asked to score them later. Do *not* provide immediate feedback or scores to the candidate during the Q&A. Simply acknowledge the answer and move to the next question or conclude the Q&A phase.
6.  **Conversation Context:** Maintain the context throughout the conversation. Refer back to previous points if necessary (e.g., "Thanks, [Name]. Now about your experience...").
7.  **Handling Edge Cases:**
    *   If the user input is unclear or irrelevant, politely ask for clarification or steer the conversation back to the required information or question. Do not deviate from the screening purpose.
    *   If the candidate asks a question outside your scope (e.g., salary, company culture), politely state that you are focused on the initial screening and those details will be discussed later by a human recruiter.
8.  **Ending the Conversation:**
    *   If the candidate uses conversation-ending keywords (like "bye", "exit", "quit", "stop", "finish", "done"), acknowledge their intent to finish.
    *   If technical questions have been asked, thank them for answering the questions.
    *   Provide a concluding statement, thanking the candidate for their time and explaining the next steps (e.g., "Thank you for completing the initial screening. A TalentScout recruiter will review your information and the interview details and will be in touch regarding the next steps.").
    *   Do *not* provide the evaluation scores directly to the candidate in the final message.

**Tone:** Professional, polite, encouraging, and efficient.
**Formatting:** Use clear paragraphs. Use numbered lists for questions. Use bold text sparingly for emphasis (like **Tech Stack**).
"""

# --- Helper Functions (Unchanged from your version, but added Evaluation Display) ---

def generate_response(messages: List[Dict]) -> str:
    """Generate response using Groq API"""
    try:
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Using a Llama 3.1 model
            messages=full_messages,
            temperature=0.6,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return "Sorry, I encountered an error trying to generate a response. Please try again."

def save_interview_data(candidate_info: Dict, qa_pairs: List[Tuple[str, str]], scores: Dict[str, int], conversation_history: List[Dict], evaluation_text: str = ""):
    """Save interview data, including conversation history and raw evaluation text, to a JSON file"""
    avg_score = 0
    if scores:
        valid_scores = [s for s in scores.values() if isinstance(s, (int, float))]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)

    data = {
        "interview_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "candidate_info": candidate_info,
        "technical_qa": [{"question": q, "answer": a} for q, a in qa_pairs],
        "evaluation_scores": scores,
        "average_score": round(avg_score, 2) if avg_score else "N/A",
        "llm_evaluation_rationale": evaluation_text, # Save the raw text
        "full_conversation_history": conversation_history
    }

    os.makedirs("interview_data", exist_ok=True)
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", candidate_info.get("Full Name", "UnknownCandidate"))
    sanitized_name = sanitized_name.replace(" ", "_")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"interview_data/{sanitized_name}_{timestamp}.json"

    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # Use Streamlit success message for better UI feedback
        st.success(f"Interview data saved successfully!") # Keep it short for the UI
        # st.caption(f"Filename: {filename}") # Optionally show filename
        return filename
    except IOError as e:
        st.error(f"Error saving interview data: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during saving: {e}")
        return None


def extract_scores_from_evaluation(evaluation_text: str) -> Dict[str, int]:
    """Extract question scores from the LLM's evaluation response."""
    scores = {}
    # More specific regex looking for "Score: X/10" or "Score: X" after a Q/A block
    score_pattern = re.compile(r"(?:score|rating)\s*[:\-]?\s*(\d{1,2})(?:\s*/\s*10)?", re.IGNORECASE)

    # Find all potential scores first
    potential_scores = score_pattern.findall(evaluation_text)

    # Assume scores appear in order of questions
    for i, score_str in enumerate(potential_scores):
        try:
            score = int(score_str)
            score = min(10, max(1, score)) # Clamp score between 1 and 10
            scores[f"Question {i+1}"] = score # Use simple Q number as key
        except ValueError:
            st.warning(f"Could not parse score: {score_str}")
            continue # Skip if score is not a valid integer

    # Fallback - if regex failed, try simpler search (less reliable)
    if not scores:
        lines = evaluation_text.split('\n')
        q_num = 1
        for line in lines:
             line_lower = line.lower()
             if "score:" in line_lower or "rating:" in line_lower:
                 score_digits = ''.join(filter(str.isdigit, line.split(':')[-1].split('/')[0]))
                 if score_digits:
                     try:
                         score = min(10, max(1, int(score_digits)))
                         scores[f"Question {q_num} (fallback)"] = score
                         q_num += 1
                     except ValueError:
                         continue
    return scores

def generate_final_evaluation(qa_pairs: List[Tuple[str, str]], messages: List[Dict]) -> Tuple[Dict[str, int], str]:
    """Ask the LLM to evaluate the answers and provide scores. Returns scores dict and raw evaluation text."""
    if not qa_pairs:
        return {}, ""

    evaluation_prompt_text = (
        "You are TalentBot Pro. Evaluate the technical accuracy and depth of the candidate's answers below. "
        "For each Q&A pair, provide a brief justification and a score from 1 (poor) to 10 (excellent).\n\n"
        "Format:\n"
        "Q#: [Question Text]\nA: [Answer Text]\nEvaluation: [Justification]\nScore: [Score]/10\n\n"
        "--- START Q&A ---"
    )
    for i, (q, a) in enumerate(qa_pairs):
        evaluation_prompt_text += f"\nQ{i+1}: {q}\nA: {a}\n"
    evaluation_prompt_text += "\n--- END Q&A ---\nPlease provide your evaluation and scores."

    try:
        # Use a focused context for evaluation
        evaluation_messages = messages[-6:] # Last few turns might be enough context
        evaluation_messages.append({"role": "user", "content": evaluation_prompt_text})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *evaluation_messages],
            temperature=0.4, # Even lower temp for objective evaluation
            max_tokens=1000
        )
        evaluation_response_text = response.choices[0].message.content

        # --- Display LLM's evaluation rationale in the UI ---
        st.markdown("---")
        st.subheader("Evaluation Summary (Internal)")
        with st.expander("Show LLM Evaluation Details"):
             st.markdown(f'<div class="evaluation-section"><pre>{evaluation_response_text}</pre></div>', unsafe_allow_html=True)
        st.markdown("---")
        # --- End Display ---

        scores = extract_scores_from_evaluation(evaluation_response_text)
        return scores, evaluation_response_text # Return both scores and the text

    except Exception as e:
        st.error(f"API Error during evaluation: {str(e)}")
        return {}, f"Error during evaluation: {str(e)}"

# extract_info_from_conversation function remains the same as your provided version
def extract_info_from_conversation(messages: List[Dict], current_info: Dict) -> bool:
    """
    Attempt to parse the conversation to extract structured candidate info.
    Uses regex for slightly more robust extraction. Checks if info already exists.
    Returns True if all required info seems collected, False otherwise.
    """
    updated_info = current_info.copy()
    info_changed = False

    # Regex patterns (simplified)
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    # Basic phone pattern (adapt for international numbers if needed)
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    # Experience pattern (looking for numbers near "year", "experience")
    exp_pattern = re.compile(r'(\d{1,2}(?:\.\d{1,2})?)\s+(?:year|yr)s?', re.IGNORECASE)

    # Process messages (focus on user messages)
    for message in reversed(messages): # Check recent messages first
        if message["role"] == "user":
            content = message["content"]
            content_lower = content.lower()

            # --- Extract specific fields if not already present ---
            # Note: This relies heavily on the LLM asking for info correctly.
            # A more robust method might involve specific prompts or function calling.

            # Full Name (Often the first thing asked)
            if not updated_info.get("Full Name"):
                 # Check if previous assistant message asked for name
                 prev_msg_idx = messages.index(message) - 1
                 if prev_msg_idx >= 0 and messages[prev_msg_idx]["role"] == "assistant":
                     assistant_prompt = messages[prev_msg_idx]["content"].lower()
                     if "full name" in assistant_prompt:
                         # Basic assumption: the user's reply is the name
                         # Avoid grabbing overly long responses as names
                         if len(content.split()) < 6:
                            updated_info["Full Name"] = content.strip()
                            info_changed = True

            # Email
            if not updated_info.get("Email Address"):
                match = email_pattern.search(content)
                if match:
                    updated_info["Email Address"] = match.group(0)
                    info_changed = True

            # Phone Number
            if not updated_info.get("Phone Number"):
                 match = phone_pattern.search(content)
                 if match:
                     updated_info["Phone Number"] = match.group(0)
                     info_changed = True
                 # Fallback: check if assistant asked for phone and user provided digits
                 elif ("phone" in content_lower or "number" in content_lower) and not any(c.isalpha() for c in content):
                     digits = "".join(filter(str.isdigit, content))
                     if len(digits) >= 7: # Basic check for enough digits (allow shorter local nums)
                         updated_info["Phone Number"] = content.strip() # Store what user typed
                         info_changed = True


            # Years of Experience
            if not updated_info.get("Years of Experience"):
                match = exp_pattern.search(content)
                if match:
                    updated_info["Years of Experience"] = match.group(1) # The captured number
                    info_changed = True
                # Fallback: check if assistant asked and user provided a number/phrase like "less than 1"
                elif "experience" in content_lower and ("year" in content_lower or any(char.isdigit() for char in content) or "less than" in content_lower):
                     # Store the whole phrase if it's not just a number
                     updated_info["Years of Experience"] = content.strip()
                     info_changed = True

            # Desired Position(s)
            if not updated_info.get("Desired Position(s)"):
                 # Check if previous assistant message asked for position
                 prev_msg_idx = messages.index(message) - 1
                 if prev_msg_idx >= 0 and messages[prev_msg_idx]["role"] == "assistant":
                     assistant_prompt = messages[prev_msg_idx]["content"].lower()
                     if "position" in assistant_prompt or "role" in assistant_prompt:
                          updated_info["Desired Position(s)"] = content.strip()
                          info_changed = True


            # Current Location
            if not updated_info.get("Current Location"):
                 # Check if previous assistant message asked for location
                 prev_msg_idx = messages.index(message) - 1
                 if prev_msg_idx >= 0 and messages[prev_msg_idx]["role"] == "assistant":
                     assistant_prompt = messages[prev_msg_idx]["content"].lower()
                     if "location" in assistant_prompt or "where are you based" in assistant_prompt:
                          updated_info["Current Location"] = content.strip()
                          info_changed = True


            # Tech Stack (Often comes after a specific prompt)
            if not updated_info.get("Tech Stack"):
                 # Check if previous assistant message asked for tech stack/skills
                 prev_msg_idx = messages.index(message) - 1
                 if prev_msg_idx >= 0 and messages[prev_msg_idx]["role"] == "assistant":
                     assistant_prompt = messages[prev_msg_idx]["content"].lower()
                     if "tech stack" in assistant_prompt or "technologies" in assistant_prompt or "skills" in assistant_prompt or "programming languages" in assistant_prompt:
                         # Assume the response lists techs, split by common delimiters
                         techs = re.split(r'[,\n;/]+|\band\b', content) # Split by comma, newline, semicolon, slash, or 'and'
                         cleaned_techs = [t.strip() for t in techs if t.strip() and len(t.strip()) > 1] # Basic filter
                         if cleaned_techs:
                             updated_info["Tech Stack"] = cleaned_techs
                             info_changed = True

    # Update session state if changes occurred
    if info_changed:
        st.session_state.candidate_info = updated_info

    # Check if all required fields are filled
    required_fields = ["Full Name", "Email Address", "Phone Number", "Years of Experience", "Desired Position(s)", "Current Location", "Tech Stack"]
    all_collected = all(updated_info.get(field) for field in required_fields)

    return all_collected


# --- Main Streamlit App Logic (Adapted for state flow) ---

def main():
    st.title("🎯 TalentScout Hiring Assistant")
    st.caption("AI-powered initial screening for technical candidates")
    st.markdown("---")

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm TalentBot Pro, your AI hiring assistant from TalentScout. I'll help with the initial screening process by asking a few questions about your background and technical skills. To start, could you please tell me your **full name**?"}
        ]
    if "stage" not in st.session_state:
        st.session_state.stage = "greeting" # Stages: greeting, info_gathering, confirmation, tech_qa, finished, error
    if "candidate_info" not in st.session_state:
        st.session_state.candidate_info = {
            "Full Name": "", "Email Address": "", "Phone Number": "",
            "Years of Experience": "", "Desired Position(s)": "",
            "Current Location": "", "Tech Stack": []
        }
    if "generated_tech_questions" not in st.session_state:
        st.session_state.generated_tech_questions = []
    if "current_tech_question_index" not in st.session_state:
        st.session_state.current_tech_question_index = 0
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "scores" not in st.session_state:
        st.session_state.scores = {}
    if "evaluation_text" not in st.session_state:
        st.session_state.evaluation_text = "" # To store raw evaluation


    # --- Display Chat History ---
    for message in st.session_state.messages:
        # Use Streamlit's built-in avatar identifiers
        avatar = "🎯" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # --- Stage-Based Logic ---
    current_stage = st.session_state.get("stage", "greeting")

    # --- Chat Input ---
    if current_stage != "finished":
        if prompt := st.chat_input("Your response..."):
            # Display user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # --- Handle Exit Condition ---
            exit_keywords = ["exit", "quit", "end", "stop", "bye", "finish", "done", "goodbye"]
            if any(keyword in prompt.lower() for keyword in exit_keywords):
                st.session_state.stage = "finished"
                final_message = "Thank you for your time today."

                if st.session_state.qa_pairs: # If questions were asked, evaluate
                    with st.spinner("Finalizing and evaluating..."):
                         st.session_state.scores, st.session_state.evaluation_text = generate_final_evaluation(
                            st.session_state.qa_pairs, st.session_state.messages
                        )
                         save_interview_data(
                            st.session_state.candidate_info, st.session_state.qa_pairs,
                            st.session_state.scores, st.session_state.messages,
                            st.session_state.evaluation_text # Pass evaluation text
                        )
                    final_message = (
                        f"Thank you for completing the initial screening, {st.session_state.candidate_info.get('Full Name', 'Candidate')}! "
                        "Your information and responses have been recorded. "
                        "A TalentScout recruiter will review the details and be in touch regarding the next steps. Have a great day!"
                    )
                else: # Save whatever info was collected if ending early
                    save_interview_data(
                        st.session_state.candidate_info, st.session_state.qa_pairs,
                        st.session_state.scores, st.session_state.messages, ""
                    )
                    final_message = "Thank you for your time. Your information has been saved. A recruiter may be in touch. Goodbye!"

                st.session_state.messages.append({"role": "assistant", "content": final_message})
                with st.chat_message("assistant", avatar="🎯"):
                    st.markdown(final_message)
                st.info("Interview finished. Refresh the page to start a new session.")
                st.stop() # Stop execution

            # --- Main State Machine ---
            with st.spinner("Thinking..."):
                # 1. Info Gathering Stage
                if current_stage in ["greeting", "info_gathering"]:
                    st.session_state.stage = "info_gathering" # Ensure stage is set
                    all_info_collected = extract_info_from_conversation(st.session_state.messages, st.session_state.candidate_info)

                    if all_info_collected:
                        # Info collected, move to confirmation stage
                        st.session_state.stage = "confirmation"
                        info = st.session_state.candidate_info
                        confirmation_prompt = (
                            "Okay, thank you! Let's quickly confirm the information I have:\n"
                            f"- **Full Name:** {info['Full Name']}\n"
                            f"- **Email:** {info['Email Address']}\n"
                            f"- **Phone:** {info['Phone Number']}\n"
                            f"- **Experience:** {info['Years of Experience']}\n"
                            f"- **Desired Position(s):** {info['Desired Position(s)']}\n"
                            f"- **Location:** {info['Current Location']}\n"
                            f"- **Tech Stack:** {', '.join(info['Tech Stack']) if isinstance(info['Tech Stack'], list) else info['Tech Stack']}\n\n"
                            "Is all of this information correct? (Yes/No)"
                        )
                        st.session_state.messages.append({"role": "assistant", "content": confirmation_prompt})
                    else:
                        # Info not yet complete, generate next question from LLM
                        response = generate_response(st.session_state.messages)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                # 2. Confirmation Stage
                elif current_stage == "confirmation":
                    if any(word in prompt.lower() for word in ["yes", "correct", "confirm", "sure", "ok", "yep"]):
                         # Info confirmed, move to tech Q&A
                         st.session_state.stage = "tech_qa"
                         tech_stack = st.session_state.candidate_info.get('Tech Stack', [])
                         if tech_stack:
                             question_gen_prompt = (
                                 f"Great, thank you for confirming. Now, based *only* on your tech stack ({', '.join(tech_stack)}), "
                                 "I will ask 3-5 technical questions. Let's start with the first one.\n\n"
                                 "*(Generating questions...)*" # Placeholder message
                             )
                             st.session_state.messages.append({"role": "assistant", "content": question_gen_prompt})
                             # Make API call to generate questions AND ask the first one
                             api_prompt_for_questions = (
                                f"Generate 3-5 technical questions based *only* on this tech stack: {', '.join(tech_stack)}. "
                                "Number them. After generating the list internally, ask **only the first question** now."
                             )
                             # Add this internal instruction to the messages sent to the API
                             response = generate_response(st.session_state.messages + [{"role": "user", "content": api_prompt_for_questions}])

                             # Attempt to parse all questions from the response for later use
                             potential_questions = re.findall(r"^\s*\d+\.\s+(.*)", response, re.MULTILINE)
                             if potential_questions:
                                 st.session_state.generated_tech_questions = [q.strip() for q in potential_questions]
                             else:
                                 # Fallback: assume the first line after potential intro is the question
                                 lines = response.split('\n')
                                 for line in lines:
                                     match = re.match(r"^\s*(?:1\.|Okay,\s*first question:|Here's the first question:)\s*(.*)", line.strip(), re.IGNORECASE)
                                     if match and match.group(1).strip():
                                         st.session_state.generated_tech_questions = [match.group(1).strip()] # Store only the first if parsing failed
                                         break
                                 if not st.session_state.generated_tech_questions: # If still no question found
                                     st.warning("Could not reliably identify the first technical question. The interview might proceed unexpectedly.")
                                     st.session_state.generated_tech_questions = [] # Ensure it's empty

                             st.session_state.current_tech_question_index = 0
                             # Replace placeholder with actual first question response
                             st.session_state.messages[-1] = {"role": "assistant", "content": response}

                         else:
                              # Tech stack missing, revert to info gathering
                              st.session_state.stage = "info_gathering"
                              response = "Apologies, it seems I don't have your tech stack. Could you please list the main technologies (programming languages, frameworks, databases, tools) you use?"
                              st.session_state.messages.append({"role": "assistant", "content": response})

                    else: # User said info is incorrect
                         st.session_state.stage = "info_gathering" # Go back to collect correct info
                         response = "Okay, my apologies. Could you please tell me which information is incorrect and provide the correction?"
                         st.session_state.messages.append({"role": "assistant", "content": response})

                # 3. Technical Q&A Stage
                elif current_stage == "tech_qa":
                    # Store the answer to the previous question
                    if st.session_state.generated_tech_questions:
                        q_index = st.session_state.current_tech_question_index
                        if q_index < len(st.session_state.generated_tech_questions):
                            question_text = st.session_state.generated_tech_questions[q_index]
                            st.session_state.qa_pairs.append((question_text, prompt))
                        else:
                             st.warning(f"Attempted to record answer for question index {q_index}, but only {len(st.session_state.generated_tech_questions)} questions were generated.")

                        # Move to next question index
                        st.session_state.current_tech_question_index += 1
                        next_q_index = st.session_state.current_tech_question_index

                        if next_q_index < len(st.session_state.generated_tech_questions):
                            # Ask the next question
                            next_question = st.session_state.generated_tech_questions[next_q_index]
                            response = f"Thank you. Next question ({next_q_index + 1}/{len(st.session_state.generated_tech_questions)}):\n\n{next_q_index + 1}. {next_question}"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            # All questions asked, finish the interview
                            st.session_state.stage = "finished"
                            st.write("Evaluating answers...") # Show status in main area
                            # Evaluate and save
                            st.session_state.scores, st.session_state.evaluation_text = generate_final_evaluation(
                                st.session_state.qa_pairs, st.session_state.messages
                            )
                            save_interview_data(
                                st.session_state.candidate_info, st.session_state.qa_pairs,
                                st.session_state.scores, st.session_state.messages,
                                st.session_state.evaluation_text
                            )
                            final_message = (
                                f"Thank you, {st.session_state.candidate_info.get('Full Name', 'Candidate')}, that concludes the technical questions. "
                                "Your responses have been recorded along with your profile information. "
                                "A TalentScout recruiter will review everything and contact you regarding the next steps. Have a great day!"
                            )
                            st.session_state.messages.append({"role": "assistant", "content": final_message})

                    else: # Should not happen if logic is correct, but handle gracefully
                        st.session_state.stage = "finished" # End interview if questions aren't available
                        response = "It seems there was an issue with the technical questions. We'll conclude the screening here. Thank you for your time. A recruiter will be in touch."
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        save_interview_data( # Save what we have
                             st.session_state.candidate_info, st.session_state.qa_pairs,
                             st.session_state.scores, st.session_state.messages, "Error: Technical questions not generated/found."
                         )


            # Rerun to display the latest messages and spinner state
            st.rerun()

    elif current_stage == "finished":
        st.info("Interview finished. Refresh the page to start a new session.")

if __name__ == "__main__":
    main()