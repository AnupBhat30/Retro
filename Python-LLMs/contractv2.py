import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import logging
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Removed WordCloud and related imports
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from collections import Counter
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from datetime import datetime, timedelta

# --- Logging Config ---
logging.basicConfig(level=logging.INFO)

# --- Environment Variables & API Key Setup ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google AI: {e}. Check your API key.")
    st.stop()

# --- PDF Text Extraction ---
def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in range(len(reader.pages)):
            page_text = reader.pages[page].extract_text()
            if page_text:
                text += page_text
        if not text:
            logging.warning(f"No text extracted from PDF: {uploaded_file.name}")
            st.warning("Could not extract text from the PDF. It might be image-based or corrupted.")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file {uploaded_file.name}: {e}")
        st.error(f"Error reading PDF: {e}. Please ensure it's a valid PDF file.")
        return None

# --- JSON Cleaning (Keep as before, helps with minor formatting issues) ---
def clean_json_array(response):
    def clean_array(match):
        array_str = match.group(0)
        elements = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)(?=(?:[^\']+\'[^\']*\')*[^\']*$)', array_str[1:-1])
        cleaned_elements = []
        for elem in elements:
            elem = elem.strip()
            if elem:
                if not ((elem.startswith('"') and elem.endswith('"')) or \
                        (elem.startswith("'") and elem.endswith("'"))):
                    elem = elem.replace('"', '\\"')
                    cleaned_elements.append(f'"{elem}"')
                else:
                    cleaned_elements.append(elem)
        return f'[{", ".join(cleaned_elements)}]'
    try:
        cleaned_response = re.sub(r'\[(.*?)\]', clean_array, response, flags=re.DOTALL)
        return cleaned_response
    except Exception as e:
        logging.error(f"Error during clean_json_array regex substitution: {e}")
        return response

# --- JSON Validation/Fallback (Keep as before, crucial for robustness) ---
def ensure_valid_json(response):
    try:
        json.loads(response)
        logging.info("Raw response parsed successfully as JSON.")
        return response
    except json.JSONDecodeError as e1:
        logging.warning(f"Initial JSON parsing failed: {e1}. Attempting cleaning and recovery...")
        cleaned = response.strip()
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        first_bracket = cleaned.find('{')
        first_square = cleaned.find('[')
        last_bracket = cleaned.rfind('}')
        last_square = cleaned.rfind(']')
        start, end = -1, -1
        if first_bracket != -1 and last_bracket != -1:
            if first_square == -1 or first_bracket < first_square:
                start, end = first_bracket, last_bracket
        elif first_square != -1 and last_square != -1:
             start, end = first_square, last_square
        if start != -1 and end != -1 and start < end :
             cleaned = cleaned[start : end + 1]
        else:
             logging.warning("Could not reliably determine JSON object boundaries.")
        try:
            json.loads(cleaned)
            logging.info("Cleaned response parsed successfully as JSON.")
            return cleaned
        except json.JSONDecodeError as e2:
            logging.error(f"Failed to parse JSON even after cleaning. Error: {e2}")
            logging.error(f"Original response snippet: {response[:500]}...")
            logging.error(f"Cleaned response snippet: {cleaned[:500]}...")
            logging.warning("Generating fallback JSON structure due to unrecoverable parsing failure.")
            fallback_data = {
                "Document Type": "Unknown - Parsing Error",
                "Key Clauses": {"Error": "Unable to extract key clauses due to response format error."},
                "Predatory Clauses": "Unable to detect predatory clauses due to response format error.",
                "Summary": f"Analysis incomplete. Raw response snippet: {response[:500]}..." if len(response) > 500 else response,
                "Risk Score": "Unknown - Parsing Error",
                "Suggestions": "No suggestions available due to response format error.",
                "Contract Details": {
                    "Value": "Unknown", "Start Date": "Unknown", "End Date": "Unknown", "Parties": [], "Review Progress": 0
                },
                "Clause Risk Levels": {},
                "Obligations": [],
                "Entity Relationships": []
            }
            return json.dumps(fallback_data, indent=3)

# --- Gemini API Call ---
def get_gemini_response(prompt):
    """Sends a prompt to the Gemini API and returns the text response."""
    try:
        # Ensure you're using a model that supports the required generation length and capabilities
        # Check Google AI Studio for available models like 'gemini-1.5-flash', 'gemini-1.5-pro', etc.
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-1.5-pro' if needed
        response = model.generate_content(prompt)

        # Add more robust error checking based on the response object structure
        if not response.parts:
             logging.warning("Gemini API returned a response with no parts.")
             # Check candidate reasons if available
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 logging.error(f"Content blocked by API. Reason: {block_reason}")
                 st.error(f"The request was blocked by the AI's safety filters (Reason: {block_reason}). Please check the contract content or try rephrasing.")
                 return None # Indicate failure due to blocking
             return "" # Return empty string if no parts but not blocked

        if not hasattr(response, 'text'):
             logging.warning("Gemini API response object does not have a 'text' attribute.")
             # Try accessing text differently if structure changed (check API docs)
             try:
                 # Example: accessing text from the first part if available
                 return response.parts[0].text
             except (IndexError, AttributeError):
                  logging.error("Could not extract text from Gemini response.")
                  return ""

        if not response.text:
            logging.warning("Gemini API returned an empty text response.")
            return ""

        return response.text

    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        # Provide more specific error guidance if possible
        if "API key not valid" in str(e):
            st.error(f"Error communicating with the AI model: Invalid API Key. Please check your GOOGLE_API_KEY.")
        elif "quota" in str(e).lower():
             st.error(f"Error communicating with the AI model: API Quota Exceeded. Please check your usage limits.")
        else:
            st.error(f"Error communicating with the AI model: {e}")
        return None

# --- Translation Function (using Gemini) with Caching ---
@st.cache_data # Cache the translation results
def translate_text(text_to_translate, target_language="Hindi"):
    """Translates text using the Gemini API."""
    if not text_to_translate or not isinstance(text_to_translate, str):
        return text_to_translate # Return original if input is invalid

    if target_language == "English": # No need to translate if target is English
        return text_to_translate

    logging.info(f"Requesting translation to {target_language} for: {text_to_translate[:50]}...")
    translate_prompt = f"""Translate the following English text into {target_language}.
Provide ONLY the translated text, without any introductory phrases like "Here is the translation:" or explanations.

English Text:
"{text_to_translate}"

{target_language} Translation:"""

    translated_text = get_gemini_response(translate_prompt)

    if translated_text:
        logging.info(f"Translation successful: {translated_text[:50]}...")
        # Basic cleaning: remove potential leading/trailing quotes sometimes added by the model
        return translated_text.strip().strip('"')
    else:
        logging.warning(f"Translation to {target_language} failed for: {text_to_translate[:50]}...")
        st.warning(f"Could not translate text segment to {target_language}. Displaying original English.", icon="⚠️")
        return text_to_translate # Return original English text on failure


# --- Analysis Prompt (Keep as before) ---
input_prompt = """
Act as a legal assistant for freelancers, explaining contract terms in simple, friendly language—like you're helping a friend. Analyze the uploaded contract and identify key clauses like payment terms, intellectual property rights, termination conditions, and liability clauses. Flag predatory clauses such as unfair non-compete agreements, delayed payment terms, or excessive penalties that could harm the freelancer—put extra emphasis on these to protect the freelancer from legal or financial trouble. Assign a risk score (Low, Medium, High) based on potential red flags. Provide a short, clear summary and practical suggestions for negotiating better terms.

Document Text: {document_text}

The response structure MUST be a valid JSON object following this schema exactly:
{{
   "Document Type": "e.g., Freelance Contract, Service Agreement",
   "Key Clauses": {{
      "Clause Name 1": "Simple explanation 1.",
      "Clause Name 2": "Simple explanation 2."
   }},
   "Predatory Clauses": "Explain ALL identified predatory clauses concisely in one paragraph, focusing on the risks. If none, state 'No significant predatory clauses detected.'.",
   "Summary": "Provide a short, easy-to-understand summary of the contract's main points (2-3 sentences).",
   "Risk Score": "Assign ONE overall risk level: Low, Medium, or High. Follow with a brief (1 sentence) justification. e.g., 'Medium - Contains standard clauses but unclear payment schedule.'",
   "Suggestions": "List 2-3 practical, actionable negotiation tips or points to clarify. Use bullet points or numbered list for clarity if possible within the string.",
   "Contract Details": {{
      "Value": "Estimated contract value (e.g., '$5000', 'Per Project', 'Unknown'). Extract if clearly stated.",
      "Start Date": "Contract start date (YYYY-MM-DD format if possible, otherwise 'Not specified' or 'Unknown').",
      "End Date": "Contract end date (YYYY-MM-DD format if possible, otherwise 'Ongoing', 'Not specified', or 'Unknown').",
      "Parties": ["Party Name 1", "Party Name 2"],
      "Review Progress": 100
   }},
   "Clause Risk Levels": {{
      "Payment Terms": "Low/Medium/High/N/A",
      "Intellectual Property": "Low/Medium/High/N/A",
      "Termination": "Low/Medium/High/N/A",
      "Liability": "Low/Medium/High/N/A",
      "Non-Compete": "Low/Medium/High/N/A",
      "Confidentiality": "Low/Medium/High/N/A",
      "Indemnification": "Low/Medium/High/N/A",
      "Warranties": "Low/Medium/High/N/A"
   }},
   "Obligations": [
      {{"party": "Freelancer/Client/Both", "description": "Concise obligation description.", "deadline": "Deadline (YYYY-MM-DD or description like 'Upon Delivery', 'Not specified').", "type": "Financial/Legal/Operational/Delivery"}},
      {{"party": "Freelancer/Client/Both", "description": "Another obligation.", "deadline": "Not specified", "type": "Operational"}}
   ],
   "Entity Relationships": [
      {{"entity1": "Entity Name A", "relationship": "e.g., Provides Service To, Pays, Owns IP", "entity2": "Entity Name B"}}
   ]
}}
Ensure all string values within the JSON are properly escaped if they contain quotes. Output ONLY the JSON object, nothing before or after.
"""


# --- VISUALIZATION FUNCTIONS (Keep as before, except WordCloud) ---

# Generate clause distribution chart
def generate_clause_chart(key_clauses, clause_risk_levels):
    if not key_clauses or not isinstance(key_clauses, dict):
        logging.warning("Clause chart skipped: Invalid key_clauses data.")
        return None
    clauses = {k: v for k, v in key_clauses.items() if k != "Error" and k!= "Note"}
    if not clauses:
         logging.warning("Clause chart skipped: No valid clauses found.")
         return None
    try:
        df = pd.DataFrame({
            'Clause': list(clauses.keys()),
            'Length': [len(str(value)) for value in clauses.values()]
        })
        risk_map = {'Low': 1, 'Medium': 2, 'High': 3, 'N/A': 0, 'Unknown': 0}
        default_risk = 'Unknown'
        if clause_risk_levels and isinstance(clause_risk_levels, dict):
             df['Risk'] = [clause_risk_levels.get(clause, default_risk) for clause in df['Clause']]
             df['Risk_Value'] = [risk_map.get(risk, 0) for risk in df['Risk']]
        else:
            df['Risk'] = default_risk
            df['Risk_Value'] = 0
        df['Risk_Color'] = df['Risk'].map({'Low': 'green', 'Medium': 'orange', 'High': 'red', 'N/A': 'grey', 'Unknown': 'grey'})
        df = df.sort_values('Risk_Value', ascending=False)
        fig = px.pie(df, values='Length', names='Clause',
                    title='Contract Clause Distribution',
                    color='Risk', color_discrete_map={'Low': '#50fa7b', 'Medium': '#ffb86c', 'High': '#ff5555', 'N/A': '#6272a4', 'Unknown': '#6272a4'},
                    hover_data=['Risk'])
        fig.update_traces(textposition='inside', textinfo='percent+label', marker_line_color='rgba(0,0,0,0.2)', marker_line_width=1)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"), margin=dict(t=50, b=50, l=20, r=20),
        )
        return fig
    except Exception as e:
        logging.error(f"Error generating clause chart: {e}")
        return None

# Generate risk analysis chart
def generate_risk_chart(clause_risk_levels):
    if not clause_risk_levels or not isinstance(clause_risk_levels, dict):
        logging.warning("Risk chart skipped: Invalid clause_risk_levels data.")
        return None
    risk_map = {'Low': 1, 'Medium': 2, 'High': 3, 'N/A': 0, 'Unknown': 0}
    clauses = list(clause_risk_levels.keys())
    risks = list(clause_risk_levels.values())
    risk_values = [risk_map.get(str(r), 0) for r in risks]
    if not clauses:
        logging.warning("Risk chart skipped: No clauses found in risk levels.")
        return None
    try:
        df = pd.DataFrame({'Clause': clauses, 'Risk': risks, 'Risk_Value': risk_values})
        df_plot = df
        df_plot = df_plot.sort_values('Risk_Value', ascending=True)
        color_map = {'Low': '#50fa7b', 'Medium': '#ffb86c', 'High': '#ff5555', 'N/A': '#6272a4', 'Unknown': '#6272a4'}
        df_plot['Color'] = df_plot['Risk'].map(color_map)
        fig = px.bar(df_plot, y='Clause', x='Risk_Value', color='Risk',
                     color_discrete_map=color_map,
                     title='Clause Risk Assessment',
                     labels={'Risk_Value': 'Risk Level', 'Clause': ''},
                     height=max(300, len(df_plot) * 35),
                     orientation='h')
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'], range=[0, 3.5]),
            yaxis=dict(automargin=True), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20), legend_title_text='Risk Level'
        )
        return fig
    except Exception as e:
        logging.error(f"Error generating risk chart: {e}")
        return None

# Generate obligations chart
def generate_obligations_chart(obligations):
    if not obligations or not isinstance(obligations, list) or not all(isinstance(o, dict) for o in obligations):
        logging.warning("Obligations chart skipped: Invalid obligations data.")
        return None
    if len(obligations) == 0:
         logging.warning("Obligations chart skipped: No obligations provided.")
         return None
    try:
        df = pd.DataFrame(obligations)
        df['party'] = df['party'].fillna('Unknown')
        df['type'] = df['type'].fillna('Unknown')
        party_counts = df['party'].value_counts().reset_index()
        party_counts.columns = ['Party', 'Count']
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']

        fig_party = px.bar(party_counts, x='Party', y='Count',
                         title='Obligations by Party', color_discrete_sequence=['#bd93f9'])
        fig_party.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"),
            margin=dict(l=20, r=20, t=50, b=40), height=350, xaxis_title=None, yaxis_title="Number of Obligations"
        )
        fig_type = px.pie(type_counts, values='Count', names='Type',
                       title='Obligation Types', color_discrete_sequence=px.colors.sequential.Viridis)
        fig_type.update_traces(textposition='inside', textinfo='percent+label', marker_line_color='rgba(0,0,0,0.2)', marker_line_width=1)
        fig_type.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"),
            margin=dict(l=20, r=20, t=50, b=40), height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        return fig_party, fig_type
    except Exception as e:
        logging.error(f"Error generating obligations charts: {e}")
        return None, None

# Generate contract timeline
def generate_timeline(contract_details):
    if not contract_details or not isinstance(contract_details, dict):
         logging.warning("Timeline generation skipped: Invalid contract_details.")
         return None, 0
    start_date_str = contract_details.get('Start Date', 'Unknown')
    end_date_str = contract_details.get('End Date', 'Unknown')
    def parse_date(date_str):
        if not date_str or date_str in ['Unknown', 'Not specified', 'Ongoing']: return None
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y', '%B %d, %Y'):
            try: return datetime.strptime(date_str, fmt)
            except ValueError: pass
        logging.warning(f"Could not parse date string: {date_str}")
        return None
    start_date, end_date = parse_date(start_date_str), parse_date(end_date_str)
    current_date = datetime.now()
    has_start, has_end = start_date is not None, end_date is not None
    timeline_data = []
    effective_start = start_date if has_start else current_date
    effective_end = end_date
    if has_start:
        timeline_data.append(dict(Task="Contract Period", Start=start_date, Finish=end_date if has_end else start_date + timedelta(days=1), Resource="Period"))
    else:
         timeline_data.append(dict(Task="Start (Unknown)", Start=current_date - timedelta(days=1), Finish=current_date, Resource="Marker"))
    if has_end:
        if not has_start:
            timeline_data.append(dict(Task="End Date", Start=end_date, Finish=end_date + timedelta(days=1), Resource="Marker"))
    elif has_start:
         timeline_data.append(dict(Task="Contract Period (Ongoing)", Start=start_date, Finish=max(current_date, start_date) + timedelta(days=30), Resource="Ongoing"))
         effective_end = None
    progress = 0
    duration_days = "N/A"
    if has_start and has_end and start_date < end_date:
        total_duration = (end_date - start_date).days
        duration_days = f"{total_duration} days"
        if total_duration > 0:
            elapsed_duration = (current_date - start_date).days
            progress = min(100, max(0, (elapsed_duration / total_duration * 100)))
    elif has_start:
        duration_days = f"{(current_date - start_date).days}+ days (Ongoing)"
        progress = 100
    if not timeline_data:
        logging.warning("Timeline generation skipped: Not enough date info.")
        return None, 0
    try:
        df = pd.DataFrame(timeline_data)
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource",
                         title=f"Contract Timeline ({duration_days})",
                         color_discrete_map={"Period": "#bd93f9", "Ongoing": "#ffb86c", "Marker": "#ff79c6"})
        fig.add_vline(x=current_date, line_width=2, line_dash="dash", line_color="#50fa7b",
                      annotation_text="Today", annotation_position="top right", annotation_font_color="#50fa7b")
        fig.update_layout(
            yaxis_title=None, yaxis_visible=False, xaxis_title="Date",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"), margin=dict(l=20, r=20, t=50, b=20),
            height=200 + len(df)*20, legend_title_text="Timeline Element"
        )
        fig.update_yaxes(autorange="reversed")
        return fig, progress
    except Exception as e:
        logging.error(f"Error generating timeline: {e}")
        return None, 0

# Generate entity relationship network (Placeholder)
def generate_relationship_network(relationships):
    if not relationships or not isinstance(relationships, list) or not all(isinstance(r, dict) for r in relationships):
         logging.warning("Relationship network skipped: Invalid relationships data.")
         return None
    if len(relationships) == 0:
         logging.warning("Relationship network skipped: No relationships provided.")
         return None
    nodes = set()
    edges_str = []
    for rel in relationships:
        e1, e2, r = rel.get('entity1'), rel.get('entity2'), rel.get('relationship', 'related to')
        if e1 and e2:
            nodes.add(e1); nodes.add(e2)
            edges_str.append(f"{e1} --[{r}]--> {e2}")
    try:
        fig = go.Figure(layout=go.Layout(
                            title='Entity Relationships (Conceptual)', showlegend=False, hovermode=False,
                            margin=dict(b=20,l=5,r=5,t=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="white"),
                            annotations=[ dict(text="Graph shows conceptual links.<br>Requires dedicated library for interactive network.",
                                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.9, font=dict(size=10) ),
                                dict(text="<br>".join(edges_str), showarrow=False, xref="paper", yref="paper", x=0.05, y=0.7, align="left", font=dict(size=12) )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0,1]),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0,1]) ))
        return fig
    except Exception as e:
        logging.error(f"Error generating relationship placeholder: {e}")
        return None


# --- CSS Styling (Incorporates Metric Card Fix) ---
st.write("""
    <style>
    /* Global styling */
    body { color: #f8f8f2; }
    .main > div { background: #282a36; }
    .stApp { background-color: #282a36; color: #f8f8f2; }
    .stMarkdown, .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div, .stFileUploader > div > div > span, .stButton > button, .stExpander > div > summary, [data-testid="stRadioButton"] label span {
        color: #f8f8f2 !important; /* Ensure radio button labels are also white */
    }
    /* Radio button styling adjustments */
    [data-testid="stRadioButton"] label {
        display: flex; align-items: center; padding: 5px 10px; margin: 2px;
        background-color: #44475a; border-radius: 6px; border: 1px solid transparent;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    [data-testid="stRadioButton"] label:has(input:checked) {
         background-color: #6272a4; border-color: #bd93f9; font-weight: bold;
     }
     [data-testid="stRadioButton"] label span { padding-left: 8px; }

    .stButton > button {
        background-color: #50fa7b; color: #282a36 !important; border: none; border-radius: 8px;
        padding: 10px 20px; font-weight: bold; transition: background-color 0.3s ease;
    }
    .stButton > button:hover { background-color: #8be9fd; }
    [data-testid="stFileUploader"] label {
        background-color: rgba(248, 248, 242, 0.1); border: 2px dashed #6272a4;
        border-radius: 8px; padding: 2rem; text-align: center;
    }
    [data-testid="stFileUploader"] label span { color: #f8f8f2 !important; }
    .gradient-header {
        padding: 2rem; text-align: center; background: linear-gradient(135deg, #44475a, #6272a4);
        border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .gradient-header h1 { font-size: 2.8rem; font-weight: 800; color: #50fa7b !important; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5); }
    .gradient-header p { font-size: 1.2rem; max-width: 800px; margin: 10px auto 0 auto; color: #f8f8f2 !important; opacity: 0.9; }
    .glass-card {
        background: rgba(68, 71, 90, 0.6); border-radius: 15px; padding: 25px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px); border: 1px solid rgba(248, 248, 242, 0.1); margin-bottom: 20px;
    }
    .warning-panel {
        background: rgba(255, 85, 85, 0.2); border-left: 8px solid #ff5555; border-radius: 10px;
        padding: 20px; margin-bottom: 20px; color: #f8f8f2 !important;
    }
     .warning-panel strong { color: #ff5555 !important; }
    h2, h3 { color: #8be9fd !important; border-bottom: 2px solid #44475a; padding-bottom: 5px; margin-top: 25px; }
    .section-header { font-size: 1.6rem !important; font-weight: bold !important; color: #8be9fd !important; margin-bottom: 15px; text-align: left; }
     .detail-item, .suggestion-item {
        margin-bottom: 15px; padding: 15px; background: rgba(68, 71, 90, 0.8);
        border-radius: 8px; border-left: 5px solid #bd93f9;
    }
    .detail-item strong, .suggestion-item strong {
        font-size: 1.1rem; font-weight: 700; color: #bd93f9 !important; display: block; margin-bottom: 5px;
    }
    .doc-type {
        background: rgba(189, 147, 249, 0.2); color: #f8f8f2 !important; padding: 1rem 1.5rem;
        border-radius: 10px; text-align: center; font-size: 1.6rem; font-weight: 700;
        margin-bottom: 20px; border: 1px solid #bd93f9;
    }
    .stAlert { border-radius: 8px; border: none; color: #f8f8f2 !important; }
    [data-testid="stAlert"] > div { color: #f8f8f2 !important; }
    .stSuccess { background-color: rgba(80, 250, 123, 0.3) !important; }
    .stError { background-color: rgba(255, 85, 85, 0.3) !important; }
    .stWarning { background-color: rgba(255, 184, 108, 0.3) !important; }
    .stInfo { background-color: rgba(139, 233, 253, 0.3) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent !important; border-bottom: 2px solid #44475a; padding-bottom: 0; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; color: #bd93f9; background-color: #44475a; padding: 10px 20px; border: none; margin-bottom: -2px; transition: background-color 0.3s ease, color 0.3s ease; }
    .stTabs [data-baseweb="tab--selected"] { color: #f8f8f2 !important; background-color: #6272a4 !important; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #f8f8f2 !important; background-color: #6272a4 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: rgba(68, 71, 90, 0.3); padding: 20px; border-radius: 0 0 10px 10px;
        border: 1px solid #44475a; border-top: none;
    }

    /* --- MODIFIED METRIC CARD STYLES --- */
    .metric-card {
        background: rgba(68, 71, 90, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px; /* Keep margin for spacing */
        border: 1px solid rgba(248, 248, 242, 0.1);
        text-align: center;
        /* height: 120px; */ /* REMOVED fixed height */
        min-height: 110px; /* Set a min-height instead to maintain some base size */
        display: flex;
        flex-direction: column;
        justify-content: center; /* Vertically center content within the card */
    }
    .metric-card .label { font-size: 0.95rem; color: #8be9fd; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.9rem; font-weight: bold; color: #50fa7b !important; line-height: 1.2; }
    .metric-card .secondary {
        font-size: 0.9rem; /* Slightly larger secondary text */
        color: #f8f8f2 !important;
        opacity: 0.8; /* Slightly more visible */
        margin-top: 8px; /* Add more space below the main value */
        line-height: 1.3; /* Adjust line spacing for readability */
        min-height: 2.6em; /* Reserve space for ~2 lines of text to help alignment */
        display: flex; /* Use flex to center content if it's short */
        align-items: center;
        justify-content: center;
    }
    .metric-card .value.risk-low { color: #50fa7b !important; }
    .metric-card .value.risk-medium { color: #ffb86c !important; }
    .metric-card .value.risk-high { color: #ff5555 !important; }
    .metric-card .value.risk-unknown { color: #8be9fd !important; }

     /* Specific styling for the progress bar inside the metric card */
     .metric-card .progress-container {
        width: 80%; margin: 8px auto 8px auto; background-color: #44475a;
        border-radius: 5px; padding: 3px; height: 18px; box-sizing: border-box;
    }
    .metric-card .progress-bar {
        height: 100%; border-radius: 3px; text-align: center; color: #282a36 !important;
        font-weight: bold; font-size: 0.8em; line-height: 12px; /* Approx center */
        background-color: #bd93f9; transition: width 0.5s ease-in-out;
        display: flex; align-items: center; justify-content: center;
    }
     .metric-card .progress-bar.progress-complete { background-color: #50fa7b; }
    /* --- END MODIFIED METRIC CARD STYLES --- */


    .styled-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; border-radius: 8px; overflow: hidden; box-shadow: 0 0 20px rgba(0, 0, 0, 0.25); }
    .styled-table thead tr { background-color: #44475a; color: #f8f8f2; text-align: left; font-weight: bold; }
    .styled-table th, .styled-table td { padding: 12px 15px; color: #f8f8f2; }
    .styled-table tbody tr { border-bottom: 1px solid #44475a; background-color: rgba(68, 71, 90, 0.5); }
    .styled-table tbody tr:nth-of-type(even) { background-color: rgba(68, 71, 90, 0.7); }
    .styled-table tbody tr:last-of-type { border-bottom: 2px solid #bd93f9; }
    .styled-table tbody tr:hover { background-color: rgba(98, 114, 164, 0.7); }
    .risk-low { color: #50fa7b !important; font-weight: bold; }
    .risk-medium { color: #ffb86c !important; font-weight: bold; }
    .risk-high { color: #ff5555 !important; font-weight: bold; }
    .risk-unknown, .risk-n/a { color: #6272a4 !important; opacity: 0.8; }
    .stExpander { border: 1px solid #44475a !important; border-radius: 8px !important; margin-bottom: 10px !important; background-color: rgba(68, 71, 90, 0.4); }
    .stExpander summary { font-size: 1.1rem !important; font-weight: bold !important; color: #ffb86c !important; /* Expander title orange */ padding: 10px 15px !important; border-radius: 8px 8px 0 0; }
     .stExpander summary:hover { background-color: rgba(98, 114, 164, 0.5); }
    .stExpander div[role="region"] { background-color: transparent; padding: 15px; border-top: 1px solid #44475a; }
    .chart-container { background: rgba(68, 71, 90, 0.4); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(248, 248, 242, 0.1); }
    .plotly-graph-div { background: transparent !important; }
    </style>
""", unsafe_allow_html=True)


# --- STREAMLIT APP LAYOUT ---

# App title and description
st.write('<div class="gradient-header"><h1>Freelancer Contract Analyzer</h1><p>Upload your contract (PDF). I\'ll analyze key terms, flag potential risks, and provide actionable insights.</p></div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload Your Contract (PDF)", type="pdf", help="Supports text-based PDF files.")

# Submit button
submit = st.button("Analyze Contract ✨")

# State management initialization
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'response_data' not in st.session_state:
    st.session_state.response_data = {}
if 'contract_text' not in st.session_state:
    st.session_state.contract_text = ""
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English' # Default language


# Processing Logic
if submit and uploaded_file is not None:
    with st.spinner("Reading and analyzing your contract... This might take a moment..."):
        try:
            # Reset state for new analysis
            st.session_state.analysis_complete = False
            st.session_state.response_data = {}
            st.session_state.contract_text = ""
            # Reset language to English for new analysis? Or keep user preference? Let's keep it.
            # st.session_state.selected_language = 'English'

            text = input_pdf_text(uploaded_file)
            st.session_state.contract_text = text

            if text:
                # Limit text length sent to API if necessary (e.g., ~30k chars for Flash, check model limits)
                max_chars = 30000
                prompt_text = text[:max_chars]
                if len(text) > max_chars:
                    logging.warning(f"Input text truncated to {max_chars} characters for API call.")
                    st.warning(f"Note: The contract text was long and has been truncated to {max_chars} characters for analysis.", icon="⚠️")

                prompt = input_prompt.format(document_text=prompt_text)
                logging.info("Sending prompt to Gemini API...")

                raw_response = get_gemini_response(prompt)

                if raw_response is not None: # Check for None explicitly (API errors)
                    if raw_response == "": # Handle empty string response
                         logging.error("Received empty string response from Gemini API after successful call.")
                         st.error("The AI model returned an empty response. This might be due to content filters or an issue with the model. Try analyzing a different document.")
                    else:
                        logging.info("Received response from Gemini API. Validating JSON...")
                        final_json_string = ensure_valid_json(raw_response)
                        response_data = json.loads(final_json_string)
                        st.session_state.response_data = response_data
                        st.session_state.analysis_complete = True
                        logging.info("JSON validated/fixed. Analysis complete.")
                        st.success("Contract Analysis Complete!")
                else:
                    # Error messages handled within get_gemini_response or here if None
                    if not st.session_state.get('api_error_shown'): # Avoid duplicate errors if already shown
                         st.error("Failed to get a response from the AI model. Please check API key, quota, and network connection.")
                         st.session_state['api_error_shown'] = True # Flag that error was shown
            else:
                # Error message handled within input_pdf_text
                st.error("Could not process the PDF. Ensure it contains selectable text.")

        except json.JSONDecodeError as json_err:
            st.session_state.analysis_complete = False
            st.session_state.response_data = {}
            logging.error(f"Fatal JSON Parsing Error after fallback: {json_err}", exc_info=True)
            st.error(f"A critical error occurred while parsing the AI's response (JSON invalid): {json_err}")
            # Optionally show the raw response snippet for debugging
            # st.code(raw_response[:1000] if 'raw_response' in locals() else "No raw response available.")

        except Exception as e:
            st.session_state.analysis_complete = False
            st.session_state.response_data = {}
            logging.error(f"An unexpected error occurred during analysis: {str(e)}", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error("Please check the PDF file or try again. If the problem persists, the AI service might be unavailable.")

        # Reset api error shown flag after processing attempt
        if 'api_error_shown' in st.session_state:
            del st.session_state['api_error_shown']


elif submit and uploaded_file is None:
    st.warning("Please upload a contract (PDF) first.")

# --- Display Results if Analysis is Complete ---
if st.session_state.analysis_complete and st.session_state.response_data:
    response_data = st.session_state.response_data
    # Removed text = st.session_state.contract_text as it's no longer needed here

    # --- Language Selection ---
    st.markdown("---") # Separator
    st.session_state.selected_language = st.radio(
        "Select Language for Textual Analysis:",
        ('English', 'Hindi'),
        key='language_selector',
        horizontal=True,
        # label_visibility="collapsed" # Use if you want to hide the "Select Language..." label
    )
    st.caption("Note: Charts and table headers will remain in English.")
    st.markdown("---") # Separator

    # --- Extract data safely using .get() with defaults ---
    doc_type = response_data.get("Document Type", "Contract")
    key_clauses_en = response_data.get("Key Clauses", {}) # Store original English
    predatory_clauses_en = response_data.get("Predatory Clauses", "Not specified.")
    summary_en = response_data.get("Summary", "No summary provided.")
    risk_score = response_data.get("Risk Score", "Unknown")
    suggestions_en = response_data.get("Suggestions", "No suggestions provided.")
    contract_details = response_data.get("Contract Details", {})
    clause_risk_levels = response_data.get("Clause Risk Levels", {})
    obligations = response_data.get("Obligations", [])
    entity_relationships = response_data.get("Entity Relationships", [])

    # --- Basic validation/cleaning of extracted data ---
    if not isinstance(key_clauses_en, dict):
        key_clauses_en = {"Error": "Clause data is not in the expected format."}
    if not isinstance(clause_risk_levels, dict): clause_risk_levels = {}
    if not isinstance(obligations, list): obligations = []
    if not isinstance(entity_relationships, list): entity_relationships = []

    # --- Translate necessary text fields based on selection ---
    current_lang = st.session_state.selected_language
    summary_display = translate_text(summary_en, current_lang)
    predatory_clauses_display = translate_text(predatory_clauses_en, current_lang)
    suggestions_display = translate_text(suggestions_en, current_lang)

    # Translate key clause *explanations* (values), keep keys in English
    key_clauses_display = {}
    if isinstance(key_clauses_en, dict):
        for clause_name, explanation_en in key_clauses_en.items():
             if clause_name == "Error": # Don't translate error messages
                 key_clauses_display[clause_name] = explanation_en
             else:
                 explanation_display = translate_text(str(explanation_en), current_lang) # Ensure explanation is string
                 key_clauses_display[clause_name] = explanation_display
    else:
         key_clauses_display = key_clauses_en # Pass through if not a dict (e.g., error message)


    # --- Create tabs ---
    tab_dashboard, tab_analysis, tab_clauses, tab_relationships, tab_compare = st.tabs([
        "📊 Dashboard", "🔍 Risk Analysis", "📝 Clauses", "🔗 Obligations", "🔄 Compare"
    ])

    # --- DASHBOARD TAB ---
    with tab_dashboard:
        st.markdown("## Overview")

        st.markdown(f"<div class='doc-type'>{doc_type}</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="glass-card"><h3>Contract Summary</h3></div>', unsafe_allow_html=True)
            # Display potentially translated summary
            st.markdown(f"> {summary_display}")


        st.markdown("### Key Metrics")
        col1, col2, col3 = st.columns(3)

        # Metric Card 1: Contract Value (Using updated HTML)
        with col1:
            contract_value = contract_details.get("Value", "Unknown")
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">CONTRACT VALUE</div>
                <div class="value">{contract_value}</div>
                <div class="secondary"> </div> <!-- Empty secondary placeholder -->
            </div>
            """, unsafe_allow_html=True)

        # Metric Card 2: Overall Risk (Using updated HTML)
        with col2:
            risk_level_raw = str(risk_score).split('-')[0].split()[0].strip().lower()
            risk_color_class = f"risk-{risk_level_raw}" if risk_level_raw in ["low", "medium", "high"] else "risk-unknown"
            risk_display_text = risk_score if risk_score != "Unknown - Parsing Error" else "Error"
            risk_level_display = risk_display_text.split('-')[0].strip()
            risk_justification = ' '.join(risk_display_text.split('-')[1:]).strip() if '-' in risk_display_text else ''
            risk_justification_display = translate_text(risk_justification, current_lang)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">OVERALL RISK</div>
                <div class="value {risk_color_class}">{risk_level_display}</div>
                <div class="secondary">{risk_justification_display if risk_justification_display else ' '}</div>
            </div>
            """, unsafe_allow_html=True)

        # Metric Card 3: Analysis Status (Using updated HTML)
        with col3:
            review_progress = contract_details.get("Review Progress", 0)
            progress_class = "progress-complete" if review_progress == 100 else ""
            status_text = 'Complete' if review_progress == 100 else 'In Progress'
            status_text_display = translate_text(status_text, current_lang)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">ANALYSIS STATUS</div>
                <div class="progress-container">
                    <div class="progress-bar {progress_class}" style="width: {review_progress}%;">
                        {review_progress}%
                    </div>
                </div>
                 <div class="secondary">{status_text_display}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Contract Timeline")
        with st.container():
            timeline_fig, progress = generate_timeline(contract_details)
            if timeline_fig: st.plotly_chart(timeline_fig, use_container_width=True)
            else: st.info("Not enough date information available to generate a timeline.")

        st.markdown("### ⚠️ Potential Red Flags")
        # Use translated predatory clauses
        base_predatory_text = predatory_clauses_display # The potentially translated text
        if "Unable to detect" in base_predatory_text or "Error" in base_predatory_text or "Not specified" in base_predatory_text or "डेटा पार्स करने में त्रुटि" in base_predatory_text: # Crude check for Hindi error msg
             st.info(base_predatory_text)
        # Need to check against original English for "No significant" condition
        elif predatory_clauses_en and predatory_clauses_en != "No significant predatory clauses detected.":
            warning_title = translate_text("Warning:", current_lang) # Translate "Warning"
            st.markdown(f"<div class='warning-panel'><strong>{warning_title}</strong> {base_predatory_text}</div>", unsafe_allow_html=True)
        else:
            success_msg = translate_text("✅ No significant predatory clauses detected based on the analysis.", current_lang)
            st.success(success_msg)


    # --- RISK ANALYSIS TAB ---
    with tab_analysis:
        st.markdown("## Risk & Clause Analysis")

        st.markdown("### Clause Risk Levels")
        with st.container():
            risk_chart = generate_risk_chart(clause_risk_levels)
            if risk_chart: st.plotly_chart(risk_chart, use_container_width=True)
            elif "Error" in str(clause_risk_levels): st.warning("Risk levels could not be determined due to a parsing error.")
            else: st.info("No specific clause risk levels were provided in the analysis.")

        st.markdown("### Clause Overview")
        with st.container():
            # Pass original English clauses for chart generation consistency
            clause_chart = generate_clause_chart(key_clauses_en, clause_risk_levels)
            if clause_chart: st.plotly_chart(clause_chart, use_container_width=True)
            elif "Error" in str(key_clauses_en): st.warning("Clause distribution could not be generated due to a parsing error.")
            else: st.info("No key clauses were identified for the distribution chart.")

        st.markdown("### 💡 Suggestions for Improvement")
        # Use translated suggestions
        base_suggestions_text = suggestions_display
        if base_suggestions_text and "No suggestions" not in base_suggestions_text and "Error" not in base_suggestions_text and "कोई सुझाव नहीं" not in base_suggestions_text:
            # Simple splitting (less reliable for translated text, but a start)
            suggestion_items = re.split(r'\n\s*[-*•–]|\n\d+\.\s*', '\n' + base_suggestions_text)
            suggestion_items = [s.strip() for s in suggestion_items if s.strip()]
            if not suggestion_items or len(suggestion_items) <= 1 and '\n' not in base_suggestions_text:
                suggestion_items = [s.strip() for s in base_suggestions_text.split('\n') if s.strip()]

            suggestion_label = translate_text("Suggestion", current_lang)
            suggestions_title = translate_text("Suggestions:", current_lang)

            if suggestion_items:
                for i, item in enumerate(suggestion_items):
                     st.markdown(f'<div class="suggestion-item"><strong>{suggestion_label} {i+1}:</strong><br>{item}</div>', unsafe_allow_html=True)
            else: # Fallback
                 st.markdown(f'<div class="suggestion-item"><strong>{suggestions_title}</strong><br>{base_suggestions_text}</div>', unsafe_allow_html=True)
        else:
            st.info(base_suggestions_text)


    # --- CONTRACT CLAUSES TAB ---
    with tab_clauses:
        st.markdown("## Detailed Clauses")

        if isinstance(key_clauses_display, dict) and "Error" not in key_clauses_display:
            search_term = st.text_input("🔍 Search clauses by keyword (searches English clause names)", key="clause_search")

            # Filter based on English keys (key_clauses_en) but display translated values (key_clauses_display)
            filtered_clauses_en_keys = key_clauses_en # Start with all English keys
            if search_term:
                search_lower = search_term.lower()
                # Filter the *original English* dictionary keys/values
                filtered_clauses_en_keys = {k: v for k, v in key_clauses_en.items()
                                            if search_lower in k.lower() or search_lower in str(v).lower()}

            if not filtered_clauses_en_keys and search_term:
                 st.warning(f"No clauses found matching '{search_term}'.")
            elif not filtered_clauses_en_keys:
                 st.info("No key clauses were identified in the analysis.")

            # Display clauses using filtered English keys, but get translated explanation from key_clauses_display
            expander_clause_label = translate_text("Clause:", current_lang)
            expander_risk_label = translate_text("Risk", current_lang)
            expander_risk_assessment_label = translate_text("Risk Assessment:", current_lang)
            expander_explanation_label = translate_text("Explanation:", current_lang)


            for key_en, _ in filtered_clauses_en_keys.items(): # Iterate using filtered English keys
                value_display = key_clauses_display.get(key_en, "Translation not available") # Get translated value
                risk_level = clause_risk_levels.get(key_en, "Unknown") # Risk is from original analysis
                risk_class = f"risk-{str(risk_level).lower().replace('/', '')}"

                # Keep clause key (name) in English for consistency
                expander_title = f"{key_en} <span class='{risk_class}' style='float: right; font-size: 0.9em; font-weight:normal;'>({risk_level} {expander_risk_label})</span>"

                with st.expander(f"{expander_clause_label} {key_en}", expanded=False):
                     st.markdown(f"**{expander_risk_assessment_label}** <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
                     st.markdown(f"**{expander_explanation_label}**")
                     st.markdown(str(value_display)) # Display potentially translated explanation

        elif isinstance(key_clauses_display, dict) and "Error" in key_clauses_display :
             st.warning(key_clauses_display["Error"]) # Display the error message directly
        else:
            st.warning("Could not display clauses due to an unexpected data format.")
            st.json(key_clauses_display)


    # --- RELATIONSHIPS & OBLIGATIONS TAB ---
    with tab_relationships:
        st.markdown("## Obligations & Relationships")

        parties = contract_details.get("Parties", [])
        if parties and isinstance(parties, list):
            st.markdown("### Contract Parties")
            st.markdown(", ".join(parties))
        elif parties:
            st.markdown("### Contract Parties")
            st.markdown(str(parties))

        st.markdown("### Obligations Analysis")
        if obligations:
            with st.container():
                obligation_charts = generate_obligations_chart(obligations)
                if obligation_charts:
                    obligation_fig_party, obligation_fig_type = obligation_charts
                    col1, col2 = st.columns(2)
                    if obligation_fig_party:
                        with col1: st.plotly_chart(obligation_fig_party, use_container_width=True)
                    if obligation_fig_type:
                        with col2: st.plotly_chart(obligation_fig_type, use_container_width=True)
                else:
                    st.info("Could not generate obligation summary charts.")

            st.markdown("#### Obligation Details")
            df_obligations = pd.DataFrame(obligations)
            for col in ['party', 'type', 'description', 'deadline']:
                if col not in df_obligations.columns: df_obligations[col] = 'Not specified'
            df_obligations.fillna('Not specified', inplace=True)

            party_options = ["All"] + sorted(df_obligations['party'].unique().tolist())
            type_options = ["All"] + sorted(df_obligations['type'].unique().tolist())
            col_f1, col_f2 = st.columns(2)
            with col_f1: party_filter = st.selectbox("Filter by Party", party_options, key="party_filter")
            with col_f2: type_filter = st.selectbox("Filter by Type", type_options, key="type_filter")

            filtered_df = df_obligations.copy()
            if party_filter != "All": filtered_df = filtered_df[filtered_df['party'] == party_filter]
            if type_filter != "All": filtered_df = filtered_df[filtered_df['type'] == type_filter]

            if not filtered_df.empty:
                 st.markdown('<div style="max-height: 400px; overflow-y: auto;">', unsafe_allow_html=True)
                 # Keep table headers in English
                 st.markdown('<table class="styled-table"><thead><tr><th>Party</th><th>Obligation Description</th><th>Type</th><th>Deadline</th></tr></thead><tbody>', unsafe_allow_html=True)
                 for _, row in filtered_df.iterrows():
                     # Translate only the description? Or keep table data English? Let's keep it English for now.
                     # desc_display = translate_text(row['description'], current_lang)
                     st.markdown(f"<tr><td>{row['party']}</td><td>{row['description']}</td><td>{row['type']}</td><td>{row['deadline']}</td></tr>", unsafe_allow_html=True)
                 st.markdown("</tbody></table></div>", unsafe_allow_html=True)
            else:
                st.info("No obligations match the current filter criteria.")

        elif "Error" in str(obligations): st.warning("Could not display obligations due to a data format error.")
        else: st.info("No specific obligations were identified in the analysis.")

        st.markdown("### Entity Relationships")
        if entity_relationships:
            with st.container():
                relationship_fig = generate_relationship_network(entity_relationships)
                if relationship_fig: st.plotly_chart(relationship_fig, use_container_width=True)

            st.markdown("#### Relationship Details")
            df_relations = pd.DataFrame(entity_relationships)
            for col in ['entity1', 'relationship', 'entity2']:
                if col not in df_relations.columns: df_relations[col] = 'Not specified'
            df_relations.fillna('Not specified', inplace=True)

            st.markdown('<div style="max-height: 300px; overflow-y: auto;">', unsafe_allow_html=True)
            # Keep table headers English
            st.markdown('<table class="styled-table"><thead><tr><th>Entity 1</th><th>Relationship</th><th>Entity 2</th></tr></thead><tbody>', unsafe_allow_html=True)
            for _, row in df_relations.iterrows():
                # Keep table data English for simplicity
                st.markdown(f"<tr><td>{row['entity1']}</td><td>{row['relationship']}</td><td>{row['entity2']}</td></tr>", unsafe_allow_html=True)
            st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        elif "Error" in str(entity_relationships): st.warning("Could not display entity relationships due to a data format error.")
        else: st.info("No specific entity relationships were identified.")


    # --- COMPARISON TAB ---
    with tab_compare:
        st.markdown("## Contract Comparison")
        st.info("ℹ️ Feature under development: Upload a second contract here to compare key clauses, risks, and terms against the first contract.")
        comparison_file = st.file_uploader("Upload Second Contract for Comparison", type="pdf", key="comparison_uploader", help="Upload another contract (PDF).")
        if comparison_file:
            st.warning("🚧 Comparison analysis is not yet implemented.")


    # Add a footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; opacity: 0.7; font-size: 0.9em;'>Contract Analyzer | Powered by Google Gemini | © 2024</div>", unsafe_allow_html=True)