import textract
import re
import json
from fuzzywuzzy import process
import spacy
from PyPDF2 import PdfReader


# Extract raw text from various types of documents
def extract_text(file_path, file_extension):
    try:
        # Support different file types by using textract
        text = textract.process(file_path).decode('utf-8')
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


# Extract email address from text using regex
def extract_email(text):
    try:
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else None
    except Exception as e:
        print(f"Error extracting email: {e}")
        return None


# Extract mobile number from text using regex (with an optional custom regex)
def extract_mobile_number(text, custom_regex=None):
    try:
        pattern = custom_regex if custom_regex else r'\+?[0-9\s-]{7,15}'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    except Exception as e:
        print(f"Error extracting mobile number: {e}")
        return None


# Extract skills based on noun chunks and skills file
def extract_skills(nlp_text, noun_chunks, skills_file):
    try:
        # Load predefined skills from file
        with open(skills_file, 'r') as file:
            skill_set = set(file.read().splitlines())

        extracted_skills = set()

        # Match individual words from text
        for token in nlp_text:
            match, score = process.extractOne(token.text.lower(), skill_set)
            if score > 85:  # Adjust fuzzy match threshold as needed
                extracted_skills.add(match)

        # Match noun phrases (more meaningful skill names)
        for chunk in noun_chunks:
            match, score = process.extractOne(chunk.text.lower(), skill_set)
            if score > 85:
                extracted_skills.add(match)

        return list(extracted_skills)

    except Exception as e:
        print(f"⚠️ Error extracting skills: {e}")
        return []


# Extract entities using a custom-trained spaCy model
def extract_entities_wih_custom_model(custom_nlp):
    try:
        entities = {}
        for ent in custom_nlp.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return {}


# Get the number of pages in a PDF document
def get_number_of_pages(file_path):
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error getting number of pages: {e}")
        return 0


# Extract named entities (like Name, Degree, etc.) from text with a trained custom model
def extract_entity_sections_grad(text_raw):
    try:
        # Placeholder function, implement actual entity extraction if needed
        # This could include logic for extracting specific sections like "Education", "Experience", etc.
        return {"section1": "sample entity"}
    except Exception as e:
        print(f"Error extracting entity sections: {e}")
        return {}


# Extract noun chunks using spaCy's NLP pipeline
def extract_noun_chunks(text):
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        return list(doc.noun_chunks)
    except Exception as e:
        print(f"Error extracting noun chunks: {e}")
        return []
