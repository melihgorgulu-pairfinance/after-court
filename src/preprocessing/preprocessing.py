from src.regexes import *
import re

replace_with_url="<URL>"
replace_with_email="<EMAIL>"
replace_with_phone_number="<PHONE>"
replace_with_number="<NUMBER>"
replace_with_digit="0"
replace_with_currency_symbol="<CUR>"
replace_with_dr_reference="<DR_REF>"
replace_with_legal_article="<LEGAL_ARTICLE>"


def normalize_whitespace(text):
    # preserve single newlines; collapse others
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_unneccessary_lines(text: str) -> str:
    lines = text.split('\n')
    
    filtered_lines = []
    for l in lines:
        if len(l)>3:
            filtered_lines.append(l)
    return '\n'.join(filtered_lines)
  

def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text using regex
    Handles both simple tags like <p> and complex tags with attributes like <div class="example">
    """
    # Pattern explanation:
    # <        - literal '<'
    # [^>]*    - any character except '>' (zero or more times)
    # >        - literal '>'
    clean_text = re.sub(r'<[^>]*>', '', text)
    return clean_text


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_unneccessary_lines(text)
    text = remove_html_tags(text)
    text = re.sub(r"[\uE000-\uF8FF]", "", text) # remove problematic emoji parses
    text = text.lower()
    return text


def replace_legal_article(text, replace_with="<LEGAL_ARTICLE>"):
    """
    Replace all legal article identifiers in ``text`` str with string specified by ``replace_with`` str.
    """
    return LEGAL_ARTICLE_REGEX.sub(replace_with, text)

def replace_dr_reference(text, replace_with="<DR_REF>"):
    """
    Replace all DR reference identifiers in ``text`` str with string specified by ``replace_with`` str.
    """
    return DR_REFERENCE_REGEX.sub(replace_with, text)

def replace_currency_symbols(text, replace_with="<CUR>"):
    """
    Replace all currency symbols in ``text`` str with string specified by ``replace_with`` str.
    Args:
        text (str): raw text
        replace_with (str): if None (default), replace symbols with
            their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
            otherwise, pass in a string with which to replace all symbols
            (e.g. "*CURRENCY*")
    """
    if replace_with is None:
        for k, v in CURRENCIES.items():
            text = text.replace(k, v)
        return text
    else:
        return CURRENCY_REGEX.sub(replace_with, text)
    
def replace_urls(text, replace_with="<URL>"):
    """
    Replace all URLs in ``text`` str with ``replace_with`` str.
    """
    return URL_REGEX.sub(replace_with, text)


def replace_emails(text, replace_with="<EMAIL>"):
    """
    Replace all emails in ``text`` str with ``replace_with`` str.
    """
    return EMAIL_REGEX.sub(replace_with, text)


def replace_phone_numbers(text, replace_with="<PHONE>"):
    """
    Replace all phone numbers in ``text`` str with ``replace_with`` str.
    """
    return PHONE_REGEX.sub(replace_with, text)


def replace_numbers(text, replace_with="<NUMBER>"):
    """
    Replace all numbers in ``text`` str with ``replace_with`` str.
    """
    return NUMBERS_REGEX.sub(replace_with, text)



def replace_with_tags(text: str) -> str:
    text = replace_legal_article(text, replace_with=replace_with_legal_article)
    text = replace_dr_reference(text, replace_with=replace_with_dr_reference)
    text = replace_currency_symbols(text, replace_with=replace_with_currency_symbol)
    text = replace_urls(text, replace_with=replace_with_url)
    text = replace_emails(text, replace_with=replace_with_email)
    text = replace_phone_numbers(text, replace_with=replace_with_phone_number)
    text = replace_numbers(text, replace_with=replace_with_number)
    return text


def preprocessing_pipeline(text: str) -> tuple[str, str]:
    cleaned_text = clean_text(text)
    text_w_tags = replace_with_tags(cleaned_text)
    return cleaned_text, text_w_tags