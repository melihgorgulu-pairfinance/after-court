import re

def remove_underscore_lines(text: str) -> str:
    lines = text.split('\n')
    filtered_lines = []
    for l in lines:
        if not re.search(r'_{10,}', l):
            filtered_lines.append(l)
    return '\n'.join(filtered_lines)

def remove_unneccessary_lines(text: str) -> str:
    lines = text.split('\n')
    
    filtered_lines = []
    for l in lines:
        if len(l)>3:
            filtered_lines.append(l)
    return '\n'.join(filtered_lines)
  
def clean_unknown_tags(text):
    # Remove HTML-encoded unknown tags
    text = text.replace('&lt;unknown&gt;', '')
    # Also remove the actual tags if present
    text = text.replace('<unknown>', '')
    return text

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

def clean_text(text):
    text = text.replace('## ', '')
    text = text.replace('\t', ' ')
    text = remove_underscore_lines(text)
    text = remove_unneccessary_lines(text)
    text = clean_unknown_tags(text)
    text = remove_html_tags(text)
    text = re.sub(r"[\uE000-\uF8FF]", "", text) # remove problematic emoji parses
    text = text.lower()
    return text