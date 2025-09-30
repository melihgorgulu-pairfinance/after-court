import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import dateparser
import spacy
from spacy.matcher import Matcher

def extract_slug_from_text(text: str) -> Optional[str]:
    """
    Extract a slug (12-13 digit number starting with 1) from given text using regex only.
    
    Args:
        text (str): Input text to extract slug from

    Returns:
        str: Extracted slug as string, or None if no valid slug found
    """
    
    def is_valid_slug_context(text: str, match_start: int, match_end: int, window: int = 20) -> bool:
        """Check if the slug is valid by examining its context"""
        # Get context around the match
        context_start = max(0, match_start - window)
        context_end = min(len(text), match_end + window)
        context = text[context_start:context_end].lower()
        context = context.replace('\n', '').replace(' ', '')
        
        # Check for invalid pattern (slug with dashes)
        pattern = r'-1[0-9]{11,12}-'
        if re.search(pattern, context):
            return False
        
        return True

    def select_best_slug(slug_candidates: list) -> Optional[str]:
        """Select the best slug from candidates"""
        # Clean slugs by removing non-digit characters
        cleaned_slugs = [re.sub(r'\D', '', slug) for slug in slug_candidates]
        
        if not cleaned_slugs:
            return None
            
        # If all slugs are the same, return the first one
        if len(set(cleaned_slugs)) == 1:
            return cleaned_slugs[0]
        else:
            # Return the last slug (bottom slug as per original logic)
            return cleaned_slugs[-1]
    
    # Find all potential slugs (12-13 digits starting with 1)
    slug_pattern = r'\b1[0-9]{11,12}\b'
    matches = list(re.finditer(slug_pattern, text))
    
    if not matches:
        return None
    
    # Filter valid slugs based on context
    valid_slugs = []
    for match in matches:
        if is_valid_slug_context(text, match.start(), match.end()):
            valid_slugs.append(match.group())
    
    if not valid_slugs:
        return None
    
    # Select the best slug
    selected_slug = select_best_slug(valid_slugs)
    return selected_slug


class DebtorNameExtractor:
    def __init__(self, nlp_model):
        """
        Initialize the debtor name extractor
        
        Args:
            nlp_model: spaCy language model (e.g., spacy.load("de_core_news_sm"))
        """
        self.nlp = nlp_model
        self.gegen_matcher = Matcher(self.nlp.vocab)
        pattern = [{"LOWER": "gegen"}]
        self.gegen_matcher.add("GEGEN_PATTERN", [pattern])
    
    def is_valid_context(self, left_ctx, right_ctx) -> bool:
        """Check if context contains valid entities (PER/LOC/ORG)"""
        left_entities_labels = [ent.label_ for ent in left_ctx.ents]
        right_entities_labels = [ent.label_ for ent in right_ctx.ents]

        if not (any(label in left_entities_labels for label in ["PER", "LOC", "ORG"]) or 
                any(label in right_entities_labels for label in ["PER", "LOC", "ORG"])):
            return False

        return True

    def get_context_window(self, doc, start, end, window=10) -> Optional[spacy.tokens.span.Span]:
        """Extract context window around the 'gegen' match"""
        left_start = max(start - window, 0)
        right_end = min(end + window, len(doc))
        left_ctx = doc[left_start:start]
        right_ctx = doc[end:right_end]
        flag = self.is_valid_context(left_ctx, right_ctx)
        if flag:
            return right_ctx
        else:
            return None

    def collect_context_windows(self, text: str) -> Optional[List[spacy.tokens.span.Span]]:
        """Find context windows around 'gegen' matches"""
        doc = self.nlp(text)
        gegen_matches = self.gegen_matcher(doc)
        if not gegen_matches:
            return None
        context_windows = []
        for _, start, end in gegen_matches:
            context = self.get_context_window(doc, start, end, window=15)
            if context:
                context_windows.append(context)
                
        if not context_windows:
            return None

        return context_windows[0]  # Return only the first valid context window

    def find_used_delimeter(self, context) -> Optional[str]:
        """Find the most frequently used delimiter in the context"""
        used_delimeter = defaultdict(int)
        for t in context:
            if t.is_punct and not t.is_space:
                used_delimeter[t.text] += 1
        # get the most frequent
        most_frequent = max(used_delimeter, key=used_delimeter.get, default=None)
        return most_frequent

    def validate_debtor_name(self, debtor_name: str) -> Optional[str]:
        """Clean and validate the extracted debtor name"""
        if not debtor_name:
            return None
            
        debtor_name = debtor_name.lower()
        debtor_name = re.sub(r'\b(herrn?|frau|fräulein)\b', '', debtor_name)
        debtor_name = debtor_name.replace('c/o', '')
        # Remove numbers and punctuation, keep only letters, spaces, and 'ß'
        debtor_name = re.sub(r'[^a-zA-ZÀ-ÿß\s-]', '', debtor_name)
        debtor_name = debtor_name.lstrip()
        # remove after \n detected
        debtor_name = re.sub(r'\n.*', '', debtor_name)
        # Clean up multiple spaces and strip
        debtor_name = re.sub(r'\s+', ' ', debtor_name).strip()
        # Name and Surname is capitalized
        debtor_name = ' '.join(word.capitalize() for word in debtor_name.split())
        return debtor_name

    def contains_salutation(self, text: str) -> bool:
        """Check if text contains German salutations"""
        salutation_patterns = [
            r'\b(herrn?|frau|fräulein)\b',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in salutation_patterns)

    def extract_debtor_name(self, context_window) -> Optional[str]:
        """Extract debtor name from context window"""
        if not context_window:
            return None
            
        debtor_name: str = None
        delimeter = self.find_used_delimeter(context_window)
        number_of_blocks = len(context_window.text.split(delimeter)) if delimeter else 3
        
        if number_of_blocks <= 3:
            # first level: extract first person entity from the span.
            per_ent = [ent for ent in context_window.ents if ent.label_ == "PER"]
            if per_ent:
                # check number of per_ent token, if it's less than two, take one more token
                first_ent = per_ent[0]
                n_of_tokens = len(first_ent)
                if self.contains_salutation(first_ent.text):
                    n_of_tokens -=1
                if n_of_tokens < 2:
                    # take until first delimeter token
                    first_delimeter_pos = next((i for i, token in enumerate(context_window) if token.text == delimeter), None)
                    debtor_name = context_window[:first_delimeter_pos].text.strip()
                else:
                    debtor_name = per_ent[0].text
                return debtor_name
        
            # second level: Find debtor name using delimeter seperation pattern (mostly ,).
            if delimeter == ',':
                debtor_name = context_window.text.split(delimeter)[0].strip()
            elif delimeter == ':':
                for idx, token in enumerate(context_window):
                    if token.is_stop:
                        if idx > 0:
                            debtor_name = context_window[:idx].text.replace(":","").strip()
                            if debtor_name == '':
                                debtor_name = None
                            break
            else:
                debtor_name = context_window.text.strip().split('\n')[0].strip()
        else:
            delimeter_i = [i for i, tok in enumerate(context_window) if tok.text == delimeter] # locations of delimeters
            if len(delimeter_i) >= 2:
                # first check if two blocks contains PER
                block_1 = context_window[:delimeter_i[0]]
                block_2 = context_window[delimeter_i[0]+1:delimeter_i[1]]
                per_ents_block_1 = [ent for ent in block_1.ents if ent.label_ == "PER"]
                per_ents_block_2 = [ent for ent in block_2.ents if ent.label_ == "PER"]
                if per_ents_block_1 and per_ents_block_2 and 'c/o' not in block_2.text.lower():
                    debtor_name = per_ents_block_1[0].text + ", " + per_ents_block_2[0].text
                    return debtor_name

                if per_ents_block_1:
                    first_per_ent_block_1 = per_ents_block_1[0]
                    n_of_tokens_block_1 = len(first_per_ent_block_1)
                    if self.contains_salutation(first_per_ent_block_1.text):
                        n_of_tokens_block_1 -= 1
                    if first_per_ent_block_1 and n_of_tokens_block_1 > 1:
                        debtor_name = first_per_ent_block_1.text
                        return debtor_name

                if block_1 and block_2:
                    debtor_name = block_1.text.strip() + ", " + block_2.text.strip()

        return debtor_name

    def valid_debtor_name_extraction(self, context_window) -> str:
        """Advanced extraction with validation"""
        debtor_name: str = self.extract_debtor_name(context_window)
        if debtor_name:
            debtor_name: str = self.validate_debtor_name(debtor_name)
        return debtor_name

    def extract_debtor_names_from_text(self, text: str) -> Optional[str]:
        """
        Main function to extract debtor names from text
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Dictionary containing extracted names and context
        """
        # Get context window
        context_window = self.collect_context_windows(text)
        
        # Extract debtor name
        valid_debtor_name = self.valid_debtor_name_extraction(context_window) if context_window else None

        return valid_debtor_name

# Example usage function
def extract_debtor_names(text: str, nlp_model=None):
    """
    Convenience function to extract debtor names from text
    
    Args:
        text (str): Input text
        nlp_model: spaCy model (if None, will try to load German model)
    
    Returns:
        dict: Extracted debtor names and context
    """
    if nlp_model is None:
        try:
            nlp_model = spacy.load("de_core_news_md")
        except OSError:
            print("German spaCy model not found. Please install it with:")
            print("python -m spacy download de_core_news_sm")
            return None
    
    extractor = DebtorNameExtractor(nlp_model)
    return extractor.extract_debtor_names_from_text(text)




class DateExtractor:
    def __init__(self):
        """Initialize the date extractor with regex patterns"""
        self.DATE_REGEX = re.compile(r"""(?x)
        (?<!\d)
        (?:
          (?:(?:0?[1-9]|[12]\d)|3[01])\s?[./:-][\s.]?(?:0?[13578]|1[02]|J(?:an(?:uar)?|uli?)|M(?:ärz?|ai)|Aug(?:ust)?|Dez(?:ember)?|Okt(?:ober)?)\s?(?:[./:-][\s.]?)?(?:[1-9]\d\d\d|\d\d)|
          (?:(?:0?[1-9]|[12]\d)|30)\s?[./:-][\s.]?(?:0?[13-9]|1[012]|J(?:an(?:uar)?|u[nl]i?)|M(?:ärz?|ai)|A(?:pr(?:il)?|ug(?:ust)?)|Sep(?:tember)?|(?:Nov|Dez)(?:ember)?|Okt(?:ober)?)\s?(?:[./:-][\s.]?)?(?:[1-9]\d\d\d|\d\d)|
          (?:0?[1-9]|[12]\d)\s?[./:-][\s.]?(?:0?2|Fe(?:b(?:ruar)?)?|(?:0?2|Feb(?:r(?:uar)?)?))\s?(?:[./:-][\s.]?)?(?:[1-9]\d(?:[02468][048]|[13579][26])|\d\d)|
          (?:0?[1-9]|[12][0-9])\s?[./:-][\s.]?(?:0?2|Fe(?:b(?:ruar)?)?|(?:0?2|Feb(?:r(?:uar)?)?))\s?(?:[./:-][\s.]?)?(?:[1-9]\d\d\d|\d\d)
        )
        (?!\d)""", re.IGNORECASE | re.VERBOSE)
        
        self.febr_pattern = re.compile(r'\bFebr\.?\b', flags=re.IGNORECASE)
        
        # Keyword patterns for context analysis
        self.keyword_patterns = {
            'vermogensauskunft': re.compile(r'\bverm[oö]gensauskunft\b', re.IGNORECASE),
            'gegen': re.compile(r'\bgegen\b', re.IGNORECASE),
            'bestimmt': re.compile(r'\bbestimmt\b', re.IGNORECASE),
            'sehr_geehrte': re.compile(r'\bsehr geehrte?\b', re.IGNORECASE),
            'mit_freundlichen': re.compile(r'\bmit freundlichen\b', re.IGNORECASE),
            'termin': re.compile(r'\btermin\b', re.IGNORECASE)
        }
        
        # Proximity thresholds for keyword analysis
        self.proximity_thresholds = {
            'vermogensauskunft': 60,
            'gegen': 60,
            'bestimmt': 90,
            'sehr_geehrte': 100,
            'mit_freundlichen': 100,
            'termin': 90
        }
        
        # Weights for keyword scoring
        self.keyword_weights = {
            'vermogensauskunft': 4,
            'gegen': 1,
            'bestimmt': 2,
            'termin': 2,
        }

    def extract_dates_with_regex(self, txt: str) -> Optional[List[Tuple[str, int, int]]]:
        """Extract all date matches from text using regex"""
        if not txt:
            return None
        
        all_dates = self.DATE_REGEX.finditer(txt)
        date_matches = list(all_dates)
        date_out = []

        for match in date_matches:
            date_out.append((match.group(), match.start(), match.end()))

        return date_out if date_out else None

    def extract_keyword_positions(self, text: str) -> Dict[str, List[Dict]]:
        """Extract positions of all keywords in the text"""
        keyword_positions = {}
        
        for keyword_type, pattern in self.keyword_patterns.items():
            matches = []
            for match in pattern.finditer(text):
                matches.append({
                    'text': match.group(),
                    'start_char': match.start(),
                    'end_char': match.end()
                })
            if matches:
                keyword_positions[keyword_type] = matches
                
        return keyword_positions

    def get_dates_keywords_statistics(self, text: str, date_matches: List[Tuple[str, int, int]]) -> Optional[List[Dict]]:
        """Compare date positions with keyword positions"""
        if not date_matches:
            return None
        
        keyword_positions = self.extract_keyword_positions(text)
        
        comparisons = []
        for date_match in date_matches:
            date_text, date_start, date_end = date_match

            comparison = {
                'date_text': date_text,
                'date_start': date_start,
                'date_end': date_end,
                'nearby_keywords': {}
            }
            
            for keyword_type, keyword_list in keyword_positions.items():
                nearby_keywords = []
                for keyword in keyword_list:
                    distance = min(
                        abs(keyword['start_char'] - date_end),
                        abs(keyword['end_char'] - date_start)
                    )
                    if distance <= self.proximity_thresholds.get(keyword_type, float('inf')):
                        nearby_keywords.append({
                            'keyword_text': keyword['text'],
                            'distance': distance,
                            'keyword_start': keyword['start_char'],
                            'keyword_end': keyword['end_char'],
                            'is_date_after_this_keyword': date_start > keyword['end_char'],
                            'is_date_before_this_keyword': date_start < keyword['start_char']
                        })
                
                if nearby_keywords:
                    comparison['nearby_keywords'][keyword_type] = nearby_keywords
            
            comparisons.append(comparison)
        
        return comparisons

    def is_date_valid(self, date_info: Dict) -> bool:
        """Validate if a date is valid based on parsing and context"""
        # 1) Check if it's parseable
        date_text = date_info['date_text']
        parsed_date = dateparser.parse(date_text, languages=['de'])
        if parsed_date is None:
            return False
        
        if '.' not in date_text:
            # Check UTC, it should be 00:00:00
            if parsed_date.hour != 0 or parsed_date.minute != 0 or parsed_date.second != 0:
                return False
        
        if parsed_date.year < 2000:
            return False
            
        # 2) Check keywords
        keywords = date_info['nearby_keywords']
        if not keywords:
            return True

        # Find the closest keyword instance by minimum distance
        # ver_list = keywords.get('vermogensauskunft', [])
        # ver = min(ver_list, key=lambda x: x.get('distance', float('inf'))) if ver_list else None
        gegen_list = keywords.get('gegen', [])
        # besti_list = keywords.get('bestimmt', [])
        # termin_list = keywords.get('termin', [])

        gegen = min(gegen_list, key=lambda x: x.get('distance', float('inf'))) if gegen_list else None
        # besti = min(besti_list, key=lambda x: x.get('distance', float('inf'))) if besti_list else None
        # termin = min(termin_list, key=lambda x: x.get('distance', float('inf'))) if termin_list else None

        mit = keywords.get('mit_freundlichen', [None])[0]
        sehr = keywords.get('sehr_geehrte', [None])[0]

        # If the date is before greeting or after goodbye, it is not our target
        if sehr:
            if sehr.get('distance', 0) < 5 or sehr.get('is_date_before_this_keyword', False):
                return False

        if mit:
            if mit.get('is_date_after_this_keyword', False):
                return False
            
        if gegen:
            if gegen.get('is_date_before_this_keyword', False):
                return False
        
        return True

    def filter_dates(self, extracted_dates: List[Dict]) -> List[Dict]:
        """Filter dates to keep only valid ones"""
        if not extracted_dates:
            return []
        
        valid_dates = []
        for date_info in extracted_dates:
            if self.is_date_valid(date_info):
                valid_dates.append(date_info)
        return valid_dates

    def select_date_by_keyword_scoring(self, dates_with_keywords: List[Dict]) -> Optional[str]:
        """Select the best date based on keyword proximity scoring"""
        best_score = -1
        best_date_text = None
        
        for date in dates_with_keywords:
            score = 0
            keywords = date.get('nearby_keywords', {})
            
            for keyword, instances in keywords.items():
                if keyword in self.keyword_weights:
                    # Use the closest instance for scoring
                    closest_instance = min(instances, key=lambda x: x['distance'])
                    score += self.keyword_weights[keyword] / ((closest_instance['distance'] + 1)**0.5)

            if score > best_score:
                best_score = score
                best_date_text = date['date_text']
        
        return best_date_text

    def select_best_date_custom(self, extracted_dates: List[Dict]) -> Optional[datetime]:
        """Select the best date using custom logic"""
        valid_dates = self.filter_dates(extracted_dates)
        selected_date = None
        
        if not valid_dates:
            return None
            
        # Selection criteria 1: Select the date with important keywords around
        important_keywords = {'vermogensauskunft', 'gegen', 'bestimmt', 'termin'}
        dates_with_important_keywords = [
            date for date in valid_dates
            if 'nearby_keywords' in date and any(
                keyword in date['nearby_keywords'] for keyword in important_keywords
            )
        ]
        
        if dates_with_important_keywords:
            if len(dates_with_important_keywords) == 1:
                selected_date = dateparser.parse(dates_with_important_keywords[0]['date_text'], languages=['de'])
            else:
                selected_date_text = self.select_date_by_keyword_scoring(dates_with_important_keywords)
                selected_date = dateparser.parse(selected_date_text, languages=['de'])

        if selected_date is None:
            # If none of the dates have keywords around, take the last date
            valid_dates_parsed = [dateparser.parse(date['date_text'], languages=['de']) for date in valid_dates]
            valid_dates_parsed = list(dict.fromkeys(valid_dates_parsed))  # unique dates while preserving order
            selected_date = valid_dates_parsed[-1]  # get the last one
            
        return selected_date

    def extract_date_from_text(self, text: str) -> Dict[str, Union[datetime, List, None]]:
        """
        Main function to extract the best date from text
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Dictionary containing extracted date and analysis data
        """
        # Extract all dates with regex
        date_matches = self.extract_dates_with_regex(text)
        
        if not date_matches:
            return {
                'extracted_date': None,
                'all_date_matches': None,
                'date_keyword_statistics': None,
                'valid_dates_count': 0
            }
        
        # Get keyword statistics for each date
        date_keyword_stats = self.get_dates_keywords_statistics(text, date_matches)
        
        # Select the best date
        final_date = self.select_best_date_custom(date_keyword_stats) if date_keyword_stats else None
        
        # Count valid dates
        valid_dates = self.filter_dates(date_keyword_stats) if date_keyword_stats else []
        
        return {
            'extracted_date': final_date,
            'all_date_matches': date_matches,
            'date_keyword_statistics': date_keyword_stats,
            'valid_dates_count': len(valid_dates)
        }


# Convenience function for easy usage
def extract_date_from_text(text: str) -> Dict[str, Union[datetime, List, None]]:
    """
    Convenience function to extract date from text
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Extracted date and analysis information
    """
    extractor = DateExtractor()
    return extractor.extract_date_from_text(text)