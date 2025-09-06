import joblib
from src.app_logger import logger
from pathlib import Path
import spacy
from spacy.tokenizer import Tokenizer
import re
import pickle


class ClassificationSpacyLemmaTokenizer:
    def __init__(self):
        self.special_tags_re = re.compile(r"<(?:EMAIL|PHONE|CUR|NUMBER|URL|DR_REF|LEGAL_ARTICLE)>")
        self.nlp = spacy.load("de_core_news_md", disable=["tagger", "parser", "ner", "attribute_ruler"])
        self.nlp.tokenizer = Tokenizer(
            self.nlp.vocab,
            rules=self.nlp.Defaults.tokenizer_exceptions,
            prefix_search=self.nlp.tokenizer.prefix_search,
            suffix_search=self.nlp.tokenizer.suffix_search,
            infix_finditer=self.nlp.tokenizer.infix_finditer,
            token_match=self.special_tags_re.match,
            url_match=self.nlp.tokenizer.url_match,
        )

    def __call__(self, text):
        doc = self.nlp(text)
        
        tokens_cleared = [
            token.lemma_.lower()
            for token in doc
            if (
                (not token.is_stop and not token.is_punct and not token.like_num)
                or token.text.lower() == "gegen"
            ) and (
                token.lemma_ != "\n" and token.lemma_ != "\n\n" and 
                token.lemma_ != " " and token.lemma_ != "" and 
                len(token.lemma_) < 45
            )
        ]
        return tokens_cleared

pipeline_path = Path(__file__).parent / "pipeline_model" / "classification_pipeline.pickle"
pipeline_path = pipeline_path.resolve()

logger.info(f"Loading classification pipeline from {pipeline_path}")
#classification_pipeline = joblib.load(str(pipeline_path))
classification_pipeline = pickle.load(open(str(pipeline_path), "rb"))
logger.info("Classification pipeline loaded successfully!")

def classify_documents(texts_w_tags: list[str]) -> list[bool]:
    return classification_pipeline.predict(texts_w_tags)
