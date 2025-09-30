import argparse
from src.app_logger import logger
from src.document_parsing.parsing_pipeline import aws_textract_pipeline, get_texts_from_textract_outputs
from src.preprocessing.preprocessing import preprocessing_pipeline
from src.document_classification.classification import classify_documents
from src.document_ent.document_entity_recognition import extract_slug_from_text, extract_debtor_names, extract_date_from_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LadungVa",
                                     description="Arg Parser for Ladung Va Classification and Information Extraction App")

    parser.add_argument('--doc_s3_key', help='S3 key of the document')
    
    args = parser.parse_args()
    
    logger.info(f"S3 key of the document: {args.doc_s3_key}")
    
    # PIPELINE 1: AWS Textract Document Parsing
    parsed_outputs = aws_textract_pipeline([args.doc_s3_key])
    extracted_texts: list[str] = get_texts_from_textract_outputs(parsed_outputs)
    print(f"Extracted Texts: {extracted_texts[0]}")
    # PIPELINE 2: Preprocessing
    preprocessed_texts = [preprocessing_pipeline(text) for text in extracted_texts]
    cleaned_texts, texts_with_tags = zip(*preprocessed_texts)
    # PIPELINE 3: Classification
    is_ladung = classify_documents(texts_with_tags)
    print(f"Is Ladung: {is_ladung}")
    # PIPELINE 4: Information Extraction
    cleaned_ladung_texts = [text for text, ladung in zip(cleaned_texts, is_ladung) if ladung]
    
    if cleaned_ladung_texts:
        logger.info("Proceeding to Information Extraction...")
        extracted_slugs = [extract_slug_from_text(txt) for txt in cleaned_ladung_texts]
        print("="*100)
        print(f"Extracted SLUG: {extracted_slugs}")
        extracted_debtor_names = [extract_debtor_names(txt) for txt in cleaned_ladung_texts]
        print(f"Extracted Debtor Names: {extracted_debtor_names}")
        extracted_dates = [extract_date_from_text(txt)['extracted_date'] for txt in cleaned_ladung_texts]
        print(f"Extracted Date: {extracted_dates}")
    else:
        logger.info("No Ladung Va documents found. Stopping further processing.")
