import argparse
from src.app_logger import logger
from src.document_parsing.parsing_pipeline import aws_textract_pipeline, get_texts_from_textract_outputs
from src.preprocessing.preprocessing import preprocessing_pipeline
from src.document_classification.classification import classify_documents

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
    print(f"Cleaned Texts: {cleaned_texts[0]}")
    print(f"Texts with Tags: {texts_with_tags[0]}")
    # PIPELINE 3: Classification
    is_ladung = classify_documents(texts_with_tags)
    print(f"Is Ladung: {is_ladung}")
    # PIPELINE 4: Information Extraction
    if is_ladung[0]:
        logger.info("Proceeding to Information Extraction...")
    else:
        logger.info("Document is not a Ladung Va. Stopping further processing.")
