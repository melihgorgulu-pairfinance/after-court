import argparse
from src.app_logger import logger
from src.document_parsing.parsing_pipeline import aws_textract_pipeline, get_texts_from_textract_outputs

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
    
    # PIPELINE 3: Ladung Va Classification
    # 3.1) Preprocessing
    
    # 1) Read the pdf document
    # 2) Parse the pdf document using AWS Textract
    # 3) Module: Ladung Va Classification: Contains: Preprocessing -> Model Prediction
    # 4) Module: Ladung VA Param Extraction: Contains 3 submodule: 4.1) Slug extraction 
    # 4.2) Debtor Name Extraction 4.3) Summon Date Extraction
    