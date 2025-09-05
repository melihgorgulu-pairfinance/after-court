import argparse
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LadungVa",
                                     description="Arg Parser for Ladung Va Classification and Information Extraction App")

    parser.add_argument('--file_path', help='File path of the document')
    
    args = parser.parse_args()
    
    print(args.file_path)
    