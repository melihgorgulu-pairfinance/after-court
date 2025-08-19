"""
Document Parsing Pipeline with AWS Textract and Docling

This module provides document parsing capabilities using AWS Textract with built-in caching support.

CACHING MECHANISM:
The AWS Textract pipeline now includes a basic caching mechanism that:
- Stores successful Textract results to avoid reprocessing the same documents
- Uses MD5 hash of object keys as cache keys for uniqueness
- Stores cache files as JSON in ./cache/textract/ directory
- Automatically skips cached documents on subsequent runs
- Handles cache corruption by removing invalid cache files

USAGE:
1. Basic usage with cache enabled (default):
   results = aws_textract_pipeline(object_keys, use_cache=True)

2. Disable caching:
   results = aws_textract_pipeline(object_keys, use_cache=False)

3. Cache management:
   manage_textract_cache('stats')  # Show cache statistics
   manage_textract_cache('clear')  # Clear all cached results

BENEFITS:
- If pipeline fails, re-running will use cached results for already processed documents
- Significant time and cost savings for large document sets
- Automatic handling of partial failures

CACHE LOCATION: ./cache/textract/
Each cached file is named with an MD5 hash of the object key and stored as JSON.
"""

import os
from typing import Any, List, Dict
import logging
import time
import json
import hashlib
from collections.abc import Iterable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import get_object_key, time_decorator
from docling.document_converter import DocumentConverter, PdfFormatOption
from textract_client import TextractClient
from config import AppConfig, ModelConfig
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions



_log = logging.getLogger(__name__)
if not _log.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    _log.addHandler(handler)
_log.setLevel(logging.INFO)


class TextractCache:
    """Simple file-based cache for Textract results."""
    
    def __init__(self, cache_dir: str = "./cache/textract"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Textract cache initialized at: {self.cache_dir}")
    
    def _get_cache_key(self, object_key: str) -> str:
        """Generate a unique cache key from the object key."""
        return hashlib.md5(object_key.encode()).hexdigest()
    
    def _get_cache_file_path(self, object_key: str) -> Path:
        """Get the cache file path for a given object key."""
        cache_key = self._get_cache_key(object_key)
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, object_key: str) -> Dict[str, Any] | None:
        """Retrieve cached result for an object key."""
        cache_file = self._get_cache_file_path(object_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                _log.info(f"Cache hit for object key: {object_key}")
                return cached_data
            except (json.JSONDecodeError, IOError) as e:
                _log.warning(f"Failed to read cache file for {object_key}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        return None
    
    def set(self, object_key: str, result: List[Dict[str, Any]]) -> None:
        """Store result in cache for an object key."""
        cache_file = self._get_cache_file_path(object_key)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            _log.info(f"Cached result for object key: {object_key}")
        except (IOError, TypeError) as e:
            _log.warning(f"Failed to cache result for {object_key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _log.info("Textract cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get basic cache statistics."""
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "total_files": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


def manage_textract_cache(action: str, cache_dir: str = "./cache/textract") -> None:
    """Utility function to manage Textract cache.
    
    Args:
        action: 'stats' to show cache statistics, 'clear' to clear cache
        cache_dir: Directory where cache is stored
    """
    cache = TextractCache(cache_dir)
    
    if action == 'stats':
        stats = cache.get_cache_stats()
        _log.info(f"Cache statistics: {stats['total_files']} files, {stats['total_size_mb']} MB")
        print(f"Cache directory: {cache.cache_dir}")
        print(f"Total cached files: {stats['total_files']}")
        print(f"Total cache size: {stats['total_size_mb']} MB")
    elif action == 'clear':
        cache.clear()
        print("Cache cleared successfully")
    else:
        print(f"Unknown action: {action}. Use 'stats' or 'clear'")


def get_texts_from_textract_output(textract_output: List[Dict[str, Any]]) -> List[str]:
    out = []
    for doc in textract_output:
        if doc["BlockType"] == "LINE":
            out.append(doc["Text"])
    return "\n".join(out)


def submit_textract_job(textract_client, model_config, doc_key: str) -> Dict[str, Any]:
    """Submit a single Textract job and return job info."""
    try:
        job_id = textract_client.submit_textract_job(
            bucket_name=model_config.ocr_s3_bucket, 
            document_key=doc_key
        )
        _log.info("Textract job submitted for document '%s' with job_id '%s'", doc_key, job_id)
        return {
            'job_id': job_id,
            'doc_key': doc_key,
            'status': 'SUBMITTED'
        }
    except Exception as e:
        _log.error("Failed to submit job for document '%s': %s", doc_key, str(e))
        return {
            'job_id': None,
            'doc_key': doc_key,
            'status': 'FAILED',
            'error': str(e)
        }


def wait_for_job_completion(textract_client, job_info: Dict[str, Any], max_wait_time: int = 100) -> Dict[str, Any]:
    """Wait for a Textract job to complete and return the result."""
    job_id = job_info['job_id']
    doc_key = job_info['doc_key']
    
    if job_info['status'] == 'FAILED':
        return {
            'doc_key': doc_key,
            'result': None,
            'status': 'FAILED',
            'error': job_info.get('error', 'Job submission failed')
        }
    
    max_count = 0
    while max_count < max_wait_time:
        try:
            time.sleep(1)
            res = textract_client.check_job_status(job_id=job_id)
            
            if res["status"] == "SUCCEEDED":
                _log.info("Job Completed for job_id '%s' (document: %s)", job_id, doc_key)
                return {
                    'doc_key': doc_key,
                    'result': res['result'],
                    'status': 'SUCCEEDED'
                }
            elif res["status"] == "FAILED":
                _log.error("Job Failed for job_id '%s' (document: %s)", job_id, doc_key)
                return {
                    'doc_key': doc_key,
                    'result': None,
                    'status': 'FAILED',
                    'error': 'Textract job failed'
                }
                
        except Exception as e:
            _log.error("Error checking job status for job_id '%s': %s", job_id, str(e))
            
        max_count += 1
    
    # Timeout reached
    _log.warning("Max retries exceeded for job_id '%s' (document: %s)", job_id, doc_key)
    return {
        'doc_key': doc_key,
        'result': None,
        'status': 'TIMEOUT',
        'error': 'Job timeout'
    }


@time_decorator
def aws_textract_pipeline(object_keys: List[str], max_workers: int = 10, use_cache: bool = True) -> List:
    """
    Enhanced AWS Textract pipeline with parallel processing and caching support.
    
    Args:
        object_keys: List of S3 object keys to process
        max_workers: Maximum number of parallel workers for job submission and monitoring
        use_cache: Whether to use caching mechanism (default: True)
    
    Returns:
        List of Textract outputs in the same order as input object_keys
    """
    app_config = AppConfig()
    app_config.init_secrets()  # Initialize secrets before using them
    model_config = ModelConfig()
    textract_client = TextractClient(app_config, model_config)
    _log.info("Textract client initialized!")
    
    # Initialize cache
    cache = TextractCache() if use_cache else None
    if cache:
        cache_stats = cache.get_cache_stats()
        _log.info(f"Cache stats: {cache_stats['total_files']} files, {cache_stats['total_size_mb']} MB")
    
    # Check cache for existing results
    cached_results = {}
    uncached_keys = []
    
    if cache:
        _log.info("Checking cache for existing results...")
        for key in object_keys:
            cached_result = cache.get(key)
            if cached_result is not None:
                cached_results[key] = cached_result
            else:
                uncached_keys.append(key)
        
        _log.info(f"Found {len(cached_results)} cached results, {len(uncached_keys)} need processing")
    else:
        uncached_keys = object_keys.copy()
    
    # Initialize results array with cached data
    textract_outputs = [None] * len(object_keys)
    for i, key in enumerate(object_keys):
        if key in cached_results:
            textract_outputs[i] = cached_results[key]
    
    # Process uncached documents if any
    if uncached_keys:
        _log.info("Starting AWS Textract Jobs with parallel processing (max_workers=%d)", max_workers)
        
        # Phase 1: Submit all jobs in parallel
        _log.info("Phase 1: Submitting %d jobs in parallel...", len(uncached_keys))
        job_infos = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_doc = {
                executor.submit(submit_textract_job, textract_client, model_config, doc_key): doc_key 
                for doc_key in uncached_keys
            }
            
            # Collect job submission results
            for future in as_completed(future_to_doc):
                job_info = future.result()
                job_infos.append(job_info)
        
        # Sort job_infos to maintain original order
        doc_key_to_job_info = {info['doc_key']: info for info in job_infos}
        ordered_job_infos = [doc_key_to_job_info[doc_key] for doc_key in uncached_keys]
        
        successful_jobs = [info for info in ordered_job_infos if info['status'] == 'SUBMITTED']
        _log.info("Successfully submitted %d/%d jobs", len(successful_jobs), len(uncached_keys))
        
        # Phase 2: Wait for all jobs to complete in parallel
        _log.info("Phase 2: Waiting for job completion...")
        job_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit monitoring tasks for all jobs
            future_to_job = {
                executor.submit(wait_for_job_completion, textract_client, job_info): job_info 
                for job_info in ordered_job_infos
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_job):
                result = future.result()
                job_results.append(result)
                completed_count += 1
                
                if completed_count % 10 == 0 or completed_count == len(uncached_keys):
                    _log.info("Completed %d/%d jobs", completed_count, len(uncached_keys))
        
        # Sort results to maintain original order
        doc_key_to_result = {result['doc_key']: result for result in job_results}
        ordered_results = [doc_key_to_result[doc_key] for doc_key in uncached_keys]
        
        # Process results and update cache and output array
        successful_count = 0
        failed_count = 0
        
        for result in ordered_results:
            # Find the index in the original object_keys list
            original_index = object_keys.index(result['doc_key'])
            
            if result['status'] == 'SUCCEEDED':
                textract_outputs[original_index] = result['result']
                successful_count += 1
                
                # Cache successful result
                if cache:
                    cache.set(result['doc_key'], result['result'])
            else:
                textract_outputs[original_index] = None
                failed_count += 1
                _log.warning("Failed to process document '%s': %s", 
                            result['doc_key'], result.get('error', 'Unknown error'))
        
        _log.info("AWS Textract pipeline completed: %d successful, %d failed", 
                  successful_count, failed_count)
    else:
        _log.info("All documents found in cache, no processing needed")
    
    # Final validation - ensure no None values where we expect results
    total_successful = sum(1 for result in textract_outputs if result is not None)
    total_failed = len(textract_outputs) - total_successful
    
    _log.info("Final results: %d successful, %d failed (including cached)", 
              total_successful, total_failed)
    
    return textract_outputs


def docling_parse_single(converter: DocumentConverter, document_path: str, output_path: str = None) -> str:
    
    result = converter.convert(document_path)
    text = result.document.export_to_markdown(
            delim="\n\n",
            from_element=0,
            to_element=1000000,
            labels=None,
            escape_underscores=False,
            image_placeholder="",
        )
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
    return text


@time_decorator
def docling_parse_pipeline_custom(document_paths: list[str], output_dir: str = None) -> List[str]:
    converter = DocumentConverter()
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_paths = [os.path.join(output_dir, os.path.basename(path).replace('.pdf','.txt')) for path in document_paths]
        
    texts = []
    n = len(document_paths)
    for idx, doc_path in enumerate(document_paths):
        if output_dir:
            output_path = output_paths[idx]
        else:
            output_path = None
            
        text = docling_parse_single(converter, doc_path, output_path)
        texts.append(text)
        print(f"Processed document {idx + 1}/{n}")
        if idx % 10 == 0:
            # clear console
            os.system('cls' if os.name == 'nt' else 'clear')

    return texts


def process_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0
    texts = []
    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem
        text = conv_res.document.export_to_markdown()
        texts.append(text)
        with (output_dir / f"{doc_filename}.txt").open("w") as fp:
            fp.write(text)
        
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            # Export Docling document format to text:

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return texts


def docling_batch_parse_pipeline(input_doc_paths, output_dir):
    logging.basicConfig(level=logging.INFO)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = False
    pipeline_options.do_ocr = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )
    start_time = time.time()
    conv_results = doc_converter.convert_all(
        input_doc_paths,
        raises_on_error=False,
    )
    texts = process_documents(conv_results, output_dir=Path(output_dir))

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    return texts



if __name__ == "__main__":
    import pandas as pd

    output_dir = "./data/demo_data/pdf_parsed/textract"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Example cache management (uncomment to use):
    manage_textract_cache('stats')  # Show cache statistics
    # manage_textract_cache('clear')  # Clear cache
        
    data = pd.read_csv(r"./data/demo_data/data_demo_pdfs_parsed.csv") # this csv contains parsed text with docling too.
    object_keys: List[str] = [get_object_key(document_file_name, created_at, doc_type) for document_file_name, created_at, doc_type in zip(data['document_file_name'], data['created_at'], data['document_type'])]
    
    max_workers = min(10, len(object_keys)) 
    _log.info("Processing %d documents with %d parallel workers", len(object_keys), max_workers)
    
    # Use cache by default, set use_cache=False to disable caching
    textract_outputs: List = aws_textract_pipeline(object_keys, max_workers=max_workers, use_cache=True)
    _log.info("Parsing AWS Texract Outputs...")
    
    # Handle None values in textract_outputs (failed documents)
    parsed_text_data: List[str] = []
    for textract_out in textract_outputs:
        if textract_out is not None:
            parsed_text_data.append(get_texts_from_textract_output(textract_out))
        else:
            parsed_text_data.append("")  # Empty string for failed documents
    
    data['textract_output'] = textract_outputs
    data['raw_parsed_textract'] = parsed_text_data
    data['object_key'] = object_keys
    
    _log.info("AWS Textract parse pipeline completed!")

    # save dataset
    data.to_csv(r"./data/demo_data/data_demo_pdfs_parsed_textract.csv", index=False)
    