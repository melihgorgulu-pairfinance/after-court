
from typing import List, Dict, Any
import time

from src.utils import time_decorator
from src.app_logger import logger
from src.document_parsing.textract_client import TextractClient, _get_textract_client

def get_texts_from_textract_outputs(textract_outputs: List[Dict[str, Any]]) -> List[str]:
    raw_text_outputs = []
    for output in textract_outputs:
        if output is None:
            raw_text_outputs.append("")
        else:
            raw_text = []
            for cur_doc in output:
                if cur_doc["BlockType"] == "LINE":
                    raw_text.append(cur_doc["Text"])    
               
            raw_text_outputs.append("\n".join(raw_text))

    return raw_text_outputs

def submit_textract_job(textract_client: TextractClient, doc_key: str) -> Dict[str, Any]:
    """Submit a single Textract job and return job info."""
    try:
        job_id = textract_client.submit_textract_job(
            bucket_name=textract_client.model_config.ocr_s3_bucket, 
            document_key=doc_key
        )
        logger.info("Textract job submitted for document '%s' with job_id '%s'", doc_key, job_id)
        return {
            'job_id': job_id,
            'doc_key': doc_key,
            'status': 'SUBMITTED'
        }
    except Exception as e:
        logger.error("Failed to submit job for document '%s': %s", doc_key, str(e))
        return {
            'job_id': None,
            'doc_key': doc_key,
            'status': 'FAILED',
            'error': str(e)
        }


def wait_for_job_completion(textract_client: TextractClient, job_info: Dict[str, Any], max_wait_time: int = 100) -> Dict[str, Any]:
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
                logger.info("Job Completed for job_id '%s' (document: %s)", job_id, doc_key)
                return {
                    'doc_key': doc_key,
                    'result': res['result'],
                    'status': 'SUCCEEDED'
                }
            elif res["status"] == "FAILED":
                logger.error("Job Failed for job_id '%s' (document: %s)", job_id, doc_key)
                return {
                    'doc_key': doc_key,
                    'result': None,
                    'status': 'FAILED',
                    'error': 'Textract job failed'
                }
                
        except Exception as e:
            logger.error("Error checking job status for job_id '%s': %s", job_id, str(e))
            
        max_count += 1
    
    # Timeout reached
    logger.warning("Max retries exceeded for job_id '%s' (document: %s)", job_id, doc_key)
    
    return {
        'doc_key': doc_key,
        'result': None,
        'status': 'TIMEOUT',
        'error': 'Job timeout'
    }


@time_decorator
def aws_textract_pipeline(object_keys: List[str], max_workers: int = 10) -> List:
    
    textract_outputs = [None] * len(object_keys)  # Placeholder for results
    
    logger.info("Starting AWS Textract Jobs with batch processing (max {} jobs at a time)", max_workers)
    batch_size = 25
    job_results = []
    total_jobs = len(object_keys)
    submitted = 0
    
    textract_client: TextractClient = _get_textract_client()

    while submitted < total_jobs:
        current_batch_keys = object_keys[submitted:submitted+batch_size]
        job_infos = []
        # Submit jobs for current batch
        for doc_key in current_batch_keys:
            job_info = submit_textract_job(textract_client, doc_key)
            job_infos.append(job_info)
            
        logger.info(f"Submitted batch {submitted//batch_size+1}: {len(current_batch_keys)} jobs")
        
        # Wait for jobs in current batch
        batch_results = []
        for job_info in job_infos:
            result = wait_for_job_completion(textract_client, job_info)
            batch_results.append(result)
    
        job_results.extend(batch_results)
        submitted += batch_size
        
        logger.info(f"Completed batch {submitted//batch_size}: {min(submitted, total_jobs)}/{total_jobs} jobs")
        
    # Sort results to maintain original order
    doc_key_to_result = {result['doc_key']: result for result in job_results}
    ordered_results = [doc_key_to_result[doc_key] for doc_key in object_keys]
    # Process results and update cache and output array
    successful_count = 0
    failed_count = 0
    
    for result in ordered_results:
        original_index = object_keys.index(result['doc_key'])
        if result['status'] == 'SUCCEEDED':
            textract_outputs[original_index] = result['result']
            successful_count += 1
        else:
            textract_outputs[original_index] = None
            failed_count += 1
            logger.warning("Failed to process document '%s': %s", result['doc_key'], result.get('error', 'Unknown error'))
            
    logger.info("AWS Textract pipeline completed: %d successful, %d failed", successful_count, failed_count)


    # Final validation - ensure no None values where we expect results
    total_successful = sum(1 for result in textract_outputs if result is not None)
    total_failed = len(textract_outputs) - total_successful

    logger.info("Final results: %d successful, %d failed (including cached)", total_successful, total_failed)

    return textract_outputs