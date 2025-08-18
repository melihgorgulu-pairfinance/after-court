import boto3


class TextractClient:
    def __init__(self, app_config, model_config):
        self.model_config = model_config
        self.app_config = app_config
        self.boto3_session = boto3.Session(region_name=self.app_config.AWS_REGION)
        self.client = self.boto3_session.client("textract")

    def submit_textract_job(self, bucket_name, document_key):
        try:
            response = self.client.start_document_text_detection(
                DocumentLocation={
                    "S3Object": {"Bucket": bucket_name, "Name": document_key}
                }
            )
            return response["JobId"]
        except Exception as e:
            print(
                f"Error submitting Textract job for document {document_key}: {str(e)}"
            )
            raise

    def check_job_status(self, job_id):
        try:
            response = self.client.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]

            if status == "SUCCEEDED":
                result = self.get_job_results(job_id)
                return {"status": status, "result": result}
            else:
                return {"status": status}

        except Exception as e:
            print(f"Error checking status for job {job_id}: {str(e)}")
            return {"status": "ERROR"}

    def get_job_results(self, job_id):
        results = []
        pagination_token = None
        while True:
            if pagination_token:
                response = self.client.get_document_text_detection(
                    JobId=job_id, NextToken=pagination_token
                )
            else:
                response = self.client.get_document_text_detection(JobId=job_id)

            results.extend(response["Blocks"])

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                break

        return results
