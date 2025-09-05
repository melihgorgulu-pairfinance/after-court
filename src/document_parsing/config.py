import awswrangler.secretsmanager as sm
import boto3


class AppConfig:
    AWS_REGION = "eu-central-1"
    CACHE_TIMEOUT = 60
    DEBUG = False
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_ID = "text-request-production"

    @classmethod
    def init_secrets(cls):
        """Initialize secrets from AWS Secrets Manager"""
        cls.SQLALCHEMY_DATABASE_URI = get_secret(
            cls.SECRET_ID, "ANALYTICS_DATABASE_URI"
        )
        cls.ZENDESK_BEARER_TOKEN = get_secret(cls.SECRET_ID, "ZENDESK_BEARER_TOKEN")
        

class ModelConfig:
    lost_condition_interval_days = 7
    scheduler_interval_lost_llm_hours = 2
    scheduler_interval_textract_send_minutes = 10
    scheduler_interval_textract_get_minutes = 10
    ocr_s3_bucket = "pair-scanner"
    ocr_source_path = "ocr_source_files"
    ocr_results_path = "ocr_prepared_output"
    irrelevant_file_extentions = ["vcf", "gif", "p7s", "asc", "ics", "bin"]
    unsupported_file_extentions = ["html", "mp4"]
    

def get_secret(secret_id, value):
    return sm.get_secret_json(
        secret_id,
        boto3_session=boto3.Session(region_name=AppConfig.AWS_REGION),
    ).get(value)