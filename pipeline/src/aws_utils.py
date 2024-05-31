from pathlib import Path
import logging
from dataclasses import dataclass
import boto3 # type: ignore
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)

def upload_artifacts(artifacts: Path, bucket:str, prefix:str) -> None:
    """Upload all the artifacts in the specified directory to S3


    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        bucket: Bucket that want to upload to 
        key: the key that we save in the bucket

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    try:
        s3_client = boto3.client('s3')

        for artifact_path in artifacts.glob('**/*'):
            if artifact_path.is_file():
                s3_key = f'{prefix}/{artifact_path.relative_to(artifacts)}'
                try:
                    s3_client.upload_file(str(artifact_path), bucket, s3_key)
                except ClientError as err:
                    logging.error(err)
    except Exception as err:
        logger.error('Error while uploading artifacts to S3')
        raise err
    logger.info('Artifacts uploaded to S3 successfully')