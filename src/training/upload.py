import logging
import boto3
import shutil
from botocore.exceptions import ClientError
import os


def upload_file(file_name, bucket, object_name=None):
    if os.path.isdir(file_name):
        zip_file_name = shutil.make_archive(file_name, 'zip', file_name)
        file_name = zip_file_name
        if object_name is None:
            object_name = os.path.basename(zip_file_name)
    else:
        if object_name is None:
            object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True