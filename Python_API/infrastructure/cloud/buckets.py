import uuid
from abc import abstractmethod

import boto3
from botocore.client import BaseClient


class BucketClient:
    def __init__(self, config):
        self._config = config

    def _generate_uuid_key(self, filename):
        unique_id = uuid.uuid4()  # Generates a random UUID.
        return f"{unique_id}{self._get_extension(filename)}"

    def _get_extension(self, filename):
        return filename[filename.rfind('.'):]

    @abstractmethod
    def upload_file(self, file, file_type, local, ignore_type_for_dir):
        pass

    @abstractmethod
    def download_file(self, s3_key, local_download_path):
        pass


class AWSBucketClient(BucketClient):
    __s3_client: BaseClient

    def __init__(self, config):
        super().__init__(config)
        self.__s3_client = boto3.client('s3',
                                        aws_access_key_id=config["AWS_IAM_Access_Key"],
                                        aws_secret_access_key=config["AWS_IAM_Secret_Key"],
                                        region_name=config["AWS_Region_Name"])
        self.__bucket_name = config["AWS_S3_Bucket_Name"]

    def upload_file(self, file, file_type="audio", local=True, ignore_type_for_dir=False):
        file_uuid = ""

        if local is True:
            file_uuid = self._generate_uuid_key(file)
            if file_type == "video":
                data = open(file, 'rb')
                self.__s3_client.put_object(Key="video/" + file_uuid, Body=data, ContentType='video/mp4',
                                     Bucket=self.__bucket_name)
                return

            if ignore_type_for_dir:
                self.__s3_client.upload_file(file, self.__bucket_name, file_type + "/" + file_uuid)
            else:
                self.__s3_client.upload_file(file_type + "/" + file, self.__bucket_name, file_type + "/" + file_uuid)
        else:
            file_uuid = self._generate_uuid_key(file.filename)
            self.__s3_client.upload_fileobj(file, self.__bucket_name, file_type + "/" + file_uuid)
        return f'https://{self.__bucket_name}.s3.amazonaws.com/{file_type}/{file_uuid}'

    def download_file(self, s3_key, local_download_path="temp"):
        self.__s3_client.download_file(self.__bucket_name, s3_key, local_download_path)
