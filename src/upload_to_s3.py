import argparse

from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd25'
YOUR_SURNAME = 'perepelyatnik'
LOCAL_FILE_PATH = ['models/gbr_model_v15.pkl']

config = dotenv_values(".env")


def main(args):
    client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=config['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=config['AWS_SECRET_ACCESS_KEY']
    )

    for file_path in args.input:
        object_name = f'{YOUR_SURNAME}/' + file_path.replace('\\', '/')
        client.upload_file(file_path, BUCKET_NAME, object_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                        help='Input local data files to upload to S3 storage',
                        default=LOCAL_FILE_PATH)
    args = parser.parse_args()
    main(args)