from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import json


class Uploader:
    def __init__(self, creds):
        self.creds = creds

    def upload_file(self):
        credentials_dict = {
            'type': self.creds['type'],
            'client_id': self.creds['client_id'],
            'client_email': self.creds['client_email'],
            'private_key_id': self.creds['private_key_id'],
            'private_key': self.creds['private_key'],
        }
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            credentials_dict
        )
        client = storage.Client(credentials=credentials, project='assignment3kassym')
        bucket = client.get_bucket('mybucket')
        blob = bucket.blob('myfile.txt')
        blob.upload_from_filename('myfile.txt')


def read_json():
    with open('assignment3kassym-a93d3bc0450a.json') as f:
        data = json.load(f)
        return data


if __name__ == '__main__':
    creds = read_json()
    uploader = Uploader(creds)
