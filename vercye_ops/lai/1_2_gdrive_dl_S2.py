# Based on https://stackoverflow.com/a/76736234
# CC BY-SA 4.0

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import click
import io
import os
from pathlib import Path
from tqdm import tqdm

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

@click.command()
@click.option('--secret-json', type=click.Path(exists=True))
@click.option('--folder-id', type=str)
@click.option('--outdir', type=click.Path(file_okay=False))
def main(secret_json, folder_id, outdir):
    """ Download files from a Google Drive Folder

    Parameters
    ----------
    secret_json: str
        Filepath to secrets JSON downloaded from setting up GDrive Python API
        https://developers.google.com/drive/api/quickstart/python
    folder_id: str
        Folder ID from Google Drive URL
    outdir: str
        Directory path to where files should be downloaded
    """

    # Obtain your Google credentials
    def get_credentials(secret_json):
        token_path = os.path.join(Path(secret_json).parent, 'token.json')
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(secret_json, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
            
        return creds

    # Build the downloader
    creds = get_credentials(secret_json)
    drive_downloader = build('drive', 'v3', credentials=creds)

    # query = f"Folder ID '{folder_id}'"  # you may get error for this line
    query = f"'{folder_id}' in parents"  # this works  ref https://stackoverflow.com/q/73119251/248616

    results = drive_downloader.files().list(q=query, pageSize=1000).execute()
    items = results.get('files', [])

    print(f"Found {len(items)} files to download.")
    print(f"Downloading to {outdir}")

    # Download the files
    for item in (pbar := tqdm(items)):
        request = drive_downloader.files().get_media(fileId=item['id'])
        f = io.FileIO(os.path.join(outdir, item['name']), 'wb')
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            pbar.set_description(f"f: {int(status.progress()*100)}%")

    print("Done!")

if __name__ == "__main__":
    main()
