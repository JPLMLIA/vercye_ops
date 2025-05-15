import os
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2 import service_account


def get_credentials(secret_json):
    """Fetch and return Google Drive API credentials"""
    SCOPES = ['https://www.googleapis.com/auth/drive']
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

def get_drive_service(secret_json):
    """Set up and return Google Drive API service"""

    # Fetch token and build the downloader
    creds = get_credentials(secret_json)
    drive_downloader = build('drive', 'v3', credentials=creds)

    return drive_downloader

def find_files_in_drive(service, folder_name, file_description):
    """Find all files in Google Drive by folder name and file description pattern"""
    # First find the folder ID
    folder_query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    folder_results = service.files().list(q=folder_query).execute()
    folder_items = folder_results.get('files', [])
    
    if not folder_items:
        print(f"Folder '{folder_name}' not found in Google Drive")
        return []
    
    folder_id = folder_items[0]['id']
    
    # Now find all files in that folder that match the description
    # Note: Exported large files will be split, so we use a pattern instead of exact match
    file_query = f"name contains '{file_description}' and '{folder_id}' in parents and trashed = false"
    file_results = service.files().list(q=file_query, fields="files(id, name, size)").execute()
    file_items = file_results.get('files', [])
    
    if not file_items:
        print(f"No files containing '{file_description}' found in folder '{folder_name}'")
        return []
    
    return file_items, folder_id  # Return all matching files and folder_id

def download_files_from_drive(service, files, download_dir):
    """Download multiple files from Google Drive"""
    os.makedirs(download_dir, exist_ok=True)
    downloaded_files = []
    
    for file_info in files:
        file_id = file_info['id']
        file_name = file_info['name']
        download_path = os.path.join(download_dir, file_name)
        
        print(f"Downloading {file_name}...")
        request = service.files().get_media(fileId=file_id)
        
        with open(download_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%")
        
        print(f"File downloaded to {download_path}")
        downloaded_files.append((file_id, download_path))
    
    return downloaded_files

def delete_files_from_drive(service, file_ids):
    """Delete multiple files from Google Drive"""
    for file_id in file_ids:
        service.files().delete(fileId=file_id).execute()
        print(f"File with ID {file_id} deleted from Google Drive")

def delete_folder_from_drive(service, folder_id):
    """Delete a folder from Google Drive"""
    service.files().delete(fileId=folder_id).execute()
    print(f"Folder with ID {folder_id} deleted from Google Drive")