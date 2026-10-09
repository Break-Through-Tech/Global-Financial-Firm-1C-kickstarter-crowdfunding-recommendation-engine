import io
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from utils.google_drive_client import get_drive_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def get_datafile(service):
    folder_id = os.environ["GOOGLE_DRIVE_DATA_FOLDER_ID"]
    file_name = os.environ["GOOGLE_DRIVE_DATA_FILE_NAME"]

    query = (
        f"name = '{file_name}' "
        f"and '{folder_id}' in parents "
        "and trashed = false"
    )

    results = (
        service.files()
        .list(
            q=query,
            pageSize=10,
            fields="files(id, name, capabilities(canDownload))",
        )
        .execute()
    )

    files = results.get("files", [])

    if not files:
        raise FileNotFoundError(
            f"Could not find '{file_name}' in the configured Google Drive folder."
        )

    return files[0]


def _download_file_to_buffer(service):
    data_file = get_datafile(service)

    if not data_file["capabilities"]["canDownload"]:
        raise PermissionError(
            f"File cannot be downloaded: {data_file['name']}"
        )

    request = service.files().get_media(
        fileId=data_file["id"]
    )

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(
        buffer,
        request,
    )

    done = False

    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)

    return buffer


def load_csv_from_drive(service):
    buffer = _download_file_to_buffer(service)

    return pd.read_csv(
        buffer,
        encoding="utf-8",
        encoding_errors="replace",
    )


def download_csv_data(service, local_path):
    buffer = _download_file_to_buffer(service)

    local_path = Path(local_path)
    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(local_path, "wb") as file:
        file.write(buffer.getvalue())


def load_csv_data(local_path):
    return pd.read_csv(
        local_path,
        encoding="utf-8",
        encoding_errors="replace",
    )


if __name__ == "__main__":
    try:
        service = get_drive_client()

        local_path = (
            PROJECT_ROOT
            / "data"
            / "Raw-data"
            / "DSI_kickstarterscrape_dataset.csv"
        )

        download_csv_data(
            service,
            local_path,
        )

        print(f"Downloaded file to: {local_path}")

    except HttpError as error:
        print(f"An error occurred: {error}")
