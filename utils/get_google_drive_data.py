import io
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.http import MediaIoBaseDownload


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.google_drive_client import get_raw_drive_client


load_dotenv(PROJECT_ROOT / ".env")


RAW_FOLDER_ID = os.getenv("GOOGLE_DRIVE_DATA_FOLDER_ID")

RAW_FILE_NAME = os.getenv(
    "GOOGLE_DRIVE_DATA_FILE_NAME",
    "DSI_kickstarterscrape_dataset.csv",
)


def escape_drive_query(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def get_drive_file_id(
    service,
    folder_id=None,
    file_name=None,
):
    if folder_id is None:
        folder_id = RAW_FOLDER_ID

    if file_name is None:
        file_name = RAW_FILE_NAME

    if not folder_id:
        raise RuntimeError(
            "GOOGLE_DRIVE_DATA_FOLDER_ID is not set in .env"
        )

    safe_file_name = escape_drive_query(file_name)

    result = (
        service.files()
        .list(
            q=(
                f"'{folder_id}' in parents "
                f"and name = '{safe_file_name}' "
                "and trashed = false"
            ),
            spaces="drive",
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )

    files = result.get("files", [])

    if not files:
        raise FileNotFoundError(
            f"Could not find {file_name!r} "
            "in the configured Google Drive Raw-data folder."
        )

    if len(files) > 1:
        raise RuntimeError(
            f"Multiple files named {file_name!r} "
            "were found in the configured Raw-data folder."
        )

    return files[0]["id"]


def download_drive_file(service, file_id):
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(
        buffer,
        request,
    )

    done = False

    while not done:
        status, done = downloader.next_chunk()

        if status:
            print(
                f"Download progress: "
                f"{int(status.progress() * 100)}%"
            )

    return buffer.getvalue()


def load_csv_from_drive(
    service=None,
    folder_id=None,
    file_name=None,
):
    """
    Download and parse the raw CSV into a pandas DataFrame.
    Used by existing notebooks.
    """
    if service is None:
        service = get_raw_drive_client()

    if folder_id is None:
        folder_id = RAW_FOLDER_ID

    if file_name is None:
        file_name = RAW_FILE_NAME

    file_id = get_drive_file_id(
        service,
        folder_id,
        file_name,
    )

    print(f"Found file: {file_name}")

    file_bytes = download_drive_file(
        service,
        file_id,
    )

    df = pd.read_csv(
        io.BytesIO(file_bytes),
        encoding="utf-8",
        encoding_errors="replace",
    )

    return df


def download_raw_data(
    service=None,
    folder_id=None,
    file_name=None,
):
    """
    Download the original CSV bytes without modifying them.
    Preserves the raw dataset's SHA-256 hash.
    """
    if service is None:
        service = get_raw_drive_client()

    if folder_id is None:
        folder_id = RAW_FOLDER_ID

    if file_name is None:
        file_name = RAW_FILE_NAME

    file_id = get_drive_file_id(
        service,
        folder_id,
        file_name,
    )

    print(f"Found file: {file_name}")

    file_bytes = download_drive_file(
        service,
        file_id,
    )

    raw_data_dir = (
        PROJECT_ROOT
        / "data"
        / "Raw-data"
    )

    raw_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = raw_data_dir / file_name

    output_path.write_bytes(file_bytes)

    print(f"Downloaded file to: {output_path}")

    return output_path


def main():
    service = get_raw_drive_client()

    download_raw_data(
        service=service,
    )


if __name__ == "__main__":
    main()
