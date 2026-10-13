import argparse
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.http import MediaFileUpload

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.google_drive_client import get_processed_drive_client

def escape_drive_query(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_existing_file(service, folder_id, filename):
    safe_name = escape_drive_query(filename)

    result = (
        service.files()
        .list(
            q=(
                f"'{folder_id}' in parents "
                f"and name = '{safe_name}' "
                "and trashed = false"
            ),
            fields="files(id, name)",
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )

    files = result.get("files", [])

    if len(files) > 1:
        raise RuntimeError(
            f"Multiple Drive files named {filename!r} exist in the "
            "Processed-Data folder. Resolve the duplicates before syncing."
        )

    return files[0] if files else None


def upload_or_update(service, folder_id, local_path, dry_run=False):
    mime_type = (
        mimetypes.guess_type(local_path.name)[0]
        or "application/octet-stream"
    )

    existing = find_existing_file(
        service,
        folder_id,
        local_path.name,
    )

    action = "UPDATE" if existing else "CREATE"
    print(f"{action}: {local_path.name}")

    if dry_run:
        return

    media = MediaFileUpload(
        str(local_path),
        mimetype=mime_type,
        resumable=True,
    )

    if existing:
        (
            service.files()
            .update(
                fileId=existing["id"],
                media_body=media,
                fields="id, name, modifiedTime",
                supportsAllDrives=True,
            )
            .execute()
        )
    else:
        (
            service.files()
            .create(
                body={
                    "name": local_path.name,
                    "parents": [folder_id],
                },
                media_body=media,
                fields="id, name, modifiedTime",
                supportsAllDrives=True,
            )
            .execute()
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sync local data/Processed-Data files to the team's "
            "Google Drive Processed-Data folder."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created or updated without uploading.",
    )

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    folder_id = os.getenv(
        "GOOGLE_DRIVE_PROCESSED_FOLDER_ID"
    )

    if not folder_id:
        raise RuntimeError(
            "GOOGLE_DRIVE_PROCESSED_FOLDER_ID is not set in .env"
        )

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "Processed-Data"
    )

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed-data directory not found: {processed_dir}"
        )

    local_files = sorted(
        path
        for path in processed_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
    )

    if not local_files:
        print(
            "No files found in data/Processed-Data. "
            "Nothing to upload."
        )
        return

    print(
        f"Found {len(local_files)} local processed file(s)."
    )

    service = get_processed_drive_client()

    for local_path in local_files:
        upload_or_update(
            service,
            folder_id,
            local_path,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("Dry run complete. No files were uploaded.")
    else:
        print("Processed-data sync complete.")


if __name__ == "__main__":
    main()
