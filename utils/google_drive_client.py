from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"

RAW_TOKEN_PATH = PROJECT_ROOT / "token_raw.json"
PROCESSED_TOKEN_PATH = PROJECT_ROOT / "token_processed.json"

RAW_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

PROCESSED_SCOPES = [
    "https://www.googleapis.com/auth/drive"
]


def _get_drive_client(scopes, token_path):
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(
            str(token_path),
            scopes=scopes,
        )

        # A valid token must also have the required scope.
        if not creds.has_scopes(scopes):
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

            if not creds.has_scopes(scopes):
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH),
                scopes=scopes,
            )

            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json())

    return build(
        "drive",
        "v3",
        credentials=creds,
    )


def get_raw_drive_client():
    return _get_drive_client(
        RAW_SCOPES,
        RAW_TOKEN_PATH,
    )


def get_processed_drive_client():
    return _get_drive_client(
        PROCESSED_SCOPES,
        PROCESSED_TOKEN_PATH,
    )


def get_drive_client():
    """
    Backward-compatible alias for existing notebooks.
    Uses read-only Google Drive authentication.
    """
    return get_raw_drive_client()
