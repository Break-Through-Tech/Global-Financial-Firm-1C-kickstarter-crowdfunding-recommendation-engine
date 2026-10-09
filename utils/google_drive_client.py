# source: https://developers.google.com/workspace/drive/api/quickstart/python

from pathlib import Path

from dotenv import load_dotenv
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"

load_dotenv(PROJECT_ROOT / ".env")


def get_drive_client():
    """Authenticate the user and return a Google Drive API client."""

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES,
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                creds = None

        if not creds:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Google OAuth credentials not found at "
                    f"'{CREDENTIALS_FILE}'."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                SCOPES,
            )

            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build(
        "drive",
        "v3",
        credentials=creds,
    )
