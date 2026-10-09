# Data Setup

The Kickstarter dataset is stored in the team's shared Google Drive rather than committed to GitHub.

The project's primary workflow loads the raw dataset directly from Google Drive into pandas. A local download option is also available for offline use, verification, and debugging.

## Google Drive Location

The dataset is stored in:

```text
1C-Kickstarter-recommendation-engine/
└── Raw-data/
    ├── DSI_kickstarterscrape_dataset.csv
    └── DSI_kickstarterscrape_dataset.zip
```

The project uses:

```text
DSI_kickstarterscrape_dataset.csv
```

The ZIP file is the original downloaded archive.

## Local Data Structure

A local raw-data copy is optional.

If downloaded, the project structure will include:

```text
data/
├── Raw-data/
│   └── DSI_kickstarterscrape_dataset.csv
├── Processed-Data/
├── Data_setup.md
└── README.md
```

### Raw-data

`data/Raw-data/` may contain an unchanged local copy of the dataset downloaded from Google Drive.

**Do not modify the raw dataset directly.**

### Processed-Data

`data/Processed-Data/` is reserved for datasets generated through cleaning, preprocessing, or feature engineering.

Processed datasets should be generated through project code rather than manually edited.

Both raw and processed dataset files are excluded from GitHub.

## 1. Set Up the Python Environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If the environment already exists:

```bash
source .venv/bin/activate
```

## 2. Set Up Google Drive Credentials

Ask a team member for the project's Google Cloud OAuth `credentials.json` file.

Place it in the project root:

```text
project-root/
├── credentials.json
├── utils/
├── data/
└── ...
```

**Do not commit `credentials.json`.**

## 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_DRIVE_DATA_FOLDER_ID=<Raw-data folder ID>
GOOGLE_DRIVE_DATA_FILE_NAME=DSI_kickstarterscrape_dataset.csv
```

`GOOGLE_DRIVE_DATA_FOLDER_ID` should contain the ID of the Google Drive `Raw-data` folder.

Do not commit `.env`.

## 4. Authenticate with Google Drive

Google Drive authentication is handled by:

```text
utils/google_drive_client.py
```

The first time the project accesses Google Drive, a browser window may open for OAuth authentication.

After successful authentication, a local:

```text
token.json
```

file is created in the project root.

Do not commit `token.json`.

If an existing token expires or becomes invalid, the authentication utility will request authorization again.

## 5. Load the Dataset Directly in a Notebook

The primary notebook workflow loads the CSV directly from Google Drive.

From a notebook:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.get_google_drive_data import load_csv_from_drive
from utils.google_drive_client import get_drive_client

service = get_drive_client()
df = load_csv_from_drive(service)
```

This loads the shared CSV directly into a pandas DataFrame without requiring a permanent local copy.

## Optional: Download a Local Copy

A local copy of the raw dataset can be downloaded for offline access, verification, or debugging.

From the project root:

```bash
python3 -m utils.get_google_drive_data
```

This downloads the dataset to:

```text
data/Raw-data/DSI_kickstarterscrape_dataset.csv
```

The downloader creates the local directory if needed.

A local copy is not required for the primary notebook workflow.

## Verify a Local Download

If a local copy was downloaded, verify its SHA-256 checksum:

```bash
shasum -a 256 data/Raw-data/DSI_kickstarterscrape_dataset.csv
```

Expected SHA-256:

```text
b788f3c4527c8203bd237367b2c142046f5dbf9bd48fe5d84ca9478126ab8c52
```

The Google Drive copy was previously verified against the original Kaggle download using this checksum.

## Notebook Workflow

### `00_load_data.ipynb`

Loads the raw dataset directly from Google Drive and performs initial validation, including:

- Checking dataset dimensions
- Verifying the expected schema
- Confirming the dataset is not empty
- Inspecting data types
- Inspecting missing values
- Inspecting campaign statuses
- Documenting known data-loading issues

No cleaning, preprocessing, feature engineering, or train/test splitting is performed in this notebook.

### `01_explore_data.ipynb`

Performs exploratory data analysis.

The current EDA workflow creates a reproducible train/test split and performs target-aware analysis primarily using the training data.

### `02_clean_data.ipynb`

Handles deterministic data-quality issues identified during exploration, including:

- Missing values
- Duplicate records
- Invalid values
- Categorical inconsistencies
- Encoding artifacts
- Location formatting

Cleaning logic should be reproducible and should not modify the original raw dataset.

### `03_preprocess_data.ipynb`

Prepares cleaned data for machine-learning models, including tasks such as:

- Encoding categorical variables
- Normalizing or standardizing numerical features
- Applying fitted preprocessing consistently across datasets

Transformations that learn parameters from the data must be fitted using training data only.

### `04_feature_engineering.ipynb`

Creates and evaluates non-leaking derived features for modeling, including:

- Feature transformations
- Interaction features
- Features derived from existing campaign variables

Feature-engineering decisions should be documented and should avoid using information unavailable at the intended prediction point.

### `05_modeling.ipynb`

Trains baseline and candidate machine-learning models using prepared training data.

### `06_evaluation_and_tuning.ipynb`

Evaluates and improves model performance through:

- Performance metrics
- Model comparison
- Hyperparameter tuning
- Model selection

### `07_interpretability.ipynb`

Analyzes model behavior and feature importance to help explain and communicate predictions.

## Important Data Workflow Rules

### Do Not Modify Raw Data

Whether accessed directly from Google Drive or downloaded locally, the raw dataset should remain unchanged.

Do not manually edit files under:

```text
data/Raw-data/
```

Cleaning should be performed through reproducible project code or notebooks.

### Do Not Commit Datasets

Dataset files are excluded from GitHub through `.gitignore`.

Do not force-add dataset files with:

```bash
git add -f
```

The repository should contain the code and documentation needed to reproduce the data workflow, not the raw or processed datasets themselves.

### Keep Data Processing Reproducible

Data-cleaning, preprocessing, and feature-engineering decisions should be implemented in shared code or notebooks so that all team members can reproduce them consistently.

Significant workflow decisions should also be documented under:

```text
docs/decisions/
```

### Prevent Data Leakage

Transformations that learn information from the dataset must not be fitted using the held-out test data.

This includes operations such as:

- Numerical scaling
- Statistical imputation
- Fitted categorical encoding
- Feature selection based on model-training data

These transformations should be fitted on the training data and then applied to the test data using the same fitted objects.

## Files Used for Google Drive Integration

- `utils/google_drive_client.py` handles Google Drive OAuth authentication.
- `utils/get_google_drive_data.py` handles direct Google Drive loading and optional local downloading.
- `utils/setup_google_drive.sh` provides a helper for local Google Drive setup and downloading.
- `.env` stores local Google Drive configuration.
- `credentials.json` contains the local Google Cloud OAuth client credentials.
- `token.json` stores the local OAuth authorization token.

The following files must not be committed:

```text
.env
credentials.json
token.json
```

## Troubleshooting

### Authentication Fails

Make sure:

1. Your Google account has access to the team's shared Google Drive data.
2. Your Google account is authorized for the Google Cloud OAuth application.
3. `credentials.json` is located in the project root.
4. `.env` is located in the project root.
5. `GOOGLE_DRIVE_DATA_FOLDER_ID` points to the correct Google Drive `Raw-data` folder.
6. `GOOGLE_DRIVE_DATA_FILE_NAME` exactly matches the CSV filename.

If `token.json` exists but authentication fails with an expired or invalid token, the authentication utility should request authorization again.

If needed, the local `token.json` can be removed and recreated through the normal OAuth login flow.

### Dataset Cannot Be Found

Verify that the shared Google Drive folder contains:

```text
DSI_kickstarterscrape_dataset.csv
```

Also verify that the values in `.env` point to the correct Google Drive folder and filename.

### Direct Notebook Loading Fails

Verify that the notebook imports:

```python
from utils.get_google_drive_data import load_csv_from_drive
from utils.google_drive_client import get_drive_client
```

and loads the dataset with:

```python
service = get_drive_client()
df = load_csv_from_drive(service)
```

### Optional Local Download Fails

Run the downloader from the project root:

```bash
python3 -m utils.get_google_drive_data
```

The expected local path is:

```text
data/Raw-data/DSI_kickstarterscrape_dataset.csv
```

### Local Download Exists but Notebook Does Not Use It

This is expected.

The primary notebook workflow loads the dataset directly from Google Drive.

The local CSV under:

```text
data/Raw-data/
```

is optional and is intended for offline access, verification, or debugging.
