# Use Shared Google Drive as the Primary Data Source for Notebooks

## Context and Problem Statement

The project dataset is stored in the team's shared Google Drive. Initially, some project code downloaded the raw CSV to `data/Raw-data/` and notebooks loaded the local copy with `pandas.read_csv()`.

Other notebooks, including the exploratory data analysis workflow, load the dataset directly from Google Drive into a pandas DataFrame.

Using different loading methods across notebooks could make the project harder to reproduce and maintain. The team needs a consistent way to access the same source dataset while still allowing a local copy to be downloaded when needed.

The raw Kickstarter CSV also contains byte sequences that are not valid UTF-8. The Google Drive copy was previously verified against the original dataset using SHA-256, confirming that the encoding issue exists in the source data and was not introduced by Google Drive.

## Considered Options

* Require every notebook to load a locally downloaded CSV from `data/Raw-data/`.
* Load the dataset directly from Google Drive into pandas for notebook work.
* Support both approaches independently in different notebooks.

## Decision Outcome

Chosen option: "Load the dataset directly from Google Drive into pandas for notebook work, while keeping local download support as an optional utility", because

* All team members can use the same shared source dataset without maintaining separate notebook-specific file paths.
* This keeps the data-loading workflow consistent with the existing exploratory data analysis notebook.
* Direct loading avoids requiring a local dataset copy before running a notebook.
* A local download option remains useful for offline access, backup, and reproducibility checks.
* Both direct and local loading can share the same Google Drive and CSV-loading utilities.

The primary notebook workflow will therefore be:

`Google Drive -> load_csv_from_drive() -> pandas DataFrame`

The optional local workflow will remain:

`Google Drive -> download_csv_data() -> local CSV -> load_csv_data()`

Both CSV-loading paths will use the same encoding behavior:

`encoding="utf-8"` with `encoding_errors="replace"`.

## Consequences

### Positive

* Team notebooks can start from the same shared Google Drive dataset.
* Notebook code no longer depends on user-specific local file paths.
* Shared loading logic reduces duplicated data-loading code.
* The direct-loading workflow is consistent across data loading, EDA, and future cleaning notebooks.
* Local downloading remains available when offline access or a local raw-data copy is useful.
* Encoding behavior is consistent between direct Google Drive loading and local CSV loading.

### Negative

* Running notebooks with the primary workflow requires an internet connection and valid Google Drive authentication.
* Each team member must have access to the shared Google Drive folder.
* Expired or revoked OAuth tokens may require the user to authenticate again.
* Direct loading downloads the file from Google Drive when the notebook creates the DataFrame, which may be slower than reading an existing local copy.

## More Information

* See `0001-google-drive-oauth-authentication.md` for the Google Drive authentication decision.
* Shared Drive client: `utils/google_drive_client.py`
* Shared data-loading utilities: `utils/get_google_drive_data.py`
* Initial loading notebook: `notebooks/00_load_data.ipynb`
* Exploratory data analysis notebook: `notebooks/01_explore_data.ipynb`
* Data-cleaning notebook: `notebooks/02_clean_data.ipynb`
* Raw local copies under `data/Raw-data/` are optional and should not be committed to Git.
