# Put the project on GitHub and publish the web app

## GitHub page vs. running app

Opening this repository on GitHub shows the source files and README. It does **not** run `app.py`. GitHub Pages serves static files; this Streamlit project needs a Python server. To use it in a browser, keep the source in GitHub and deploy it with [Streamlit Community Cloud](https://share.streamlit.io/). Streamlit Community Cloud reads the repository, installs `requirements.txt`, and runs the selected Python file.

## Files to upload

Upload the source, trained model, documentation, requirements, and Streamlit theme:

```text
app.py
lung_svm/
requirements.txt
.streamlit/config.toml
artifacts/lung_svm_model.joblib
README.md
PROJECT_REPORT.md
GITHUB_DEPLOYMENT.md
REPORT_TEMPLATE.md
```

Do not upload `images/`, `labels.csv`, `.venv/`, or cached HOG arrays. The web app only needs the trained model; the raw training set is not needed for a user's prediction. The `.gitignore` is set up to exclude those large inputs and caches while allowing the trained model and report figures/metrics into Git.

The trained model is about 4.3 MB. Keep the scikit-learn, NumPy, and other dependency versions in `requirements.txt` aligned with the training environment because the model file was serialized with those packages.

## Upload using GitHub Desktop

1. Create a **public** repository on GitHub. A public repository is the simplest option for a free Community Cloud deployment.
2. In GitHub Desktop, choose **File → Add Local Repository** and select `C:\ML_Project`. If it says this folder is not a repository, choose **Create a Repository** for this existing project folder.
3. Review the changed-file list. It should include `artifacts/lung_svm_model.joblib` and should not include the `images` folder, `labels.csv`, `.venv`, or `artifacts/hog_features.npy`.
4. Enter a commit message such as `Add LungLens Streamlit app`, click **Commit to main**, then click **Publish repository** / **Push origin**.

Alternatively, create a GitHub repository and use Git from a PowerShell terminal in `C:\ML_Project`:

```powershell
git init -b main
git add .
git status --short
git commit -m "Add LungLens Streamlit app"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If this folder already has a Git repository, skip `git init`. If a remote is already configured, skip `git remote add origin` and push to that remote. Before committing, use `git status --short` to confirm the training images, `labels.csv`, and `.venv` are excluded.

## Publish with Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) using the GitHub account that owns the repository.
2. Choose **Create app**, select the GitHub repository, branch `main`, and entrypoint file `app.py`.
3. In **Advanced settings**, select Python **3.14**, matching the environment that produced the included model.
4. Deploy. When the build finishes, Streamlit gives you a `*.streamlit.app` URL. That is the public webpage to open and share; the GitHub repository itself remains the code page.

The repository root contains `requirements.txt`, the `lung_svm/` package, `.streamlit/config.toml`, and the model at the path expected by `app.py`. Streamlit Community Cloud runs the app from the repository root. Later GitHub commits are picked up by the deployed app.

## Uploads and privacy

If hosted online, the uploaded X-ray is sent to the app host for processing. The app removes its temporary image file after feature extraction, but avoid uploading identifiable or sensitive health information to a public demo. The classifier is an educational project, not a medical diagnostic service.
