# Upload this project to GitHub

Repo: **https://github.com/Adithkumarrai/Apple.com**

## Files to upload (required)

Upload these from `c:\Users\dell\Downloads\deployment`:

| File | Required |
|------|----------|
| `app.py` | Yes |
| `model_utils.py` | Yes |
| `requirements.txt` | Yes |
| `README.md` | Yes |
| `features.pkl` | Yes |
| `scaler.pkl` | Yes |
| `rf_model.pkl` | Yes |
| `xgb_model.pkl` | Yes |
| `lstm_model.keras` | Yes |
| `sample_features.csv` | Optional |
| `.gitignore` | Optional |

**Do not upload:** `.venv`, `__pycache__`, `repo.zip`, `Apple.com-main`

Total size ~4.5 MB (GitHub allows up to 100 MB per file).

---

## Option A — Upload in browser (easiest, no Git install)

1. Open https://github.com/Adithkumarrai/Apple.com
2. Click **Add file** → **Upload files**
3. Drag all required files listed above into the page
4. Commit message: `Add Streamlit app and ML models`
5. Click **Commit changes**

---

## Option B — Git commands (if Git is installed)

```powershell
cd c:\Users\dell\Downloads\deployment
git init
git add app.py model_utils.py requirements.txt README.md .gitignore sample_features.csv
git add features.pkl scaler.pkl rf_model.pkl xgb_model.pkl lstm_model.keras
git commit -m "Add Streamlit Apple stock predictor"
git branch -M main
git remote add origin https://github.com/Adithkumarrai/Apple.com.git
git push -u origin main
```

If the remote already has a README, use:

```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## Deploy on Streamlit Cloud (after upload)

1. Go to https://share.streamlit.io and sign in with GitHub
2. **New app** → repository: `Adithkumarrai/Apple.com`
3. **Main file path:** `app.py`
4. **Deploy**

Your live app URL will look like: `https://apple-com-xxxxx.streamlit.app`
