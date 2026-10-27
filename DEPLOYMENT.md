# Deploying the app for class

## Option A — Streamlit Community Cloud (recommended)

1. Create a new GitHub repository, for example `hea-hardness-class`.
2. Upload the full contents of this folder to the repository root.
3. Confirm that `app.py`, `requirements.txt`, `data/`, and `models/` are visible in GitHub.
4. Sign in to Streamlit Community Cloud with GitHub.
5. Choose **Create app** / **Deploy an app**.
6. Select your GitHub repository and branch.
7. Set the entry point to:

   ```text
   app.py
   ```

8. Deploy.
9. Open the generated `*.streamlit.app` address in a private/incognito browser to check the student view.
10. Share that fixed URL or a QR code with students.

The model file is about 6 MB, so it is small enough to keep directly in the GitHub repository; Git LFS is not needed for this package.

## Updating the class app

Edit files locally or in GitHub, commit, and push. Streamlit Community Cloud watches the connected repository and redeploys changes.

Before class, check:

- Stage 1 accepts the correct VEC and δ values.
- Stage 3 identifies H and I as borderline ML passes.
- Stage 4 reveals H and I as false passes.
- Stage 5 shows only experimentally feasible candidates J–O.
- The final recommendation CSV downloads correctly.

## Option B — run locally on the lecturer computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

This is useful for development, but Community Cloud is easier for a room full of students because everyone uses the same URL.

## Package versions

The supplied `requirements.txt` pins the environment used for the model package, including scikit-learn 1.8.0. Keeping scikit-learn pinned avoids `joblib` model-compatibility surprises.
