# 🚀 Push to GitHub — Step by Step

## Step 1 — Create the repo on GitHub (2 min)

1. Go to **github.com** → sign in
2. Click the **+** icon top right → **New repository**
3. Fill in:
   - Repository name: `aml-detection-engine`
   - Description: `Blockchain AML fraud detection engine. 28 scoring rules, AI layer. Historical detection figures are not reproducible from this repo.`
   - Visibility: **Public** (this is your portfolio)
   - ❌ Do NOT check "Add README" — we already have one
4. Click **Create repository**
5. GitHub shows you a page with setup commands — copy your repo URL (looks like `https://github.com/YOUR_USERNAME/aml-detection-engine.git`)

---

## Step 2 — Push from your computer (terminal)

Open Terminal (Mac) or Command Prompt / PowerShell (Windows).

Navigate to the `github_repo` folder inside your Cloude 2026 folder:
```bash
cd "path/to/Cloude 2026/github_repo"
```

Then run these commands one by one:

```bash
git init
git add .
git commit -m "Initial commit — NEXUS-RISK AML Engine v11, AI Layer, Triage System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/aml-detection-engine.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

GitHub will ask for your username + password (use a Personal Access Token as password — see below).

---

## Step 3 — Personal Access Token (GitHub password replacement)

GitHub no longer accepts your password in terminal. You need a token:

1. GitHub → top right avatar → **Settings**
2. Left sidebar → **Developer settings** (bottom)
3. **Personal access tokens** → **Tokens (classic)** → **Generate new token**
4. Name it: `aml-engine-push`
5. Expiration: 90 days
6. Check: **repo** (full control of private repositories)
7. Click **Generate token**
8. **COPY IT NOW** — GitHub only shows it once
9. Use this token as your password when `git push` asks

---

## Step 4 — Verify

Go to `github.com/YOUR_USERNAME/aml-detection-engine`

You should see:
- ✅ README.md rendered with badges and tables
- ✅ `engine/` folder with engine_v11_blockchain.py
- ✅ `ai_layer/` folder with aml_ai_layer.py + triage_labeler.py
- ✅ `dashboard/` folder with nexus_dashboard.html
- ✅ `data/` folder with sample_transactions.csv

That URL is now your portfolio link. Put it in your LinkedIn bio, blog posts, and resume.

---

## Step 5 — Future updates (any time you improve the engine)

```bash
cd "path/to/Cloude 2026/github_repo"
git add .
git commit -m "Add Streamlit demo / new rule / fix bug"
git push
```

GitHub auto-updates. The commit history shows your progression v6 → v11 → AI layer. That's the work log.

---

## What the repo looks like to someone visiting it

They land on the README:
- See the 94.9% detection badge immediately
- Read the case studies table (Tornado, Ronin, Lazarus, Wormhole, Nomad)
- Understand the 3-layer architecture
- Can clone it and run it on their own data in 5 minutes
- See you wrote it — Bionic Banker, bionicbanker.tech

This is a stronger signal than a resume line. It's proof.
