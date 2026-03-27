# Book Friend Finder - Deployment Guide

Complete guide to deploy the Book Friend Finder app, optimized for Google Cloud Run.

## Why Pre-Computed Recommendations?

This deployment uses pre-computed recommendations instead of running ML on the server:

| Metric | ML on Server | Pre-Computed |
|--------|-------------|--------------|
| Startup time | 30 seconds | <1 second |
| Memory needed | 1GB | 128MB |
| Container size | ~1GB | ~100MB |
| Dependencies | TF+NumPy+sklearn | Flask only |
| Response time | <100ms | <50ms |
| Cold start | 30 seconds | 1 second |
| Cost (always-on) | ~$15/mo | ~$5/mo |

## Prerequisites

- Google account (Gmail)
- Credit card (required for Google Cloud, but free tier is generous)
- Terminal access
- Pre-computed data files generated locally

## Part 1: Google Cloud Account Setup

### Step 1: Create Google Cloud Account

1. Go to https://cloud.google.com
2. Click **"Get started for free"** or **"Console"**
3. Sign in with your Google account
4. Accept terms of service
5. Set up billing:
   - Enter credit card information
   - You get **$300 free credits** for 90 days
   - Cloud Run has a permanent free tier (2M requests/month)

### Step 2: Create a New Project

1. Go to https://console.cloud.google.com
2. Click the project dropdown (top left, next to "Google Cloud")
3. Click **"New Project"**
4. Enter project details:
   - **Project name:** `book-friend-finder` (or your choice)
   - **Project ID:** Will be auto-generated (e.g., `book-friend-finder-123456`)
   - **Organization:** Leave as "No organization"
5. Click **"Create"**
6. Wait for project creation (takes ~30 seconds)
7. **Important:** Note your **Project ID** - you'll need it later

### Step 3: Enable Required APIs

1. Go to https://console.cloud.google.com/apis/library
2. Make sure your new project is selected (check top bar)
3. Search for and enable these APIs:
   - **Cloud Run API** - Click "Enable"
   - **Cloud Build API** - Click "Enable"
   - **Artifact Registry API** - Click "Enable"

Each takes ~30 seconds to enable.

## Part 2: Install and Configure gcloud CLI

### Step 4: Install gcloud CLI

**For macOS:**
```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your terminal, then run:
exec -l $SHELL

# Verify installation
gcloud --version
```

**For Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud --version
```

**For Windows:**
Download installer from: https://cloud.google.com/sdk/docs/install

### Step 5: Authenticate and Configure

```bash
# Login to your Google account
gcloud auth login
# This opens a browser window - follow the prompts to authenticate

# Set your project (replace with YOUR project ID from Step 2)
gcloud config set project book-friend-finder-123456

# Verify configuration
gcloud config list

# Set default region (choose one close to you)
gcloud config set run/region us-central1
# Options: us-central1, us-east1, europe-west1, asia-east1
```

## Part 3: Prepare Your Application

### Step 6: Pre-compute Recommendations

Run this on your local machine (where you have TensorFlow):

```bash
cd /Users/robin/git/hardcover-live
python3 precompute_all.py
```

This will:
- Load your data from `~/data/hardcover/`
- Train the TensorFlow model
- Generate recommendations for all users
- Save to `webapp/data/`:
  - `recommendations.json` (~5-10MB)
  - `users.json` (~100KB)
  - `clusters.json` (~50KB)

**Time:** ~30-40 seconds

### Step 7: Verify Your Files

**Required files checklist:**
- `webapp/app.py` - Flask application
- `webapp/requirements.txt` - Python dependencies
- `webapp/templates/` - HTML templates
- `webapp/data/recommendations.json` - Pre-computed data
- `webapp/data/users.json` - User list
- `webapp/data/clusters.json` - Cluster info

Verify data files exist:
```bash
ls -lh webapp/data/
# Should show:
# recommendations.json (~903KB)
# users.json (~18KB)
# clusters.json (~26KB)
```

### Step 8: Test Locally

```bash
cd webapp
python3 app.py
# Open http://localhost:8000 - should start instantly!
```

### Step 9: Create Dockerfile (Optional but Recommended)

Cloud Run can auto-detect Python apps, but a Dockerfile gives you more control:

```bash
# Create Dockerfile in webapp directory
cat > webapp/Dockerfile << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
EOF
```

### Step 10: Create .gcloudignore

Prevent unnecessary files from being uploaded:

```bash
cat > webapp/.gcloudignore << 'EOF'
.git
.gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
env/
venv/
.DS_Store
README.md
DEPLOYMENT.md
EOF
```

## Part 4: Deploy to Cloud Run

### Step 11: Initial Deployment

From the `hardcover-live` directory:

```bash
cd /Users/robin/git/hardcover-live

# Deploy to Cloud Run
gcloud run deploy book-friend-finder \
  --source ./webapp \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 10
```

**What this does:**
- `book-friend-finder` - Your service name
- `--source ./webapp` - Deploy from webapp directory
- `--region us-central1` - Where to deploy
- `--allow-unauthenticated` - Public access (no login required)
- `--memory 512Mi` - 512MB RAM (plenty for our app)
- `--cpu 1` - 1 vCPU
- `--timeout 300` - 5 minute timeout (generous)
- `--max-instances 10` - Scale up to 10 containers max

**Deployment process:**
1. Uploads your code (~1-2 minutes)
2. Builds container image (~2-3 minutes)
3. Deploys to Cloud Run (~1 minute)
4. **Total time: 4-6 minutes**

**Optional: Keep 1 instance warm (no cold starts):**

```bash
gcloud run deploy book-friend-finder \
  --source ./webapp \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --min-instances 1
```

Cost: ~$7/month for always-on instance.

### Step 12: Get Your URL

After deployment completes, you'll see:

```
Service [book-friend-finder] revision [book-friend-finder-00001-xxx] has been deployed and is serving 100 percent of traffic.
Service URL: https://book-friend-finder-xxxxx-uc.a.run.app
```

**Copy this URL!** This is your live application.

### Step 13: Test Your Deployment

Open the Service URL in your browser:
```bash
# Or use curl to test
curl https://book-friend-finder-xxxxx-uc.a.run.app
```

Test the app:
1. Select a user from dropdown
2. Click "Find My Book Friends!"
3. Verify recommendations load
4. Click "View Full Cluster" to test cluster page

## Part 5: Updating Your App

When you have new data (more users, updated recommendations):

```bash
# 1. Run pre-compute locally
cd /Users/robin/git/hardcover-live
python3 precompute_all.py

# 2. Verify new data files were created
ls -lh webapp/data/

# 3. Re-deploy (uses same command as initial deploy)
gcloud run deploy book-friend-finder \
  --source ./webapp \
  --region us-central1 \
  --allow-unauthenticated
```

The deployment automatically:
- Detects changes in your files
- Creates a new revision
- Switches traffic to new revision
- Keeps old revision as backup

## Part 6: Monitoring and Management

### View Logs

```bash
# Stream live logs
gcloud run services logs tail book-friend-finder --region us-central1

# View recent logs in console
# Go to: https://console.cloud.google.com/run
# Click your service → Logs tab
```

### Check Metrics

1. Go to https://console.cloud.google.com/run
2. Click **"book-friend-finder"**
3. Click **"Metrics"** tab
4. See:
   - Request count
   - Latency
   - Error rate
   - Container CPU/memory usage

### Manage Revisions

```bash
# List all revisions
gcloud run revisions list --service book-friend-finder --region us-central1

# Roll back to previous revision if needed
gcloud run services update-traffic book-friend-finder \
  --to-revisions book-friend-finder-00001-xxx=100 \
  --region us-central1
```

## Part 7: Cost Management

### Understanding Costs

**Free Tier (per month):**
- 2,000,000 requests
- 360,000 GB-seconds of memory
- 180,000 vCPU-seconds
- 1 GB network egress (North America)

**Your app usage (estimated):**
- 512MB memory × 1 second per request = 0.5 GB-seconds per request
- Free tier: 360,000 GB-seconds = ~720,000 requests/month FREE
- After that: ~$0.00001 per request (1 cent per 1000 requests)

**Example scenarios:**
- 100 requests/day = 3,000/month → **$0.00** (free tier)
- 1,000 requests/day = 30,000/month → **$0.00** (free tier)
- 10,000 requests/day = 300,000/month → **$0.00** (free tier)
- 100,000 requests/day = 3M/month → **~$23/month**

### Set Up Budget Alerts

1. Go to https://console.cloud.google.com/billing/budgets
2. Click **"Create Budget"**
3. Set amount: **$10/month**
4. Set alert threshold: **50%, 90%, 100%**
5. Add your email for notifications

### Minimize Costs

```bash
# Reduce max instances if not expecting much traffic
gcloud run services update book-friend-finder \
  --max-instances 3 \
  --region us-central1

# Reduce memory if app runs fine with less
gcloud run services update book-friend-finder \
  --memory 256Mi \
  --region us-central1
```

## Part 8: Custom Domain (Optional)

If you own a domain (e.g., `bookfriends.com`):

1. Go to https://console.cloud.google.com/run
2. Click **"book-friend-finder"**
3. Click **"Manage Custom Domains"**
4. Click **"Add Mapping"**
5. Select your service
6. Enter your domain
7. Follow DNS instructions to verify ownership

Cloud Run provides free SSL certificates automatically.

## Part 9: Alternative Deployment Options

While Cloud Run is recommended, here are other options:

### Railway
- Simple git-based deployment
- Free tier available
- `railway up`

### Render
- Free tier
- Auto-deploys from GitHub
- `render.yaml` config

### Heroku
- Classic PaaS
- `git push heroku main`

## Part 10: Troubleshooting

### Common Issues

**Issue: "Permission denied" during deploy**
```bash
# Re-authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**Issue: "API not enabled"**
```bash
# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

**Issue: "Deployment failed - build error"**
```bash
# Check build logs
gcloud builds list --limit 5

# View specific build
gcloud builds log BUILD_ID
```

**Issue: App returns 500 errors**
```bash
# Check application logs
gcloud run services logs tail book-friend-finder --region us-central1

# Common causes:
# - Missing data files (check webapp/data/)
# - Wrong Python version in requirements.txt
# - Import errors
```

**Issue: "Out of memory" errors**
```bash
# Increase memory allocation
gcloud run services update book-friend-finder \
  --memory 1Gi \
  --region us-central1
```

**Issue: recommendations.json not found**
- Run `python3 precompute_all.py` first

**Issue: App starts but shows no users**
- Check that `webapp/data/*.json` files exist
- Verify files were generated correctly

**Issue: Users list is empty**
- Re-run `python3 precompute_all.py`
- Check ~/data/hardcover/ has valid data

## Part 11: Security Best Practices

### Environment Variables

If you add sensitive config later:

```bash
# Set secret environment variable
gcloud run services update book-friend-finder \
  --set-env-vars "SECRET_KEY=your-secret-here" \
  --region us-central1

# Or use Secret Manager for better security
gcloud secrets create api-token --data-file=secret.txt
gcloud run services update book-friend-finder \
  --set-secrets "API_TOKEN=api-token:latest" \
  --region us-central1
```

### Rate Limiting

Add rate limiting to prevent abuse:
```python
# In app.py, add:
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

## Quick Command Reference

```bash
# Deploy/Update
gcloud run deploy book-friend-finder --source ./webapp --region us-central1 --allow-unauthenticated

# View logs
gcloud run services logs tail book-friend-finder --region us-central1

# List services
gcloud run services list

# Delete service
gcloud run services delete book-friend-finder --region us-central1

# Describe service (get URL, config)
gcloud run services describe book-friend-finder --region us-central1

# Update memory
gcloud run services update book-friend-finder --memory 1Gi --region us-central1

# Update CPU
gcloud run services update book-friend-finder --cpu 2 --region us-central1

# Update max instances
gcloud run services update book-friend-finder --max-instances 5 --region us-central1
```

## Success Checklist

- [ ] Google Cloud account created
- [ ] Billing enabled (credit card added)
- [ ] Project created and APIs enabled
- [ ] gcloud CLI installed and authenticated
- [ ] Pre-computed data files generated
- [ ] Successfully deployed to Cloud Run
- [ ] Tested live URL
- [ ] Budget alerts configured
- [ ] Monitoring set up

## Support

- **Cloud Run Docs:** https://cloud.google.com/run/docs
- **Pricing Calculator:** https://cloud.google.com/products/calculator
- **Community Support:** https://stackoverflow.com/questions/tagged/google-cloud-run
- **Your app logs:** https://console.cloud.google.com/run

---

**Estimated deployment time:** 20-30 minutes (first time)

**Estimated cost:** $0-5/month (likely $0 with free tier)

**Difficulty:** Beginner-friendly with this guide
