# ==============================================================================
# Setup & Push Script for CampusX Deep Learning Companion Repository
# ==============================================================================
param (
    [string]$RemoteUrl = ""
)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  CampusX 100 Days of Deep Learning - GitHub Repo Setup" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

$RepoDir = "c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes"
Set-Location -Path $RepoDir

# 1. Verify Git Installation
try {
    $gitVersion = git --version
    Write-Host "[OK] Git detected: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Ensure .gitignore exists
$GitIgnorePath = Join-Path $RepoDir ".gitignore"
if (-not (Test-Path $GitIgnorePath)) {
    $GitIgnoreContent = @"
# Python bytecode & cache
__pycache__/
*.py[cod]
*$py.class
*.ipynb_checkpoints

# OS / Editor temporary files
.DS_Store
Thumbs.db
.vscode/
.idea/
"@
    Set-Content -Path $GitIgnorePath -Value $GitIgnoreContent -Encoding UTF8
    Write-Host "[OK] Created .gitignore" -ForegroundColor Green
} else {
    Write-Host "[OK] Existing .gitignore found." -ForegroundColor Yellow
}

# 3. Initialize Git Repository if needed
if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    Write-Host "[INFO] Initializing new git repository with main branch..." -ForegroundColor Cyan
    git init -b main
} else {
    Write-Host "[OK] Git repository already initialized." -ForegroundColor Green
}

# 4. Stage and commit files
Write-Host "[INFO] Staging all lecture notes, slides, notebooks, and README..." -ForegroundColor Cyan
git add .
$status = git status --porcelain

if ($status) {
    Write-Host "[INFO] Committing files..." -ForegroundColor Cyan
    git commit -m "feat: complete 100 Days of Deep Learning comprehensive notes, Colab archive, and companion guide"
    Write-Host "[OK] Commit complete." -ForegroundColor Green
} else {
    Write-Host "[OK] Working tree clean, no uncommitted changes." -ForegroundColor Green
}

# 5. Remote Configuration & Instructions
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Yellow
Write-Host "  NEXT STEPS TO PUBLISH ON GITHUB" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Yellow

if ($RemoteUrl -ne "") {
    Write-Host "[INFO] Setting remote origin to: $RemoteUrl" -ForegroundColor Cyan
    git remote remove origin 2>$null
    git remote add origin $RemoteUrl
    Write-Host "[INFO] Pushing to GitHub..." -ForegroundColor Cyan
    git push -u origin main
    Write-Host "[SUCCESS] Repository successfully pushed to $RemoteUrl" -ForegroundColor Green
} else {
    Write-Host "1. Create a new EMPTY repository on your GitHub account:" -ForegroundColor White
    Write-Host "   URL: https://github.com/new" -ForegroundColor Cyan
    Write-Host "   Name suggestion: campusx-100-days-deep-learning-notes" -ForegroundColor White
    Write-Host "   (Leave 'Add README', '.gitignore', and 'license' UNCHECKED)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "2. Then run these commands in PowerShell:" -ForegroundColor White
    Write-Host "   cd c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes" -ForegroundColor Green
    Write-Host "   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/campusx-100-days-deep-learning-notes.git" -ForegroundColor Green
    Write-Host "   git push -u origin main" -ForegroundColor Green
    Write-Host ""
    Write-Host "   Or re-run this script with -RemoteUrl:" -ForegroundColor White
    Write-Host "   .\setup_github_repo.ps1 -RemoteUrl 'https://github.com/<USER>/<REPO>.git'" -ForegroundColor Green
}
Write-Host "==============================================================" -ForegroundColor Yellow
