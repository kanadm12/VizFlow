# VizFlow Setup Script
# This script will help you set up the entire VizFlow project

Write-Host "🚀 Starting VizFlow Setup..." -ForegroundColor Cyan
Write-Host ""

# Check Node.js
Write-Host "Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js $nodeVersion is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js is not installed. Please install Node.js from https://nodejs.org" -ForegroundColor Red
    exit 1
}

# Check MongoDB
Write-Host ""
Write-Host "Checking MongoDB installation..." -ForegroundColor Yellow
try {
    $mongoVersion = mongod --version
    Write-Host "✅ MongoDB is installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  MongoDB is not installed or not in PATH" -ForegroundColor Yellow
    Write-Host "Please install MongoDB from https://www.mongodb.com/try/download/community" -ForegroundColor Yellow
    Write-Host "Or use MongoDB Atlas (cloud): https://www.mongodb.com/cloud/atlas" -ForegroundColor Yellow
}

# Install frontend dependencies
Write-Host ""
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Cyan
npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install frontend dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green

# Install backend dependencies
Write-Host ""
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Cyan
Set-Location server
npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install backend dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Backend dependencies installed" -ForegroundColor Green

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "📝 Creating .env file..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env file created" -ForegroundColor Green
    Write-Host "⚠️  Please edit server/.env and add your MongoDB URI and JWT secret" -ForegroundColor Yellow
}

Set-Location ..

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit server/.env with your MongoDB URI and JWT secret" -ForegroundColor White
Write-Host "2. Start MongoDB: mongod" -ForegroundColor White
Write-Host "3. Start backend: cd server && npm run dev" -ForegroundColor White
Write-Host "4. Start frontend: npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🎉" -ForegroundColor Cyan
