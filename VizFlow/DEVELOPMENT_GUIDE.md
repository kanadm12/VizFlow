# VizFlow Development Guide

## 🏗️ Architecture Overview

### Frontend (React + Vite)
- **Location**: `/src`
- **Tech Stack**: React, Tailwind CSS, Framer Motion, Lucide Icons
- **Port**: 5174

### Backend (Node.js + Express)
- **Location**: `/server`
- **Tech Stack**: Express, MongoDB, JWT, bcryptjs
- **Port**: 5000

### Database (MongoDB)
- **Collections**: Users, Projects
- **Port**: 27017 (default)

## 📁 Project Structure

```
VizFlow/
├── src/                        # Frontend source
│   ├── components/            # React components
│   │   ├── AuthPage.jsx      # Login/Signup
│   │   ├── Toolbar.jsx       # Top navigation
│   │   ├── CodeEditor.jsx    # Code editor
│   │   ├── Terminal.jsx      # Terminal output
│   │   └── ...
│   ├── utils/                # Utilities
│   │   └── api.js           # API client
│   └── hooks/               # Custom hooks
│
├── server/                   # Backend source
│   ├── models/              # MongoDB models
│   │   ├── User.js         # User schema
│   │   └── Project.js      # Project schema
│   ├── controllers/        # Route controllers
│   ├── routes/            # API routes
│   ├── middleware/        # Auth middleware
│   └── server.js         # Express server
│
└── .vscode/              # VS Code settings

## 🔑 Key Features Implemented

### 1. Authentication System
- User registration with email/password
- Secure login with JWT tokens
- Password hashing with bcryptjs
- Protected routes with middleware

### 2. Project Management
- Create and save projects
- Store code files and folders
- Track AI provider usage per project
- Save execution history

### 3. User Tracking
- Track which AI provider each user prefers
- Monitor usage statistics
- Last active timestamp
- Total runs counter

### 4. Terminal Integration
- Real-time output display
- Command history
- Copy output functionality
- Minimize/maximize toggle

## 🔐 Environment Variables

Create `server/.env`:

```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/vizflow
JWT_SECRET=your_super_secret_key_change_this_in_production
NODE_ENV=development
```

## 🗄️ Database Schema

### User Model
```javascript
{
  username: String (unique),
  email: String (unique),
  password: String (hashed),
  aiProvider: String (claude/gemini/copilot/chatgpt/none),
  aiApiKey: String (encrypted),
  projects: [ObjectId],
  usageStats: {
    totalRuns: Number,
    lastActive: Date,
    aiProviderUsage: {
      claude: Number,
      gemini: Number,
      copilot: Number,
      chatgpt: Number
    }
  }
}
```

### Project Model
```javascript
{
  userId: ObjectId,
  name: String,
  description: String,
  files: [{
    name: String,
    content: String,
    language: String
  }],
  folders: [{
    name: String,
    files: [...]
  }],
  aiProviderUsed: String,
  executionHistory: [{
    timestamp: Date,
    output: String,
    success: Boolean
  }]
}
```

## 🌐 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user (protected)

### Code Management
- `POST /api/code/save` - Save project (protected)
- `GET /api/code/projects` - Get user's projects (protected)
- `GET /api/code/project/:id` - Get single project (protected)
- `DELETE /api/code/project/:id` - Delete project (protected)
- `POST /api/code/execution` - Save execution history (protected)

### User Management
- `PUT /api/user/ai-provider` - Update AI provider (protected)
- `GET /api/user/stats` - Get usage statistics (protected)
- `PUT /api/user/profile` - Update profile (protected)

## 🚀 Development Workflow

1. **Start MongoDB**:
   ```bash
   mongod
   ```

2. **Start Backend** (Terminal 1):
   ```bash
   cd server
   npm run dev
   ```

3. **Start Frontend** (Terminal 2):
   ```bash
   npm run dev
   ```

4. **Access Application**:
   - Frontend: http://localhost:5174
   - Backend API: http://localhost:5000

## 🧪 Testing the Features

1. **User Registration**:
   - Create a new account
   - Login with credentials

2. **Code Editor**:
   - Write Python code
   - Upload files
   - Run code to see output in terminal

3. **Project Saving**:
   - Write code
   - Click Save button
   - Project is stored in MongoDB

4. **AI Provider Tracking**:
   - Select AI provider from toolbar
   - Usage is tracked in database
   - View stats via API

## 🔧 Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running
- Check MONGODB_URI in .env
- Verify port 27017 is not blocked

### Authentication Errors
- Clear localStorage
- Check JWT_SECRET in .env
- Verify token in browser DevTools

### CORS Errors
- Backend allows CORS from all origins in development
- For production, configure specific origins

## 📈 Future Enhancements

- Real Python code execution
- Collaborative coding
- Advanced AI integrations
- Project sharing
- Export functionality
- Docker deployment
- CI/CD pipeline

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## 📝 License

MIT License
