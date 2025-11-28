const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    minlength: 3
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  aiProvider: {
    type: String,
    enum: ['claude', 'gemini', 'copilot', 'chatgpt', 'none'],
    default: 'none'
  },
  aiApiKey: {
    type: String,
    default: ''
  },
  projects: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Project'
  }],
  usageStats: {
    totalRuns: { type: Number, default: 0 },
    lastActive: { type: Date, default: Date.now },
    aiProviderUsage: {
      claude: { type: Number, default: 0 },
      gemini: { type: Number, default: 0 },
      copilot: { type: Number, default: 0 },
      chatgpt: { type: Number, default: 0 }
    }
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

// Compare password method
userSchema.methods.comparePassword = async function(candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.model('User', userSchema);
