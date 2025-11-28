const mongoose = require('mongoose');

const fileSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  content: {
    type: String,
    required: true
  },
  language: {
    type: String,
    default: 'python'
  },
  lastModified: {
    type: Date,
    default: Date.now
  }
});

const folderSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  files: [fileSchema],
  createdAt: {
    type: Date,
    default: Date.now
  }
});

const projectSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: {
    type: String,
    required: true,
    trim: true
  },
  description: {
    type: String,
    default: ''
  },
  files: [fileSchema],
  folders: [folderSchema],
  aiProviderUsed: {
    type: String,
    enum: ['claude', 'gemini', 'copilot', 'chatgpt', 'none'],
    default: 'none'
  },
  executionHistory: [{
    timestamp: { type: Date, default: Date.now },
    output: String,
    success: Boolean
  }],
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

projectSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Project', projectSchema);
