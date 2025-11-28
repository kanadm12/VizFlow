const User = require('../models/User');

// Update AI provider settings
exports.updateAiProvider = async (req, res) => {
  try {
    const { aiProvider, aiApiKey } = req.body;

    const user = await User.findById(req.user._id);
    
    user.aiProvider = aiProvider;
    if (aiApiKey) {
      user.aiApiKey = aiApiKey;
    }

    await user.save();

    res.json({
      message: 'AI provider updated successfully',
      aiProvider: user.aiProvider
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to update AI provider', message: error.message });
  }
};

// Get user statistics
exports.getUserStats = async (req, res) => {
  try {
    const user = await User.findById(req.user._id)
      .select('usageStats aiProvider')
      .populate('projects');

    const stats = {
      totalRuns: user.usageStats.totalRuns,
      lastActive: user.usageStats.lastActive,
      aiProviderUsage: user.usageStats.aiProviderUsage,
      currentAiProvider: user.aiProvider,
      totalProjects: user.projects.length
    };

    res.json({ stats });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch statistics', message: error.message });
  }
};

// Update user profile
exports.updateProfile = async (req, res) => {
  try {
    const { username, email } = req.body;
    const user = await User.findById(req.user._id);

    if (username) user.username = username;
    if (email) user.email = email;

    await user.save();

    res.json({
      message: 'Profile updated successfully',
      user: {
        id: user._id,
        username: user.username,
        email: user.email
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to update profile', message: error.message });
  }
};
