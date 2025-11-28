const Project = require('../models/Project');
const User = require('../models/User');

// Save or update project
exports.saveProject = async (req, res) => {
  try {
    const { projectId, name, description, files, folders, aiProviderUsed } = req.body;
    const userId = req.user._id;

    let project;

    if (projectId) {
      // Update existing project
      project = await Project.findOne({ _id: projectId, userId });
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }

      project.name = name || project.name;
      project.description = description || project.description;
      project.files = files || project.files;
      project.folders = folders || project.folders;
      project.aiProviderUsed = aiProviderUsed || project.aiProviderUsed;
    } else {
      // Create new project
      project = new Project({
        userId,
        name,
        description,
        files: files || [],
        folders: folders || [],
        aiProviderUsed: aiProviderUsed || 'none'
      });

      // Add project to user's projects
      await User.findByIdAndUpdate(userId, {
        $push: { projects: project._id }
      });
    }

    await project.save();

    res.json({
      message: 'Project saved successfully',
      project
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to save project', message: error.message });
  }
};

// Get user's projects
exports.getProjects = async (req, res) => {
  try {
    const projects = await Project.find({ userId: req.user._id })
      .sort({ updatedAt: -1 });

    res.json({ projects });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch projects', message: error.message });
  }
};

// Get single project
exports.getProject = async (req, res) => {
  try {
    const project = await Project.findOne({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    res.json({ project });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch project', message: error.message });
  }
};

// Delete project
exports.deleteProject = async (req, res) => {
  try {
    const project = await Project.findOneAndDelete({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    // Remove from user's projects
    await User.findByIdAndUpdate(req.user._id, {
      $pull: { projects: project._id }
    });

    res.json({ message: 'Project deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete project', message: error.message });
  }
};

// Save execution history
exports.saveExecution = async (req, res) => {
  try {
    const { projectId, output, success } = req.body;

    const project = await Project.findOne({
      _id: projectId,
      userId: req.user._id
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    project.executionHistory.push({
      output,
      success,
      timestamp: new Date()
    });

    // Track AI provider usage
    if (project.aiProviderUsed && project.aiProviderUsed !== 'none') {
      await User.findByIdAndUpdate(req.user._id, {
        $inc: {
          'usageStats.totalRuns': 1,
          [`usageStats.aiProviderUsage.${project.aiProviderUsed}`]: 1
        }
      });
    }

    await project.save();

    res.json({ message: 'Execution saved successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to save execution', message: error.message });
  }
};
