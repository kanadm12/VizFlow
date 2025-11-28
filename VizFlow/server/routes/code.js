const express = require('express');
const router = express.Router();
const codeController = require('../controllers/codeController');
const auth = require('../middleware/auth');

// All routes are protected
router.use(auth);

router.post('/save', codeController.saveProject);
router.get('/projects', codeController.getProjects);
router.get('/project/:id', codeController.getProject);
router.delete('/project/:id', codeController.deleteProject);
router.post('/execution', codeController.saveExecution);

module.exports = router;
