const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const auth = require('../middleware/auth');

// All routes are protected
router.use(auth);

router.put('/ai-provider', userController.updateAiProvider);
router.get('/stats', userController.getUserStats);
router.put('/profile', userController.updateProfile);

module.exports = router;
