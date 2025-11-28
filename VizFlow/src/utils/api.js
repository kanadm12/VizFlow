import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  register: (userData) => api.post('/auth/register', userData),
  login: (credentials) => api.post('/auth/login', credentials),
  getCurrentUser: () => api.get('/auth/me'),
};

export const codeAPI = {
  saveProject: (projectData) => api.post('/code/save', projectData),
  getProjects: () => api.get('/code/projects'),
  getProject: (id) => api.get(`/code/project/${id}`),
  deleteProject: (id) => api.delete(`/code/project/${id}`),
  saveExecution: (executionData) => api.post('/code/execution', executionData),
};

export const userAPI = {
  updateAiProvider: (providerData) => api.put('/user/ai-provider', providerData),
  getUserStats: () => api.get('/user/stats'),
  updateProfile: (profileData) => api.put('/user/profile', profileData),
};

export default api;
