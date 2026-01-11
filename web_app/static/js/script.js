/**
 * Advanced Football Betting Prediction System
 * Main JavaScript File
 * Version: 1.0.0
 */

// Global Variables
let currentUser = null;
let predictionChart = null;
let statsChart = null;
let isLoading = false;
let refreshInterval = null;

// Utility Functions
const Utils = {
    /**
     * Format number with commas
     */
    formatNumber: (num) => {
        return new Intl.NumberFormat('en-US').format(num);
    },

    /**
     * Format currency
     */
    formatCurrency: (amount, currency = 'USD') => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2
        }).format(amount);
    },

    /**
     * Format percentage
     */
    formatPercentage: (value, decimals = 1) => {
        return `${value.toFixed(decimals)}%`;
    },

    /**
     * Format date
     */
    formatDate: (date, format = 'short') => {
        const options = format === 'short' 
            ? { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }
            : { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        
        return new Date(date).toLocaleDateString('en-US', options);
    },

    /**
     * Debounce function
     */
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Show loading spinner
     */
    showLoading: (element) => {
        const spinner = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2 text-muted">Loading...</p>
            </div>
        `;
        element.innerHTML = spinner;
    },

    /**
     * Show error message
     */
    showError: (element, message) => {
        const error = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Error:</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        element.innerHTML = error;
    },

    /**
     * Show success message
     */
    showSuccess: (element, message) => {
        const success = `
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="fas fa-check-circle me-2"></i>
                <strong>Success:</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        element.innerHTML = success;
    },

    /**
     * Validate email
     */
    validateEmail: (email) => {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(String(email).toLowerCase());
    },

    /**
     * Validate password
     */
    validatePassword: (password) => {
        return password.length >= 8 && 
               /[A-Z]/.test(password) && 
               /[a-z]/.test(password) && 
               /[0-9]/.test(password);
    },

    /**
     * Generate random ID
     */
    generateId: () => {
        return Math.random().toString(36).substr(2, 9);
    },

    /**
     * Store data in localStorage
     */
    storeData: (key, data) => {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('Error storing data:', e);
            return false;
        }
    },

    /**
     * Retrieve data from localStorage
     */
    getData: (key) => {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('Error retrieving data:', e);
            return null;
        }
    },

    /**
     * Remove data from localStorage
     */
    removeData: (key) => {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('Error removing data:', e);
            return false;
        }
    }
};

// API Service
const APIService = {
    baseURL: window.location.origin + '/api',
    
    /**
     * Make API request
     */
    request: async (endpoint, options = {}) => {
        const url = `${APIService.baseURL}${endpoint}`;
        const config = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add authentication token if available
        const token = Utils.getData('authToken');
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    /**
     * Get matches
     */
    getMatches: (date, league) => {
        const params = new URLSearchParams();
        if (date) params.append('date', date);
        if (league) params.append('league', league);
        
        return APIService.request(`/matches?${params}`);
    },

    /**
     * Get statistics
     */
    getStats: () => {
        return APIService.request('/stats');
    },

    /**
     * Make prediction
     */
    makePrediction: (predictionData) => {
        return APIService.request('/predict', {
            method: 'POST',
            body: JSON.stringify(predictionData)
        });
    },

    /**
     * Get user predictions
     */
    getUserPredictions: (userId) => {
        return APIService.request(`/users/${userId}/predictions`);
    },

    /**
     * Update user preferences
     */
    updateUserPreferences: (userId, preferences) => {
        return APIService.request(`/users/${userId}/preferences`, {
            method: 'PUT',
            body: JSON.stringify(preferences)
        });
    }
};

// Chart Service
const ChartService = {
    /**
     * Create prediction accuracy chart
     */
    createAccuracyChart: (canvas, data) => {
        const ctx = canvas.getContext('2d');
        
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Incorrect'],
                datasets: [{
                    data: [data.correct, data.incorrect],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(220, 53, 69, 0.8)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    },

    /**
     * Create profit/loss chart
     */
    createProfitChart: (canvas, data) => {
        const ctx = canvas.getContext('2d');
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Profit/Loss',
                    data: data.values,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return Utils.formatCurrency(value);
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Profit: ${Utils.formatCurrency(context.parsed.y)}`;
                            }
                        }
                    }
                }
            }
        });
    },

    /**
     * Create model performance chart
     */
    createModelPerformanceChart: (canvas, data) => {
        const ctx = canvas.getContext('2d');
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Accuracy',
                        data: data.accuracy,
                        backgroundColor: 'rgba(40, 167, 69, 0.8)'
                    },
                    {
                        label: 'ROI',
                        data: data.roi,
                        backgroundColor: 'rgba(23, 162, 184, 0.8)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
};

// Prediction Engine
const PredictionEngine = {
    /**
     * Calculate expected value
     */
    calculateExpectedValue: (probability, odds) => {
        return (probability * odds) - (1 - probability);
    },

    /**
     * Calculate Kelly Criterion
     */
    calculateKellyCriterion: (probability, odds) => {
        const b = odds - 1;
        const p = probability;
        const q = 1 - p;
        
        return (b * p - q) / b;
    },

    /**
     * Assess risk level
     */
    assessRisk: (confidence, odds, stake) => {
        const riskScore = (1 - confidence) + (odds / 10) + (stake / 100);
        
        if (riskScore < 0.5) return 'low';
        if (riskScore < 1.0) return 'medium';
        return 'high';
    },

    /**
     * Generate prediction summary
     */
    generateSummary: (prediction) => {
        const { match, prediction_type, confidence, odds, stake } = prediction;
        
        return {
            expectedValue: PredictionEngine.calculateExpectedValue(confidence, odds),
            kellyCriterion: PredictionEngine.calculateKellyCriterion(confidence, odds),
            riskLevel: PredictionEngine.assessRisk(confidence, odds, stake),
            recommendedStake: Math.min(stake * PredictionEngine.calculateKellyCriterion(confidence, odds), stake)
        };
    }
};

// User Interface
const UI = {
    /**
     * Initialize the application
     */
    init: () => {
        console.log('Initializing Football Prediction System...');
        
        // Check authentication status
        UI.checkAuthStatus();
        
        // Initialize event listeners
        UI.initEventListeners();
        
        // Load initial data
        UI.loadInitialData();
        
        // Start auto-refresh
        UI.startAutoRefresh();
        
        console.log('Application initialized successfully');
    },

    /**
     * Check authentication status
     */
    checkAuthStatus: () => {
        const token = Utils.getData('authToken');
        const user = Utils.getData('currentUser');
        
        if (token && user) {
            currentUser = user;
            UI.updateUIForAuthenticatedUser();
        }
    },

    /**
     * Initialize event listeners
     */
    initEventListeners: () => {
        // Login form
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', UI.handleLogin);
        }

        // Registration form
        const registerForm = document.getElementById('registerForm');
        if (registerForm) {
            registerForm.addEventListener('submit', UI.handleRegistration);
        }

        // Prediction forms
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm) {
            predictionForm.addEventListener('submit', UI.handlePrediction);
        }

        // Real-time odds updates
        const oddsElements = document.querySelectorAll('.odds-display, .odds-small');
        oddsElements.forEach(element => {
            element.addEventListener('click', UI.showOddsHistory);
        });

        // Auto-complete for team search
        const teamSearch = document.getElementById('teamSearch');
        if (teamSearch) {
            teamSearch.addEventListener('input', Utils.debounce(UI.handleTeamSearch, 300));
        }

        // Responsive navigation
        const navbarToggler = document.querySelector('.navbar-toggler');
        if (navbarToggler) {
            navbarToggler.addEventListener('click', UI.handleNavbarToggle);
        }

        // Window resize handler
        window.addEventListener('resize', Utils.debounce(UI.handleResize, 250));
    },

    /**
     * Load initial data
     */
    loadInitialData: () => {
        // Load today's matches
        UI.loadTodaysMatches();
        
        // Load statistics
        UI.loadStatistics();
        
        // Load user predictions if authenticated
        if (currentUser) {
            UI.loadUserPredictions();
        }
    },

    /**
     * Load today's matches
     */
    loadTodaysMatches: async () => {
        const matchesContainer = document.getElementById('matchesContainer');
        if (!matchesContainer) return;

        try {
            Utils.showLoading(matchesContainer);
            
            const today = new Date().toISOString().split('T')[0];
            const response = await APIService.getMatches(today);
            
            if (response.matches && response.matches.length > 0) {
                UI.renderMatches(response.matches, matchesContainer);
            } else {
                matchesContainer.innerHTML = `
                    <div class="text-center py-5">
                        <i class="fas fa-calendar-times fa-3x text-muted mb-3"></i>
                        <h4 class="text-muted">No matches scheduled for today</h4>
                        <p class="text-muted">Check back tomorrow for new predictions.</p>
                    </div>
                `;
            }
        } catch (error) {
            Utils.showError(matchesContainer, 'Failed to load matches. Please try again.');
        }
    },

    /**
     * Render matches
     */
    renderMatches: (matches, container) => {
        const matchesHTML = matches.map(match => `
            <div class="col-lg-6 mb-4">
                <div class="card match-card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span class="league-badge">${match.league}</span>
                        <span class="match-time">${new Date(match.match_date).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                    <div class="card-body">
                        <div class="row align-items-center">
                            <div class="col-5 text-center">
                                <h5 class="team-name">${match.home_team}</h5>
                                <small class="text-muted">Home</small>
                            </div>
                            <div class="col-2 text-center">
                                <div class="vs-small">VS</div>
                            </div>
                            <div class="col-5 text-center">
                                <h5 class="team-name">${match.away_team}</h5>
                                <small class="text-muted">Away</small>
                            </div>
                        </div>
                        
                        <div class="row mt-3">
                            <div class="col-4 text-center">
                                <small class="text-muted">Home</small>
                                <div class="odds-small">${match.home_odds.toFixed(2)}</div>
                            </div>
                            <div class="col-4 text-center">
                                <small class="text-muted">Draw</small>
                                <div class="odds-small">${match.draw_odds.toFixed(2)}</div>
                            </div>
                            <div class="col-4 text-center">
                                <small class="text-muted">Away</small>
                                <div class="odds-small">${match.away_odds.toFixed(2)}</div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer text-center">
                        <button class="btn btn-primary btn-sm predict-btn" data-match-id="${match.match_id}">
                            <i class="fas fa-chart-line"></i> Get Prediction
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = `<div class="row">${matchesHTML}</div>`;
        
        // Add event listeners to prediction buttons
        container.querySelectorAll('.predict-btn').forEach(btn => {
            btn.addEventListener('click', UI.handlePredictionButtonClick);
        });
    },

    /**
     * Load statistics
     */
    loadStatistics: async () => {
        const statsContainer = document.getElementById('statsContainer');
        if (!statsContainer) return;

        try {
            const response = await APIService.getStats();
            UI.renderStatistics(response, statsContainer);
        } catch (error) {
            console.error('Failed to load statistics:', error);
        }
    },

    /**
     * Render statistics
     */
    renderStatistics: (stats, container) => {
        const statsHTML = `
            <div class="row">
                <div class="col-md-3 mb-3">
                    <div class="card text-center bg-primary text-white">
                        <div class="card-body">
                            <i class="fas fa-chart-line fa-3x mb-3"></i>
                            <h3>${Utils.formatPercentage(stats.win_rate)}%</h3>
                            <p class="mb-0">Prediction Accuracy</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card text-center bg-success text-white">
                        <div class="card-body">
                            <i class="fas fa-trophy fa-3x mb-3"></i>
                            <h3>${Utils.formatNumber(stats.won_predictions)}</h3>
                            <p class="mb-0">Winning Predictions</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card text-center bg-info text-white">
                        <div class="card-body">
                            <i class="fas fa-users fa-3x mb-3"></i>
                            <h3>${Utils.formatNumber(stats.total_predictions)}</h3>
                            <p class="mb-0">Total Predictions</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card text-center bg-warning text-white">
                        <div class="card-body">
                            <i class="fas fa-coins fa-3x mb-3"></i>
                            <h3>${Utils.formatCurrency(stats.total_profit || 0)}</h3>
                            <p class="mb-0">Total Profit</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = statsHTML;
    },

    /**
     * Handle login
     */
    handleLogin: async (event) => {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const loginData = {
            username: formData.get('username'),
            password: formData.get('password')
        };

        try {
            // Simulate login (replace with actual API call)
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(loginData)
            });

            if (response.ok) {
                const data = await response.json();
                
                // Store user data
                Utils.storeData('authToken', data.token);
                Utils.storeData('currentUser', data.user);
                currentUser = data.user;
                
                // Update UI
                UI.updateUIForAuthenticatedUser();
                
                // Show success message
                Utils.showSuccess(document.body, 'Login successful!');
                
                // Redirect to dashboard
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            } else {
                throw new Error('Login failed');
            }
        } catch (error) {
            Utils.showError(document.body, 'Invalid username or password');
        }
    },

    /**
     * Handle registration
     */
    handleRegistration: async (event) => {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const registerData = {
            username: formData.get('username'),
            email: formData.get('email'),
            password: formData.get('password'),
            confirmPassword: formData.get('confirm_password')
        };

        // Validate input
        if (!Utils.validateEmail(registerData.email)) {
            Utils.showError(document.body, 'Please enter a valid email address');
            return;
        }

        if (registerData.password !== registerData.confirmPassword) {
            Utils.showError(document.body, 'Passwords do not match');
            return;
        }

        if (!Utils.validatePassword(registerData.password)) {
            Utils.showError(document.body, 'Password must be at least 8 characters with uppercase, lowercase, and numbers');
            return;
        }

        try {
            // Simulate registration (replace with actual API call)
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(registerData)
            });

            if (response.ok) {
                Utils.showSuccess(document.body, 'Registration successful! Please log in.');
                setTimeout(() => {
                    window.location.href = '/login';
                }, 2000);
            } else {
                throw new Error('Registration failed');
            }
        } catch (error) {
            Utils.showError(document.body, 'Registration failed. Please try again.');
        }
    },

    /**
     * Handle prediction
     */
    handlePrediction: async (event) => {
        event.preventDefault();
        
        if (!currentUser) {
            Utils.showError(document.body, 'Please log in to make predictions');
            return;
        }

        const form = event.target;
        const formData = new FormData(form);
        const predictionData = {
            match_id: formData.get('match_id'),
            prediction_type: formData.get('prediction_type'),
            predicted_outcome: formData.get('predicted_outcome'),
            confidence: parseFloat(formData.get('confidence')),
            odds: parseFloat(formData.get('odds')),
            stake: parseFloat(formData.get('stake'))
        };

        // Validate stake
        if (predictionData.stake <= 0) {
            Utils.showError(document.body, 'Please enter a valid stake amount');
            return;
        }

        try {
            const response = await APIService.makePrediction(predictionData);
            
            if (response.success) {
                // Update balance
                const balanceElement = document.getElementById('userBalance');
                if (balanceElement) {
                    balanceElement.textContent = Utils.formatCurrency(response.new_balance);
                }
                
                Utils.showSuccess(document.body, `Prediction placed successfully! Potential win: ${Utils.formatCurrency(response.potential_win)}`);
                
                // Reset form
                form.reset();
                
                // Reload predictions
                UI.loadUserPredictions();
            } else {
                throw new Error(response.error || 'Prediction failed');
            }
        } catch (error) {
            Utils.showError(document.body, error.message || 'Failed to place prediction');
        }
    },

    /**
     * Handle prediction button click
     */
    handlePredictionButtonClick: (event) => {
        const button = event.target;
        const matchId = button.dataset.matchId;
        
        // Open prediction modal or redirect to prediction page
        window.location.href = `/predict?match=${matchId}`;
    },

    /**
     * Update UI for authenticated user
     */
    updateUIForAuthenticatedUser: () => {
        // Update navigation
        const userNav = document.querySelector('.navbar-nav.ms-auto');
        if (userNav) {
            userNav.innerHTML = `
                <li class="nav-item dropdown">
                    <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown">
                        <i class="fas fa-user"></i> ${currentUser.username}
                    </a>
                    <ul class="dropdown-menu dropdown-menu-end">
                        <li><a class="dropdown-item" href="/profile">
                            <i class="fas fa-user-circle me-2"></i> Profile
                        </a></li>
                        <li><a class="dropdown-item" href="#">
                            <i class="fas fa-cog me-2"></i> Settings
                        </a></li>
                        <li><hr class="dropdown-divider"></li>
                        ${currentUser.is_admin ? `
                        <li><a class="dropdown-item" href="/admin">
                            <i class="fas fa-cogs me-2"></i> Admin Panel
                        </a></li>
                        <li><hr class="dropdown-divider"></li>
                        ` : ''}
                        <li><a class="dropdown-item" href="#" onclick="UI.handleLogout()">
                            <i class="fas fa-sign-out-alt me-2"></i> Logout
                        </a></li>
                    </ul>
                </li>
            `;
        }
        
        // Update balance display
        const balanceElement = document.getElementById('userBalance');
        if (balanceElement && currentUser.balance !== undefined) {
            balanceElement.textContent = Utils.formatCurrency(currentUser.balance);
        }
    },

    /**
     * Handle logout
     */
    handleLogout: () => {
        Utils.removeData('authToken');
        Utils.removeData('currentUser');
        currentUser = null;
        window.location.href = '/';
    },

    /**
     * Load user predictions
     */
    loadUserPredictions: async () => {
        if (!currentUser) return;
        
        const predictionsContainer = document.getElementById('userPredictions');
        if (!predictionsContainer) return;

        try {
            const response = await APIService.getUserPredictions(currentUser.id);
            UI.renderUserPredictions(response.predictions, predictionsContainer);
        } catch (error) {
            console.error('Failed to load user predictions:', error);
        }
    },

    /**
     * Render user predictions
     */
    renderUserPredictions: (predictions, container) => {
        if (!predictions || predictions.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="fas fa-chart-line fa-3x text-muted mb-3"></i>
                    <h4 class="text-muted">No predictions yet</h4>
                    <p class="text-muted">Start making predictions to see your results here.</p>
                    <a href="/predict" class="btn btn-primary">
                        <i class="fas fa-plus"></i> Make Your First Prediction
                    </a>
                </div>
            `;
            return;
        }

        const predictionsHTML = predictions.map(prediction => `
            <tr class="${prediction.result === 'win' ? 'table-success' : prediction.result === 'loss' ? 'table-danger' : ''}">
                <td>
                    <strong>${prediction.match.home_team} vs ${prediction.match.away_team}</strong>
                    <br>
                    <small class="text-muted">${prediction.match.league}</small>
                </td>
                <td>
                    <span class="badge bg-primary">${prediction.predicted_outcome}</span>
                    <br>
                    <small class="text-muted">${prediction.prediction_type}</small>
                </td>
                <td>
                    <div class="confidence-indicator">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${prediction.confidence * 100}%"></div>
                        </div>
                        <small>${Utils.formatPercentage(prediction.confidence * 100)}%</small>
                    </div>
                </td>
                <td>${prediction.odds.toFixed(2)}</td>
                <td>
                    ${prediction.result === 'win' 
                        ? '<span class="badge bg-success"><i class="fas fa-check"></i> Win</span>'
                        : prediction.result === 'loss'
                        ? '<span class="badge bg-danger"><i class="fas fa-times"></i> Loss</span>'
                        : '<span class="badge bg-warning"><i class="fas fa-clock"></i> Pending</span>'
                    }
                </td>
                <td>
                    ${prediction.profit > 0 
                        ? `<span class="text-success">+${Utils.formatCurrency(prediction.profit)}</span>`
                        : prediction.profit < 0
                        ? `<span class="text-danger">${Utils.formatCurrency(prediction.profit)}</span>`
                        : '<span class="text-muted">$0.00</span>'
                    }
                </td>
            </tr>
        `).join('');

        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-hover prediction-table">
                    <thead class="table-dark">
                        <tr>
                            <th>Match</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                            <th>Odds</th>
                            <th>Result</th>
                            <th>Profit</th>
                        </tr>
                    </thead>
                    <tbody>${predictionsHTML}</tbody>
                </table>
            </div>
        `;
    },

    /**
     * Start auto-refresh
     */
    startAutoRefresh: () => {
        // Refresh matches every 5 minutes
        refreshInterval = setInterval(() => {
            UI.loadTodaysMatches();
            UI.loadStatistics();
        }, 300000); // 5 minutes
    },

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh: () => {
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    },

    /**
     * Handle window resize
     */
    handleResize: () => {
        // Redraw charts if needed
        if (predictionChart) {
            predictionChart.resize();
        }
        if (statsChart) {
            statsChart.resize();
        }
    },

    /**
     * Handle navbar toggle
     */
    handleNavbarToggle: () => {
        const navbarCollapse = document.querySelector('.navbar-collapse');
        navbarCollapse.classList.toggle('show');
    }
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', UI.init);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    UI.stopAutoRefresh();
});

// Export for use in other modules
window.FootballPredictionSystem = {
    Utils,
    APIService,
    ChartService,
    PredictionEngine,
    UI
};