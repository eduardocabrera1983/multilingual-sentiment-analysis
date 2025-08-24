/**
 * Main JavaScript for Multilingual Sentiment Analysis Flask App
 * Handles common functionality, animations, and interactive elements
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize common functionality
    initializeAnimations();
    initializeTooltips();
    initializeFormValidation();
    initializeFileUpload();
    checkModelStatus();
});

/**
 * Initialize scroll-triggered animations
 */
function initializeAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.feature-card, .bi-card, .result-section').forEach(element => {
        observer.observe(element);
    });
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation and enhancement
 */
function initializeFormValidation() {
    // Real-time validation for text analysis forms
    const textInputs = document.querySelectorAll('textarea[required]');
    textInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateTextInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateTextInput(this);
        });
    });
    
    // Form submission prevention for empty required fields
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
                showAlert('Please fill in all required fields correctly.', 'error');
            }
        });
    });
}

/**
 * Validate individual text input
 */
function validateTextInput(input) {
    const value = input.value.trim();
    const minLength = 10;
    const maxLength = 5000;
    
    // Remove existing validation classes
    input.classList.remove('is-valid', 'is-invalid');
    
    if (value.length === 0) {
        input.classList.add('is-invalid');
        updateValidationFeedback(input, 'Please enter some text to analyze.');
        return false;
    } else if (value.length < minLength) {
        input.classList.add('is-invalid');
        updateValidationFeedback(input, `Text must be at least ${minLength} characters long.`);
        return false;
    } else if (value.length > maxLength) {
        input.classList.add('is-invalid');
        updateValidationFeedback(input, `Text must be no more than ${maxLength} characters long.`);
        return false;
    } else {
        input.classList.add('is-valid');
        updateValidationFeedback(input, 'Ready for analysis!');
        return true;
    }
}

/**
 * Update validation feedback message
 */
function updateValidationFeedback(input, message) {
    let feedback = input.parentElement.querySelector('.invalid-feedback, .valid-feedback');
    
    if (!feedback) {
        feedback = document.createElement('div');
        input.parentElement.appendChild(feedback);
    }
    
    feedback.className = input.classList.contains('is-valid') ? 'valid-feedback' : 'invalid-feedback';
    feedback.textContent = message;
}

/**
 * Validate entire form
 */
function validateForm(form) {
    let isValid = true;
    const requiredInputs = form.querySelectorAll('[required]');
    
    requiredInputs.forEach(input => {
        if (input.type === 'textarea' || input.tagName === 'TEXTAREA') {
            if (!validateTextInput(input)) {
                isValid = false;
            }
        } else if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        }
    });
    
    return isValid;
}

/**
 * Initialize file upload functionality
 */
function initializeFileUpload() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        // Drag and drop events
        area.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        
        area.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelection(files[0], this);
            }
        });
        
        // Click to upload
        area.addEventListener('click', function() {
            const fileInput = this.querySelector('input[type="file"]') || 
                             document.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.click();
            }
        });
    });
    
    // File input change handler
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            if (this.files.length > 0) {
                handleFileSelection(this.files[0]);
            }
        });
    });
}

/**
 * Handle file selection and validation
 */
function handleFileSelection(file, uploadArea = null) {
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    // Validate file type
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx|xls)$/i)) {
        showAlert('Please upload a CSV or Excel file (.csv, .xlsx, .xls)', 'error');
        return false;
    }
    
    // Validate file size
    if (file.size > maxSize) {
        showAlert('File size must be less than 16MB', 'error');
        return false;
    }
    
    // Update UI to show selected file
    if (uploadArea) {
        const fileName = uploadArea.querySelector('.file-name');
        if (fileName) {
            fileName.textContent = file.name;
            fileName.style.display = 'block';
        }
    }
    
    // Show file info
    showAlert(`File selected: ${file.name} (${formatFileSize(file.size)})`, 'success');
    
    return true;
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show alert messages
 */
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert.alert-auto');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alert = document.createElement('div');
    alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show alert-auto`;
    alert.innerHTML = `
        <i class="bi bi-${getAlertIcon(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(alert, main.firstChild);
    } else {
        document.body.insertBefore(alert, document.body.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

/**
 * Get appropriate icon for alert type
 */
function getAlertIcon(type) {
    switch(type) {
        case 'success': return 'check-circle-fill';
        case 'error': 
        case 'danger': return 'exclamation-circle-fill';
        case 'warning': return 'exclamation-triangle-fill';
        default: return 'info-circle-fill';
    }
}

/**
 * Check model status and update UI accordingly
 */
function checkModelStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            updateModelStatusIndicators(data);
        })
        .catch(error => {
            console.warn('Could not check model status:', error);
        });
}

/**
 * Update model status indicators throughout the app
 */
function updateModelStatusIndicators(healthData) {
    const indicators = document.querySelectorAll('.model-status-indicator');
    
    indicators.forEach(indicator => {
        if (healthData.models_loaded) {
            indicator.className = 'model-status-indicator text-success';
            indicator.innerHTML = '<i class="bi bi-check-circle-fill me-1"></i>Enhanced Models Active';
        } else {
            indicator.className = 'model-status-indicator text-warning';
            indicator.innerHTML = '<i class="bi bi-exclamation-triangle-fill me-1"></i>Basic Mode';
        }
    });
}

/**
 * API helper functions
 */
const API = {
    /**
     * Analyze single text via API
     */
    analyzeSingleText: async function(text, language = 'auto') {
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    language: language
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    /**
     * Analyze batch of texts via API
     */
    analyzeBatchTexts: async function(texts) {
        try {
            const response = await fetch('/api/batch-analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    texts: texts
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Batch analysis failed');
            }
            
            return data;
        } catch (error) {
            console.error('Batch API Error:', error);
            throw error;
        }
    }
};

/**
 * Utility functions
 */
const Utils = {
    /**
     * Debounce function to limit rapid function calls
     */
    debounce: function(func, wait) {
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
     * Copy text to clipboard
     */
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            showAlert('Copied to clipboard!', 'success');
        }).catch(() => {
            showAlert('Could not copy to clipboard', 'error');
        });
    },
    
    /**
     * Format percentage for display
     */
    formatPercentage: function(value, decimals = 1) {
        return `${(value * 100).toFixed(decimals)}%`;
    },
    
    /**
     * Get sentiment color class
     */
    getSentimentColor: function(sentiment) {
        switch(sentiment.toLowerCase()) {
            case 'positive': return 'success';
            case 'negative': return 'danger';
            default: return 'secondary';
        }
    },
    
    /**
     * Get priority color class
     */
    getPriorityColor: function(priority) {
        switch(priority.toUpperCase()) {
            case 'HIGH': return 'danger';
            case 'MEDIUM': return 'warning';
            case 'LOW': return 'success';
            default: return 'secondary';
        }
    }
};

// Make API and Utils available globally
window.SentimentAPI = API;
window.SentimentUtils = Utils;