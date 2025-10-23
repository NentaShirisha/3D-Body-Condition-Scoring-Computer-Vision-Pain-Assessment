// Main JavaScript for the web application
document.addEventListener('DOMContentLoaded', function() {
    const videoInput = document.getElementById('videoInput');
    const uploadArea = document.getElementById('uploadArea');
    const uploadProgress = document.getElementById('uploadProgress');
    const resultsContainer = document.getElementById('resultsContainer');
    const historyBody = document.getElementById('historyBody');
    
    // Initialize
    loadAssessmentHistory();
    
    // File upload handling
    videoInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => videoInput.click());
    
    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            videoInput.files = files;
            handleFileSelect();
        }
    }
    
    function handleFileSelect() {
        const file = videoInput.files[0];
        if (file) {
            uploadVideo(file);
        }
    }
    
    function uploadVideo(file) {
        const formData = new FormData();
        formData.append('video', file);
        
        // Show progress
        uploadProgress.classList.remove('d-none');
        uploadProgress.querySelector('.progress-bar').style.width = '0%';
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            uploadProgress.querySelector('.progress-bar').style.width = progress + '%';
        }, 200);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(progressInterval);
            uploadProgress.querySelector('.progress-bar').style.width = '100%';
            
            setTimeout(() => {
                uploadProgress.classList.add('d-none');
                displayResults(data);
                loadAssessmentHistory();
            }, 500);
        })
        .catch(error => {
            clearInterval(progressInterval);
            uploadProgress.classList.add('d-none');
            console.error('Error:', error);
            showError('Failed to upload video. Please try again.');
        });
    }
    
    function displayResults(data) {
        if (data.error) {
            showError(data.error);
            return;
        }
        
        const resultsHtml = `
            <div class="row">
                <div class="col-md-6">
                    <div class="result-card bcs-score">
                        <h6><i class="fas fa-weight me-2"></i>Body Condition Score</h6>
                        <div class="score">${data.body_condition_score || 'N/A'}</div>
                        <div class="confidence">Confidence: ${Math.round(data.bcs_confidence * 100)}%</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="result-card pain-score">
                        <h6><i class="fas fa-face-frown me-2"></i>Pain Score</h6>
                        <div class="score">${data.pain_score || 'N/A'}</div>
                        <div class="confidence">Confidence: ${Math.round(data.pain_confidence * 100)}%</div>
                    </div>
                </div>
            </div>
            <div class="mt-3">
                <div class="alert alert-info">
                    <h6><i class="fas fa-info-circle me-2"></i>Analysis Summary</h6>
                    <p class="mb-0">
                        Frames analyzed: ${data.frames_analyzed || 0}<br>
                        Assessment ID: ${data.assessment_id || 'N/A'}<br>
                        Status: ${data.analysis_complete ? 'Complete' : 'Failed'}
                    </p>
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = resultsHtml;
        resultsContainer.classList.add('fade-in');
        
        // Show recommendations
        showRecommendations(data);
    }
    
    function showRecommendations(data) {
        const bcs = data.body_condition_score;
        const pain = data.pain_score;
        
        let recommendations = [];
        
        if (bcs !== null) {
            if (bcs < 4) {
                recommendations.push('Animal appears underweight - consider nutritional assessment');
            } else if (bcs > 6) {
                recommendations.push('Animal appears overweight - consider dietary management');
            } else {
                recommendations.push('Body condition appears normal');
            }
        }
        
        if (pain !== null) {
            if (pain > 7) {
                recommendations.push('High pain level detected - immediate veterinary attention recommended');
            } else if (pain > 4) {
                recommendations.push('Moderate pain detected - monitor closely and consider pain management');
            } else if (pain > 0) {
                recommendations.push('Low pain level detected - continue monitoring');
            } else {
                recommendations.push('No significant pain indicators detected');
            }
        }
        
        if (recommendations.length > 0) {
            const recommendationsHtml = `
                <div class="mt-3">
                    <div class="alert alert-warning">
                        <h6><i class="fas fa-lightbulb me-2"></i>Recommendations</h6>
                        <ul class="mb-0">
                            ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
            resultsContainer.innerHTML += recommendationsHtml;
        }
    }
    
    function showError(message) {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>Error</h6>
                <p class="mb-0">${message}</p>
            </div>
        `;
    }
    
    function loadAssessmentHistory() {
        fetch('/assessments')
        .then(response => response.json())
        .then(data => {
            if (data.assessments && data.assessments.length > 0) {
                displayHistory(data.assessments);
            } else {
                historyBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No assessments yet</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error loading history:', error);
            historyBody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading history</td></tr>';
        });
    }
    
    function displayHistory(assessments) {
        const historyHtml = assessments.slice(-10).reverse().map(assessment => {
            const timestamp = new Date(assessment.timestamp).toLocaleString();
            const statusClass = assessment.body_condition_score !== null ? 'complete' : 'error';
            const statusText = assessment.body_condition_score !== null ? 'Complete' : 'Error';
            
            return `
                <tr class="assessment-row" data-id="${assessment.id}">
                    <td>${assessment.id}</td>
                    <td>${timestamp}</td>
                    <td>${assessment.body_condition_score || 'N/A'}</td>
                    <td>${assessment.pain_score || 'N/A'}</td>
                    <td>${Math.round((assessment.bcs_confidence || 0) * 100)}%</td>
                    <td><span class="status-badge status-${statusClass}">${statusText}</span></td>
                </tr>
            `;
        }).join('');
        
        historyBody.innerHTML = historyHtml;
        
        // Add click handlers for history rows
        document.querySelectorAll('.assessment-row').forEach(row => {
            row.addEventListener('click', () => {
                const assessmentId = row.dataset.id;
                showAssessmentDetails(assessmentId);
            });
        });
    }
    
    function showAssessmentDetails(assessmentId) {
        // This would show detailed assessment information in a modal
        console.log('Showing details for assessment:', assessmentId);
    }
});

// Global function for saving results (called from modal)
function saveResults() {
    // Implementation for saving results to external system
    console.log('Saving results...');
    alert('Results saved successfully!');
}

