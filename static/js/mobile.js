// Mobile app JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const videoElement = document.getElementById('videoElement');
    const canvasElement = document.getElementById('canvasElement');
    const startCameraBtn = document.getElementById('startCamera');
    const captureVideoBtn = document.getElementById('captureVideo');
    const analyzeVideoBtn = document.getElementById('analyzeVideo');
    const resultsSection = document.getElementById('resultsSection');
    
    // Mobile navbar elements
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const curvedMenu = document.getElementById('curvedMenu');
    const closeBtn = document.getElementById('closeBtn');
    
    let mediaStream = null;
    let recordedChunks = [];
    let mediaRecorder = null;
    let isRecording = false;
    
    // Initialize
    loadMobileHistory();
    initializeMobileNavbar();
    
    // Mobile navbar functionality
    function initializeMobileNavbar() {
        if (hamburgerBtn && curvedMenu && closeBtn) {
            hamburgerBtn.addEventListener('click', toggleMenu);
            closeBtn.addEventListener('click', closeMenu);
            
            // Close menu when clicking outside
            document.addEventListener('click', function(e) {
                if (!curvedMenu.contains(e.target) && !hamburgerBtn.contains(e.target)) {
                    closeMenu();
                }
            });
            
            // Close menu on escape key
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    closeMenu();
                }
            });
        }
    }
    
    function toggleMenu() {
        curvedMenu.classList.toggle('active');
        hamburgerBtn.classList.toggle('active');
        
        // Add overlay
        let overlay = document.querySelector('.menu-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'menu-overlay';
            document.body.appendChild(overlay);
        }
        overlay.classList.toggle('active');
    }
    
    function closeMenu() {
        curvedMenu.classList.remove('active');
        hamburgerBtn.classList.remove('active');
        
        const overlay = document.querySelector('.menu-overlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }
    
    // Camera controls
    startCameraBtn.addEventListener('click', startCamera);
    captureVideoBtn.addEventListener('click', toggleRecording);
    analyzeVideoBtn.addEventListener('click', analyzeVideo);
    
    async function startCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Use back camera on mobile
                },
                audio: false
            });
            
            videoElement.srcObject = mediaStream;
            startCameraBtn.disabled = true;
            captureVideoBtn.disabled = false;
            
            // Initialize MediaRecorder
            mediaRecorder = new MediaRecorder(mediaStream, {
                mimeType: 'video/webm;codecs=vp9'
            });
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                recordedChunks = [];
                analyzeVideoBlob(blob);
            };
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Unable to access camera. Please check permissions.');
        }
    }
    
    function toggleRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }
    
    function startRecording() {
        recordedChunks = [];
        mediaRecorder.start();
        isRecording = true;
        captureVideoBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Stop Recording';
        captureVideoBtn.classList.remove('btn-success');
        captureVideoBtn.classList.add('btn-danger');
        captureVideoBtn.classList.add('recording');
        
        // Auto-stop after 10 seconds
        setTimeout(() => {
            if (isRecording) {
                stopRecording();
            }
        }, 10000);
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
        
        isRecording = false;
        captureVideoBtn.innerHTML = '<i class="fas fa-video me-2"></i>Record Video';
        captureVideoBtn.classList.remove('btn-danger');
        captureVideoBtn.classList.add('btn-success');
        captureVideoBtn.classList.remove('recording');
        analyzeVideoBtn.disabled = false;
    }
    
    function analyzeVideo() {
        if (recordedChunks.length > 0) {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            analyzeVideoBlob(blob);
        } else {
            alert('Please record a video first.');
        }
    }
    
    function analyzeVideoBlob(blob) {
        const formData = new FormData();
        formData.append('video', blob, 'recording.webm');
        
        // Show loading state
        analyzeVideoBtn.innerHTML = '<span class="loading-spinner me-2"></span>Analyzing...';
        analyzeVideoBtn.disabled = true;
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayMobileResults(data);
            loadMobileHistory();
        })
        .catch(error => {
            console.error('Error analyzing video:', error);
            alert('Failed to analyze video. Please try again.');
        })
        .finally(() => {
            analyzeVideoBtn.innerHTML = '<i class="fas fa-brain me-2"></i>Analyze';
            analyzeVideoBtn.disabled = false;
        });
    }
    
    function displayMobileResults(data) {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Update result cards
        document.getElementById('bcsScore').textContent = data.body_condition_score || 'N/A';
        document.getElementById('bcsConfidence').textContent = `Confidence: ${Math.round((data.bcs_confidence || 0) * 100)}%`;
        document.getElementById('painScore').textContent = data.pain_score || 'N/A';
        document.getElementById('painConfidence').textContent = `Confidence: ${Math.round((data.pain_confidence || 0) * 100)}%`;
        
        // Show recommendations
        showMobileRecommendations(data);
        
        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function showMobileRecommendations(data) {
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
        
        const recommendationsHtml = `
            <h6><i class="fas fa-lightbulb me-2"></i>Recommendations</h6>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
        
        document.getElementById('recommendations').innerHTML = recommendationsHtml;
    }
    
    function loadMobileHistory() {
        fetch('/assessments')
        .then(response => response.json())
        .then(data => {
            if (data.assessments && data.assessments.length > 0) {
                displayMobileHistory(data.assessments);
            } else {
                document.getElementById('mobileHistory').innerHTML = '<p class="text-center text-muted">No assessments yet</p>';
            }
        })
        .catch(error => {
            console.error('Error loading mobile history:', error);
            document.getElementById('mobileHistory').innerHTML = '<p class="text-center text-danger">Error loading history</p>';
        });
    }
    
    function displayMobileHistory(assessments) {
        const historyHtml = assessments.slice(-5).reverse().map(assessment => {
            const timestamp = new Date(assessment.timestamp).toLocaleString();
            
            return `
                <div class="history-item">
                    <h6>Assessment #${assessment.id}</h6>
                    <div class="timestamp">${timestamp}</div>
                    <div class="scores">
                        <div class="score-item">
                            <div>BCS</div>
                            <div class="score-value">${assessment.body_condition_score || 'N/A'}</div>
                        </div>
                        <div class="score-item">
                            <div>Pain</div>
                            <div class="score-value">${assessment.pain_score || 'N/A'}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        document.getElementById('mobileHistory').innerHTML = historyHtml;
    }
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', function() {
        if (document.hidden && mediaStream) {
            // Pause camera when page is hidden to save battery
            videoElement.pause();
        } else if (!document.hidden && mediaStream) {
            videoElement.play();
        }
    });
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
    });
});

