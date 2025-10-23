import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import logging

# Import custom modules
from computer_vision.body_condition_scorer import AdvancedBodyConditionScorer
from computer_vision.pain_assessment import AdvancedPainAssessment
from computer_vision.three_d_scanner import ThreeDScanner
from data_manager import AssessmentDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize advanced models
bcs_scorer = AdvancedBodyConditionScorer()
pain_assessor = AdvancedPainAssessment()
three_d_scanner = ThreeDScanner()
data_manager = AssessmentDataManager()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and analysis"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save uploaded video temporarily
        video_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_file.save(video_path)
        
        # Process video
        results = process_video(video_path)
        
        # Clean up temporary file
        os.remove(video_path)
        
        # Log results using advanced data manager
        assessment_id = data_manager.save_assessment(results)
        results['assessment_id'] = assessment_id
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_video(video_path):
    """Process video for BCS and pain assessment"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        bcs_scores = []
        pain_scores = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze every 10th frame for efficiency
            if frame_count % 10 == 0:
                # Body condition scoring
                bcs_result = bcs_scorer.analyze_body_condition(frame)
                if bcs_result['bcs_score'] is not None:
                    bcs_scores.append(bcs_result['bcs_score'])
                
                # Pain assessment
                pain_result = pain_assessor.assess_pain_level(frame)
                if pain_result['pain_score'] is not None:
                    pain_scores.append(pain_result['pain_score'])
            
            frame_count += 1
        
        cap.release()
        
        # Calculate average scores
        avg_bcs = np.mean(bcs_scores) if bcs_scores else None
        avg_pain = np.mean(pain_scores) if pain_scores else None
        
        return {
            'body_condition_score': round(avg_bcs, 1) if avg_bcs else None,
            'pain_score': round(avg_pain, 1) if avg_pain else None,
            'frames_analyzed': frame_count,
            'bcs_confidence': 0.85 if avg_bcs else 0.0,
            'pain_confidence': 0.80 if avg_pain else 0.0,
            'analysis_complete': True
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'body_condition_score': None,
            'pain_score': None,
            'error': str(e),
            'analysis_complete': False
        }

@app.route('/assessments')
def get_assessments():
    """Get all logged assessments"""
    try:
        assessments = data_manager.get_all_assessments()
        return jsonify({'assessments': assessments})
    except Exception as e:
        logger.error(f"Error retrieving assessments: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/assessments/<assessment_id>')
def get_assessment(assessment_id):
    """Get specific assessment by ID"""
    try:
        assessment = data_manager.get_assessment(assessment_id)
        if assessment:
            return jsonify(assessment)
        else:
            return jsonify({'error': 'Assessment not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving assessment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics')
def get_statistics():
    """Get assessment statistics"""
    try:
        stats = data_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/scan', methods=['POST'])
def start_3d_scan():
    """Start 3D scanning process"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save uploaded video temporarily
        video_path = f"temp_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_file.save(video_path)
        
        # Process video for 3D scanning
        cap = cv2.VideoCapture(video_path)
        scan_results = three_d_scanner.start_3d_scan(cap)
        cap.release()
        
        # Clean up temporary file
        os.remove(video_path)
        
        # Save scan results
        scan_id = data_manager.save_scan(scan_results)
        scan_results['scan_id'] = scan_id
        
        return jsonify(scan_results)
        
    except Exception as e:
        logger.error(f"Error processing 3D scan: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/scans')
def get_scans():
    """Get all 3D scans"""
    try:
        scans = data_manager.get_all_scans()
        return jsonify({'scans': scans})
    except Exception as e:
        logger.error(f"Error retrieving scans: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/<data_type>')
def export_data(data_type):
    """Export data in JSON format"""
    try:
        filename = data_manager.export_data('json', data_type)
        if filename:
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename), as_attachment=True)
        else:
            return jsonify({'error': 'Export failed'}), 500
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/report/<report_type>')
def generate_report(report_type):
    """Generate assessment report"""
    try:
        report = data_manager.generate_report(report_type)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/mobile')
def mobile_app():
    """Mobile app interface"""
    return render_template('mobile.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
