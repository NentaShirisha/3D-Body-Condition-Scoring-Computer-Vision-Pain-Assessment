import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedBodyConditionScorer:
    """Advanced computer vision model for precise body condition scoring"""
    
    def __init__(self):
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize custom BCS model
        self.bcs_model = self._create_bcs_model()
        self._load_bcs_weights()
        
        # Body condition scoring parameters
        self.bcs_criteria = {
            'rib_visibility': {'weight': 0.3, 'thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]},
            'waist_definition': {'weight': 0.25, 'thresholds': [0.2, 0.4, 0.6, 0.8, 1.0]},
            'hip_prominence': {'weight': 0.2, 'thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]},
            'muscle_definition': {'weight': 0.15, 'thresholds': [0.2, 0.4, 0.6, 0.8, 1.0]},
            'overall_shape': {'weight': 0.1, 'thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]}
        }
    
    def _create_bcs_model(self):
        """Create neural network model for BCS prediction"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(33, 3)),  # 33 pose landmarks
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Output BCS score (normalized)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _load_bcs_weights(self):
        """Load pre-trained weights for BCS model"""
        weights_path = 'models/bcs_model_weights.h5'
        if os.path.exists(weights_path):
            try:
                self.bcs_model.load_weights(weights_path)
                logger.info("BCS model weights loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load BCS weights: {e}")
        else:
            logger.info("No pre-trained BCS weights found, using random initialization")
    
    def analyze_body_condition(self, frame: np.ndarray) -> Dict:
        """Comprehensive body condition analysis"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose landmarks
            pose_results = self.pose_detector.process(rgb_frame)
            
            if not pose_results.pose_landmarks:
                return {
                    'bcs_score': None,
                    'confidence': 0.0,
                    'landmarks_detected': False,
                    'error': 'No pose landmarks detected'
                }
            
            # Extract landmarks
            landmarks = pose_results.pose_landmarks.landmark
            
            # Calculate BCS using multiple methods
            bcs_scores = []
            confidences = []
            
            # Method 1: Neural network prediction
            nn_score, nn_confidence = self._neural_network_bcs(landmarks)
            if nn_score is not None:
                bcs_scores.append(nn_score)
                confidences.append(nn_confidence)
            
            # Method 2: Geometric analysis
            geo_score, geo_confidence = self._geometric_bcs(landmarks, frame.shape)
            if geo_score is not None:
                bcs_scores.append(geo_score)
                confidences.append(geo_confidence)
            
            # Method 3: Contour analysis
            contour_score, contour_confidence = self._contour_bcs(frame, landmarks)
            if contour_score is not None:
                bcs_scores.append(contour_score)
                confidences.append(contour_confidence)
            
            # Calculate weighted average
            if bcs_scores:
                final_score = np.average(bcs_scores, weights=confidences)
                final_confidence = np.mean(confidences)
                
                # Ensure score is within valid range (1-9)
                final_score = max(1.0, min(9.0, final_score))
                
                return {
                    'bcs_score': round(final_score, 1),
                    'confidence': round(final_confidence, 2),
                    'landmarks_detected': True,
                    'analysis_methods': len(bcs_scores),
                    'individual_scores': {
                        'neural_network': nn_score,
                        'geometric': geo_score,
                        'contour': contour_score
                    },
                    'analysis_timestamp': self._get_timestamp()
                }
            else:
                return {
                    'bcs_score': None,
                    'confidence': 0.0,
                    'landmarks_detected': True,
                    'error': 'All analysis methods failed'
                }
                
        except Exception as e:
            logger.error(f"Error in body condition analysis: {str(e)}")
            return {
                'bcs_score': None,
                'confidence': 0.0,
                'landmarks_detected': False,
                'error': str(e)
            }
    
    def _neural_network_bcs(self, landmarks: List) -> Tuple[Optional[float], float]:
        """Use neural network to predict BCS"""
        try:
            # Convert landmarks to input format
            landmark_data = []
            for landmark in landmarks:
                landmark_data.append([landmark.x, landmark.y, landmark.visibility])
            
            landmark_array = np.array(landmark_data).reshape(1, -1)
            
            # Predict BCS (normalized to 0-1, then scaled to 1-9)
            prediction = self.bcs_model.predict(landmark_array, verbose=0)[0][0]
            bcs_score = 1 + (prediction * 8)  # Scale to 1-9 range
            
            # Calculate confidence based on landmark visibility
            avg_visibility = np.mean([lm.visibility for lm in landmarks])
            confidence = min(avg_visibility * 1.2, 1.0)
            
            return bcs_score, confidence
            
        except Exception as e:
            logger.error(f"Neural network BCS error: {e}")
            return None, 0.0
    
    def _geometric_bcs(self, landmarks: List, frame_shape: Tuple) -> Tuple[Optional[float], float]:
        """Calculate BCS using geometric measurements"""
        try:
            # Extract key body points
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            
            # Calculate body measurements
            shoulder_width = abs(right_shoulder.x - left_shoulder.x) * frame_shape[1]
            hip_width = abs(right_hip.x - left_hip.x) * frame_shape[1]
            body_height = abs((left_shoulder.y + right_shoulder.y) / 2 - 
                            (left_ankle.y + right_ankle.y) / 2) * frame_shape[0]
            
            # Calculate ratios
            waist_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0
            body_mass_index = (shoulder_width * hip_width) / (body_height ** 2) if body_height > 0 else 1.0
            
            # Map to BCS scale based on veterinary standards
            if waist_hip_ratio > 1.3 and body_mass_index > 0.8:
                bcs_score = 8.0  # Overweight
            elif waist_hip_ratio > 1.2 and body_mass_index > 0.7:
                bcs_score = 7.0  # Slightly overweight
            elif waist_hip_ratio > 1.0 and body_mass_index > 0.6:
                bcs_score = 6.0  # Good condition
            elif waist_hip_ratio > 0.9 and body_mass_index > 0.5:
                bcs_score = 5.0  # Ideal
            elif waist_hip_ratio > 0.8 and body_mass_index > 0.4:
                bcs_score = 4.0  # Slightly underweight
            elif waist_hip_ratio > 0.7 and body_mass_index > 0.3:
                bcs_score = 3.0  # Underweight
            else:
                bcs_score = 2.0  # Very underweight
            
            # Calculate confidence based on landmark visibility
            key_points = [left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle]
            avg_visibility = np.mean([point.visibility for point in key_points])
            confidence = min(avg_visibility * 1.1, 1.0)
            
            return bcs_score, confidence
            
        except Exception as e:
            logger.error(f"Geometric BCS error: {e}")
            return None, 0.0
    
    def _contour_bcs(self, frame: np.ndarray, landmarks: List) -> Tuple[Optional[float], float]:
        """Calculate BCS using body contour analysis"""
        try:
            # Create mask from pose segmentation
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Draw body contour based on landmarks
            body_points = []
            for landmark in landmarks:
                if landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    body_points.append([x, y])
            
            if len(body_points) < 10:
                return None, 0.0
            
            # Create convex hull
            hull = cv2.convexHull(np.array(body_points))
            cv2.fillPoly(mask, [hull], 255)
            
            # Analyze body shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, 0.0
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape descriptors
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate compactness (shape regularity)
            compactness = (perimeter ** 2) / area if area > 0 else 0
            
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1
            
            # Map to BCS based on shape characteristics
            if compactness > 20 and aspect_ratio > 0.8:
                bcs_score = 7.0  # Overweight (rounder shape)
            elif compactness > 15 and aspect_ratio > 0.7:
                bcs_score = 6.0  # Slightly overweight
            elif compactness > 10 and aspect_ratio > 0.6:
                bcs_score = 5.0  # Ideal
            elif compactness > 8 and aspect_ratio > 0.5:
                bcs_score = 4.0  # Slightly underweight
            elif compactness > 6 and aspect_ratio > 0.4:
                bcs_score = 3.0  # Underweight
            else:
                bcs_score = 2.0  # Very underweight
            
            # Calculate confidence based on contour quality
            confidence = min(area / (frame.shape[0] * frame.shape[1] * 0.1), 1.0)
            
            return bcs_score, confidence
            
        except Exception as e:
            logger.error(f"Contour BCS error: {e}")
            return None, 0.0
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def train_model(self, training_data: List[Dict]):
        """Train the BCS model with new data"""
        try:
            X = []
            y = []
            
            for data_point in training_data:
                landmarks = data_point['landmarks']
                bcs_score = data_point['bcs_score']
                
                # Convert landmarks to input format
                landmark_data = []
                for landmark in landmarks:
                    landmark_data.append([landmark['x'], landmark['y'], landmark['visibility']])
                
                X.append(landmark_data)
                y.append((bcs_score - 1) / 8)  # Normalize to 0-1 range
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            self.bcs_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
            # Save updated weights
            os.makedirs('models', exist_ok=True)
            self.bcs_model.save_weights('models/bcs_model_weights.h5')
            
            logger.info("BCS model trained and weights saved")
            return True
            
        except Exception as e:
            logger.error(f"Error training BCS model: {e}")
            return False

