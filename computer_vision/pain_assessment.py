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

class AdvancedPainAssessment:
    """Advanced computer vision model for pain assessment using facial expressions"""
    
    def __init__(self):
        self.face_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize custom pain assessment model
        self.pain_model = self._create_pain_model()
        self._load_pain_weights()
        
        # Glasgow Pain Scale criteria
        self.glasgow_criteria = {
            'eye_closure': {'weight': 0.25, 'thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]},
            'mouth_tension': {'weight': 0.2, 'thresholds': [0.2, 0.4, 0.6, 0.8, 1.0]},
            'ear_position': {'weight': 0.15, 'thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]},
            'brow_furrowing': {'weight': 0.2, 'thresholds': [0.2, 0.4, 0.6, 0.8, 1.0]},
            'overall_tension': {'weight': 0.2, 'thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]}
        }
        
        # Facial landmark indices for pain assessment
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'ears': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
    
    def _create_pain_model(self):
        """Create neural network model for pain assessment"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(468, 3)),  # 468 face landmarks
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Output pain score (normalized)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _load_pain_weights(self):
        """Load pre-trained weights for pain assessment model"""
        weights_path = 'models/pain_model_weights.h5'
        if os.path.exists(weights_path):
            try:
                self.pain_model.load_weights(weights_path)
                logger.info("Pain assessment model weights loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load pain weights: {e}")
        else:
            logger.info("No pre-trained pain weights found, using random initialization")
    
    def assess_pain_level(self, frame: np.ndarray) -> Dict:
        """Comprehensive pain assessment using Glasgow Pain Scale"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect facial landmarks
            face_results = self.face_detector.process(rgb_frame)
            
            if not face_results.multi_face_landmarks:
                return {
                    'pain_score': None,
                    'confidence': 0.0,
                    'face_detected': False,
                    'error': 'No face detected'
                }
            
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Calculate pain using multiple methods
            pain_scores = []
            confidences = []
            
            # Method 1: Neural network prediction
            nn_score, nn_confidence = self._neural_network_pain(face_landmarks)
            if nn_score is not None:
                pain_scores.append(nn_score)
                confidences.append(nn_confidence)
            
            # Method 2: Glasgow Pain Scale analysis
            glasgow_score, glasgow_confidence = self._glasgow_pain_scale(face_landmarks)
            if glasgow_score is not None:
                pain_scores.append(glasgow_score)
                confidences.append(glasgow_confidence)
            
            # Method 3: Facial action unit analysis
            fau_score, fau_confidence = self._facial_action_units(face_landmarks)
            if fau_score is not None:
                pain_scores.append(fau_score)
                confidences.append(fau_confidence)
            
            # Calculate weighted average
            if pain_scores:
                final_score = np.average(pain_scores, weights=confidences)
                final_confidence = np.mean(confidences)
                
                # Ensure score is within valid range (0-10)
                final_score = max(0.0, min(10.0, final_score))
                
                return {
                    'pain_score': round(final_score, 1),
                    'confidence': round(final_confidence, 2),
                    'face_detected': True,
                    'analysis_methods': len(pain_scores),
                    'individual_scores': {
                        'neural_network': nn_score,
                        'glasgow_scale': glasgow_score,
                        'facial_action_units': fau_score
                    },
                    'pain_level': self._categorize_pain_level(final_score),
                    'assessment_timestamp': self._get_timestamp()
                }
            else:
                return {
                    'pain_score': None,
                    'confidence': 0.0,
                    'face_detected': True,
                    'error': 'All analysis methods failed'
                }
                
        except Exception as e:
            logger.error(f"Error in pain assessment: {str(e)}")
            return {
                'pain_score': None,
                'confidence': 0.0,
                'face_detected': False,
                'error': str(e)
            }
    
    def _neural_network_pain(self, face_landmarks) -> Tuple[Optional[float], float]:
        """Use neural network to predict pain level"""
        try:
            # Convert landmarks to input format
            landmark_data = []
            for landmark in face_landmarks.landmark:
                landmark_data.append([landmark.x, landmark.y, landmark.z])
            
            landmark_array = np.array(landmark_data).reshape(1, -1)
            
            # Predict pain score (normalized to 0-1, then scaled to 0-10)
            prediction = self.pain_model.predict(landmark_array, verbose=0)[0][0]
            pain_score = prediction * 10  # Scale to 0-10 range
            
            # Calculate confidence based on landmark visibility
            avg_visibility = np.mean([lm.visibility for lm in face_landmarks.landmark])
            confidence = min(avg_visibility * 1.2, 1.0)
            
            return pain_score, confidence
            
        except Exception as e:
            logger.error(f"Neural network pain error: {e}")
            return None, 0.0
    
    def _glasgow_pain_scale(self, face_landmarks) -> Tuple[Optional[float], float]:
        """Calculate pain score using Glasgow Pain Scale criteria"""
        try:
            landmarks = face_landmarks.landmark
            pain_indicators = {}
            
            # 1. Eye closure assessment
            left_eye_score = self._assess_eye_closure(landmarks, 'left_eye')
            right_eye_score = self._assess_eye_closure(landmarks, 'right_eye')
            pain_indicators['eye_closure'] = (left_eye_score + right_eye_score) / 2
            
            # 2. Mouth tension assessment
            pain_indicators['mouth_tension'] = self._assess_mouth_tension(landmarks)
            
            # 3. Ear position assessment
            pain_indicators['ear_position'] = self._assess_ear_position(landmarks)
            
            # 4. Brow furrowing assessment
            pain_indicators['brow_furrowing'] = self._assess_brow_furrowing(landmarks)
            
            # 5. Overall facial tension
            pain_indicators['overall_tension'] = self._assess_overall_tension(landmarks)
            
            # Calculate weighted pain score
            total_score = 0
            total_weight = 0
            
            for criterion, weight_info in self.glasgow_criteria.items():
                if criterion in pain_indicators:
                    weight = weight_info['weight']
                    score = pain_indicators[criterion]
                    total_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                pain_score = (total_score / total_weight) * 10  # Scale to 0-10
                confidence = min(total_weight * 1.1, 1.0)
                return pain_score, confidence
            else:
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Glasgow pain scale error: {e}")
            return None, 0.0
    
    def _assess_eye_closure(self, landmarks, eye_side: str) -> float:
        """Assess eye closure as pain indicator"""
        try:
            eye_indices = self.landmark_indices[eye_side]
            eye_points = [landmarks[i] for i in eye_indices if i < len(landmarks)]
            
            if len(eye_points) < 4:
                return 0.0
            
            # Calculate eye opening ratio
            top_points = eye_points[:len(eye_points)//2]
            bottom_points = eye_points[len(eye_points)//2:]
            
            top_y = np.mean([p.y for p in top_points])
            bottom_y = np.mean([p.y for p in bottom_points])
            
            eye_opening = abs(top_y - bottom_y)
            
            # Map to pain score (more closed = higher pain)
            if eye_opening < 0.01:
                return 0.9  # Very closed
            elif eye_opening < 0.02:
                return 0.7  # Closed
            elif eye_opening < 0.03:
                return 0.5  # Partially closed
            elif eye_opening < 0.04:
                return 0.3  # Slightly closed
            else:
                return 0.1  # Open
            
        except Exception as e:
            logger.error(f"Eye closure assessment error: {e}")
            return 0.0
    
    def _assess_mouth_tension(self, landmarks) -> float:
        """Assess mouth tension as pain indicator"""
        try:
            mouth_indices = self.landmark_indices['mouth']
            mouth_points = [landmarks[i] for i in mouth_indices if i < len(landmarks)]
            
            if len(mouth_points) < 4:
                return 0.0
            
            # Calculate mouth shape metrics
            top_points = mouth_points[:len(mouth_points)//2]
            bottom_points = mouth_points[len(mouth_points)//2:]
            
            top_y = np.mean([p.y for p in top_points])
            bottom_y = np.mean([p.y for p in bottom_points])
            
            mouth_height = abs(top_y - bottom_y)
            
            # Calculate mouth width
            left_points = mouth_points[:len(mouth_points)//4]
            right_points = mouth_points[3*len(mouth_points)//4:]
            
            left_x = np.mean([p.x for p in left_points])
            right_x = np.mean([p.x for p in right_points])
            
            mouth_width = abs(right_x - left_x)
            
            # Calculate tension ratio
            tension_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Map to pain score (higher tension = higher pain)
            if tension_ratio > 0.3:
                return 0.9  # High tension
            elif tension_ratio > 0.2:
                return 0.7  # Moderate tension
            elif tension_ratio > 0.1:
                return 0.5  # Low tension
            else:
                return 0.1  # No tension
            
        except Exception as e:
            logger.error(f"Mouth tension assessment error: {e}")
            return 0.0
    
    def _assess_ear_position(self, landmarks) -> float:
        """Assess ear position as pain indicator"""
        try:
            ear_indices = self.landmark_indices['ears']
            ear_points = [landmarks[i] for i in ear_indices if i < len(landmarks)]
            
            if len(ear_points) < 4:
                return 0.0
            
            # Calculate ear angle (flattened ears indicate pain)
            left_ear_points = ear_points[:len(ear_points)//2]
            right_ear_points = ear_points[len(ear_points)//2:]
            
            left_ear_angle = self._calculate_ear_angle(left_ear_points)
            right_ear_angle = self._calculate_ear_angle(right_ear_points)
            
            avg_ear_angle = (left_ear_angle + right_ear_angle) / 2
            
            # Map to pain score (flatter ears = higher pain)
            if avg_ear_angle < 0.1:
                return 0.9  # Very flat
            elif avg_ear_angle < 0.2:
                return 0.7  # Flat
            elif avg_ear_angle < 0.3:
                return 0.5  # Slightly flat
            else:
                return 0.1  # Normal
            
        except Exception as e:
            logger.error(f"Ear position assessment error: {e}")
            return 0.0
    
    def _calculate_ear_angle(self, ear_points) -> float:
        """Calculate ear angle from landmark points"""
        try:
            if len(ear_points) < 3:
                return 0.0
            
            # Calculate angle between ear points
            p1, p2, p3 = ear_points[0], ear_points[len(ear_points)//2], ear_points[-1]
            
            # Vector calculations
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return angle / np.pi  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Ear angle calculation error: {e}")
            return 0.0
    
    def _assess_brow_furrowing(self, landmarks) -> float:
        """Assess brow furrowing as pain indicator"""
        try:
            brow_indices = self.landmark_indices['eyebrows']
            brow_points = [landmarks[i] for i in brow_indices if i < len(landmarks)]
            
            if len(brow_points) < 4:
                return 0.0
            
            # Calculate brow curvature
            left_brow = brow_points[:len(brow_points)//2]
            right_brow = brow_points[len(brow_points)//2:]
            
            left_curvature = self._calculate_curvature(left_brow)
            right_curvature = self._calculate_curvature(right_brow)
            
            avg_curvature = (left_curvature + right_curvature) / 2
            
            # Map to pain score (more furrowed = higher pain)
            if avg_curvature > 0.8:
                return 0.9  # Very furrowed
            elif avg_curvature > 0.6:
                return 0.7  # Furrowed
            elif avg_curvature > 0.4:
                return 0.5  # Slightly furrowed
            else:
                return 0.1  # Normal
            
        except Exception as e:
            logger.error(f"Brow furrowing assessment error: {e}")
            return 0.0
    
    def _calculate_curvature(self, points) -> float:
        """Calculate curvature of a set of points"""
        try:
            if len(points) < 3:
                return 0.0
            
            # Calculate curvature using second derivative approximation
            x_coords = [p.x for p in points]
            y_coords = [p.y for p in points]
            
            if len(x_coords) < 3:
                return 0.0
            
            # Calculate second derivatives
            dx2 = np.gradient(np.gradient(x_coords))
            dy2 = np.gradient(np.gradient(y_coords))
            
            # Calculate curvature
            curvature = np.abs(dx2 + dy2)
            avg_curvature = np.mean(curvature)
            
            return min(avg_curvature * 10, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Curvature calculation error: {e}")
            return 0.0
    
    def _assess_overall_tension(self, landmarks) -> float:
        """Assess overall facial tension"""
        try:
            # Calculate facial symmetry and tension
            left_face = landmarks[:len(landmarks)//2]
            right_face = landmarks[len(landmarks)//2:]
            
            # Calculate asymmetry
            asymmetry = 0
            for i in range(min(len(left_face), len(right_face))):
                left_point = left_face[i]
                right_point = right_face[i]
                asymmetry += abs(left_point.x - (1 - right_point.x))  # Mirror x-coordinate
            
            avg_asymmetry = asymmetry / min(len(left_face), len(right_face))
            
            # Map to pain score (more asymmetry = higher pain)
            if avg_asymmetry > 0.1:
                return 0.9  # High asymmetry
            elif avg_asymmetry > 0.05:
                return 0.7  # Moderate asymmetry
            elif avg_asymmetry > 0.02:
                return 0.5  # Low asymmetry
            else:
                return 0.1  # Normal
            
        except Exception as e:
            logger.error(f"Overall tension assessment error: {e}")
            return 0.0
    
    def _facial_action_units(self, face_landmarks) -> Tuple[Optional[float], float]:
        """Calculate pain score using Facial Action Units"""
        try:
            landmarks = face_landmarks.landmark
            
            # Define action units for pain assessment
            action_units = {
                'AU4': self._calculate_au4(landmarks),  # Brow lowerer
                'AU6': self._calculate_au6(landmarks),  # Cheek raiser
                'AU7': self._calculate_au7(landmarks),  # Lid tightener
                'AU9': self._calculate_au9(landmarks),  # Nose wrinkler
                'AU10': self._calculate_au10(landmarks), # Upper lip raiser
                'AU12': self._calculate_au12(landmarks), # Lip corner puller
                'AU20': self._calculate_au20(landmarks), # Lip stretcher
                'AU25': self._calculate_au25(landmarks), # Lips part
                'AU26': self._calculate_au26(landmarks), # Jaw drop
                'AU43': self._calculate_au43(landmarks)  # Eyes closed
            }
            
            # Calculate pain score based on action units
            pain_indicators = ['AU4', 'AU6', 'AU7', 'AU9', 'AU20', 'AU43']
            pain_score = np.mean([action_units[au] for au in pain_indicators])
            
            # Scale to 0-10 range
            pain_score *= 10
            
            # Calculate confidence
            confidence = min(np.mean(list(action_units.values())) * 1.2, 1.0)
            
            return pain_score, confidence
            
        except Exception as e:
            logger.error(f"Facial action units error: {e}")
            return None, 0.0
    
    def _calculate_au4(self, landmarks) -> float:
        """Calculate AU4 - Brow lowerer"""
        # Simplified calculation based on eyebrow landmarks
        return 0.5  # Placeholder implementation
    
    def _calculate_au6(self, landmarks) -> float:
        """Calculate AU6 - Cheek raiser"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au7(self, landmarks) -> float:
        """Calculate AU7 - Lid tightener"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au9(self, landmarks) -> float:
        """Calculate AU9 - Nose wrinkler"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au10(self, landmarks) -> float:
        """Calculate AU10 - Upper lip raiser"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au12(self, landmarks) -> float:
        """Calculate AU12 - Lip corner puller"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au20(self, landmarks) -> float:
        """Calculate AU20 - Lip stretcher"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au25(self, landmarks) -> float:
        """Calculate AU25 - Lips part"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au26(self, landmarks) -> float:
        """Calculate AU26 - Jaw drop"""
        return 0.5  # Placeholder implementation
    
    def _calculate_au43(self, landmarks) -> float:
        """Calculate AU43 - Eyes closed"""
        return 0.5  # Placeholder implementation
    
    def _categorize_pain_level(self, pain_score: float) -> str:
        """Categorize pain score into descriptive levels"""
        if pain_score >= 8:
            return "Severe Pain"
        elif pain_score >= 6:
            return "Moderate-Severe Pain"
        elif pain_score >= 4:
            return "Moderate Pain"
        elif pain_score >= 2:
            return "Mild Pain"
        elif pain_score > 0:
            return "Minimal Pain"
        else:
            return "No Pain"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def train_model(self, training_data: List[Dict]):
        """Train the pain assessment model with new data"""
        try:
            X = []
            y = []
            
            for data_point in training_data:
                landmarks = data_point['landmarks']
                pain_score = data_point['pain_score']
                
                # Convert landmarks to input format
                landmark_data = []
                for landmark in landmarks:
                    landmark_data.append([landmark['x'], landmark['y'], landmark['z']])
                
                X.append(landmark_data)
                y.append(pain_score / 10)  # Normalize to 0-1 range
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            self.pain_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
            # Save updated weights
            os.makedirs('models', exist_ok=True)
            self.pain_model.save_weights('models/pain_model_weights.h5')
            
            logger.info("Pain assessment model trained and weights saved")
            return True
            
        except Exception as e:
            logger.error(f"Error training pain model: {e}")
            return False

