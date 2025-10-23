import cv2
import numpy as np
import mediapipe as mp
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ThreeDScanner:
    """3D scanning capabilities for mobile devices using computer vision"""
    
    def __init__(self):
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.face_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 3D reconstruction parameters
        self.scan_data = {
            'frames': [],
            'landmarks_3d': [],
            'depth_estimates': [],
            'mesh_points': [],
            'scan_quality': 0.0
        }
        
        # Camera calibration parameters (estimated for mobile cameras)
        self.camera_params = {
            'focal_length': 1000,  # Estimated focal length
            'principal_point': (320, 240),  # Image center
            'baseline': 0.1  # Estimated stereo baseline
        }
    
    def start_3d_scan(self, video_stream) -> Dict:
        """Start 3D scanning process from video stream"""
        try:
            scan_results = {
                'scan_id': self._generate_scan_id(),
                'status': 'scanning',
                'frames_processed': 0,
                'quality_score': 0.0,
                'mesh_data': None,
                'measurements': {},
                'timestamp': datetime.now().isoformat()
            }
            
            frame_count = 0
            max_frames = 100  # Limit scanning to prevent memory issues
            
            while frame_count < max_frames:
                ret, frame = video_stream.read()
                if not ret:
                    break
                
                # Process frame for 3D reconstruction
                frame_data = self._process_frame_for_3d(frame)
                if frame_data:
                    self.scan_data['frames'].append(frame_data)
                    scan_results['frames_processed'] += 1
                
                frame_count += 1
            
            # Generate 3D mesh from collected data
            if len(self.scan_data['frames']) > 10:
                mesh_data = self._generate_3d_mesh()
                measurements = self._calculate_3d_measurements(mesh_data)
                
                scan_results.update({
                    'status': 'completed',
                    'mesh_data': mesh_data,
                    'measurements': measurements,
                    'quality_score': self._calculate_scan_quality()
                })
            else:
                scan_results['status'] = 'failed'
                scan_results['error'] = 'Insufficient frames for 3D reconstruction'
            
            return scan_results
            
        except Exception as e:
            logger.error(f"3D scanning error: {e}")
            return {
                'scan_id': self._generate_scan_id(),
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_frame_for_3d(self, frame: np.ndarray) -> Optional[Dict]:
        """Process individual frame for 3D reconstruction"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose landmarks
            pose_results = self.pose_detector.process(rgb_frame)
            
            # Detect facial landmarks
            face_results = self.face_detector.process(rgb_frame)
            
            frame_data = {
                'timestamp': datetime.now().isoformat(),
                'pose_landmarks': None,
                'face_landmarks': None,
                'depth_map': None,
                'quality_score': 0.0
            }
            
            # Extract pose landmarks
            if pose_results.pose_landmarks:
                landmarks_3d = []
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks_3d.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['pose_landmarks'] = landmarks_3d
            
            # Extract facial landmarks
            if face_results.multi_face_landmarks:
                face_landmarks_3d = []
                for landmark in face_results.multi_face_landmarks[0].landmark:
                    face_landmarks_3d.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                frame_data['face_landmarks'] = face_landmarks_3d
            
            # Estimate depth map
            depth_map = self._estimate_depth_map(frame)
            frame_data['depth_map'] = depth_map
            
            # Calculate frame quality
            frame_data['quality_score'] = self._calculate_frame_quality(frame_data)
            
            return frame_data if frame_data['quality_score'] > 0.3 else None
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None
    
    def _estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from single image using monocular depth estimation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate gradients
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize to depth range (0-1)
            depth_map = gradient_magnitude / np.max(gradient_magnitude)
            
            # Apply smoothing
            depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)
    
    def _generate_3d_mesh(self) -> Dict:
        """Generate 3D mesh from collected frame data"""
        try:
            if len(self.scan_data['frames']) < 5:
                return None
            
            # Combine all landmarks from all frames
            all_landmarks = []
            for frame_data in self.scan_data['frames']:
                if frame_data['pose_landmarks']:
                    all_landmarks.extend(frame_data['pose_landmarks'])
            
            if len(all_landmarks) < 10:
                return None
            
            # Create 3D point cloud
            points_3d = []
            for landmark in all_landmarks:
                if landmark['visibility'] > 0.5:
                    points_3d.append([
                        landmark['x'],
                        landmark['y'],
                        landmark['z']
                    ])
            
            points_3d = np.array(points_3d)
            
            # Generate mesh using Delaunay triangulation
            mesh_data = self._create_mesh_from_points(points_3d)
            
            return mesh_data
            
        except Exception as e:
            logger.error(f"3D mesh generation error: {e}")
            return None
    
    def _create_mesh_from_points(self, points_3d: np.ndarray) -> Dict:
        """Create mesh from 3D points using triangulation"""
        try:
            # Project 3D points to 2D for triangulation
            points_2d = points_3d[:, :2]
            
            # Create Delaunay triangulation
            from scipy.spatial import Delaunay
            tri = Delaunay(points_2d)
            
            # Extract triangles
            triangles = tri.simplices
            
            # Create mesh data
            mesh_data = {
                'vertices': points_3d.tolist(),
                'triangles': triangles.tolist(),
                'vertex_count': len(points_3d),
                'triangle_count': len(triangles),
                'bounding_box': {
                    'min': points_3d.min(axis=0).tolist(),
                    'max': points_3d.max(axis=0).tolist()
                }
            }
            
            return mesh_data
            
        except Exception as e:
            logger.error(f"Mesh creation error: {e}")
            return None
    
    def _calculate_3d_measurements(self, mesh_data: Dict) -> Dict:
        """Calculate 3D measurements from mesh data"""
        try:
            if not mesh_data or not mesh_data['vertices']:
                return {}
            
            vertices = np.array(mesh_data['vertices'])
            
            measurements = {}
            
            # Calculate body dimensions
            if len(vertices) > 0:
                # Body length (height)
                measurements['body_length'] = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
                
                # Body width
                measurements['body_width'] = float(np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
                
                # Body depth
                measurements['body_depth'] = float(np.max(vertices[:, 2]) - np.min(vertices[:, 2]))
                
                # Volume estimation (simplified)
                measurements['estimated_volume'] = float(
                    measurements['body_length'] * 
                    measurements['body_width'] * 
                    measurements['body_depth']
                )
                
                # Surface area estimation
                measurements['estimated_surface_area'] = float(
                    2 * (measurements['body_length'] * measurements['body_width'] +
                         measurements['body_length'] * measurements['body_depth'] +
                         measurements['body_width'] * measurements['body_depth'])
                )
            
            return measurements
            
        except Exception as e:
            logger.error(f"3D measurements calculation error: {e}")
            return {}
    
    def _calculate_frame_quality(self, frame_data: Dict) -> float:
        """Calculate quality score for individual frame"""
        try:
            quality_score = 0.0
            
            # Pose landmarks quality
            if frame_data['pose_landmarks']:
                pose_quality = np.mean([lm['visibility'] for lm in frame_data['pose_landmarks']])
                quality_score += pose_quality * 0.6
            
            # Face landmarks quality
            if frame_data['face_landmarks']:
                face_quality = len(frame_data['face_landmarks']) / 468  # Max face landmarks
                quality_score += face_quality * 0.3
            
            # Depth map quality
            if frame_data['depth_map'] is not None:
                depth_variance = np.var(frame_data['depth_map'])
                depth_quality = min(depth_variance * 10, 1.0)
                quality_score += depth_quality * 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Frame quality calculation error: {e}")
            return 0.0
    
    def _calculate_scan_quality(self) -> float:
        """Calculate overall scan quality"""
        try:
            if not self.scan_data['frames']:
                return 0.0
            
            # Average frame quality
            avg_frame_quality = np.mean([frame['quality_score'] for frame in self.scan_data['frames']])
            
            # Frame count factor
            frame_count_factor = min(len(self.scan_data['frames']) / 50, 1.0)
            
            # Landmark consistency factor
            landmark_consistency = self._calculate_landmark_consistency()
            
            # Overall quality score
            overall_quality = (avg_frame_quality * 0.5 + 
                              frame_count_factor * 0.3 + 
                              landmark_consistency * 0.2)
            
            return min(overall_quality, 1.0)
            
        except Exception as e:
            logger.error(f"Scan quality calculation error: {e}")
            return 0.0
    
    def _calculate_landmark_consistency(self) -> float:
        """Calculate consistency of landmarks across frames"""
        try:
            if len(self.scan_data['frames']) < 2:
                return 0.0
            
            # Extract landmarks from all frames
            all_landmarks = []
            for frame in self.scan_data['frames']:
                if frame['pose_landmarks']:
                    landmarks = [(lm['x'], lm['y']) for lm in frame['pose_landmarks']]
                    all_landmarks.append(landmarks)
            
            if len(all_landmarks) < 2:
                return 0.0
            
            # Calculate variance in landmark positions
            landmark_variance = []
            for i in range(min(len(all_landmarks[0]), len(all_landmarks[1]))):
                pos1 = all_landmarks[0][i]
                pos2 = all_landmarks[1][i]
                variance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                landmark_variance.append(variance)
            
            # Consistency score (lower variance = higher consistency)
            avg_variance = np.mean(landmark_variance)
            consistency_score = max(0, 1 - avg_variance * 10)
            
            return min(consistency_score, 1.0)
            
        except Exception as e:
            logger.error(f"Landmark consistency calculation error: {e}")
            return 0.0
    
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID"""
        return f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def export_3d_data(self, scan_results: Dict, format: str = 'json') -> str:
        """Export 3D scan data in specified format"""
        try:
            scan_id = scan_results['scan_id']
            
            if format == 'json':
                filename = f"data/scans/{scan_id}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, 'w') as f:
                    json.dump(scan_results, f, indent=2)
                
                return filename
            
            elif format == 'obj':
                filename = f"data/scans/{scan_id}.obj"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                self._export_obj_file(scan_results, filename)
                return filename
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Export error: {e}")
            return None
    
    def _export_obj_file(self, scan_results: Dict, filename: str):
        """Export mesh data as OBJ file"""
        try:
            mesh_data = scan_results['mesh_data']
            if not mesh_data:
                return
            
            with open(filename, 'w') as f:
                # Write vertices
                for vertex in mesh_data['vertices']:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                
                # Write faces
                for triangle in mesh_data['triangles']:
                    f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
                    
        except Exception as e:
            logger.error(f"OBJ export error: {e}")
    
    def get_scan_statistics(self) -> Dict:
        """Get statistics about current scan session"""
        return {
            'frames_collected': len(self.scan_data['frames']),
            'total_landmarks': sum(len(frame.get('pose_landmarks', [])) for frame in self.scan_data['frames']),
            'average_quality': np.mean([frame['quality_score'] for frame in self.scan_data['frames']]) if self.scan_data['frames'] else 0,
            'scan_duration': len(self.scan_data['frames']) * 0.1,  # Assuming 10 FPS
            'memory_usage': len(str(self.scan_data))  # Rough estimate
        }
    
    def reset_scan(self):
        """Reset scan data for new scanning session"""
        self.scan_data = {
            'frames': [],
            'landmarks_3d': [],
            'depth_estimates': [],
            'mesh_points': [],
            'scan_quality': 0.0
        }

