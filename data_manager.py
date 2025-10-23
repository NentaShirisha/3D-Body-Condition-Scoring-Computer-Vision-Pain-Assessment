import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class AssessmentDataManager:
    """Manages assessment data storage, retrieval, and analysis"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.assessments_file = os.path.join(data_dir, 'assessments.json')
        self.scans_file = os.path.join(data_dir, 'scans.json')
        self.statistics_file = os.path.join(data_dir, 'statistics.json')
        
        # Create data directory structure
        self._create_data_structure()
        
        # Initialize data files
        self._initialize_data_files()
    
    def _create_data_structure(self):
        """Create necessary data directories"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, 'assessments'),
            os.path.join(self.data_dir, 'scans'),
            os.path.join(self.data_dir, 'models'),
            os.path.join(self.data_dir, 'exports'),
            os.path.join(self.data_dir, 'reports')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_data_files(self):
        """Initialize data files if they don't exist"""
        if not os.path.exists(self.assessments_file):
            self._save_data(self.assessments_file, {'assessments': []})
        
        if not os.path.exists(self.scans_file):
            self._save_data(self.scans_file, {'scans': []})
        
        if not os.path.exists(self.statistics_file):
            self._save_data(self.statistics_file, {
                'total_assessments': 0,
                'total_scans': 0,
                'average_bcs': 0.0,
                'average_pain': 0.0,
                'last_updated': datetime.now().isoformat()
            })
    
    def save_assessment(self, assessment_data: Dict) -> str:
        """Save assessment data and return assessment ID"""
        try:
            # Load existing assessments
            assessments = self._load_data(self.assessments_file)
            
            # Generate assessment ID
            assessment_id = self._generate_assessment_id()
            
            # Add metadata
            assessment_data.update({
                'id': assessment_id,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            })
            
            # Add to assessments list
            assessments['assessments'].append(assessment_data)
            
            # Save updated data
            self._save_data(self.assessments_file, assessments)
            
            # Update statistics
            self._update_statistics()
            
            logger.info(f"Assessment saved with ID: {assessment_id}")
            return assessment_id
            
        except Exception as e:
            logger.error(f"Error saving assessment: {e}")
            return None
    
    def save_scan(self, scan_data: Dict) -> str:
        """Save 3D scan data and return scan ID"""
        try:
            # Load existing scans
            scans = self._load_data(self.scans_file)
            
            # Generate scan ID
            scan_id = self._generate_scan_id()
            
            # Add metadata
            scan_data.update({
                'id': scan_id,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            })
            
            # Add to scans list
            scans['scans'].append(scan_data)
            
            # Save updated data
            self._save_data(self.scans_file, scans)
            
            # Update statistics
            self._update_statistics()
            
            logger.info(f"Scan saved with ID: {scan_id}")
            return scan_id
            
        except Exception as e:
            logger.error(f"Error saving scan: {e}")
            return None
    
    def get_assessment(self, assessment_id: str) -> Optional[Dict]:
        """Get specific assessment by ID"""
        try:
            assessments = self._load_data(self.assessments_file)
            
            for assessment in assessments['assessments']:
                if assessment['id'] == assessment_id:
                    return assessment
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving assessment: {e}")
            return None
    
    def get_scan(self, scan_id: str) -> Optional[Dict]:
        """Get specific scan by ID"""
        try:
            scans = self._load_data(self.scans_file)
            
            for scan in scans['scans']:
                if scan['id'] == scan_id:
                    return scan
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving scan: {e}")
            return None
    
    def get_all_assessments(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all assessments with optional limit"""
        try:
            assessments = self._load_data(self.assessments_file)
            
            assessment_list = assessments['assessments']
            
            if limit:
                assessment_list = assessment_list[-limit:]
            
            return assessment_list
            
        except Exception as e:
            logger.error(f"Error retrieving assessments: {e}")
            return []
    
    def get_all_scans(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all scans with optional limit"""
        try:
            scans = self._load_data(self.scans_file)
            
            scan_list = scans['scans']
            
            if limit:
                scan_list = scan_list[-limit:]
            
            return scan_list
            
        except Exception as e:
            logger.error(f"Error retrieving scans: {e}")
            return []
    
    def get_assessments_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get assessments within date range"""
        try:
            assessments = self.get_all_assessments()
            
            filtered_assessments = []
            for assessment in assessments:
                assessment_date = assessment['timestamp'][:10]  # Extract date part
                if start_date <= assessment_date <= end_date:
                    filtered_assessments.append(assessment)
            
            return filtered_assessments
            
        except Exception as e:
            logger.error(f"Error filtering assessments by date: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get assessment and scan statistics"""
        try:
            stats = self._load_data(self.statistics_file)
            
            # Calculate additional statistics
            assessments = self.get_all_assessments()
            scans = self.get_all_scans()
            
            # BCS statistics
            bcs_scores = [a.get('body_condition_score') for a in assessments if a.get('body_condition_score') is not None]
            pain_scores = [a.get('pain_score') for a in assessments if a.get('pain_score') is not None]
            
            stats.update({
                'total_assessments': len(assessments),
                'total_scans': len(scans),
                'average_bcs': float(np.mean(bcs_scores)) if bcs_scores else 0.0,
                'average_pain': float(np.mean(pain_scores)) if pain_scores else 0.0,
                'bcs_distribution': self._calculate_distribution(bcs_scores),
                'pain_distribution': self._calculate_distribution(pain_scores),
                'last_updated': datetime.now().isoformat()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def generate_report(self, report_type: str = 'summary') -> Dict:
        """Generate assessment report"""
        try:
            stats = self.get_statistics()
            assessments = self.get_all_assessments()
            
            if report_type == 'summary':
                report = {
                    'report_type': 'summary',
                    'generated_at': datetime.now().isoformat(),
                    'total_assessments': stats['total_assessments'],
                    'total_scans': stats['total_scans'],
                    'average_bcs': stats['average_bcs'],
                    'average_pain': stats['average_pain'],
                    'bcs_distribution': stats['bcs_distribution'],
                    'pain_distribution': stats['pain_distribution'],
                    'recent_assessments': assessments[-10:] if assessments else []
                }
            
            elif report_type == 'detailed':
                report = {
                    'report_type': 'detailed',
                    'generated_at': datetime.now().isoformat(),
                    'statistics': stats,
                    'all_assessments': assessments,
                    'trends': self._calculate_trends(assessments),
                    'recommendations': self._generate_recommendations(stats)
                }
            
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Save report
            report_filename = f"data/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._save_data(report_filename, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
    
    def export_data(self, export_format: str = 'json', data_type: str = 'assessments') -> str:
        """Export data in specified format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if data_type == 'assessments':
                data = self.get_all_assessments()
                filename = f"data/exports/assessments_{timestamp}.{export_format}"
            elif data_type == 'scans':
                data = self.get_all_scans()
                filename = f"data/exports/scans_{timestamp}.{export_format}"
            elif data_type == 'statistics':
                data = self.get_statistics()
                filename = f"data/exports/statistics_{timestamp}.{export_format}"
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            if export_format == 'json':
                self._save_data(filename, data)
            elif export_format == 'csv':
                self._export_csv(data, filename)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None
    
    def _load_data(self, filename: str) -> Dict:
        """Load data from JSON file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            return {}
    
    def _save_data(self, filename: str, data: Dict):
        """Save data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {e}")
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID"""
        return f"assess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID"""
        return f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _update_statistics(self):
        """Update statistics file"""
        try:
            stats = self.get_statistics()
            self._save_data(self.statistics_file, stats)
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def _calculate_distribution(self, scores: List[float]) -> Dict:
        """Calculate score distribution"""
        if not scores:
            return {}
        
        # Define bins for distribution
        bins = {
            'low': 0,
            'medium': 0,
            'high': 0
        }
        
        for score in scores:
            if score < 3:
                bins['low'] += 1
            elif score < 7:
                bins['medium'] += 1
            else:
                bins['high'] += 1
        
        # Convert to percentages
        total = len(scores)
        for key in bins:
            bins[key] = round((bins[key] / total) * 100, 1)
        
        return bins
    
    def _calculate_trends(self, assessments: List[Dict]) -> Dict:
        """Calculate trends from assessment data"""
        try:
            if len(assessments) < 2:
                return {}
            
            # Sort by timestamp
            sorted_assessments = sorted(assessments, key=lambda x: x['timestamp'])
            
            # Extract BCS and pain scores
            bcs_scores = [a.get('body_condition_score') for a in sorted_assessments if a.get('body_condition_score') is not None]
            pain_scores = [a.get('pain_score') for a in sorted_assessments if a.get('pain_score') is not None]
            
            trends = {}
            
            # Calculate BCS trend
            if len(bcs_scores) >= 2:
                bcs_trend = np.polyfit(range(len(bcs_scores)), bcs_scores, 1)[0]
                trends['bcs_trend'] = round(bcs_trend, 3)
            
            # Calculate pain trend
            if len(pain_scores) >= 2:
                pain_trend = np.polyfit(range(len(pain_scores)), pain_scores, 1)[0]
                trends['pain_trend'] = round(pain_trend, 3)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {}
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on statistics"""
        recommendations = []
        
        # BCS recommendations
        avg_bcs = stats.get('average_bcs', 0)
        if avg_bcs < 4:
            recommendations.append("Average body condition score is low - consider nutritional assessment")
        elif avg_bcs > 6:
            recommendations.append("Average body condition score is high - consider dietary management")
        
        # Pain recommendations
        avg_pain = stats.get('average_pain', 0)
        if avg_pain > 5:
            recommendations.append("Average pain score is high - consider pain management strategies")
        
        # Assessment frequency recommendations
        total_assessments = stats.get('total_assessments', 0)
        if total_assessments < 10:
            recommendations.append("Consider increasing assessment frequency for better monitoring")
        
        return recommendations
    
    def _export_csv(self, data: List[Dict], filename: str):
        """Export data as CSV file"""
        try:
            import pandas as pd
            
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
            else:
                # For single dictionary or empty data
                df = pd.DataFrame([data] if data else [])
                df.to_csv(filename, index=False)
                
        except ImportError:
            logger.error("pandas not available for CSV export")
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old assessment and scan data"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            # Clean up assessments
            assessments = self._load_data(self.assessments_file)
            original_count = len(assessments['assessments'])
            
            assessments['assessments'] = [
                a for a in assessments['assessments']
                if datetime.fromisoformat(a['timestamp']).timestamp() > cutoff_date
            ]
            
            removed_assessments = original_count - len(assessments['assessments'])
            self._save_data(self.assessments_file, assessments)
            
            # Clean up scans
            scans = self._load_data(self.scans_file)
            original_scan_count = len(scans['scans'])
            
            scans['scans'] = [
                s for s in scans['scans']
                if datetime.fromisoformat(s['timestamp']).timestamp() > cutoff_date
            ]
            
            removed_scans = original_scan_count - len(scans['scans'])
            self._save_data(self.scans_file, scans)
            
            logger.info(f"Cleaned up {removed_assessments} assessments and {removed_scans} scans")
            
            return {
                'removed_assessments': removed_assessments,
                'removed_scans': removed_scans
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}

