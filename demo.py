#!/usr/bin/env python3
"""
Demo script for 3D Body Condition Scoring & Computer Vision Pain Assessment
This script demonstrates the capabilities of the system with sample data.
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import time

# Import our custom modules
from computer_vision.body_condition_scorer import AdvancedBodyConditionScorer
from computer_vision.pain_assessment import AdvancedPainAssessment
from computer_vision.three_d_scanner import ThreeDScanner
from data_manager import AssessmentDataManager

def create_sample_video():
    """Create a sample video for demonstration"""
    print("Creating sample video...")
    
    # Create a simple video with moving shapes to simulate an animal
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_animal.mp4', fourcc, 10.0, (640, 480))
    
    for i in range(50):  # 5 seconds at 10 FPS
        # Create a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(255)  # White background
        
        # Draw a simple animal-like shape
        center_x = 320 + int(50 * np.sin(i * 0.2))
        center_y = 240 + int(30 * np.cos(i * 0.15))
        
        # Body (ellipse)
        cv2.ellipse(frame, (center_x, center_y), (80, 60), 0, 0, 360, (100, 150, 200), -1)
        
        # Head (circle)
        cv2.circle(frame, (center_x, center_y - 40), 30, (120, 180, 220), -1)
        
        # Eyes
        cv2.circle(frame, (center_x - 15, center_y - 45), 5, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + 15, center_y - 45), 5, (0, 0, 0), -1)
        
        # Legs
        cv2.rectangle(frame, (center_x - 60, center_y + 20), (center_x - 50, center_y + 80), (100, 150, 200), -1)
        cv2.rectangle(frame, (center_x - 20, center_y + 20), (center_x - 10, center_y + 80), (100, 150, 200), -1)
        cv2.rectangle(frame, (center_x + 10, center_y + 20), (center_x + 20, center_y + 80), (100, 150, 200), -1)
        cv2.rectangle(frame, (center_x + 50, center_y + 20), (center_x + 60, center_y + 80), (100, 150, 200), -1)
        
        out.write(frame)
    
    out.release()
    print("Sample video created: sample_animal.mp4")

def demo_body_condition_scoring():
    """Demonstrate body condition scoring"""
    print("\n" + "="*50)
    print("DEMO: Body Condition Scoring")
    print("="*50)
    
    # Initialize BCS scorer
    bcs_scorer = AdvancedBodyConditionScorer()
    
    # Load sample video
    cap = cv2.VideoCapture('sample_animal.mp4')
    
    frame_count = 0
    bcs_scores = []
    
    while cap.isOpened() and frame_count < 10:  # Analyze first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\nAnalyzing frame {frame_count + 1}...")
        
        # Analyze body condition
        result = bcs_scorer.analyze_body_condition(frame)
        
        if result['bcs_score'] is not None:
            bcs_scores.append(result['bcs_score'])
            print(f"  BCS Score: {result['bcs_score']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Analysis Methods: {result['analysis_methods']}")
            
            if 'individual_scores' in result:
                print("  Individual Scores:")
                for method, score in result['individual_scores'].items():
                    if score is not None:
                        print(f"    {method}: {score}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        frame_count += 1
    
    cap.release()
    
    if bcs_scores:
        avg_bcs = np.mean(bcs_scores)
        print(f"\nAverage BCS Score: {avg_bcs:.1f}")
        print(f"BCS Range: {min(bcs_scores):.1f} - {max(bcs_scores):.1f}")
        
        # Interpret BCS score
        if avg_bcs < 4:
            interpretation = "Underweight - nutritional assessment recommended"
        elif avg_bcs > 6:
            interpretation = "Overweight - dietary management recommended"
        else:
            interpretation = "Ideal body condition"
        
        print(f"Interpretation: {interpretation}")
    
    return bcs_scores

def demo_pain_assessment():
    """Demonstrate pain assessment"""
    print("\n" + "="*50)
    print("DEMO: Pain Assessment")
    print("="*50)
    
    # Initialize pain assessor
    pain_assessor = AdvancedPainAssessment()
    
    # Load sample video
    cap = cv2.VideoCapture('sample_animal.mp4')
    
    frame_count = 0
    pain_scores = []
    
    while cap.isOpened() and frame_count < 10:  # Analyze first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\nAnalyzing frame {frame_count + 1}...")
        
        # Assess pain level
        result = pain_assessor.assess_pain_level(frame)
        
        if result['pain_score'] is not None:
            pain_scores.append(result['pain_score'])
            print(f"  Pain Score: {result['pain_score']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Pain Level: {result['pain_level']}")
            print(f"  Analysis Methods: {result['analysis_methods']}")
            
            if 'individual_scores' in result:
                print("  Individual Scores:")
                for method, score in result['individual_scores'].items():
                    if score is not None:
                        print(f"    {method}: {score}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        frame_count += 1
    
    cap.release()
    
    if pain_scores:
        avg_pain = np.mean(pain_scores)
        print(f"\nAverage Pain Score: {avg_pain:.1f}")
        print(f"Pain Range: {min(pain_scores):.1f} - {max(pain_scores):.1f}")
        
        # Interpret pain score
        if avg_pain >= 8:
            interpretation = "Severe pain - immediate veterinary attention required"
        elif avg_pain >= 6:
            interpretation = "Moderate-severe pain - pain management recommended"
        elif avg_pain >= 4:
            interpretation = "Moderate pain - monitor closely"
        elif avg_pain >= 2:
            interpretation = "Mild pain - continue monitoring"
        else:
            interpretation = "No significant pain detected"
        
        print(f"Interpretation: {interpretation}")
    
    return pain_scores

def demo_3d_scanning():
    """Demonstrate 3D scanning capabilities"""
    print("\n" + "="*50)
    print("DEMO: 3D Scanning")
    print("="*50)
    
    # Initialize 3D scanner
    scanner = ThreeDScanner()
    
    # Load sample video
    cap = cv2.VideoCapture('sample_animal.mp4')
    
    print("Starting 3D scan...")
    scan_results = scanner.start_3d_scan(cap)
    
    print(f"Scan Status: {scan_results['status']}")
    print(f"Frames Processed: {scan_results['frames_processed']}")
    print(f"Quality Score: {scan_results['quality_score']:.2f}")
    
    if scan_results['status'] == 'completed':
        measurements = scan_results['measurements']
        print("\n3D Measurements:")
        for measurement, value in measurements.items():
            print(f"  {measurement}: {value:.3f}")
        
        # Export scan data
        export_path = scanner.export_3d_data(scan_results, 'json')
        if export_path:
            print(f"\nScan data exported to: {export_path}")
    
    # Get scan statistics
    stats = scanner.get_scan_statistics()
    print(f"\nScan Statistics:")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")
    
    return scan_results

def demo_data_management():
    """Demonstrate data management capabilities"""
    print("\n" + "="*50)
    print("DEMO: Data Management")
    print("="*50)
    
    # Initialize data manager
    data_manager = AssessmentDataManager()
    
    # Create sample assessment data
    sample_assessment = {
        'body_condition_score': 5.2,
        'pain_score': 2.1,
        'bcs_confidence': 0.85,
        'pain_confidence': 0.80,
        'analysis_methods': 3,
        'individual_scores': {
            'neural_network': 5.1,
            'geometric': 5.3,
            'contour': 5.2
        }
    }
    
    # Save assessment
    assessment_id = data_manager.save_assessment(sample_assessment)
    print(f"Assessment saved with ID: {assessment_id}")
    
    # Retrieve assessment
    retrieved_assessment = data_manager.get_assessment(assessment_id)
    if retrieved_assessment:
        print("Assessment retrieved successfully:")
        print(f"  BCS Score: {retrieved_assessment['body_condition_score']}")
        print(f"  Pain Score: {retrieved_assessment['pain_score']}")
        print(f"  Timestamp: {retrieved_assessment['timestamp']}")
    
    # Get statistics
    stats = data_manager.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Total Assessments: {stats['total_assessments']}")
    print(f"  Total Scans: {stats['total_scans']}")
    print(f"  Average BCS: {stats['average_bcs']:.1f}")
    print(f"  Average Pain: {stats['average_pain']:.1f}")
    
    # Generate report
    report = data_manager.generate_report('summary')
    print(f"\nReport generated:")
    print(f"  Report Type: {report['report_type']}")
    print(f"  Generated At: {report['generated_at']}")
    print(f"  Total Assessments: {report['total_assessments']}")
    
    return assessment_id

def demo_export_capabilities():
    """Demonstrate data export capabilities"""
    print("\n" + "="*50)
    print("DEMO: Export Capabilities")
    print("="*50)
    
    data_manager = AssessmentDataManager()
    
    # Export assessments
    assessments_file = data_manager.export_data('json', 'assessments')
    if assessments_file:
        print(f"Assessments exported to: {assessments_file}")
    
    # Export statistics
    stats_file = data_manager.export_data('json', 'statistics')
    if stats_file:
        print(f"Statistics exported to: {stats_file}")
    
    # Generate detailed report
    detailed_report = data_manager.generate_report('detailed')
    print(f"Detailed report generated with {len(detailed_report.get('all_assessments', []))} assessments")

def cleanup_demo_files():
    """Clean up demo files"""
    demo_files = ['sample_animal.mp4']
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

def main():
    """Main demo function"""
    print("3D Body Condition Scoring & Computer Vision Pain Assessment")
    print("DEMONSTRATION SCRIPT")
    print("="*70)
    
    try:
        # Create sample video
        create_sample_video()
        
        # Run demonstrations
        bcs_scores = demo_body_condition_scoring()
        pain_scores = demo_pain_assessment()
        scan_results = demo_3d_scanning()
        assessment_id = demo_data_management()
        demo_export_capabilities()
        
        # Summary
        print("\n" + "="*70)
        print("DEMO SUMMARY")
        print("="*70)
        print(f"✓ Body Condition Scoring: {len(bcs_scores)} frames analyzed")
        print(f"✓ Pain Assessment: {len(pain_scores)} frames analyzed")
        print(f"✓ 3D Scanning: {scan_results['frames_processed']} frames processed")
        print(f"✓ Data Management: Assessment {assessment_id} saved")
        print(f"✓ Export Capabilities: Data exported successfully")
        
        print("\nAll demonstrations completed successfully!")
        print("\nTo run the full application:")
        print("  python app.py")
        print("  Then visit: http://localhost:5000")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        return False
    
    finally:
        # Clean up
        cleanup_demo_files()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

