#!/usr/bin/env python3
"""
Setup script for 3D Body Condition Scoring & Computer Vision Pain Assessment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'data/assessments',
        'data/scans',
        'data/models',
        'data/exports',
        'data/reports',
        'static/icons',
        'static/screenshots',
        'computer_vision',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_dependencies():
    """Install required Python packages"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def create_sample_data():
    """Create sample data files"""
    # Sample assessment data
    sample_assessment = {
        "assessments": [
            {
                "id": "assess_20240101_120000",
                "timestamp": "2024-01-01T12:00:00",
                "body_condition_score": 5.2,
                "pain_score": 2.1,
                "bcs_confidence": 0.85,
                "pain_confidence": 0.80,
                "analysis_methods": 3,
                "individual_scores": {
                    "neural_network": 5.1,
                    "geometric": 5.3,
                    "contour": 5.2
                },
                "version": "1.0"
            }
        ]
    }
    
    with open('data/assessments.json', 'w') as f:
        json.dump(sample_assessment, f, indent=2)
    
    # Sample scan data
    sample_scan = {
        "scans": [
            {
                "id": "scan_20240101_120000",
                "timestamp": "2024-01-01T12:00:00",
                "status": "completed",
                "frames_processed": 50,
                "quality_score": 0.78,
                "measurements": {
                    "body_length": 0.45,
                    "body_width": 0.25,
                    "body_depth": 0.15,
                    "estimated_volume": 0.017,
                    "estimated_surface_area": 0.34
                },
                "version": "1.0"
            }
        ]
    }
    
    with open('data/scans.json', 'w') as f:
        json.dump(sample_scan, f, indent=2)
    
    # Sample statistics
    sample_stats = {
        "total_assessments": 1,
        "total_scans": 1,
        "average_bcs": 5.2,
        "average_pain": 2.1,
        "bcs_distribution": {
            "low": 0.0,
            "medium": 100.0,
            "high": 0.0
        },
        "pain_distribution": {
            "low": 100.0,
            "medium": 0.0,
            "high": 0.0
        },
        "last_updated": "2024-01-01T12:00:00"
    }
    
    with open('data/statistics.json', 'w') as f:
        json.dump(sample_stats, f, indent=2)
    
    print("Sample data files created")

def create_config_file():
    """Create configuration file"""
    config = {
        "app": {
            "name": "Animal Health Assessment",
            "version": "1.0.0",
            "debug": True,
            "host": "0.0.0.0",
            "port": 5000
        },
        "computer_vision": {
            "pose_detection_confidence": 0.7,
            "face_detection_confidence": 0.7,
            "bcs_model_path": "models/bcs_model_weights.h5",
            "pain_model_path": "models/pain_model_weights.h5"
        },
        "data": {
            "max_assessments": 1000,
            "max_scans": 100,
            "cleanup_days": 30,
            "export_formats": ["json", "csv"]
        },
        "mobile": {
            "pwa_enabled": True,
            "offline_support": True,
            "camera_resolution": "1280x720",
            "max_recording_duration": 30
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Configuration file created")

def create_startup_script():
    """Create startup script"""
    startup_script = """#!/bin/bash
# Startup script for Animal Health Assessment

echo "Starting Animal Health Assessment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the application
echo "Starting Flask application..."
python app.py
"""
    
    with open('start.sh', 'w') as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod('start.sh', 0o755)
    print("Startup script created")

def create_windows_startup():
    """Create Windows startup script"""
    windows_script = """@echo off
echo Starting Animal Health Assessment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is required but not installed.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Start the application
echo Starting Flask application...
python app.py

pause
"""
    
    with open('start.bat', 'w') as f:
        f.write(windows_script)
    
    print("Windows startup script created")

def main():
    """Main setup function"""
    print("Setting up 3D Body Condition Scoring & Computer Vision Pain Assessment")
    print("=" * 70)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    if not install_dependencies():
        print("Failed to install dependencies. Please check your Python environment.")
        return False
    
    # Create sample data
    print("\n3. Creating sample data...")
    create_sample_data()
    
    # Create configuration
    print("\n4. Creating configuration...")
    create_config_file()
    
    # Create startup scripts
    print("\n5. Creating startup scripts...")
    create_startup_script()
    create_windows_startup()
    
    print("\n" + "=" * 70)
    print("Setup completed successfully!")
    print("\nTo start the application:")
    print("  Linux/Mac: ./start.sh")
    print("  Windows: start.bat")
    print("  Or manually: python app.py")
    print("\nAccess the application at: http://localhost:5000")
    print("Mobile app: http://localhost:5000/mobile")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

