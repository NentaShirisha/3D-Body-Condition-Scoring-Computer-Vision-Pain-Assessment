# 3D Body Condition Scoring & Computer Vision Pain Assessment

A comprehensive system for automated animal health assessment using computer vision and 3D scanning technology. This system replaces manual Body Condition Score (BCS) checks and subjective pain scale scoring with automated computer vision analysis.

## 🚀 Features

### Core Capabilities
- **Automated Body Condition Scoring (BCS)**: Uses multiple computer vision methods to calculate BCS (1-9 scale)
- **Pain Assessment**: Implements Glasgow Pain Scale using facial expression analysis
- **3D Body Scanning**: Mobile device integration for 3D body reconstruction
- **Real-time Analysis**: Video processing for immediate assessment results
- **Objective Data Logging**: Comprehensive data storage and tracking system

### Advanced Computer Vision
- **Multi-method BCS Analysis**: Neural networks, geometric analysis, and contour analysis
- **Facial Expression Recognition**: Eye closure, mouth tension, ear position, brow furrowing
- **3D Pose Estimation**: MediaPipe integration for accurate landmark detection
- **Depth Estimation**: Monocular depth estimation for 3D reconstruction

### Mobile & Web Interface
- **Progressive Web App (PWA)**: Installable mobile app with offline support
- **Real-time Camera**: Mobile camera integration for video capture
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Offline Capability**: Service worker for offline functionality

### Data Management
- **Comprehensive Logging**: All assessments stored with metadata
- **Statistical Analysis**: Trends, distributions, and recommendations
- **Export Capabilities**: JSON and CSV export formats
- **Report Generation**: Summary and detailed reports

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, MediaPipe, TensorFlow
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **3D Processing**: MediaPipe 3D pose estimation, depth estimation
- **Web Framework**: Flask, Bootstrap 5
- **Mobile**: Progressive Web App (PWA)
- **Data Storage**: JSON-based with comprehensive management

## 📦 Installation

### Quick Setup
```bash
# Clone or download the project
cd 3d-body-condition-scoring

# Run setup script
python setup.py

# Start the application
python app.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/{assessments,scans,models,exports,reports}

# Start the application
python app.py
```

## 🚀 Usage

### Web Interface
1. Open browser to `http://localhost:5000`
2. Upload video file or use mobile interface
3. View automated BCS and pain assessment results
4. Access assessment history and statistics

### Mobile App
1. Visit `http://localhost:5000/mobile`
2. Allow camera permissions
3. Record video of animal
4. Get real-time analysis results
5. Install as PWA for offline use

### API Endpoints
- `POST /upload` - Upload video for analysis
- `GET /assessments` - Get all assessments
- `GET /assessments/<id>` - Get specific assessment
- `POST /scan` - Start 3D scanning
- `GET /statistics` - Get system statistics
- `GET /export/<type>` - Export data
- `GET /report/<type>` - Generate reports

## 🧪 Demonstration

Run the demo script to see all capabilities:
```bash
python demo.py
```

This will:
- Create sample video data
- Demonstrate BCS scoring
- Show pain assessment
- Perform 3D scanning
- Test data management
- Export capabilities

## 📊 Assessment Methods

### Body Condition Scoring (BCS)
1. **Neural Network**: Trained model using pose landmarks
2. **Geometric Analysis**: Body measurements and ratios
3. **Contour Analysis**: Body shape and compactness

### Pain Assessment
1. **Glasgow Pain Scale**: Eye closure, mouth tension, ear position
2. **Facial Action Units**: Comprehensive facial expression analysis
3. **Neural Network**: Deep learning pain recognition

### 3D Scanning
1. **Pose Landmark Collection**: Multi-frame landmark tracking
2. **Depth Estimation**: Monocular depth estimation
3. **Mesh Generation**: 3D reconstruction using triangulation

## 📈 Data Output

### Assessment Results
```json
{
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
  "recommendations": [
    "Body condition appears normal",
    "No significant pain indicators detected"
  ]
}
```

### 3D Scan Results
```json
{
  "status": "completed",
  "frames_processed": 50,
  "quality_score": 0.78,
  "measurements": {
    "body_length": 0.45,
    "body_width": 0.25,
    "body_depth": 0.15,
    "estimated_volume": 0.017,
    "estimated_surface_area": 0.34
  }
}
```

## 🔧 Configuration

Edit `config.json` to customize:
- Detection confidence thresholds
- Model paths
- Data retention policies
- Mobile app settings

## 📱 Mobile App Features

- **Camera Integration**: Real-time video capture
- **Offline Support**: Works without internet connection
- **PWA Installation**: Install as native app
- **Push Notifications**: Assessment alerts
- **Background Sync**: Sync data when online

## 🎯 Use Cases

- **Veterinary Clinics**: Automated health assessments
- **Animal Shelters**: Regular health monitoring
- **Research**: Objective health data collection
- **Pet Owners**: Home health monitoring
- **Livestock Management**: Farm animal health tracking

## 🔬 Technical Details

### Computer Vision Pipeline
1. **Input**: Video frames or images
2. **Preprocessing**: Color space conversion, normalization
3. **Detection**: Pose and facial landmark detection
4. **Analysis**: Multi-method scoring algorithms
5. **Post-processing**: Confidence weighting, result aggregation

### 3D Reconstruction
1. **Landmark Tracking**: Multi-frame pose consistency
2. **Depth Estimation**: Gradient-based depth mapping
3. **Mesh Generation**: Delaunay triangulation
4. **Measurement**: 3D geometric calculations

## 📋 Requirements

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- TensorFlow 2.13+
- Flask 2.3+
- Modern web browser with camera support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the demo script: `python demo.py`
2. Review the API documentation
3. Check the logs in the console
4. Create an issue with detailed information

## 🔮 Future Enhancements

- **Species-specific Models**: Tailored models for different animal types
- **Cloud Integration**: Cloud-based processing and storage
- **Machine Learning Training**: Continuous model improvement
- **Integration APIs**: Third-party system integration
- **Advanced Analytics**: Predictive health modeling
