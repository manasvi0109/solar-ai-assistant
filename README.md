# ☀️ Solar Industry AI Assistant

<div align="center">

![Solar AI Assistant](https://img.shields.io/badge/Solar-AI%20Assistant-brightgreen?style=for-the-badge&logo=solar-power)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv)

**Professional Solar Installation Analysis & ROI Calculator**

*An AI-powered tool that analyzes satellite rooftop images to assess solar panel installation potential, providing comprehensive financial analysis and industry-compliant recommendations.*

</div>

---

## 🎯 **Project Overview**

This comprehensive Solar Industry AI Assistant combines **computer vision**, **artificial intelligence**, and **solar industry expertise** to provide professional-grade rooftop analysis for solar panel installations. Built for the Solar Industry AI Internship Assessment, it demonstrates deep integration of AI technologies with real-world solar industry knowledge.

### **Key Capabilities**
- 🖼️ **Advanced Image Analysis**: OpenCV + AI-powered rooftop detection
- 🔍 **Structured AI Analysis**: Confidence-scored obstacle detection and roof assessment
- 💰 **Comprehensive Financial Modeling**: NPV, IRR, and 25-year ROI projections
- 🏗️ **Industry Expertise**: Real solar panel specifications and installation costs
- 📋 **Regulatory Compliance**: Indian solar regulations, permits, and incentives
- 🔧 **Professional UI**: Interactive configuration and detailed reporting

---

## 🏗️ **Architecture & Tech Stack**

### **Core Technologies**
- **Frontend**: Streamlit (Professional UI with interactive controls)
- **Computer Vision**: OpenCV (Image processing and rooftop detection)
- **AI Integration**: OpenRouter API with Gemini Pro Vision
- **Backend**: Python 3.8+ (Modular architecture)
- **Data Processing**: Pandas, NumPy (Financial calculations)

### **AI & Machine Learning**
- **Vision AI**: Structured output extraction with confidence scoring
- **Prompt Engineering**: Advanced prompts for consistent JSON responses
- **Fallback Processing**: Robust parsing when AI responses vary
- **Multi-source Analysis**: OpenCV + AI combined insights

---

## 📂 **Project Structure**

```
solar-ai-assistant/
├── 📱 app.py                 # Main Streamlit application
├── 🤖 ai_module.py           # AI analysis with structured output
├── 🖼️ opencv_module.py       # Image processing with OpenCV
├── ⚙️ utils.py               # Solar industry calculations & data
├── 📋 requirements.txt       # Python dependencies
├── 📖 README.md              # This documentation
├── 📁 assets/                # Sample images and resources
└── 🐍 venv/                  # Virtual environment
```

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Internet connection (for AI analysis)

### **Installation**

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd solar-ai-assistant
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

---

## 🎮 **How to Use**

### **Step 1: Configure System Parameters**
Use the **sidebar controls** to customize your analysis:
- **Panel Type**: Choose from Monocrystalline, Polycrystalline, or Thin-film
- **Region**: Select your location in India for accurate solar irradiance
- **System Components**: Configure inverter and mounting types
- **Electricity Rate**: Set your current electricity cost (₹/kWh)

### **Step 2: Upload Rooftop Image**
- Click **"Browse files"** to upload a satellite/aerial rooftop image
- Supported formats: JPG, JPEG, PNG
- Maximum file size: 10MB

### **Step 3: Choose Analysis Method**
- **🔍 OpenCV Analysis**: Fast computer vision-based detection
- **🤖 AI Analysis (Enhanced)**: Comprehensive AI-powered assessment with confidence scoring

### **Step 4: Review Results**
The application provides detailed analysis including:
- **System Overview**: Panel count, capacity, and specifications
- **Financial Analysis**: Cost breakdown, ROI, and savings projections
- **Maintenance Schedule**: Required maintenance tasks and costs
- **Regulatory Information**: Permits, incentives, and compliance requirements

---

## 🧪 **Features & Capabilities**

### **🖼️ Image Analysis**
- **OpenCV Processing**: Edge detection, thresholding, and contour analysis
- **AI-Powered Assessment**: Structured analysis with confidence scoring
- **Obstacle Detection**: Identification of chimneys, vents, AC units, water tanks
- **Roof Characterization**: Type, material, condition, and orientation analysis

### **☀️ Solar System Design**
- **Multiple Panel Types**:
  - Monocrystalline (22% efficiency, ₹45/watt)
  - Polycrystalline (18% efficiency, ₹35/watt)
  - Thin-film (12% efficiency, ₹25/watt)
- **System Components**: String inverters, power optimizers, microinverters
- **Mounting Options**: Roof-mounted, ground-mounted, tracking systems

### **💰 Financial Analysis**
- **Comprehensive Cost Breakdown**: Panels, inverters, mounting, electrical, labor
- **Advanced ROI Calculations**: NPV, IRR, payback period
- **25-Year Projections**: Savings with degradation and rate escalation
- **Regional Factors**: India-specific solar irradiance and efficiency factors

### **📋 Industry Compliance**
- **Required Permits**: Building approvals, electrical clearances, grid connections
- **Building Codes**: IS 16221, CEA standards, state guidelines
- **Safety Standards**: Earthing, lightning protection, fire safety
- **Incentives**: Central subsidies, accelerated depreciation, net metering

---

## � **Technical Specifications**

### **Solar Panel Database**
| Panel Type | Efficiency | Cost/Watt | Area/Panel | Lifespan | Degradation |
|------------|------------|-----------|------------|----------|-------------|
| Monocrystalline | 22% | ₹45 | 1.7 m² | 25 years | 0.5%/year |
| Polycrystalline | 18% | ₹35 | 1.8 m² | 25 years | 0.7%/year |
| Thin-film | 12% | ₹25 | 2.5 m² | 20 years | 0.8%/year |

### **Regional Solar Data (India)**
| Region | Peak Sun Hours | Temperature Factor | Dust Factor |
|--------|----------------|-------------------|-------------|
| North | 4.5 kWh/m²/day | 0.82 | 0.93 |
| South | 5.5 kWh/m²/day | 0.85 | 0.95 |
| East | 4.8 kWh/m²/day | 0.84 | 0.92 |
| West | 5.2 kWh/m²/day | 0.83 | 0.90 |
| Central | 5.0 kWh/m²/day | 0.84 | 0.92 |

---

## 🔧 **Configuration**

### **API Configuration**
The application uses OpenRouter API for enhanced AI analysis. The API key is pre-configured, but you can update it in `ai_module.py`:

```python
OPENROUTER_API_KEY = "your-api-key-here"
```

### **Customization Options**
- **Panel Specifications**: Modify panel data in `utils.py`
- **Regional Factors**: Update solar irradiance data for different locations
- **Cost Parameters**: Adjust installation costs and component pricing
- **UI Settings**: Customize the Streamlit interface in `app.py`

---

## 📈 **Sample Analysis Results**

### **Example: 100m² Rooftop Analysis**
- **Usable Area**: 85 m² (after obstacles)
- **Panel Count**: 42 panels (Monocrystalline)
- **System Capacity**: 16.8 kW
- **Installation Cost**: ₹12,50,000
- **25-Year Savings**: ₹28,75,000
- **Payback Period**: 6.2 years
- **NPV**: ₹15,25,000
- **IRR**: 14.8%

---

## 🔬 **Testing & Validation**

The application includes comprehensive validation:
- **Input Validation**: File size, format, and parameter ranges
- **Data Validation**: Confidence scores, numerical bounds, and consistency checks
- **Error Handling**: Graceful fallbacks for API failures and processing errors
- **Edge Cases**: Zero area, extreme values, and malformed responses

---

## 🚀 **Deployment Options**

### **Local Development**
```bash
streamlit run app.py
```

### **Streamlit Cloud**
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## 🔮 **Future Enhancements**

### **Planned Features**
- 🗺️ **3D Visualization**: Interactive 3D rooftop models
- 📊 **Advanced Analytics**: Machine learning for better predictions
- 📱 **Mobile App**: React Native mobile application
- 🔗 **API Integration**: RESTful API for third-party integrations
- 📄 **Report Generation**: PDF reports with detailed analysis
- 🔄 **Real-time Updates**: Live solar irradiance and pricing data

### **Technical Improvements**
- ⚡ **Performance Optimization**: Faster image processing
- 🧠 **Enhanced AI**: Custom-trained models for rooftop detection
- 🔐 **Security**: Enhanced API key management and user authentication
- 📈 **Scalability**: Multi-user support and cloud deployment

---

## 🤝 **Contributing**

This project was developed for the Solar Industry AI Internship Assessment. While primarily for evaluation purposes, suggestions and improvements are welcome.

### **Development Guidelines**
- Follow PEP 8 Python style guidelines
- Add comprehensive docstrings to functions
- Include unit tests for new features
- Update documentation for any changes

---

## 📄 **License**

This project is developed for educational and assessment purposes. It demonstrates the integration of AI technologies with solar industry expertise for professional rooftop analysis applications.

---

## 🙏 **Acknowledgments**

- **Solar Industry Data**: Based on current Indian solar market conditions
- **AI Integration**: Powered by OpenRouter and Gemini Pro Vision
- **Computer Vision**: Built with OpenCV for robust image processing
- **UI Framework**: Streamlit for rapid prototyping and deployment

---

<div align="center">

**Built with ❤️ for the Solar Industry AI Internship Assessment**

*Demonstrating the future of AI-powered solar installation analysis*

</div>