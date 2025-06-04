"""
Hugging Face Spaces Deployment Version
Simplified version for deployment on Hugging Face Spaces with reduced dependencies
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import requests
import base64
import json
import re
from typing import Dict, Any

# Configuration for Hugging Face Spaces
st.set_page_config(
    page_title="Solar Industry AI Assistant",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
OPENROUTER_API_KEY = "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"

def encode_image(image_file):
    """Encode image file to base64."""
    return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_ai(image_file):
    """Analyze image using OpenRouter API."""
    base64_image = encode_image(image_file)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-pro-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Analyze this rooftop image for solar panel installation. Provide: "
                        "1. Usable area in square meters "
                        "2. Number of obstacles (chimneys, vents, AC units) "
                        "3. Roof condition and type "
                        "4. Estimated number of solar panels that can fit "
                        "5. Overall suitability rating (excellent/good/fair/poor)"
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                               json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Analysis failed: {str(e)}"

def process_image_opencv(image_file):
    """Process image using OpenCV for rooftop detection."""
    try:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 5000
        max_area = 50000
        usable_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        # Draw contours on original image
        result_img = img.copy()
        cv2.drawContours(result_img, usable_contours, -1, (0, 255, 0), 2)
        
        # Calculate total usable area in pixels
        total_usable_pixels = sum(cv2.contourArea(c) for c in usable_contours)
        
        return result_img, total_usable_pixels
        
    except Exception as e:
        st.error(f"OpenCV processing failed: {str(e)}")
        return None, 0

def calculate_solar_metrics(usable_area_m2, panel_type="monocrystalline"):
    """Calculate solar installation metrics."""
    
    # Panel specifications
    panel_specs = {
        "monocrystalline": {"efficiency": 0.22, "cost_per_watt": 45, "area": 1.7, "wattage": 400},
        "polycrystalline": {"efficiency": 0.18, "cost_per_watt": 35, "area": 1.8, "wattage": 350},
        "thin_film": {"efficiency": 0.12, "cost_per_watt": 25, "area": 2.5, "wattage": 300}
    }
    
    specs = panel_specs[panel_type]
    
    # Calculate panel count
    panel_count = int(usable_area_m2 / (specs["area"] * 1.2))  # 20% spacing factor
    
    # Calculate system capacity
    system_capacity_kw = panel_count * specs["wattage"] / 1000
    
    # Calculate costs
    panel_cost = system_capacity_kw * 1000 * specs["cost_per_watt"]
    inverter_cost = system_capacity_kw * 8000  # ₹8000 per kW
    installation_cost = panel_cost * 0.3  # 30% of panel cost
    total_cost = panel_cost + inverter_cost + installation_cost
    
    # Calculate savings (25 years)
    annual_generation = system_capacity_kw * 1500  # kWh per year
    electricity_rate = 8.5  # ₹ per kWh
    annual_savings = annual_generation * electricity_rate
    total_savings = annual_savings * 25 * 0.9  # 10% degradation factor
    
    # Calculate payback period
    payback_years = total_cost / annual_savings if annual_savings > 0 else 0
    
    return {
        "panel_count": panel_count,
        "system_capacity": system_capacity_kw,
        "total_cost": total_cost,
        "annual_savings": annual_savings,
        "total_savings": total_savings,
        "payback_years": payback_years,
        "annual_generation": annual_generation
    }

# Main App
st.title("☀️ Solar Industry AI Assistant")
st.markdown("### Professional Solar Installation Analysis & ROI Calculator")
st.markdown("*Powered by AI and Computer Vision for Hugging Face Spaces*")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

panel_type = st.sidebar.selectbox(
    "Panel Type:",
    ["monocrystalline", "polycrystalline", "thin_film"],
    format_func=lambda x: x.title()
)

analysis_method = st.sidebar.radio(
    "Analysis Method:",
    ["AI Analysis", "OpenCV Analysis", "Both"]
)

# File upload
uploaded_file = st.file_uploader(
    "Upload a rooftop image", 
    type=["jpg", "png", "jpeg"],
    help="Upload a satellite or aerial view of the rooftop"
)

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Rooftop Image", use_container_width=True)
    
    # Analysis buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze Rooftop", use_container_width=True):
            
            if analysis_method in ["OpenCV Analysis", "Both"]:
                st.markdown("## 🖼️ OpenCV Analysis")
                with st.spinner("Processing with OpenCV..."):
                    uploaded_file.seek(0)
                    result_img, usable_pixels = process_image_opencv(uploaded_file)
                    
                    if result_img is not None:
                        # Convert pixels to area (calibration factor)
                        usable_area = usable_pixels * 0.05  # 1 pixel ≈ 0.05 m²
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.image(result_img, caption="Detected Rooftop Areas", channels="BGR")
                        with col2:
                            st.metric("Usable Area", f"{usable_area:.1f} m²")
                            st.metric("Detected Pixels", f"{usable_pixels:,}")
                        
                        # Calculate solar metrics
                        metrics = calculate_solar_metrics(usable_area, panel_type)
                        
                        # Display results
                        st.markdown("### 📊 Solar Installation Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Panel Count", metrics["panel_count"])
                        with col2:
                            st.metric("System Capacity", f"{metrics['system_capacity']:.1f} kW")
                        with col3:
                            st.metric("Total Investment", f"₹{metrics['total_cost']:,.0f}")
                        with col4:
                            st.metric("Payback Period", f"{metrics['payback_years']:.1f} years")
                        
                        # Financial breakdown
                        with st.expander("💰 Financial Details"):
                            st.write(f"**Annual Generation:** {metrics['annual_generation']:,.0f} kWh")
                            st.write(f"**Annual Savings:** ₹{metrics['annual_savings']:,.0f}")
                            st.write(f"**25-Year Savings:** ₹{metrics['total_savings']:,.0f}")
                            st.write(f"**ROI:** {(metrics['total_savings'] / metrics['total_cost'] - 1) * 100:.1f}%")
            
            if analysis_method in ["AI Analysis", "Both"]:
                st.markdown("## 🤖 AI Analysis")
                with st.spinner("Analyzing with AI..."):
                    uploaded_file.seek(0)
                    ai_result = analyze_image_with_ai(uploaded_file)
                    
                    st.markdown("### 🧠 AI Assessment")
                    st.write(ai_result)
                    
                    # Try to extract area from AI response
                    area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:square\s*)?(?:meters?|m²)', ai_result.lower())
                    if area_match:
                        ai_area = float(area_match.group(1))
                        st.success(f"AI detected usable area: {ai_area} m²")
                        
                        # Calculate metrics based on AI area
                        ai_metrics = calculate_solar_metrics(ai_area, panel_type)
                        
                        st.markdown("### 📈 AI-Based Calculations")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("AI Panel Count", ai_metrics["panel_count"])
                        with col2:
                            st.metric("AI System Capacity", f"{ai_metrics['system_capacity']:.1f} kW")
                        with col3:
                            st.metric("AI Investment", f"₹{ai_metrics['total_cost']:,.0f}")

    with col2:
        if st.button("📋 Generate Report", use_container_width=True):
            st.markdown("## 📄 Solar Installation Report")
            
            # Sample report
            st.markdown(f"""
            ### Executive Summary
            
            **Property Analysis Date:** {st.session_state.get('analysis_date', 'Today')}
            
            **Rooftop Assessment:**
            - Panel Type: {panel_type.title()}
            - Analysis Method: {analysis_method}
            
            **Recommendations:**
            - ✅ Rooftop suitable for solar installation
            - ✅ Good solar exposure potential
            - ✅ Estimated 15-20 year payback period
            - ⚠️ Professional site survey recommended
            
            **Next Steps:**
            1. Detailed structural assessment
            2. Electrical system evaluation
            3. Permit applications
            4. Installation scheduling
            
            **Disclaimer:** This analysis is for preliminary assessment only. 
            Professional evaluation required for final system design.
            """)

else:
    # Information when no file uploaded
    st.info("👆 Please upload a rooftop image to begin analysis")
    
    # Sample information
    with st.expander("📋 About This Tool"):
        st.markdown("""
        **Solar Industry AI Assistant** combines computer vision and artificial intelligence 
        to analyze rooftop images for solar panel installation potential.
        
        **Features:**
        - 🖼️ **OpenCV Analysis**: Computer vision-based rooftop detection
        - 🤖 **AI Analysis**: Advanced AI-powered assessment
        - 💰 **Financial Modeling**: ROI and payback calculations
        - 📊 **Professional Reports**: Comprehensive analysis results
        
        **Supported Panel Types:**
        - **Monocrystalline**: Highest efficiency (22%), premium cost
        - **Polycrystalline**: Balanced efficiency (18%), moderate cost  
        - **Thin-film**: Lower efficiency (12%), lowest cost
        
        **Analysis Methods:**
        - **OpenCV**: Fast, reliable computer vision processing
        - **AI**: Detailed assessment with natural language insights
        - **Both**: Combined analysis for maximum accuracy
        """)
    
    # Sample images section
    st.markdown("### 📸 Sample Analysis")
    st.markdown("Try uploading a satellite or aerial image of a rooftop to see the analysis in action!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌞 <strong>Solar Industry AI Assistant</strong> - Built for Hugging Face Spaces</p>
    <p>Demonstrating AI-powered solar installation analysis</p>
</div>
""", unsafe_allow_html=True)
