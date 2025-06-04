import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Solar Industry AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ai_module import analyze_rooftop_with_structured_output
from opencv_module import process_image_for_rooftop
from utils import (
    estimate_panels, estimate_installation_cost, estimate_savings, calculate_roi,
    get_maintenance_schedule, get_regulatory_requirements, PANEL_TYPES, REGIONAL_FACTORS
)

# Hugging Face integration (optional)
try:
    from huggingface_module import get_huggingface_analyzer, SOLAR_KNOWLEDGE_BASE
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    st.warning("⚠️ Hugging Face models not available. Install transformers and torch for enhanced AI features.")

st.title("☀️ Solar Industry AI Assistant")
st.markdown("### Professional Solar Installation Analysis & ROI Calculator")
st.write("Upload a satellite rooftop image to get comprehensive solar installation recommendations, cost analysis, and ROI estimates.")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Panel type selection
panel_type = st.sidebar.selectbox(
    "Select Panel Type:",
    options=list(PANEL_TYPES.keys()),
    format_func=lambda x: f"{x.title()} - {PANEL_TYPES[x]['description']}"
)

# Region selection
region = st.sidebar.selectbox(
    "Select Region:",
    options=list(REGIONAL_FACTORS["solar_irradiance"].keys()),
    format_func=lambda x: f"{x.title()} India ({REGIONAL_FACTORS['solar_irradiance'][x]} kWh/m²/day)"
)

# System configuration
inverter_type = st.sidebar.selectbox(
    "Inverter Type:",
    options=["string_inverter", "power_optimizer", "microinverter"],
    format_func=lambda x: x.replace("_", " ").title()
)

mounting_type = st.sidebar.selectbox(
    "Mounting Type:",
    options=["roof_mounted", "ground_mounted", "tracking"],
    format_func=lambda x: x.replace("_", " ").title()
)

electricity_rate = st.sidebar.number_input(
    "Current Electricity Rate (₹/kWh):",
    min_value=1.0,
    max_value=20.0,
    value=8.5,
    step=0.5
)

# Hugging Face options
if HUGGINGFACE_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.header("🤗 Hugging Face AI")

    enable_hf_analysis = st.sidebar.checkbox(
        "Enable Advanced AI Analysis",
        help="Use Hugging Face models for enhanced rooftop analysis"
    )

    enable_report_generation = st.sidebar.checkbox(
        "Generate AI Report",
        help="Create comprehensive analysis report using NLP models"
    )

    enable_knowledge_search = st.sidebar.checkbox(
        "Enable Knowledge Search",
        help="Semantic search through solar industry knowledge base"
    )
else:
    enable_hf_analysis = False
    enable_report_generation = False
    enable_knowledge_search = False

# Main content
uploaded_file = st.file_uploader("Upload a rooftop image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Rooftop Image", use_container_width=True)

    # Analysis options
    if HUGGINGFACE_AVAILABLE:
        col1, col2, col3 = st.columns(3)

        with col1:
            opencv_analysis = st.button("🔍 OpenCV Analysis", use_container_width=True)

        with col2:
            ai_analysis = st.button("🤖 AI Analysis (Enhanced)", use_container_width=True)

        with col3:
            hf_analysis = st.button("🤗 Hugging Face AI", use_container_width=True)
    else:
        col1, col2 = st.columns(2)

        with col1:
            opencv_analysis = st.button("🔍 OpenCV Analysis", use_container_width=True)

        with col2:
            ai_analysis = st.button("🤖 AI Analysis (Enhanced)", use_container_width=True)

        hf_analysis = False

    # OpenCV Analysis
    if opencv_analysis:
        with st.spinner("Processing image with OpenCV..."):
            try:
                result_img, usable_px = process_image_for_rooftop(uploaded_file)

                # Convert pixels to area (calibration factor)
                usable_area = round(usable_px * 0.05, 2)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.image(result_img, caption="Rooftop Detection (OpenCV)", channels="BGR", use_container_width=True)

                with col2:
                    st.metric("Detected Area", f"{usable_area} m²")
                    st.metric("Detected Pixels", f"{usable_px:,}")

                # Perform comprehensive analysis
                perform_comprehensive_analysis(usable_area, panel_type, region, inverter_type, mounting_type, electricity_rate)

            except Exception as e:
                st.error(f"OpenCV analysis failed: {str(e)}")

    # AI Analysis
    if ai_analysis:
        with st.spinner("Getting enhanced AI analysis..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                ai_result = analyze_rooftop_with_structured_output(uploaded_file)

                if "error" in ai_result:
                    st.error(f"AI analysis failed: {ai_result['message']}")
                else:
                    display_ai_analysis(ai_result)

                    # Use AI-detected area for calculations
                    ai_usable_area = ai_result.get("rooftop_analysis", {}).get("usable_area_m2", 50.0)
                    perform_comprehensive_analysis(ai_usable_area, panel_type, region, inverter_type, mounting_type, electricity_rate)

            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")

    # Hugging Face Analysis
    if hf_analysis and HUGGINGFACE_AVAILABLE:
        with st.spinner("Running Hugging Face AI analysis..."):
            try:
                # Get Hugging Face analyzer
                hf_analyzer = get_huggingface_analyzer()

                if hf_analyzer:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    from PIL import Image
                    pil_image = Image.open(uploaded_file)

                    # Perform Hugging Face analysis
                    hf_result = hf_analyzer.analyze_rooftop_structure(pil_image)

                    if "error" not in hf_result:
                        display_huggingface_analysis(hf_result)

                        # Extract area estimate from description or use default
                        estimated_area = extract_area_from_hf_analysis(hf_result)

                        if estimated_area > 0:
                            perform_comprehensive_analysis(estimated_area, panel_type, region, inverter_type, mounting_type, electricity_rate)

                            # Generate AI report if enabled
                            if enable_report_generation:
                                st.markdown("---")
                                st.markdown("## 📄 AI-Generated Report")

                                # Combine all analysis data
                                panel_data = estimate_panels(estimated_area, panel_type=panel_type)
                                cost_data = estimate_installation_cost(panel_data, inverter_type=inverter_type, mounting_type=mounting_type)
                                savings_data = estimate_savings(panel_data, region=region, electricity_rate=electricity_rate)

                                combined_data = {
                                    "rooftop_analysis": {"usable_area_m2": estimated_area},
                                    "panel_data": panel_data,
                                    "cost_data": cost_data,
                                    "savings_data": savings_data,
                                    "hf_analysis": hf_result
                                }

                                report = hf_analyzer.generate_analysis_report(combined_data)
                                st.markdown(report)
                    else:
                        st.error(f"Hugging Face analysis failed: {hf_result['error']}")
                else:
                    st.error("Failed to initialize Hugging Face analyzer")

            except Exception as e:
                st.error(f"Hugging Face analysis failed: {str(e)}")

else:
    # Show information when no file is uploaded
    st.info("👆 Please upload a rooftop image to begin analysis")

    # Knowledge search feature (available without image upload)
    if HUGGINGFACE_AVAILABLE and enable_knowledge_search:
        st.markdown("---")
        st.markdown("## 🔍 Solar Knowledge Search")

        search_query = st.text_input(
            "Search solar industry knowledge:",
            placeholder="e.g., 'best panel type for hot climate' or 'roof angle optimization'"
        )

        if search_query:
            with st.spinner("Searching knowledge base..."):
                try:
                    hf_analyzer = get_huggingface_analyzer()
                    if hf_analyzer:
                        results = hf_analyzer.semantic_search(search_query, SOLAR_KNOWLEDGE_BASE)

                        if results:
                            st.markdown("### 📚 Search Results")
                            for i, (doc, score) in enumerate(results[:5], 1):
                                confidence = "🟢 High" if score > 0.7 else "🟡 Medium" if score > 0.5 else "🔴 Low"
                                st.markdown(f"**{i}.** {doc}")
                                st.caption(f"Relevance: {confidence} ({score:.2f})")
                                st.markdown("")
                        else:
                            st.info("No relevant results found. Try different keywords.")
                    else:
                        st.error("Knowledge search not available")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")

    # Display sample information
    with st.expander("📋 What this tool analyzes"):
        st.markdown("""
        **Image Analysis:**
        - Rooftop area detection and measurement
        - Obstacle identification (chimneys, vents, AC units)
        - Roof condition and suitability assessment
        - Shading analysis and orientation detection

        **Solar System Design:**
        - Panel count and capacity estimation
        - System component recommendations
        - Installation complexity assessment

        **Financial Analysis:**
        - Detailed cost breakdown
        - 25-year savings projection
        - ROI and payback period calculation
        - Maintenance cost estimation

        **Regulatory Compliance:**
        - Required permits and approvals
        - Safety standards and building codes
        - Available incentives and subsidies
        """)


def perform_comprehensive_analysis(usable_area, panel_type, region, inverter_type, mounting_type, electricity_rate):
    """Perform comprehensive solar analysis and display results."""

    st.markdown("---")
    st.markdown("## 📊 Comprehensive Solar Analysis")

    # Step 1: Panel estimation
    panel_data = estimate_panels(usable_area, panel_type=panel_type)

    if panel_data["count"] == 0:
        st.warning("⚠️ Insufficient rooftop area for solar installation")
        return

    # Step 2: Cost estimation
    cost_data = estimate_installation_cost(panel_data, inverter_type=inverter_type, mounting_type=mounting_type)

    # Step 3: Savings estimation
    savings_data = estimate_savings(panel_data, region=region, electricity_rate=electricity_rate)

    # Step 4: ROI calculation
    roi_data = calculate_roi(cost_data, savings_data)

    # Step 5: Maintenance schedule
    maintenance_data = get_maintenance_schedule(panel_data)

    # Step 6: Regulatory information
    regulatory_data = get_regulatory_requirements()

    # Display results in organized sections
    display_system_overview(panel_data, usable_area)
    display_financial_analysis(cost_data, savings_data, roi_data)
    display_maintenance_schedule(maintenance_data)
    display_regulatory_info(regulatory_data)


def display_ai_analysis(ai_result):
    """Display AI analysis results in a structured format."""
    st.markdown("## 🤖 AI Analysis Results")

    # Overall confidence
    overall_confidence = ai_result.get("overall_confidence", 0.0)
    confidence_color = "green" if overall_confidence > 0.7 else "orange" if overall_confidence > 0.4 else "red"
    st.markdown(f"**Overall Confidence:** :{confidence_color}[{overall_confidence:.1%}]")

    # Rooftop Analysis
    rooftop = ai_result.get("rooftop_analysis", {})
    if rooftop:
        st.markdown("### 🏠 Rooftop Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Roof Area", f"{rooftop.get('total_roof_area_m2', 0):.1f} m²")
        with col2:
            st.metric("Usable Area", f"{rooftop.get('usable_area_m2', 0):.1f} m²")
        with col3:
            st.metric("Usable %", f"{rooftop.get('usable_percentage', 0):.1f}%")

    # Obstacles
    obstacles = ai_result.get("obstacles_detected", {})
    if obstacles:
        st.markdown("### 🚧 Obstacles Detected")
        obstacle_cols = st.columns(5)
        obstacle_items = [
            ("Chimneys", obstacles.get("chimneys", 0)),
            ("Vents", obstacles.get("vents", 0)),
            ("AC Units", obstacles.get("ac_units", 0)),
            ("Water Tanks", obstacles.get("water_tanks", 0)),
            ("Skylights", obstacles.get("skylights", 0))
        ]
        for i, (name, count) in enumerate(obstacle_items):
            with obstacle_cols[i]:
                st.metric(name, count)

    # Roof Characteristics
    roof_char = ai_result.get("roof_characteristics", {})
    if roof_char:
        st.markdown("### 🏗️ Roof Characteristics")
        char_col1, char_col2, char_col3 = st.columns(3)
        with char_col1:
            st.write(f"**Type:** {roof_char.get('roof_type', 'Unknown').title()}")
            st.write(f"**Material:** {roof_char.get('roof_material', 'Unknown').title()}")
        with char_col2:
            st.write(f"**Condition:** {roof_char.get('roof_condition', 'Unknown').title()}")
            st.write(f"**Orientation:** {roof_char.get('orientation', 'Unknown').title()}")
        with char_col3:
            st.write(f"**Shading:** {roof_char.get('shading_issues', 'Unknown').title()}")
            st.write(f"**Tilt Angle:** {roof_char.get('tilt_angle', 0)}°")

    # Solar Suitability
    suitability = ai_result.get("solar_suitability", {})
    if suitability:
        st.markdown("### ☀️ Solar Suitability")
        suit_col1, suit_col2 = st.columns(2)
        with suit_col1:
            rating = suitability.get('overall_rating', 'Unknown')
            rating_color = "green" if rating == "excellent" else "blue" if rating == "good" else "orange" if rating == "fair" else "red"
            st.markdown(f"**Overall Rating:** :{rating_color}[{rating.title()}]")
            st.write(f"**Recommended Panels:** {suitability.get('recommended_panel_count', 0)}")
            st.write(f"**Estimated Capacity:** {suitability.get('estimated_capacity_kw', 0):.1f} kW")
        with suit_col2:
            complexity = suitability.get('installation_complexity', 'Unknown')
            complexity_color = "green" if complexity == "low" else "orange" if complexity == "medium" else "red"
            st.markdown(f"**Installation Complexity:** :{complexity_color}[{complexity.title()}]")

            considerations = suitability.get('special_considerations', [])
            if considerations:
                st.write("**Special Considerations:**")
                for consideration in considerations:
                    st.write(f"• {consideration}")


def display_system_overview(panel_data, usable_area):
    """Display system overview section."""
    st.markdown("### 🔧 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Usable Area", f"{usable_area} m²")

    with col2:
        st.metric("Panel Count", panel_data["count"])

    with col3:
        st.metric("System Capacity", f"{panel_data['total_capacity']:.1f} kW")

    with col4:
        panel_specs = panel_data["panel_specs"]
        if panel_specs:
            st.metric("Panel Efficiency", f"{panel_specs['efficiency']:.1%}")

    # Panel specifications
    if panel_specs:
        with st.expander("📋 Panel Specifications"):
            spec_col1, spec_col2 = st.columns(2)
            with spec_col1:
                st.write(f"**Panel Type:** {panel_data['panel_type'].title()}")
                st.write(f"**Wattage per Panel:** {panel_specs['wattage']} W")
                st.write(f"**Area per Panel:** {panel_specs['area_per_panel']} m²")
            with spec_col2:
                st.write(f"**Efficiency:** {panel_specs['efficiency']:.1%}")
                st.write(f"**Lifespan:** {panel_specs['lifespan']} years")
                st.write(f"**Degradation Rate:** {panel_specs['degradation_rate']:.1%}/year")


def display_financial_analysis(cost_data, savings_data, roi_data):
    """Display financial analysis section."""
    st.markdown("### 💰 Financial Analysis")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Investment", f"₹{cost_data['total_cost']:,.0f}")

    with col2:
        st.metric("25-Year Savings", f"₹{savings_data['total_savings']:,.0f}")

    with col3:
        payback = roi_data['payback_period']
        st.metric("Payback Period", f"{payback} years" if payback != "N/A" else "N/A")

    with col4:
        st.metric("Annual Generation", f"{savings_data['annual_generation']:,.0f} kWh")

    # Detailed cost breakdown
    with st.expander("💸 Detailed Cost Breakdown"):
        breakdown = cost_data['breakdown']
        cost_col1, cost_col2 = st.columns(2)

        with cost_col1:
            st.write(f"**Solar Panels:** ₹{breakdown['panels']:,.0f}")
            st.write(f"**Inverter:** ₹{breakdown['inverter']:,.0f}")
            st.write(f"**Mounting System:** ₹{breakdown['mounting']:,.0f}")
            st.write(f"**Electrical Components:** ₹{breakdown['electrical']:,.0f}")

        with cost_col2:
            st.write(f"**Installation Labor:** ₹{breakdown['labor']:,.0f}")
            st.write(f"**Permits & Inspections:** ₹{breakdown['permits']:,.0f}")
            st.write(f"**Contingency:** ₹{breakdown['contingency']:,.0f}")
            st.write(f"**Cost per Watt:** ₹{cost_data['cost_per_watt']}")

    # ROI Analysis
    with st.expander("📈 ROI Analysis"):
        roi_col1, roi_col2 = st.columns(2)

        with roi_col1:
            st.write(f"**Net Present Value (NPV):** ₹{roi_data['npv']:,.0f}")
            st.write(f"**Internal Rate of Return (IRR):** {roi_data['irr']:.1f}%")
            st.write(f"**Cost per kWh:** ₹{roi_data['cost_per_kwh']:.2f}")

        with roi_col2:
            st.write(f"**Gross Savings:** ₹{savings_data['gross_savings']:,.0f}")
            st.write(f"**Maintenance Costs:** ₹{savings_data['maintenance_costs']:,.0f}")
            st.write(f"**Average Annual Savings:** ₹{savings_data['breakdown']['average_annual_savings']:,.0f}")

    # Savings projection chart
    if 'yearly_details' in savings_data['breakdown']:
        st.markdown("#### 📊 5-Year Savings Projection")
        yearly_data = savings_data['breakdown']['yearly_details']

        import pandas as pd
        df = pd.DataFrame(yearly_data)
        st.line_chart(df.set_index('year')[['generation', 'savings']])


def display_maintenance_schedule(maintenance_data):
    """Display maintenance schedule section."""
    st.markdown("### 🔧 Maintenance Schedule")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Annual Maintenance Cost", f"₹{maintenance_data['annual_cost']:,.0f}")

    with col2:
        st.metric("25-Year Maintenance Total", f"₹{maintenance_data['total_25_year_cost']:,.0f}")

    # Maintenance tasks
    with st.expander("📋 Maintenance Tasks"):
        for task in maintenance_data['schedule']:
            st.markdown(f"**{task['task']}** ({task['frequency']})")
            st.write(f"• {task['description']}")
            st.write(f"• Annual Cost: ₹{task['cost_per_year']:,.0f}")
            st.write("")


def display_regulatory_info(regulatory_data):
    """Display regulatory information section."""
    st.markdown("### 📋 Regulatory Information")

    # Permits required
    with st.expander("📄 Required Permits"):
        for permit in regulatory_data['permits_required']:
            st.write(f"• {permit}")

    # Building codes
    with st.expander("🏗️ Building Codes & Standards"):
        for code in regulatory_data['building_codes']:
            st.write(f"• {code}")

    # Incentives
    with st.expander("💰 Available Incentives"):
        incentives = regulatory_data['incentives']

        st.markdown("**Central Government Subsidies:**")
        central = incentives['central_subsidy']
        st.write(f"• Up to 3kW: {central['residential_up_to_3kw']:.0%} subsidy")
        st.write(f"• 3kW to 10kW: {central['residential_3kw_to_10kw']:.0%} subsidy")
        st.write(f"• {central['description']}")

        st.markdown("**Tax Benefits:**")
        accel_dep = incentives['accelerated_depreciation']
        st.write(f"• Accelerated Depreciation: {accel_dep['rate']:.0%} in first year")
        st.write(f"• {accel_dep['description']}")

        st.markdown("**Net Metering:**")
        net_meter = incentives['net_metering']
        if net_meter['available']:
            st.write(f"• ✅ {net_meter['description']}")
        else:
            st.write(f"• ❌ Not available in this region")

    # Safety standards
    with st.expander("⚡ Safety Standards"):
        for standard in regulatory_data['safety_standards']:
            st.write(f"• {standard}")


def display_huggingface_analysis(hf_result):
    """Display Hugging Face analysis results."""
    st.markdown("## 🤗 Hugging Face AI Analysis")

    # Roof classification
    if "roof_classification" in hf_result:
        st.markdown("### 🏠 Roof Classification")
        classifications = hf_result["roof_classification"]
        for i, cls in enumerate(classifications, 1):
            confidence = cls["score"]
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"**{i}.** {cls['label']} - :{confidence_color}[{confidence:.1%}]")

    # Object detection
    if "detected_objects" in hf_result:
        st.markdown("### 🔍 Detected Objects")
        objects = hf_result["detected_objects"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Obstacles:**")
            if objects.get("obstacles"):
                for obj in objects["obstacles"]:
                    st.write(f"• {obj['type'].title()} ({obj['confidence']:.1%})")
            else:
                st.write("• No obstacles detected")

        with col2:
            st.markdown("**Roof Features:**")
            if objects.get("roof_features"):
                for obj in objects["roof_features"]:
                    st.write(f"• {obj['type'].title()} ({obj['confidence']:.1%})")
            else:
                st.write("• No specific features detected")

    # Description
    if "description" in hf_result:
        st.markdown("### 📝 AI Description")
        st.write(hf_result["description"])

    # Confidence score
    if "confidence_score" in hf_result:
        confidence = hf_result["confidence_score"]
        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
        st.markdown(f"**Overall Confidence:** :{confidence_color}[{confidence:.1%}]")


def extract_area_from_hf_analysis(hf_result):
    """Extract usable area estimate from Hugging Face analysis."""
    # Try to extract area from description using regex
    import re

    description = hf_result.get("description", "")

    # Look for area mentions in the description
    area_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:square\s*)?(?:meters?|m²|m2)',
        r'(\d+(?:\.\d+)?)\s*(?:m²|m2)',
        r'area.*?(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*m)'
    ]

    for pattern in area_patterns:
        match = re.search(pattern, description.lower())
        if match:
            try:
                area = float(match.group(1))
                if 10 <= area <= 1000:  # Reasonable range
                    return area
            except (ValueError, IndexError):
                continue

    # Fallback: estimate based on detected objects and classification
    base_area = 80.0  # Default assumption

    # Adjust based on roof classification confidence
    if "confidence_score" in hf_result:
        confidence = hf_result["confidence_score"]
        if confidence > 0.8:
            base_area *= 1.1  # Increase if very confident
        elif confidence < 0.5:
            base_area *= 0.8  # Decrease if low confidence

    # Adjust based on detected obstacles
    if "detected_objects" in hf_result:
        obstacles = hf_result["detected_objects"].get("obstacles", [])
        obstacle_count = len(obstacles)
        if obstacle_count > 0:
            base_area *= (1 - obstacle_count * 0.05)  # Reduce by 5% per obstacle

    return max(20.0, base_area)  # Minimum 20 m²
