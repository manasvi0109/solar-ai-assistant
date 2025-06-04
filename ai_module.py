import requests
import base64
import json
import re
from typing import Dict, Any

OPENROUTER_API_KEY = "your-api-key-here"

def encode_image(image_file):
    """Encode image file to base64."""
    return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_openrouter(image_file):
    """
    Original function for backward compatibility.
    Provides basic text analysis of rooftop image.
    """
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
                        "This is a satellite image of a building. Analyze it and identify the usable rooftop area "
                        "for solar panel installation. Detect any visible obstacles (e.g., AC units, water tanks), "
                        "and estimate how many solar panels (standard 1.7 m² each) can fit. Provide estimated usable area in m²."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


def analyze_rooftop_with_structured_output(image_file) -> Dict[str, Any]:
    """
    Enhanced function with structured output and confidence scoring.

    Returns:
        dict: Structured analysis with confidence scores
    """
    base64_image = encode_image(image_file)

    # Enhanced prompt for structured output
    structured_prompt = """
    Analyze this satellite/aerial rooftop image for solar panel installation potential.
    Provide your analysis in the following JSON format:

    {
        "rooftop_analysis": {
            "total_roof_area_m2": <estimated total roof area in square meters>,
            "usable_area_m2": <usable area for solar panels in square meters>,
            "usable_percentage": <percentage of roof that's usable>,
            "confidence_score": <confidence in analysis from 0.0 to 1.0>
        },
        "obstacles_detected": {
            "chimneys": <number of chimneys>,
            "vents": <number of vents>,
            "ac_units": <number of AC units>,
            "water_tanks": <number of water tanks>,
            "skylights": <number of skylights>,
            "other_obstacles": ["<list of other obstacles>"],
            "confidence_score": <confidence in obstacle detection from 0.0 to 1.0>
        },
        "roof_characteristics": {
            "roof_type": "<flat/pitched/complex>",
            "roof_material": "<tiles/metal/concrete/other>",
            "roof_condition": "<excellent/good/fair/poor>",
            "shading_issues": "<none/minimal/moderate/significant>",
            "orientation": "<north/south/east/west/mixed>",
            "tilt_angle": <estimated tilt angle in degrees>,
            "confidence_score": <confidence in roof characteristics from 0.0 to 1.0>
        },
        "solar_suitability": {
            "overall_rating": "<excellent/good/fair/poor>",
            "recommended_panel_count": <estimated number of panels>,
            "estimated_capacity_kw": <estimated system capacity in kW>,
            "installation_complexity": "<low/medium/high>",
            "special_considerations": ["<list of special considerations>"],
            "confidence_score": <confidence in suitability assessment from 0.0 to 1.0>
        },
        "recommendations": {
            "proceed_with_installation": <true/false>,
            "recommended_next_steps": ["<list of recommended actions>"],
            "potential_challenges": ["<list of potential challenges>"],
            "estimated_installation_timeline": "<timeframe estimate>"
        }
    }

    Ensure all numerical values are realistic and based on what you can observe in the image.
    Be conservative with confidence scores - only use high scores when you're very certain.
    """

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
                    {"type": "text", "text": structured_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.3,  # Lower temperature for more consistent output
        "max_tokens": 2000
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               json=data, headers=headers, timeout=30)

        if response.status_code == 200:
            ai_response = response.json()["choices"][0]["message"]["content"]
            return parse_structured_response(ai_response)
        else:
            return create_error_response(f"API Error: {response.status_code}")

    except Exception as e:
        return create_error_response(f"Analysis failed: {str(e)}")


def parse_structured_response(ai_response: str) -> Dict[str, Any]:
    """
    Parse AI response and extract structured data.
    """
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)

            # Validate and clean the data
            validated_data = validate_ai_response(parsed_data)
            return validated_data
        else:
            # Fallback: parse key information manually
            return parse_fallback_response(ai_response)

    except json.JSONDecodeError:
        return parse_fallback_response(ai_response)


def validate_ai_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean AI response data.
    """
    # Ensure required structure exists
    required_sections = ["rooftop_analysis", "obstacles_detected",
                        "roof_characteristics", "solar_suitability", "recommendations"]

    for section in required_sections:
        if section not in data:
            data[section] = {}

    # Validate numerical values
    rooftop = data.get("rooftop_analysis", {})
    if "total_roof_area_m2" in rooftop:
        rooftop["total_roof_area_m2"] = max(0, min(10000, rooftop["total_roof_area_m2"]))
    if "usable_area_m2" in rooftop:
        rooftop["usable_area_m2"] = max(0, min(rooftop.get("total_roof_area_m2", 1000),
                                              rooftop["usable_area_m2"]))

    # Validate confidence scores
    for section_name, section_data in data.items():
        if isinstance(section_data, dict) and "confidence_score" in section_data:
            confidence = section_data["confidence_score"]
            section_data["confidence_score"] = max(0.0, min(1.0, float(confidence)))

    # Calculate overall confidence
    confidence_scores = []
    for section_name, section_data in data.items():
        if isinstance(section_data, dict) and "confidence_score" in section_data:
            confidence_scores.append(section_data["confidence_score"])

    data["overall_confidence"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

    return data


def parse_fallback_response(ai_response: str) -> Dict[str, Any]:
    """
    Fallback parser for when structured JSON parsing fails.
    """
    # Extract basic information using regex patterns
    area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m²|square meters?|sq\.?\s*m)', ai_response, re.IGNORECASE)
    panel_match = re.search(r'(\d+)\s*(?:panels?|solar panels?)', ai_response, re.IGNORECASE)

    usable_area = float(area_match.group(1)) if area_match else 50.0
    panel_count = int(panel_match.group(1)) if panel_match else max(1, int(usable_area / 2))

    return {
        "rooftop_analysis": {
            "total_roof_area_m2": usable_area * 1.2,
            "usable_area_m2": usable_area,
            "usable_percentage": 80.0,
            "confidence_score": 0.6
        },
        "obstacles_detected": {
            "chimneys": 0,
            "vents": 1,
            "ac_units": 1,
            "water_tanks": 0,
            "skylights": 0,
            "other_obstacles": [],
            "confidence_score": 0.5
        },
        "roof_characteristics": {
            "roof_type": "flat",
            "roof_material": "concrete",
            "roof_condition": "good",
            "shading_issues": "minimal",
            "orientation": "south",
            "tilt_angle": 0,
            "confidence_score": 0.5
        },
        "solar_suitability": {
            "overall_rating": "good",
            "recommended_panel_count": panel_count,
            "estimated_capacity_kw": panel_count * 0.4,
            "installation_complexity": "medium",
            "special_considerations": ["Professional assessment recommended"],
            "confidence_score": 0.6
        },
        "recommendations": {
            "proceed_with_installation": True,
            "recommended_next_steps": ["Get professional site survey", "Check local regulations"],
            "potential_challenges": ["Weather conditions", "Grid connection"],
            "estimated_installation_timeline": "2-4 weeks"
        },
        "overall_confidence": 0.55,
        "raw_response": ai_response
    }


def create_error_response(error_message: str) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": True,
        "message": error_message,
        "rooftop_analysis": {"confidence_score": 0.0},
        "obstacles_detected": {"confidence_score": 0.0},
        "roof_characteristics": {"confidence_score": 0.0},
        "solar_suitability": {"confidence_score": 0.0},
        "overall_confidence": 0.0
    }

