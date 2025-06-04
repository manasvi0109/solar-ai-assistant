"""
Hugging Face Integration Module for Solar Industry AI Assistant
Provides advanced computer vision and NLP capabilities using pre-trained models
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st

# Lazy imports to handle missing dependencies gracefully
try:
    from transformers import (
        AutoImageProcessor, AutoModelForImageClassification,
        AutoTokenizer, AutoModelForCausalLM,
        pipeline, BlipProcessor, BlipForConditionalGeneration
    )
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

class HuggingFaceRooftopAnalyzer:
    """Advanced rooftop analysis using Hugging Face models."""
    
    def __init__(self):
        self.models_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configurations
        self.models = {
            "image_classifier": "microsoft/resnet-50",
            "object_detector": "facebook/detr-resnet-50",
            "image_captioner": "Salesforce/blip-image-captioning-base",
            "text_generator": "microsoft/DialoGPT-medium",
            "embeddings": "all-MiniLM-L6-v2"
        }
        
        # Initialize models lazily
        self._image_classifier = None
        self._object_detector = None
        self._image_captioner = None
        self._text_generator = None
        self._embeddings_model = None
    
    def check_availability(self) -> bool:
        """Check if Hugging Face dependencies are available."""
        return HUGGINGFACE_AVAILABLE
    
    @st.cache_resource
    def load_models(_self):
        """Load Hugging Face models with caching."""
        if not HUGGINGFACE_AVAILABLE:
            return False
        
        try:
            # Load image classification model
            _self._image_classifier = pipeline(
                "image-classification",
                model=_self.models["image_classifier"],
                device=0 if _self.device == "cuda" else -1
            )
            
            # Load object detection model
            _self._object_detector = pipeline(
                "object-detection",
                model=_self.models["object_detector"],
                device=0 if _self.device == "cuda" else -1
            )
            
            # Load image captioning model
            _self._image_captioner = pipeline(
                "image-to-text",
                model=_self.models["image_captioner"],
                device=0 if _self.device == "cuda" else -1
            )
            
            # Load text generation model
            _self._text_generator = pipeline(
                "text-generation",
                model=_self.models["text_generator"],
                device=0 if _self.device == "cuda" else -1
            )
            
            # Load sentence embeddings model
            _self._embeddings_model = SentenceTransformer(_self.models["embeddings"])
            
            _self.models_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading Hugging Face models: {str(e)}")
            return False
    
    def analyze_rooftop_structure(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze rooftop structure using computer vision models.
        
        Args:
            image (PIL.Image): Input rooftop image
            
        Returns:
            dict: Analysis results with structure information
        """
        if not self.models_loaded:
            if not self.load_models():
                return {"error": "Models not available"}
        
        try:
            results = {}
            
            # Image classification for roof type
            classification = self._image_classifier(image)
            results["roof_classification"] = classification[:3]  # Top 3 predictions
            
            # Object detection for obstacles and features
            objects = self._object_detector(image)
            results["detected_objects"] = self._process_detections(objects)
            
            # Image captioning for overall description
            caption = self._image_captioner(image)
            results["description"] = caption[0]["generated_text"] if caption else "No description available"
            
            # Calculate confidence scores
            results["confidence_score"] = self._calculate_confidence(classification, objects)
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _process_detections(self, detections: List[Dict]) -> Dict[str, Any]:
        """Process object detection results for solar-relevant objects."""
        relevant_objects = {
            "obstacles": [],
            "roof_features": [],
            "potential_issues": []
        }
        
        # Map detected objects to solar installation categories
        obstacle_keywords = ["person", "car", "truck", "bicycle", "motorcycle", "bus"]
        roof_keywords = ["roof", "building", "house", "chimney", "window", "door"]
        
        for detection in detections:
            label = detection["label"].lower()
            score = detection["score"]
            
            if score > 0.5:  # Only consider high-confidence detections
                if any(keyword in label for keyword in obstacle_keywords):
                    relevant_objects["obstacles"].append({
                        "type": label,
                        "confidence": score,
                        "bbox": detection["box"]
                    })
                elif any(keyword in label for keyword in roof_keywords):
                    relevant_objects["roof_features"].append({
                        "type": label,
                        "confidence": score,
                        "bbox": detection["box"]
                    })
        
        return relevant_objects
    
    def _calculate_confidence(self, classification: List[Dict], objects: List[Dict]) -> float:
        """Calculate overall confidence score for the analysis."""
        if not classification:
            return 0.0
        
        # Base confidence from top classification
        base_confidence = classification[0]["score"]
        
        # Adjust based on number of detected objects
        object_confidence = min(len(objects) * 0.1, 0.3)
        
        return min(base_confidence + object_confidence, 1.0)
    
    def generate_analysis_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report using NLP models.
        
        Args:
            analysis_data (dict): Combined analysis data from all sources
            
        Returns:
            str: Generated report text
        """
        if not self.models_loaded:
            if not self.load_models():
                return "Report generation not available - models not loaded."
        
        try:
            # Create prompt for report generation
            prompt = self._create_report_prompt(analysis_data)
            
            # Generate report using text generation model
            generated = self._text_generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            if generated:
                return generated[0]["generated_text"][len(prompt):].strip()
            else:
                return self._create_fallback_report(analysis_data)
                
        except Exception as e:
            return self._create_fallback_report(analysis_data)
    
    def _create_report_prompt(self, data: Dict[str, Any]) -> str:
        """Create a prompt for report generation."""
        prompt = "Solar Rooftop Analysis Report:\n\n"
        
        if "rooftop_analysis" in data:
            area = data["rooftop_analysis"].get("usable_area_m2", 0)
            prompt += f"Usable rooftop area: {area} square meters.\n"
        
        if "panel_data" in data:
            count = data["panel_data"].get("count", 0)
            capacity = data["panel_data"].get("total_capacity", 0)
            prompt += f"Estimated {count} solar panels with {capacity:.1f} kW capacity.\n"
        
        prompt += "\nBased on this analysis, the solar installation recommendations are:\n"
        return prompt
    
    def _create_fallback_report(self, data: Dict[str, Any]) -> str:
        """Create a fallback report when AI generation fails."""
        report = "## Solar Rooftop Analysis Report\n\n"
        
        if "rooftop_analysis" in data:
            rooftop = data["rooftop_analysis"]
            report += f"**Rooftop Assessment:**\n"
            report += f"- Total roof area: {rooftop.get('total_roof_area_m2', 'N/A')} m²\n"
            report += f"- Usable area: {rooftop.get('usable_area_m2', 'N/A')} m²\n"
            report += f"- Usable percentage: {rooftop.get('usable_percentage', 'N/A')}%\n\n"
        
        if "panel_data" in data:
            panels = data["panel_data"]
            report += f"**System Design:**\n"
            report += f"- Panel count: {panels.get('count', 'N/A')}\n"
            report += f"- System capacity: {panels.get('total_capacity', 'N/A')} kW\n"
            report += f"- Panel type: {panels.get('panel_type', 'N/A')}\n\n"
        
        if "cost_data" in data:
            cost = data["cost_data"]
            report += f"**Financial Summary:**\n"
            report += f"- Total investment: ₹{cost.get('total_cost', 'N/A'):,}\n"
            report += f"- Cost per watt: ₹{cost.get('cost_per_watt', 'N/A')}\n\n"
        
        report += "**Recommendation:** This analysis provides a comprehensive assessment of the rooftop's solar potential. "
        report += "Professional site survey recommended for final system design."
        
        return report
    
    def semantic_search(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Perform semantic search on documents using sentence embeddings.
        
        Args:
            query (str): Search query
            documents (List[str]): List of documents to search
            
        Returns:
            List[Tuple[str, float]]: Ranked documents with similarity scores
        """
        if not self.models_loaded or not self._embeddings_model:
            return []
        
        try:
            # Encode query and documents
            query_embedding = self._embeddings_model.encode([query])
            doc_embeddings = self._embeddings_model.encode(documents)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                torch.tensor(query_embedding),
                torch.tensor(doc_embeddings)
            )
            
            # Rank documents by similarity
            ranked_docs = [
                (doc, float(sim)) for doc, sim in zip(documents, similarities)
            ]
            ranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            return ranked_docs
            
        except Exception as e:
            st.error(f"Semantic search failed: {str(e)}")
            return []


def get_huggingface_analyzer() -> Optional[HuggingFaceRooftopAnalyzer]:
    """Get Hugging Face analyzer instance with error handling."""
    try:
        analyzer = HuggingFaceRooftopAnalyzer()
        if analyzer.check_availability():
            return analyzer
        else:
            return None
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face analyzer: {str(e)}")
        return None


# Knowledge base for semantic search
SOLAR_KNOWLEDGE_BASE = [
    "Monocrystalline solar panels offer the highest efficiency at 22% but cost more per watt.",
    "Polycrystalline panels provide good balance of cost and efficiency at 18% efficiency.",
    "Thin-film panels are flexible and work better in low light but have lower efficiency.",
    "South-facing roofs receive maximum solar irradiance in the Northern Hemisphere.",
    "Roof tilt angle should ideally match the latitude for optimal energy generation.",
    "Shading from trees or buildings can significantly reduce solar panel performance.",
    "String inverters are cost-effective for simple installations without shading issues.",
    "Power optimizers help maximize energy harvest when partial shading occurs.",
    "Microinverters provide panel-level optimization and monitoring capabilities.",
    "Net metering allows selling excess solar energy back to the electrical grid.",
    "Solar panel degradation rate is typically 0.5-0.8% per year over 25 years.",
    "Proper maintenance includes regular cleaning and electrical system inspections."
]
