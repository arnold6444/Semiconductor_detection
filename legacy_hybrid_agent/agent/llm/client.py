"""
LLM API client for external decision support using Luxia Cloud.
"""

import os
import json
import time
import base64
import requests
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from io import BytesIO

from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from ..config import get_config

logger = logging.getLogger(__name__)

# Luxia Cloud API configuration
LUXIA_BASE_URL = "https://bridge.luxiacloud.com/luxia/v1"
LUXIA_DOCUMENT_AI_ENDPOINT = f"{LUXIA_BASE_URL}/document-ai"
LUXIA_CHAT_ENDPOINT = f"{LUXIA_BASE_URL}/chat/completions"


class LLMClient:
    """
    Client for external LLM decision support using Luxia Cloud.
    
    Uses Luxia Cloud Document AI for multimodal image analysis.
    Implements retry with exponential backoff.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "luxia-document-ai",
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.api_key = api_key or os.environ.get("LUXIA_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning(
                "LUXIA_API_KEY not set. LLM decision support will be disabled."
            )
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    
    def _parse_response(self, response_text: str) -> Tuple[Optional[int], Optional[float]]:
        """
        Parse LLM response to extract label and confidence.
        
        Returns:
            Tuple of (label, confidence) or (None, None) on parse failure
        """
        try:
            # Clean response text
            text = response_text.strip().lower()
            
            # Try to find JSON in response
            if "{" in text and "}" in text:
                # Extract JSON part
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
                
                try:
                    data = json.loads(json_str)
                    label = int(data.get("label", -1))
                    confidence = float(data.get("confidence", 0.8))
                    
                    if label in [0, 1]:
                        return label, min(max(confidence, 0.0), 1.0)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse text for keywords
            if "abnormal" in text or "defect" in text or "anomal" in text:
                # Look for confidence indicators
                if "high" in text or "certain" in text or "clear" in text:
                    return 1, 0.9
                elif "low" in text or "uncertain" in text:
                    return 1, 0.6
                return 1, 0.75
            elif "normal" in text or "good" in text or "no defect" in text:
                if "high" in text or "certain" in text or "clear" in text:
                    return 0, 0.9
                return 0, 0.75
            
            logger.warning(f"Could not parse response: {response_text[:200]}")
            return None, None
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None, None
    
    def _call_document_ai(self, image: Image.Image) -> Tuple[Optional[str], Optional[str]]:
        """
        Call Luxia Document AI endpoint.
        
        Returns:
            Tuple of (response_content, error)
        """
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Convert image to base64
        image_b64 = self._image_to_base64(image)
        
        payload = {
            "image": image_b64
        }
        
        try:
            response = requests.post(
                LUXIA_DOCUMENT_AI_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("content", "")
            return content, None
            
        except requests.exceptions.RequestException as e:
            return None, f"Request error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"
    
    def _call_chat_api(self, image: Image.Image) -> Tuple[Optional[str], Optional[str]]:
        """
        Call Luxia Chat API with image for analysis.
        Uses the chat completions endpoint with multimodal support.
        
        Returns:
            Tuple of (response_content, error)
        """
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Convert image to base64
        image_b64 = self._image_to_base64(image)
        
        # Build chat request with system prompt and image
        payload = {
            "model": "luxia3",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_PROMPT_TEMPLATE
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_b64
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        try:
            response = requests.post(
                LUXIA_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content, None
            
        except requests.exceptions.RequestException as e:
            return None, f"Request error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"
    
    def analyze_image(
        self,
        image: Image.Image
    ) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """
        Analyze a semiconductor image using the LLM.
        
        Tries Document AI first, then falls back to Chat API if needed.
        
        Args:
            image: PIL Image to analyze
        
        Returns:
            Tuple of (label, confidence, error) where error is None on success
        """
        if not self.api_key:
            return None, None, "LLM client not initialized (API key missing?)"
        
        # Convert image to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Try Chat API first (sends prompt + image for better accuracy)
                content, error = self._call_chat_api(image)
                
                if content:
                    # Parse the response to get label and confidence
                    label, confidence = self._parse_response(content)
                    if label is not None:
                        logger.info(f"LLM response: label={label}, confidence={confidence:.2f}")
                        return label, confidence, None
                
                # Fallback to Document AI if Chat API fails
                if error:
                    logger.warning(f"Chat API failed: {error}, trying Document AI...")
                    content, error = self._call_document_ai(image)
                    if content:
                        label, confidence = self._parse_response(content)
                        if label is not None:
                            logger.info(f"Document AI response: label={label}, confidence={confidence:.2f}")
                            return label, confidence, None
                
                last_error = error or "Failed to get valid response"
                
            except Exception as e:
                last_error = f"LLM API error: {e}"
                logger.warning(f"LLM attempt {attempt + 1}/{self.max_retries}: {last_error}")
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        return None, None, last_error
    
    def get_probability(self, image: Image.Image) -> Tuple[float, Optional[str]]:
        """
        Get abnormal probability from LLM.
        
        Returns:
            Tuple of (probability, error) where error is None on success
        """
        label, confidence, error = self.analyze_image(image)
        
        if error is not None:
            return 0.5, error  # Return neutral probability on error
        
        # Convert label + confidence to probability
        # If label=1 (abnormal), use confidence directly
        # If label=0 (normal), probability = 1 - confidence
        if label == 1:
            prob = confidence
        else:
            prob = 1.0 - confidence
        
        return prob, None
    
    def is_available(self) -> bool:
        """Check if LLM is available for use."""
        return self.api_key is not None
