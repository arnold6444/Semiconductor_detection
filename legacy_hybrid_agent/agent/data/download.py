"""
Image download module with retry logic and caching.
"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"


def get_cache_path(url: str, cache_dir: Optional[Path] = None) -> Path:
    """Generate a unique cache path for a URL."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    # Create hash of URL for unique filename
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # Try to preserve extension from URL
    ext = ".jpg"
    if "." in url.split("/")[-1]:
        ext = "." + url.split(".")[-1].split("?")[0][:4]
    
    return cache_dir / f"{url_hash}{ext}"


def download_image(
    url: str,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
    timeout: int = 30,
    use_cache: bool = True
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Download an image from URL with retry logic.
    
    Args:
        url: Image URL
        cache_dir: Directory to cache downloaded images
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        use_cache: Whether to use cached images
    
    Returns:
        Tuple of (PIL Image or None, error message or None)
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    cache_path = get_cache_path(url, cache_dir)
    
    # Check cache first
    if use_cache and cache_path.exists():
        try:
            img = Image.open(cache_path).convert("RGB")
            logger.debug(f"Loaded from cache: {cache_path}")
            return img, None
        except Exception as e:
            logger.warning(f"Failed to load cached image: {e}")
            cache_path.unlink(missing_ok=True)
    
    # Download with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Load image
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Cache the image
            os.makedirs(cache_dir, exist_ok=True)
            img.save(cache_path, "JPEG", quality=95)
            logger.debug(f"Downloaded and cached: {url}")
            
            return img, None
            
        except requests.exceptions.RequestException as e:
            last_error = f"Request error (attempt {attempt + 1}/{max_retries}): {e}"
            logger.warning(last_error)
        except Exception as e:
            last_error = f"Image processing error: {e}"
            logger.error(last_error)
            break
    
    return None, last_error


def download_images_batch(
    urls: list,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
    show_progress: bool = True
) -> list:
    """
    Download multiple images with progress bar.
    
    Returns:
        List of tuples (url, image or None, error or None)
    """
    from tqdm import tqdm
    
    results = []
    iterator = tqdm(urls, desc="Downloading images") if show_progress else urls
    
    for url in iterator:
        img, error = download_image(url, cache_dir, max_retries)
        results.append((url, img, error))
    
    return results
