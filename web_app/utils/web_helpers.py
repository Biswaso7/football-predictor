#!/usr/bin/env python3
"""
Web Application Utilities
Helper functions for the web application
Author: AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
import jwt
import redis
from flask import request, jsonify, session, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import requests
from urllib.parse import urlparse, urljoin
import re
from bs4 import BeautifulSoup
import markdown
import bleach
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebHelpers:
    """Web application helper functions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize web helpers"""
        self.config = config or {}
        self.redis_client = self._init_redis()
        
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
            return redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return None
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return check_password_hash(password_hash, password)
    
    @staticmethod
    def generate_jwt_token(user_id: int, secret_key: str, expires_in: int = 3600) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    @staticmethod
    def decode_jwt_token(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
        """Decode and verify JWT token"""
        try:
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Limit length
        text = text[:max_length]
        
        # Remove potentially harmful characters
        text = bleach.clean(text, tags=[], attributes={}, strip=True)
        
        return text
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate username format"""
        if not username:
            return False, "Username cannot be empty"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(username) > 20:
            return False, "Username must be no more than 20 characters long"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        
        if username[0].isdigit():
            return False, "Username cannot start with a number"
        
        return True, "Valid username"
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must be no more than 128 characters long"
        
        # Check for uppercase letter
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for lowercase letter
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for digit
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        # Check for special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, "Valid password"
    
    @staticmethod
    def format_currency(amount: float, currency: str = 'USD') -> str:
        """Format amount as currency"""
        try:
            if currency == 'USD':
                return f"${amount:,.2f}"
            elif currency == 'EUR':
                return f"€{amount:,.2f}"
            elif currency == 'GBP':
                return f"£{amount:,.2f}"
            else:
                return f"{currency} {amount:,.2f}"
        except Exception:
            return f"${amount:.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format value as percentage"""
        try:
            return f"{value:.{decimals}f}%"
        except Exception:
            return f"{value:.1f}%"
    
    @staticmethod
    def format_number(number: float, decimals: int = 0) -> str:
        """Format number with commas"""
        try:
            return f"{number:,.{decimals}f}"
        except Exception:
            return str(number)
    
    @staticmethod
    def format_date(date: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format datetime object"""
        try:
            return date.strftime(format_str)
        except Exception:
            return str(date)
    
    @staticmethod
    def parse_date(date_string: str, format_str: str = '%Y-%m-%d') -> Optional[datetime]:
        """Parse date string to datetime object"""
        try:
            return datetime.strptime(date_string, format_str)
        except Exception:
            return None
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """Convert datetime to human-readable time ago format"""
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 0:
            if diff.days == 1:
                return "1 day ago"
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            if hours == 1:
                return "1 hour ago"
            return f"{hours} hours ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            if minutes == 1:
                return "1 minute ago"
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
        """Truncate text to maximum length"""
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9]+', '-', text)
        
        # Remove leading/trailing hyphens
        text = text.strip('-')
        
        return text
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""
    
    @staticmethod
    def is_safe_url(url: str, allowed_domains: List[str] = None) -> bool:
        """Check if URL is safe"""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check domain if provided
            if allowed_domains:
                domain = parsed.netloc
                if domain not in allowed_domains:
                    return False
            
            # Additional safety checks
            if '@' in url or '..' in parsed.path:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def generate_gravatar_url(email: str, size: int = 200) -> str:
        """Generate Gravatar URL from email"""
        try:
            email_hash = hashlib.md5(email.lower().encode('utf-8')).hexdigest()
            return f"https://www.gravatar.com/avatar/{email_hash}?s={size}&d=identicon"
        except Exception:
            return ""
    
    @staticmethod
    def resize_image(image_data: bytes, max_width: int = 800, max_height: int = 600,
                    quality: int = 85) -> bytes:
        """Resize image while maintaining aspect ratio"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Calculate new dimensions
            width, height = image.size
            
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                # Resize image
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image_data
    
    @staticmethod
    def encode_image_base64(image_data: bytes) -> str:
        """Encode image data to base64 string"""
        try:
            return base64.b64encode(image_data).decode('utf-8')
        except Exception:
            return ""
    
    @staticmethod
    def decode_image_base64(base64_string: str) -> bytes:
        """Decode base64 string to image data"""
        try:
            return base64.b64decode(base64_string)
        except Exception:
            return b""
    
    @staticmethod
    def generate_qr_code(data: str, size: int = 200) -> str:
        """Generate QR code as base64 string"""
        try:
            import qrcode
            from qrcode.image.pure import PyPNGImage
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white", image_factory=PyPNGImage)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except ImportError:
            logger.warning("qrcode library not installed")
            return ""
        except Exception as e:
            logger.error(f"Error generating QR code: {e}")
            return ""
    
    @staticmethod
    def send_email(to_email: str, subject: str, body: str, 
                  from_email: str = None, smtp_config: Dict[str, Any] = None) -> bool:
        """Send email notification"""
        try:
            if not smtp_config:
                logger.warning("SMTP configuration not provided")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email or smtp_config.get('from_email', 'noreply@example.com')
            msg['To'] = to_email
            
            # Create HTML version of the body
            html_body = markdown.markdown(body)
            
            # Attach parts
            text_part = MIMEText(body, 'plain')
            html_part = MIMEText(html_body, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    @staticmethod
    def validate_file_upload(file, allowed_extensions: List[str], 
                           max_size_mb: int = 10) -> Tuple[bool, str]:
        """Validate file upload"""
        try:
            if not file:
                return False, "No file provided"
            
            # Check file extension
            filename = file.filename
            if '.' not in filename:
                return False, "File has no extension"
            
            extension = filename.rsplit('.', 1)[1].lower()
            if extension not in allowed_extensions:
                return False, f"File extension not allowed. Allowed: {', '.join(allowed_extensions)}"
            
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            max_size_bytes = max_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return False, f"File too large. Maximum size: {max_size_mb}MB"
            
            return True, "Valid file"
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False, "Error validating file"
    
    @staticmethod
    def secure_filename(filename: str) -> str:
        """Secure filename for upload"""
        return secure_filename(filename)
    
    @staticmethod
    def generate_csv_download(data: List[Dict[str, Any]], filename: str = None) -> Tuple[bytes, str]:
        """Generate CSV file for download"""
        try:
            if not filename:
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create CSV string
            csv_string = df.to_csv(index=False)
            
            return csv_string.encode('utf-8'), filename
            
        except Exception as e:
            logger.error(f"Error generating CSV: {e}")
            return b"", "error.csv"
    
    @staticmethod
    def generate_json_download(data: List[Dict[str, Any]], filename: str = None) -> Tuple[bytes, str]:
        """Generate JSON file for download"""
        try:
            if not filename:
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to JSON string
            json_string = json.dumps(data, indent=2, default=str)
            
            return json_string.encode('utf-8'), filename
            
        except Exception as e:
            logger.error(f"Error generating JSON: {e}")
            return b"", "error.json"
    
    @staticmethod
    def generate_excel_download(data: List[Dict[str, Any]], filename: str = None) -> Tuple[bytes, str]:
        """Generate Excel file for download"""
        try:
            if not filename:
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create Excel buffer
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            return buffer.getvalue(), filename
            
        except ImportError:
            logger.warning("openpyxl library not installed")
            return b"", "error.xlsx"
        except Exception as e:
            logger.error(f"Error generating Excel: {e}")
            return b"", "error.xlsx"
    
    @staticmethod
    def parse_user_agent(user_agent: str) -> Dict[str, str]:
        """Parse user agent string"""
        try:
            from user_agents import parse
            
            ua = parse(user_agent)
            
            return {
                'browser': ua.browser.family,
                'browser_version': ua.browser.version_string,
                'os': ua.os.family,
                'os_version': ua.os.version_string,
                'device': ua.device.family,
                'is_mobile': ua.is_mobile,
                'is_tablet': ua.is_tablet,
                'is_pc': ua.is_pc
            }
            
        except ImportError:
            logger.warning("user_agents library not installed")
            return {
                'browser': 'Unknown',
                'browser_version': 'Unknown',
                'os': 'Unknown',
                'os_version': 'Unknown',
                'device': 'Unknown',
                'is_mobile': False,
                'is_tablet': False,
                'is_pc': True
            }
        except Exception as e:
            logger.error(f"Error parsing user agent: {e}")
            return {
                'browser': 'Unknown',
                'browser_version': 'Unknown',
                'os': 'Unknown',
                'os_version': 'Unknown',
                'device': 'Unknown',
                'is_mobile': False,
                'is_tablet': False,
                'is_pc': True
            }
    
    @staticmethod
    def get_client_ip(request) -> str:
        """Get client IP address from request"""
        try:
            # Check for proxy headers
            if request.headers.get('X-Forwarded-For'):
                return request.headers.get('X-Forwarded-For').split(',')[0].strip()
            elif request.headers.get('X-Real-IP'):
                return request.headers.get('X-Real-IP')
            else:
                return request.remote_addr or '127.0.0.1'
        except Exception:
            return '127.0.0.1'
    
    @staticmethod
    def get_geolocation(ip_address: str) -> Dict[str, Any]:
        """Get geolocation information for IP address"""
        try:
            # Use free IP geolocation service
            response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    return {
                        'country': data.get('country', 'Unknown'),
                        'country_code': data.get('countryCode', 'Unknown'),
                        'region': data.get('regionName', 'Unknown'),
                        'city': data.get('city', 'Unknown'),
                        'latitude': data.get('lat', 0),
                        'longitude': data.get('lon', 0),
                        'timezone': data.get('timezone', 'Unknown'),
                        'isp': data.get('isp', 'Unknown')
                    }
            
            return {
                'country': 'Unknown',
                'country_code': 'Unknown',
                'region': 'Unknown',
                'city': 'Unknown',
                'latitude': 0,
                'longitude': 0,
                'timezone': 'Unknown',
                'isp': 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting geolocation: {e}")
            return {
                'country': 'Unknown',
                'country_code': 'Unknown',
                'region': 'Unknown',
                'city': 'Unknown',
                'latitude': 0,
                'longitude': 0,
                'timezone': 'Unknown',
                'isp': 'Unknown'
            }
    
    @staticmethod
    def is_spam(content: str, spam_keywords: List[str] = None) -> bool:
        """Check if content contains spam keywords"""
        if not content:
            return False
        
        if spam_keywords is None:
            spam_keywords = [
                'viagra', 'cialis', 'casino', 'poker', 'lottery',
                'winner', 'congratulations', 'million dollars',
                'nigerian prince', 'inheritance', 'tax refund'
            ]
        
        content_lower = content.lower()
        
        for keyword in spam_keywords:
            if keyword.lower() in content_lower:
                return True
        
        return False
    
    @staticmethod
    def rate_limit_key(identifier: str, action: str) -> str:
        """Generate rate limiting key"""
        return f"rate_limit:{action}:{identifier}"
    
    def check_rate_limit(self, identifier: str, action: str, 
                        limit: int = 10, window: int = 3600) -> Tuple[bool, int]:
        """Check if request is within rate limit"""
        if not self.redis_client:
            return True, limit  # Allow if Redis not available
        
        try:
            key = self.rate_limit_key(identifier, action)
            
            # Get current count
            current = self.redis_client.get(key)
            current_count = int(current) if current else 0
            
            if current_count >= limit:
                # Get TTL for retry-after header
                ttl = self.redis_client.ttl(key)
                return False, ttl
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            pipe.execute()
            
            return True, limit - current_count - 1
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True, limit  # Allow on error
    
    def cache_set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set cache value"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.setex(key, expire, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def cache_delete(self, key: str) -> bool:
        """Delete cache key"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False
    
    @staticmethod
    def generate_sitemap(urls: List[Dict[str, Any]]) -> str:
        """Generate XML sitemap"""
        try:
            xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
            xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            
            for url_info in urls:
                xml += '  <url>\n'
                xml += f'    <loc>{url_info["url"]}</loc>\n'
                
                if 'lastmod' in url_info:
                    xml += f'    <lastmod>{url_info["lastmod"]}</lastmod>\n'
                
                if 'changefreq' in url_info:
                    xml += f'    <changefreq>{url_info["changefreq"]}</changefreq>\n'
                
                if 'priority' in url_info:
                    xml += f'    <priority>{url_info["priority"]}</priority>\n'
                
                xml += '  </url>\n'
            
            xml += '</urlset>'
            
            return xml
            
        except Exception as e:
            logger.error(f"Error generating sitemap: {e}")
            return ""
    
    @staticmethod
    def generate_robots_txt(allow_paths: List[str] = None, 
                          disallow_paths: List[str] = None,
                          sitemap_url: str = None) -> str:
        """Generate robots.txt content"""
        try:
            robots = "User-agent: *\n"
            
            if disallow_paths:
                for path in disallow_paths:
                    robots += f"Disallow: {path}\n"
            
            if allow_paths:
                for path in allow_paths:
                    robots += f"Allow: {path}\n"
            
            if sitemap_url:
                robots += f"\nSitemap: {sitemap_url}\n"
            
            return robots
            
        except Exception as e:
            logger.error(f"Error generating robots.txt: {e}")
            return "User-agent: *\nDisallow: /"

# Flask decorators and utilities
def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(action: str, limit: int = 10, window: int = 3600):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client identifier (IP + user agent)
            identifier = f"{request.remote_addr}:{request.headers.get('User-Agent', '')}"
            
            # Check rate limit
            helpers = WebHelpers()
            allowed, remaining = helpers.check_rate_limit(identifier, action, limit, window)
            
            if not allowed:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': remaining
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def cache_result(expire: int = 3600):
    """Cache function result decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"cache:{f.__name__}:{hash(str(args))}:{hash(str(kwargs))}"
            
            helpers = WebHelpers()
            
            # Check cache
            cached_result = helpers.cache_get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Cache result
            helpers.cache_set(cache_key, result, expire)
            
            return result
        return decorated_function
    return decorator

# API response utilities
def success_response(data: Any = None, message: str = "Success", 
                    status_code: int = 200) -> Tuple[Dict[str, Any], int]:
    """Create success response"""
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    }, status_code

def error_response(message: str = "Error", error_code: str = None,
                  status_code: int = 400, details: Any = None) -> Tuple[Dict[str, Any], int]:
    """Create error response"""
    return {
        'success': False,
        'message': message,
        'error_code': error_code,
        'details': details,
        'timestamp': datetime.utcnow().isoformat()
    }, status_code

def validation_error(errors: Dict[str, List[str]]) -> Tuple[Dict[str, Any], int]:
    """Create validation error response"""
    return {
        'success': False,
        'message': 'Validation failed',
        'errors': errors,
        'timestamp': datetime.utcnow().isoformat()
    }, 400

# Pagination utilities
def paginate_query(query, page: int = 1, per_page: int = 20):
    """Paginate SQLAlchemy query"""
    try:
        page = max(1, page)
        per_page = max(1, min(per_page, 100))  # Limit to 100 items per page
        
        total = query.count()
        items = query.offset((page - 1) * per_page).limit(per_page).all()
        
        total_pages = (total + per_page - 1) // per_page
        
        return {
            'items': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'has_prev': page > 1,
            'has_next': page < total_pages
        }
        
    except Exception as e:
        logger.error(f"Error paginating query: {e}")
        return {
            'items': [],
            'total': 0,
            'page': 1,
            'per_page': per_page,
            'total_pages': 0,
            'has_prev': False,
            'has_next': False
        }

# Data validation utilities
def validate_bet_data(data: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
    """Validate bet data"""
    errors = {}
    
    # Required fields
    required_fields = ['match_id', 'prediction_type', 'stake', 'odds']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.setdefault(field, []).append(f"{field.replace('_', ' ').title()} is required")
    
    # Validate stake
    if 'stake' in data:
        try:
            stake = float(data['stake'])
            if stake <= 0:
                errors.setdefault('stake', []).append("Stake must be positive")
            if stake > 10000:  # Max stake
                errors.setdefault('stake', []).append("Stake exceeds maximum allowed")
        except (ValueError, TypeError):
            errors.setdefault('stake', []).append("Stake must be a valid number")
    
    # Validate odds
    if 'odds' in data:
        try:
            odds = float(data['odds'])
            if odds < 1.01 or odds > 1000:
                errors.setdefault('odds', []).append("Odds must be between 1.01 and 1000")
        except (ValueError, TypeError):
            errors.setdefault('odds', []).append("Odds must be a valid number")
    
    # Validate prediction type
    valid_types = ['match_result', 'over_under', 'both_teams_to_score', 'correct_score']
    if 'prediction_type' in data and data['prediction_type'] not in valid_types:
        errors.setdefault('prediction_type', []).append(f"Prediction type must be one of: {', '.join(valid_types)}")
    
    return len(errors) == 0, errors

def validate_user_data(data: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
    """Validate user registration data"""
    errors = {}
    
    # Validate username
    if 'username' in data:
        is_valid, message = WebHelpers.validate_username(data['username'])
        if not is_valid:
            errors.setdefault('username', []).append(message)
    
    # Validate email
    if 'email' in data:
        if not WebHelpers.validate_email(data['email']):
            errors.setdefault('email', []).append("Invalid email format")
    
    # Validate password
    if 'password' in data:
        is_valid, message = WebHelpers.validate_password(data['password'])
        if not is_valid:
            errors.setdefault('password', []).append(message)
    
    # Check password confirmation
    if 'password' in data and 'confirm_password' in data:
        if data['password'] != data['confirm_password']:
            errors.setdefault('confirm_password', []).append("Passwords do not match")
    
    return len(errors) == 0, errors

# Security utilities
def generate_csrf_token() -> str:
    """Generate CSRF token"""
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, session_token: str) -> bool:
    """Verify CSRF token"""
    return secrets.compare_digest(token, session_token)

def sanitize_html(html: str) -> str:
    """Sanitize HTML content"""
    allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                   'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'img']
    
    allowed_attributes = {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title', 'width', 'height']
    }
    
    return bleach.clean(html, tags=allowed_tags, attributes=allowed_attributes)

# Data processing utilities
def process_match_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process and clean match data"""
    processed = {}
    
    # Basic fields
    processed['match_id'] = data.get('match_id', '')
    processed['home_team'] = data.get('home_team', '').strip()
    processed['away_team'] = data.get('away_team', '').strip()
    processed['league'] = data.get('league', '').strip()
    
    # Parse date
    if 'date' in data:
        processed['date'] = WebHelpers.parse_date(data['date'])
    else:
        processed['date'] = None
    
    # Numeric fields
    processed['home_odds'] = float(data.get('home_odds', 0))
    processed['draw_odds'] = float(data.get('draw_odds', 0))
    processed['away_odds'] = float(data.get('away_odds', 0))
    
    # Validate odds
    if processed['home_odds'] <= 0 or processed['draw_odds'] <= 0 or processed['away_odds'] <= 0:
        raise ValueError("Odds must be positive numbers")
    
    return processed

def calculate_betting_stats(bets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive betting statistics"""
    if not bets:
        return {
            'total_bets': 0,
            'won_bets': 0,
            'lost_bets': 0,
            'win_rate': 0,
            'total_stake': 0,
            'total_profit': 0,
            'roi': 0,
            'average_odds': 0,
            'average_stake': 0
        }
    
    total_bets = len(bets)
    won_bets = sum(1 for bet in bets if bet.get('profit', 0) > 0)
    lost_bets = sum(1 for bet in bets if bet.get('profit', 0) < 0)
    
    total_stake = sum(bet.get('stake', 0) for bet in bets)
    total_profit = sum(bet.get('profit', 0) for bet in bets)
    
    win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
    
    odds = [bet.get('odds', 0) for bet in bets if bet.get('odds', 0) > 0]
    average_odds = np.mean(odds) if odds else 0
    
    stakes = [bet.get('stake', 0) for bet in bets]
    average_stake = np.mean(stakes) if stakes else 0
    
    return {
        'total_bets': total_bets,
        'won_bets': won_bets,
        'lost_bets': lost_bets,
        'win_rate': win_rate,
        'total_stake': total_stake,
        'total_profit': total_profit,
        'roi': roi,
        'average_odds': average_odds,
        'average_stake': average_stake
    }

# Example usage
if __name__ == "__main__":
    # Initialize helpers
    helpers = WebHelpers()
    
    # Test some functions
    print("Testing WebHelpers...")
    
    # Test token generation
    token = helpers.generate_secure_token()
    print(f"Generated token: {token}")
    
    # Test password hashing
    password = "TestPassword123!"
    hashed = helpers.hash_password(password)
    print(f"Password hash: {hashed}")
    print(f"Password verification: {helpers.verify_password(password, hashed)}")
    
    # Test email validation
    print(f"Email validation: {helpers.validate_email('test@example.com')}")
    
    # Test username validation
    is_valid, message = helpers.validate_username("test_user")
    print(f"Username validation: {is_valid} - {message}")
    
    # Test currency formatting
    print(f"Currency formatting: {helpers.format_currency(1234.56)}")
    
    # Test time ago
    past_date = datetime.utcnow() - timedelta(hours=2)
    print(f"Time ago: {helpers.time_ago(past_date)}")