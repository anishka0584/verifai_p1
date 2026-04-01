import os
import io
import tempfile
import hashlib
import traceback
import base64
import json
import csv
import mimetypes
import logging
import re
import time
import uuid
import threading
import functools
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, session, send_file, send_from_directory, Response
from flask_cors import CORS
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import requests
from langdetect import detect, DetectorFactory
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from zoneinfo import ZoneInfo  # Python 3.9+

# ========================================================================
# USER CREDENTIALS WITH ROLES - Add/Remove users here
# Format: "username": {"password": "password", "role": "admin" or "user"}
# 
# ROLES:
#   - admin: Full access to all endpoints, can view all users' data
#   - user: Can only verify documents and view their own results
# ========================================================================
USERS = {
    "admin": {
        "password": "********",
        "role": "admin",
        "full_name": "System Administrator"
    },
    "user": {
        "password": "****",
        "role": "user",
        "full_name": "Inital Testing User"
    },
    "extuser": {
        "password": "*****",
        "role": "user",
        "full_name": "Extranal API connections endpoint"
    },
}
# ========================================================================

IST = ZoneInfo("Asia/Kolkata")

# Set seed for consistent language detection
DetectorFactory.seed = 0

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('tempercheck.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
APP_SECRET = os.environ.get("APP_SECRET", "*****")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "******")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "pdf"}
MAX_WORKERS = 4
GEMINI_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging files
CSV_AUDIT_FILE = "verifai_audit_log.csv"
JSON_LOG_FILE = "verifai_json_log.json"

# CSV Headers for audit log
CSV_HEADERS = [
    "Request ID",
    "Timestamp",
    "Username",
    "Document Name",
    "Document Type",
    "Status",
    "Category",
    "Confidence (%)",
    "Reasoning",
    "Extracted Text",
    "Translated Text",
    "Language",
    "Security Features",
    "Format Match",
    "Text Legible",
    "No Tampering",
    "Quality OK",
    "Dates Valid",
    "Processing Time (ms)",
    "Saved File Path",
]

# Storage for requests (in production, use a database)
REQUEST_STORAGE = {}
STORAGE_FILE = "request_storage.json"

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")
else:
    logger.error("GEMINI_API_KEY not found!")

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = APP_SECRET
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS

LAST_REPORT_KEY = "last_report"

# Thread lock for file writing
csv_lock = threading.Lock()
json_lock = threading.Lock()
storage_lock = threading.Lock()

# ========================================================================
# ROLE-BASED AUTHENTICATION FUNCTIONS
# ========================================================================

def check_basic_auth(username, password):
    """Verify username and password against USERS dictionary"""
    if username in USERS:
        user_data = USERS[username]
        return user_data.get("password") == password
    return False

def get_user_role(username):
    """Get the role of a user"""
    if username in USERS:
        return USERS[username].get("role", "user")
    return None

def get_user_info(username):
    """Get full user info"""
    if username in USERS:
        return {
            "username": username,
            "role": USERS[username].get("role", "user"),
            "full_name": USERS[username].get("full_name", username)
        }
    return None

def is_admin(username):
    """Check if user is admin"""
    return get_user_role(username) == "admin"

def get_basic_auth_credentials():
    """Extract credentials from Basic Auth header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None, None
    
    try:
        if auth_header.startswith('Basic '):
            credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
            username, password = credentials.split(':', 1)
            return username, password
    except Exception as e:
        logger.error(f"Failed to parse Basic Auth header: {e}")
    
    return None, None

def is_authenticated():
    """Check if request is authenticated via session or Basic Auth"""
    # Check session first (for web UI)
    if session.get('authenticated') and session.get('username'):
        username = session.get('username')
        role = session.get('role', get_user_role(username))
        return True, username, role
    
    # Check Basic Auth (for API/curl)
    username, password = get_basic_auth_credentials()
    if username and password and check_basic_auth(username, password):
        role = get_user_role(username)
        return True, username, role
    
    return False, None, None

def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        authenticated, username, role = is_authenticated()
        if not authenticated:
            return Response(
                json.dumps({"error": "Authentication required", "message": "Please provide valid credentials"}),
                status=401,
                mimetype='application/json',
                headers={'WWW-Authenticate': 'Basic realm="VerifAI API"'}
            )
        # Add user info to request context
        request.authenticated_user = username
        request.user_role = role
        request.is_admin = (role == "admin")
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    """Decorator to require admin role for endpoints"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        authenticated, username, role = is_authenticated()
        if not authenticated:
            return Response(
                json.dumps({"error": "Authentication required", "message": "Please provide valid credentials"}),
                status=401,
                mimetype='application/json',
                headers={'WWW-Authenticate': 'Basic realm="VerifAI API"'}
            )
        if role != "admin":
            logger.warning(f"[AUTH] Non-admin user '{username}' attempted to access admin endpoint")
            return Response(
                json.dumps({"error": "Access denied", "message": "Admin privileges required for this endpoint"}),
                status=403,
                mimetype='application/json'
            )
        # Add user info to request context
        request.authenticated_user = username
        request.user_role = role
        request.is_admin = True
        return f(*args, **kwargs)
    return decorated

# ---------- File Saving Functions ----------
def save_uploaded_file(file_path, request_id, original_filename):
    """Save uploaded file to uploads folder with request_id as filename."""
    try:
        ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "bin"
        saved_filename = f"{request_id}.{ext}"
        saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        
        import shutil
        shutil.copy2(file_path, saved_path)
        
        logger.info(f"[FILE_SAVE] Saved file: {original_filename} -> {saved_path}")
        return saved_path
    
    except Exception as e:
        logger.error(f"[FILE_SAVE] Failed to save file: {e}")
        return None

def save_file_bytes(file_bytes, request_id, extension):
    """Save file bytes directly to uploads folder."""
    try:
        saved_filename = f"{request_id}.{extension}"
        saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        
        with open(saved_path, 'wb') as f:
            f.write(file_bytes)
        
        logger.info(f"[FILE_SAVE] Saved file bytes: {saved_path}")
        return saved_path
    
    except Exception as e:
        logger.error(f"[FILE_SAVE] Failed to save file bytes: {e}")
        return None

# ---------- Enhanced Logging Functions ----------
def append_csv_log_from_result(request_id, result, username="unknown", saved_path=""):
    """Append analysis result to CSV audit log"""
    checks = result.get("verificationChecks", {})

    row = {
        "Request ID": request_id,
        "Timestamp": result.get("processDateTime", ""),
        "Username": username,
        "Document Name": result.get("documentName", ""),
        "Document Type": result.get("documentType", ""),
        "Status": result.get("status", ""),
        "Category": result.get("category", ""),
        "Confidence (%)": result.get("confidence", ""),
        "Reasoning": result.get("reasoning", "")[:500] if result.get("reasoning") else "",
        "Extracted Text": result.get("extractedText", "")[:1000] if result.get("extractedText") else "",
        "Translated Text": result.get("translatedText", "")[:1000] if result.get("translatedText") else "",
        "Language": result.get("detectedLanguage", ""),
        "Security Features": "Yes" if checks.get("has_security_features") else "No",
        "Format Match": "Yes" if checks.get("format_matches_template") else "No",
        "Text Legible": "Yes" if checks.get("text_is_legible") else "No",
        "No Tampering": "Yes" if checks.get("no_tampering_signs") else "No",
        "Quality OK": "Yes" if checks.get("quality_acceptable") else "No",
        "Dates Valid": "Yes" if checks.get("dates_are_valid") else "No",
        "Processing Time (ms)": result.get("processingTimeMs", ""),
        "Saved File Path": saved_path,
    }

    file_exists = os.path.exists(CSV_AUDIT_FILE)
    with csv_lock:
        with open(CSV_AUDIT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

def append_json_log(log_data):
    """Append a log entry to JSON log file"""
    try:
        logs = []
        
        if os.path.exists(JSON_LOG_FILE):
            with json_lock:
                try:
                    with open(JSON_LOG_FILE, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = []
                except (json.JSONDecodeError, FileNotFoundError):
                    logs = []
        
        logs.append(log_data)
        
        with json_lock:
            with open(JSON_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"JSON log entry added: {log_data.get('request_id', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to write JSON log: {e}")

def log_request(endpoint, request_id, client_ip, username="unknown", filename="", 
                doc_type="", file_size=0, processing_time=0, status="STARTED", 
                error="", saved_path=""):
    """Log request details to JSON"""
    timestamp = datetime.now(IST).isoformat()
    
    json_entry = {
        "timestamp": timestamp,
        "request_id": request_id,
        "endpoint": endpoint,
        "client_ip": client_ip,
        "username": username,
        "filename": filename,
        "document_type": doc_type,
        "file_size_bytes": file_size,
        "processing_time_ms": processing_time,
        "status": status,
        "error": error,
        "saved_file_path": saved_path,
        "user_agent": request.headers.get("User-Agent", ""),
        "http_method": request.method
    }
    
    append_json_log(json_entry)
    return json_entry

# ---------- Storage Functions ----------
def load_storage():
    """Load request storage from file"""
    global REQUEST_STORAGE
    try:
        if os.path.exists(STORAGE_FILE):
            with storage_lock:
                with open(STORAGE_FILE, 'r') as f:
                    REQUEST_STORAGE = json.load(f)
            logger.info(f"Loaded {len(REQUEST_STORAGE)} stored requests")
    except Exception as e:
        logger.error(f"Failed to load storage: {e}")
        REQUEST_STORAGE = {}

def save_storage():
    """Save request storage to file"""
    try:
        with storage_lock:
            with open(STORAGE_FILE, 'w') as f:
                json.dump(REQUEST_STORAGE, f, indent=2)
        logger.debug(f"Saved {len(REQUEST_STORAGE)} requests to storage")
    except Exception as e:
        logger.error(f"Failed to save storage: {e}")

def generate_request_id():
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def store_request(request_id, data):
    """Store request data"""
    REQUEST_STORAGE[request_id] = data
    save_storage()

def get_request(request_id):
    """Retrieve request data"""
    return REQUEST_STORAGE.get(request_id)

def get_user_requests(username):
    """Get all requests for a specific user"""
    user_requests = {}
    for req_id, data in REQUEST_STORAGE.items():
        if data.get("username") == username:
            user_requests[req_id] = data
    return user_requests

def get_all_requests():
    """Get all requests (admin only)"""
    return REQUEST_STORAGE

def get_user_statistics(username=None):
    """Get statistics for a user or all users (if username is None)"""
    requests_to_analyze = REQUEST_STORAGE if username is None else get_user_requests(username)
    
    stats = {
        "total_requests": len(requests_to_analyze),
        "legit": 0,
        "suspicious": 0,
        "not_legit": 0,
        "error": 0,
        "by_document_type": {},
        "by_category": {},
        "avg_confidence": 0,
        "avg_processing_time": 0,
        "recent_requests": []
    }
    
    total_confidence = 0
    total_processing_time = 0
    confidence_count = 0
    
    sorted_requests = sorted(
        requests_to_analyze.items(), 
        key=lambda x: x[1].get("timestamp", ""), 
        reverse=True
    )
    
    for req_id, data in sorted_requests:
        result = data.get("result", {})
        status = result.get("status", "ERROR")
        category = result.get("category", "ERROR")
        doc_type = data.get("document_type", "Unknown")
        confidence = result.get("confidence", 0)
        processing_time = data.get("processing_time_ms", 0)
        
        # Count statuses
        if status == "LEGIT":
            stats["legit"] += 1
        elif status == "SUSPICIOUS":
            stats["suspicious"] += 1
        elif status == "NOT_LEGIT":
            stats["not_legit"] += 1
        else:
            stats["error"] += 1
        
        # Count by document type
        if doc_type not in stats["by_document_type"]:
            stats["by_document_type"][doc_type] = {"total": 0, "legit": 0, "suspicious": 0, "not_legit": 0}
        stats["by_document_type"][doc_type]["total"] += 1
        if status == "LEGIT":
            stats["by_document_type"][doc_type]["legit"] += 1
        elif status == "SUSPICIOUS":
            stats["by_document_type"][doc_type]["suspicious"] += 1
        elif status == "NOT_LEGIT":
            stats["by_document_type"][doc_type]["not_legit"] += 1
        
        # Count by category
        if category not in stats["by_category"]:
            stats["by_category"][category] = 0
        stats["by_category"][category] += 1
        
        # Calculate averages
        if confidence and isinstance(confidence, (int, float)):
            total_confidence += confidence
            confidence_count += 1
        
        if processing_time and isinstance(processing_time, (int, float)):
            total_processing_time += processing_time
        
        # Add to recent requests (max 10)
        if len(stats["recent_requests"]) < 10:
            stats["recent_requests"].append({
                "request_id": req_id,
                "timestamp": data.get("timestamp"),
                "document_name": result.get("documentName"),
                "document_type": doc_type,
                "status": status,
                "category": category,
                "confidence": confidence
            })
    
    # Calculate averages
    if confidence_count > 0:
        stats["avg_confidence"] = round(total_confidence / confidence_count, 2)
    if stats["total_requests"] > 0:
        stats["avg_processing_time"] = round(total_processing_time / stats["total_requests"], 2)
    
    return stats

# ---------- Document Templates ----------
DOCUMENT_TEMPLATES = {
    "Passport": {
        "front": {
            "required_fields": ["passport number", "surname", "given names", "date of birth", "sex", "date of issue", "date of expiry", "issuing state"],
            "keywords": ["passport", "travel document", "issuing state"]
        },
        "back": {
            "expected_elements": ["machine readable zone", "barcode", "instructions", "security patterns"],
            "keywords": ["authority", "endorsement", "instructions"]
        }
    },
    "PAN Card": {
        "front": {
            "required_fields": ["pan number", "name", "date of birth"],
            "keywords": ["income tax", "permanent account number"]
        },
        "back": {
            "expected_elements": ["qr code", "signature strip", "instructions"],
            "keywords": ["signature", "instructions"]
        }
    },
    "Voter ID Card": {
        "front": {
            "required_fields": ["voter id", "name", "date of birth"],
            "keywords": ["election commission", "voter"]
        },
        "back": {
            "expected_elements": ["address", "barcode", "issuing authority"],
            "keywords": ["address", "constituency"]
        }
    },
    "Driving License": {
        "front": {
            "required_fields": ["license number", "name", "date of birth", "date of issue", "date of expiry"],
            "keywords": ["driving license", "dl"]
        },
        "back": {
            "expected_elements": ["address", "vehicle categories", "qr code"],
            "keywords": ["address", "transport authority"]
        }
    },
    "Bank Statement": {
        "front": {
            "required_fields": ["bank name", "account number", "account holder", "statement period"],
            "keywords": ["bank", "statement", "account"]
        },
        "back": {
            "expected_elements": ["terms", "disclaimer", "barcode", "continuation notice"],
            "keywords": ["terms", "conditions", "disclaimer"]
        }
    },
    "Death Certificate": {
        "front": {
            "required_fields": ["deceased name", "date of death", "age", "registration number"],
            "keywords": ["death certificate", "registration"]
        },
        "back": {
            "expected_elements": ["issuing authority", "seal", "instructions"],
            "keywords": ["authority", "registrar"]
        }
    },
    "Bills": {
        "front": {
            "required_fields": ["customer name", "account number", "bill date", "amount due"],
            "keywords": ["bill", "invoice"]
        },
        "back": {
            "expected_elements": ["payment instructions", "barcode", "customer service details"],
            "keywords": ["payment", "instructions"]
        }
    }
}

# ---------- Utilities ----------
def allowed_filename(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return ""

def combine_bank_statement_results(first, last):
    """Conservative combination logic. One bad page = downgrade."""
    categories_priority = ["FRAUD", "FORMAT_INVALID", "DAMAGED", "PHOTOCOPY", "SUSPICIOUS", "LEGIT"]

    def worst_category(a, b):
        return min(a, b, key=lambda x: categories_priority.index(x) if x in categories_priority else 0)

    combined_category = worst_category(first.get("category", "ERROR"), last.get("category", "ERROR"))

    if combined_category == "LEGIT":
        status = "LEGIT"
    elif combined_category in ["SUSPICIOUS", "PHOTOCOPY"]:
        status = "SUSPICIOUS"
    else:
        status = "NOT_LEGIT"

    combined_confidence = min(first.get("confidence", 0), last.get("confidence", 0))

    return {
        "status": status,
        "category": combined_category,
        "confidence": combined_confidence,
        "reasoning": (
            f"First page analysis: {first.get('reasoning', 'N/A')} | "
            f"Last page analysis: {last.get('reasoning', 'N/A')}"
        ),
        "extracted_text": (
            first.get("extracted_text", "") + "\n---\n" + last.get("extracted_text", "")
        ),
        "translated_text": (
            first.get("translated_text", "") + "\n---\n" + last.get("translated_text", "")
        ),
        "detected_language": first.get("detected_language", "unknown"),
        "verification_checks": {
            "has_security_features": first.get("verification_checks", {}).get("has_security_features", False)
                                      and last.get("verification_checks", {}).get("has_security_features", False),
            "format_matches_template": first.get("verification_checks", {}).get("format_matches_template", False)
                                       and last.get("verification_checks", {}).get("format_matches_template", False),
            "text_is_legible": first.get("verification_checks", {}).get("text_is_legible", False)
                               and last.get("verification_checks", {}).get("text_is_legible", False),
            "no_tampering_signs": first.get("verification_checks", {}).get("no_tampering_signs", False)
                                  and last.get("verification_checks", {}).get("no_tampering_signs", False),
            "quality_acceptable": first.get("verification_checks", {}).get("quality_acceptable", False)
                                   and last.get("verification_checks", {}).get("quality_acceptable", False),
            "dates_are_valid": last.get("verification_checks", {}).get("dates_are_valid", False)
        }
    }

def extract_first_last_images_from_pdf(pdf_path):
    """Returns (first_page_img_bytes, last_page_img_bytes)"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            return None, None

        first_pix = doc[0].get_pixmap(dpi=200)
        first_img = first_pix.tobytes("jpeg")

        if total_pages > 1:
            last_pix = doc[total_pages - 1].get_pixmap(dpi=200)
            last_img = last_pix.tobytes("jpeg")
        else:
            last_img = first_img

        doc.close()
        return first_img, last_img

    except Exception as e:
        logger.error(f"Failed to extract first/last pages: {e}")
        return None, None

def data_uri_to_file(data_uri, out_dir=None):
    if not data_uri.startswith("data:"):
        raise ValueError("Not a data URI")
    header, b64 = data_uri.split(",", 1)
    mimetype = header.split(";")[0].split(":", 1)[1]
    ext = mimetypes.guess_extension(mimetype) or ".bin"
    if ext == ".jpe":
        ext = ".jpg"
    out_dir = out_dir or tempfile.gettempdir()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=out_dir)
    tmp.write(base64.b64decode(b64))
    tmp.flush()
    tmp.close()
    return tmp.name, "uploaded" + ext

def download_file_from_url(url):
    """Download file from URL and return temp file path"""
    try:
        logger.info(f"Downloading file from URL: {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        filename = url.split("/")[-1].split("?")[0]
        if "Content-Disposition" in response.headers:
            cd = response.headers["Content-Disposition"]
            if "filename=" in cd:
                filename = cd.split("filename=")[1].strip('"')
        
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "bin"
        if ext not in ALLOWED_EXT:
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                ext = "pdf"
            elif "image" in content_type:
                ext = "jpg"
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        
        logger.info(f"Downloaded file: {filename} ({os.path.getsize(tmp.name)} bytes)")
        return tmp.name, filename
        
    except Exception as e:
        logger.error(f"Failed to download file from URL: {e}")
        raise

# ---------- PDF Report Generation ----------
def generate_pdf_report(results, request_id):
    """Generate PDF report for analysis results"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        story.append(Paragraph("VerifAI Document Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(f"<b>Request ID:</b> {request_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Documents:</b> {len(results)}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        legit_count = sum(1 for r in results if r.get('status') == 'LEGIT')
        suspicious_count = sum(1 for r in results if r.get('status') == 'SUSPICIOUS')
        not_legit_count = sum(1 for r in results if r.get('status') == 'NOT_LEGIT')
        
        summary_data = [
            ['Status', 'Count'],
            ['LEGIT', str(legit_count)],
            ['SUSPICIOUS', str(suspicious_count)],
            ['NOT LEGIT', str(not_legit_count)]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Summary", heading_style))
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        story.append(Paragraph("Detailed Results", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        for idx, result in enumerate(results, 1):
            story.append(Paragraph(f"<b>Document {idx}: {result.get('documentName', 'Unknown')}</b>", styles['Heading3']))
            
            status = result.get('status', 'UNKNOWN')
            if status == 'LEGIT':
                status_color = colors.green
            elif status == 'SUSPICIOUS':
                status_color = colors.orange
            else:
                status_color = colors.red
            
            doc_data = [
                ['Field', 'Value'],
                ['Document Type', result.get('documentType', 'N/A')],
                ['Status', result.get('status', 'N/A')],
                ['Category', result.get('category', 'N/A')],
                ['Confidence', f"{result.get('confidence', 0)}%"],
                ['Detected Language', result.get('detectedLanguage', 'unknown')],
            ]
            
            doc_table = Table(doc_data, colWidths=[2*inch, 4*inch])
            doc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (1, 2), (1, 2), status_color),
                ('TEXTCOLOR', (1, 2), (1, 2), colors.whitesmoke),
            ]))
            
            story.append(doc_table)
            story.append(Spacer(1, 0.1*inch))
            
            reasoning = result.get('reasoning', 'No reasoning provided')
            story.append(Paragraph(f"<b>Analysis:</b>", styles['Normal']))
            story.append(Paragraph(reasoning[:2000], styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            checks = result.get('verificationChecks', {})
            if checks:
                check_data = [['Verification Check', 'Result']]
                check_data.append(['Security Features', 'PASS' if checks.get('has_security_features') else 'FAIL'])
                check_data.append(['Format Match', 'PASS' if checks.get('format_matches_template') else 'FAIL'])
                check_data.append(['Text Legible', 'PASS' if checks.get('text_is_legible') else 'FAIL'])
                check_data.append(['No Tampering', 'PASS' if checks.get('no_tampering_signs') else 'FAIL'])
                check_data.append(['Quality OK', 'PASS' if checks.get('quality_acceptable') else 'FAIL'])
                check_data.append(['Dates Valid', 'PASS' if checks.get('dates_are_valid') else 'FAIL'])
                
                check_table = Table(check_data, colWidths=[3*inch, 1*inch])
                check_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(Paragraph("<b>Verification Checks:</b>", styles['Normal']))
                story.append(check_table)
            
            if idx < len(results):
                story.append(PageBreak())
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        logger.error(traceback.format_exc())
        raise

def generate_user_aggregate_report(username, stats, user_requests):
    """Generate aggregate PDF report for a user"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        subtitle_style = ParagraphStyle(
            'SubTitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4a5568'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Title
        story.append(Paragraph("VerifAI User Activity Report", title_style))
        
        user_info = get_user_info(username)
        full_name = user_info.get("full_name", username) if user_info else username
        story.append(Paragraph(f"User: {full_name} ({username})", subtitle_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Verifications:</b> {stats['total_requests']}", styles['Normal']))
        story.append(Paragraph(f"<b>Average Confidence:</b> {stats['avg_confidence']}%", styles['Normal']))
        story.append(Paragraph(f"<b>Average Processing Time:</b> {stats['avg_processing_time']}ms", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary Statistics
        story.append(Paragraph("Verification Summary", heading_style))
        
        summary_data = [
            ['Status', 'Count', 'Percentage'],
            ['LEGIT', str(stats['legit']), f"{(stats['legit']/max(stats['total_requests'],1)*100):.1f}%"],
            ['SUSPICIOUS', str(stats['suspicious']), f"{(stats['suspicious']/max(stats['total_requests'],1)*100):.1f}%"],
            ['NOT LEGIT', str(stats['not_legit']), f"{(stats['not_legit']/max(stats['total_requests'],1)*100):.1f}%"],
            ['ERROR', str(stats['error']), f"{(stats['error']/max(stats['total_requests'],1)*100):.1f}%"],
            ['TOTAL', str(stats['total_requests']), '100%']
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#48bb78')),  # Green for LEGIT
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#ed8936')),  # Orange for SUSPICIOUS
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f56565')),  # Red for NOT_LEGIT
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#a0aec0')),  # Gray for ERROR
            ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#4a5568')),  # Dark for TOTAL
            ('TEXTCOLOR', (0, 5), (-1, 5), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 5), (-1, 5), 'Helvetica-Bold'),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # By Document Type
        if stats['by_document_type']:
            story.append(Paragraph("By Document Type", heading_style))
            
            doc_type_data = [['Document Type', 'Total', 'Legit', 'Suspicious', 'Not Legit']]
            for doc_type, counts in stats['by_document_type'].items():
                doc_type_data.append([
                    doc_type,
                    str(counts['total']),
                    str(counts['legit']),
                    str(counts['suspicious']),
                    str(counts['not_legit'])
                ])
            
            doc_type_table = Table(doc_type_data, colWidths=[2*inch, 1*inch, 1*inch, 1.2*inch, 1.2*inch])
            doc_type_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#edf2f7')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(doc_type_table)
            story.append(Spacer(1, 0.3*inch))
        
        # By Category
        if stats['by_category']:
            story.append(Paragraph("By Category", heading_style))
            
            category_data = [['Category', 'Count']]
            for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
                category_data.append([category, str(count)])
            
            category_table = Table(category_data, colWidths=[3*inch, 1.5*inch])
            category_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#edf2f7')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(category_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Recent Requests
        if stats['recent_requests']:
            story.append(PageBreak())
            story.append(Paragraph("Recent Verifications (Last 10)", heading_style))
            
            recent_data = [['Date/Time', 'Document', 'Type', 'Status', 'Conf.']]
            for req in stats['recent_requests']:
                timestamp = req.get('timestamp', '')[:16] if req.get('timestamp') else 'N/A'
                doc_name = req.get('document_name', 'Unknown')
                if len(doc_name) > 20:
                    doc_name = doc_name[:17] + '...'
                recent_data.append([
                    timestamp,
                    doc_name,
                    req.get('document_type', 'N/A')[:15],
                    req.get('status', 'N/A'),
                    f"{req.get('confidence', 0)}%"
                ])
            
            recent_table = Table(recent_data, colWidths=[1.3*inch, 1.5*inch, 1.2*inch, 1*inch, 0.7*inch])
            recent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#edf2f7')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            
            story.append(recent_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Failed to generate user aggregate report: {e}")
        logger.error(traceback.format_exc())
        raise

# ---------- Gemini Analysis Function ----------
def analyze_document_with_gemini(image_bytes, document_type="", retry_count=0):
    """Use Gemini to analyze document"""
    if not GEMINI_API_KEY or not image_bytes:
        logger.error("Gemini API key not configured or no image bytes provided")
        return {
            "status": "ERROR",
            "category": "ERROR",
            "confidence": 0,
            "reasoning": "API configuration error",
            "extracted_text": "",
            "translated_text": "",
            "detected_language": "unknown",
            "verification_checks": {}
        }
    
    try:
        logger.info(f"[GEMINI] Starting analysis for document type: {document_type} (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
        
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = f"""
You are performing HIGH-RISK KYC DOCUMENT VERIFICATION.
This task requires EXTREME CAUTION. False positives are unacceptable.

DOCUMENT TYPE: {document_type}

You MUST strictly follow the document template for this document type,
including BOTH FRONT and BACK side expectations if applicable.
CRITICAL RULES (NON-NEGOTIABLE):
1. DO NOT assume authenticity.
2. DO NOT infer missing details.
3. DO NOT guess text that is not clearly visible.
4. If evidence is insufficient, you MUST downgrade.
5. LEGIT is allowed ONLY when evidence is strong and tampering risk is minimal.
IMPORTANT:
- A BACK SIDE is NOT invalid.
- A BACK SIDE must still be checked for tampering.
- A document may be FRONT, BACK, or UNKNOWN side.
DOCUMENT SIDE IDENTIFICATION (MANDATORY)
Determine which side of the document is visible.
FRONT SIDE indicators:
- Primary identifiers (name, ID number, account number)
- Issue / expiry dates
- Photograph (if applicable)
- Document title or issuing authority

BACK SIDE indicators:
- Address block
- Instructions / terms
- Barcode / QR / MICR
- Security patterns
- Signature strip

If the side cannot be determined → UNKNOWN.

CATEGORIES (CHOOSE EXACTLY ONE)

LEGIT  
→ Clear document (front or back), readable, correct layout,
   no signs of tampering, matches template expectations

SUSPICIOUS  
→ Real document but incomplete (e.g., back side only),
   minor quality issues, or limited verifiable fields

PHOTOCOPY  
→ Scan or copy with no depth or security features

FORMAT_INVALID  
→ Not a real document, wrong layout, unrelated image,
   heavily cropped, rotated, or blank

FRAUD  
→ Clear evidence of tampering, editing, altered text,
   inconsistent fonts, manipulated numbers or patterns

DAMAGED  
→ Unreadable due to blur, tearing, over/under exposure
OUTPUT FORMAT (STRICT)
Respond ONLY with valid JSON.
{{
    "document_side": "FRONT" | "BACK" | "UNKNOWN",
    "status": "LEGIT" | "SUSPICIOUS" | "NOT_LEGIT",
    "category": "LEGIT" | "SUSPICIOUS" | "PHOTOCOPY" |
                "FORMAT_INVALID" | "FRAUD" | "DAMAGED",
    "confidence": 0-100,
    "reasoning": "Explain findings, explicitly mentioning FRONT/BACK and tampering",
    "extracted_text": "visible text only or empty",
    "detected_language": "language code or unknown",
    "translated_text": "English translation or empty",
    "verification_checks": {{
        "has_security_features": true/false,
        "format_matches_template": true/false,
        "text_is_legible": true/false,
        "no_tampering_signs": true/false,
        "quality_acceptable": true/false,
        "dates_are_valid": true/false
    }}
}}
"""

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_base64},
            prompt
        ])
        
        logger.info(f"[GEMINI] API call successful")
        
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
            
            result = {
                "status": result_data.get("status", "NOT_LEGIT"),
                "category": result_data.get("category", "ERROR"),
                "confidence": result_data.get("confidence", 0),
                "reasoning": result_data.get("reasoning", ""),
                "extracted_text": result_data.get("extracted_text", ""),
                "translated_text": result_data.get("translated_text", result_data.get("extracted_text", "")),
                "detected_language": result_data.get("detected_language", "unknown"),
                "verification_checks": result_data.get("verification_checks", {})
            }
            
            logger.info(f"[GEMINI] Analysis complete - {result['category']}, {result['confidence']}%")
            return result
        else:
            logger.error("[GEMINI] Could not parse JSON")
            return {
                "status": "ERROR",
                "category": "ERROR",
                "confidence": 0,
                "reasoning": "Failed to parse API response",
                "extracted_text": "",
                "translated_text": "",
                "detected_language": "unknown",
                "verification_checks": {}
            }
    
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"[GEMINI] Quota exceeded: {str(e)}")
        
        if retry_count < MAX_RETRIES:
            delay = RETRY_DELAY * (2 ** retry_count)
            logger.warning(f"[GEMINI] Retrying in {delay}s...")
            time.sleep(delay)
            return analyze_document_with_gemini(image_bytes, document_type, retry_count + 1)
        else:
            return {
                "status": "ERROR",
                "category": "ERROR",
                "confidence": 0,
                "reasoning": "API quota exceeded",
                "extracted_text": "",
                "translated_text": "",
                "detected_language": "unknown",
                "verification_checks": {}
            }
    
    except Exception as e:
        logger.error(f"[GEMINI] Error: {str(e)}")
        
        if retry_count < MAX_RETRIES:
            delay = RETRY_DELAY * (2 ** retry_count)
            time.sleep(delay)
            return analyze_document_with_gemini(image_bytes, document_type, retry_count + 1)
        else:
            return {
                "status": "ERROR",
                "category": "ERROR",
                "confidence": 0,
                "reasoning": f"API error: {str(e)}",
                "extracted_text": "",
                "translated_text": "",
                "detected_language": "unknown",
                "verification_checks": {}
            }

# ---------- File Processing Function ----------
def process_single_file(file_data, document_type, request_id=None, username="unknown"):
    """Process a single file"""
    file_id = file_data.get("id", request_id)
    filename = file_data.get("name")
    file_path = file_data.get("path")
    
    logger.info(f"[PROCESS:{file_id}] Starting: {filename} (User: {username})")
    
    start_time = time.time()
    saved_path = ""
    
    try:
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
        img_bytes = None
        analysis_result = None
        
        # Save the uploaded file first
        saved_path = save_uploaded_file(file_path, file_id, filename)
        
        if ext == "pdf" and document_type.lower() == "bank statement":
            first_img, last_img = extract_first_last_images_from_pdf(file_path)

            if not first_img or not last_img:
                raise Exception("Could not extract first/last pages from PDF")

            first_result = analyze_document_with_gemini(first_img, document_type + " (First Page)")
            last_result = analyze_document_with_gemini(last_img, document_type + " (Last Page)")
            analysis_result = combine_bank_statement_results(first_result, last_result)

            doc = fitz.open(file_path)
            if len(doc) > 0:
                pix = doc[0].get_pixmap()
                img_bytes = pix.tobytes("jpeg")
            doc.close()
            
        elif ext == "pdf":
            doc = fitz.open(file_path)
            if len(doc) == 0:
                raise Exception("Empty PDF")

            pix = doc[0].get_pixmap(dpi=200)
            img_bytes = pix.tobytes("jpeg")
            doc.close()
            
            analysis_result = analyze_document_with_gemini(img_bytes, document_type)
        else:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            
            analysis_result = analyze_document_with_gemini(img_bytes, document_type)
        
        if not analysis_result:
            raise Exception("Analysis returned no result")
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "id": file_id,
            "documentName": filename,
            "documentType": document_type,
            "status": analysis_result["status"],
            "category": analysis_result["category"],
            "confidence": analysis_result["confidence"],
            "reasoning": analysis_result["reasoning"],
            "extractedText": analysis_result["extracted_text"],
            "translatedText": analysis_result["translated_text"],
            "detectedLanguage": analysis_result["detected_language"],
            "verificationChecks": analysis_result["verification_checks"],
            "processDateTime": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "processingTimeMs": f"{processing_time:.2f}",
            "savedFilePath": saved_path
        }
        
        # Log successful result
        append_csv_log_from_result(file_id, result, username, saved_path)
        
        # Store in request storage
        store_request(file_id, {
            "request_id": file_id,
            "timestamp": result["processDateTime"],
            "document_type": document_type,
            "filename": filename,
            "username": username,
            "result": result,
            "processing_time_ms": processing_time,
            "saved_file_path": saved_path
        })
        
        return result
        
    except Exception as e:
        logger.error(f"[PROCESS:{file_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        
        processing_time = (time.time() - start_time) * 1000
        
        error_result = {
            "id": file_id,
            "documentName": filename,
            "documentType": document_type,
            "status": "ERROR",
            "category": "ERROR",
            "confidence": 0,
            "reasoning": f"Processing error: {str(e)}",
            "extractedText": "",
            "translatedText": "",
            "detectedLanguage": "unknown",
            "verificationChecks": {},
            "processDateTime": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "processingTimeMs": f"{processing_time:.2f}",
            "savedFilePath": saved_path
        }
        
        append_csv_log_from_result(file_id, error_result, username, saved_path)
        
        return error_result

# ========================================================================
# AUTHENTICATION ENDPOINTS
# ========================================================================

@app.route("/api/login", methods=["POST"])
def api_login():
    """Login endpoint for web UI"""
    try:
        data = request.get_json(force=True)
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        if check_basic_auth(username, password):
            user_info = get_user_info(username)
            
            session['authenticated'] = True
            session['username'] = username
            session['role'] = user_info['role']
            session.permanent = True
            
            logger.info(f"[AUTH] User logged in: {username} (role: {user_info['role']}) from {request.remote_addr}")
            
            return jsonify({
                "success": True,
                "message": "Login successful",
                "username": username,
                "role": user_info['role'],
                "full_name": user_info['full_name']
            })
        else:
            logger.warning(f"[AUTH] Failed login attempt for user: {username} from {request.remote_addr}")
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        logger.error(f"[AUTH] Login error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Logout endpoint"""
    username = session.get('username', 'unknown')
    session.clear()
    logger.info(f"[AUTH] User logged out: {username}")
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Check authentication status"""
    authenticated, username, role = is_authenticated()
    
    response_data = {
        "authenticated": authenticated,
        "username": username,
        "role": role
    }
    
    if authenticated and username:
        user_info = get_user_info(username)
        if user_info:
            response_data["full_name"] = user_info.get("full_name")
    
    return jsonify(response_data)

# ========================================================================
# USER ENDPOINTS (Available to all authenticated users)
# ========================================================================

@app.route("/api/verify", methods=["POST"])
@require_auth
def api_verify():
    """Unified endpoint for document verification"""
    logger.info("=" * 80)
    username = getattr(request, 'authenticated_user', 'unknown')
    logger.info(f"[API_VERIFY] New verification request from {request.remote_addr} (User: {username})")
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        document_type = None
        filename = None
        file_path = None
        file_size = 0
        
        # ---------- MULTIPART FILE UPLOAD ----------
        if request.files:
            logger.info("[API_VERIFY] Processing multipart upload")
            uploaded_file = request.files.get("file")
            document_type = request.form.get("document_type", "Unknown")
            
            if not uploaded_file:
                error_msg = "No file provided"
                log_request("/api/verify", request_id, request.remote_addr, username,
                           status="ERROR", error=error_msg)
                return jsonify({"error": error_msg}), 400
            
            filename = uploaded_file.filename
            if not filename or not allowed_filename(filename):
                error_msg = f"Invalid file type. Allowed: {', '.join(ALLOWED_EXT)}"
                log_request("/api/verify", request_id, request.remote_addr, username,
                           filename=filename, doc_type=document_type,
                           status="ERROR", error=error_msg)
                return jsonify({"error": error_msg}), 400
            
            log_request("/api/verify", request_id, request.remote_addr, username,
                       filename=filename, doc_type=document_type, status="PROCESSING")
            
            ext = filename.rsplit(".", 1)[1].lower()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            uploaded_file.save(tmp.name)
            file_path = tmp.name
            file_size = os.path.getsize(file_path)
        
        # ---------- JSON (URL / BASE64) ----------
        elif request.is_json:
            logger.info("[API_VERIFY] Processing JSON request")
            data = request.get_json(force=True)
            
            document_type = data.get("document_type", "Unknown")
            filename = data.get("filename")
            
            log_request("/api/verify", request_id, request.remote_addr, username,
                       filename=filename or "unknown", doc_type=document_type, 
                       status="PROCESSING")
            
            if "url" in data:
                logger.info(f"[API_VERIFY] Downloading from URL")
                try:
                    file_path, url_filename = download_file_from_url(data["url"])
                    if not filename:
                        filename = url_filename
                    file_size = os.path.getsize(file_path)
                except Exception as e:
                    error_msg = f"Failed to download from URL: {str(e)}"
                    log_request("/api/verify", request_id, request.remote_addr, username,
                               filename=filename, doc_type=document_type,
                               status="ERROR", error=error_msg)
                    return jsonify({"error": error_msg}), 400
                
            elif "data" in data:
                logger.info("[API_VERIFY] Processing base64 payload")
                try:
                    file_path, _ = data_uri_to_file(data["data"])
                    if not filename:
                        filename = "uploaded_file"
                    file_size = os.path.getsize(file_path)
                except Exception as e:
                    error_msg = f"Failed to process base64 data: {str(e)}"
                    log_request("/api/verify", request_id, request.remote_addr, username,
                               filename=filename, doc_type=document_type,
                               status="ERROR", error=error_msg)
                    return jsonify({"error": error_msg}), 400
                
            else:
                error_msg = "No file, url, or data provided"
                log_request("/api/verify", request_id, request.remote_addr, username,
                           status="ERROR", error=error_msg)
                return jsonify({"error": error_msg}), 400
        
        else:
            error_msg = "Invalid request format. Use multipart/form-data or JSON"
            log_request("/api/verify", request_id, request.remote_addr, username,
                       status="ERROR", error=error_msg)
            return jsonify({"error": error_msg}), 400
        
        if not file_path or not os.path.exists(file_path):
            error_msg = "Failed to process file"
            log_request("/api/verify", request_id, request.remote_addr, username,
                       filename=filename, doc_type=document_type,
                       status="ERROR", error=error_msg)
            return jsonify({"error": error_msg}), 400
        
        logger.info(f"[API_VERIFY] Processing: {filename} ({document_type})")
        
        # ---------- PROCESS ----------
        file_data = {
            "id": request_id,
            "name": filename,
            "path": file_path
        }
        
        result = process_single_file(file_data, document_type, request_id, username)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"[API_VERIFY] Complete - Request ID: {request_id} ({processing_time:.2f}ms)")
        logger.info("=" * 80)
        
        # Clean up temp file
        try:
            if file_path and os.path.exists(file_path) and tempfile.gettempdir() in file_path:
                os.remove(file_path)
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup temp file: {cleanup_err}")
        
        return jsonify({
            "success": True,
            "request_id": request_id,
            "result": result,
            "message": "Document analyzed successfully",
            "processing_time_ms": f"{processing_time:.2f}",
            "retrieve_url": f"/api/result/{request_id}"
        })
        
    except Exception as e:
        logger.error(f"[API_VERIFY] Error: {str(e)}")
        logger.error(traceback.format_exc())
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "error": f"Verification failed: {str(e)}",
            "request_id": request_id
        }), 500

@app.route("/api/upload", methods=["POST"])
@require_auth
def api_upload():
    """Legacy upload endpoint"""
    logger.info("="*80)
    username = getattr(request, 'authenticated_user', 'unknown')
    logger.info(f"[API_UPLOAD] Incoming upload request from {request.remote_addr} (User: {username})")
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        if not request.data:
            return jsonify({"error": "No data received"}), 400
        
        payload = request.get_json(force=True)
        files_data = payload.get("files", [])
        
        if not files_data:
            return jsonify({"error": "No files provided"}), 400
        
        uploaded_files = []
        
        for idx, file_info in enumerate(files_data):
            name = file_info.get("name")
            data_uri = file_info.get("dataUri")
            
            if not name or not data_uri:
                continue
            
            try:
                tmp_path, _ = data_uri_to_file(data_uri)
                file_id = hashlib.md5(f"{name}{datetime.now(IST).isoformat()}{uuid.uuid4()}".encode()).hexdigest()[:12]
                file_size = os.path.getsize(tmp_path)
                
                saved_path = save_uploaded_file(tmp_path, file_id, name)
                
                uploaded_files.append({
                    "id": file_id,
                    "name": name,
                    "path": tmp_path,
                    "size": file_size,
                    "saved_path": saved_path
                })
                
                logger.info(f"[API_UPLOAD] File {idx+1}/{len(files_data)} uploaded: {name}")
                
            except Exception as e:
                logger.error(f"[API_UPLOAD] Error processing file {name}: {str(e)}")
                continue
        
        if not uploaded_files:
            return jsonify({"error": "No files could be processed"}), 400
        
        session["uploaded_files"] = uploaded_files
        session["upload_username"] = username
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"[API_UPLOAD] Upload complete: {len(uploaded_files)} file(s)")
        logger.info("="*80)
        
        return jsonify({
            "success": True,
            "files": [{"id": f["id"], "name": f["name"], "size": f["size"]} for f in uploaded_files]
        })
        
    except Exception as e:
        logger.error(f"[API_UPLOAD] Error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/api/analyze", methods=["POST"])
@require_auth
def api_analyze():
    """Legacy analyze endpoint"""
    logger.info("="*80)
    username = getattr(request, 'authenticated_user', 'unknown')
    logger.info(f"[API_ANALYZE] Incoming analysis request from {request.remote_addr} (User: {username})")
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        payload = request.get_json(force=True)
        document_type = payload.get("documentType", "Unknown")
        
        uploaded_files = session.get("uploaded_files", [])
        
        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400
        
        logger.info(f"[API_ANALYZE] Processing {len(uploaded_files)} file(s)")
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_single_file, file_data, document_type, file_data["id"], username): file_data
                for file_data in uploaded_files
            }
            
            for future in as_completed(future_to_file):
                file_data = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[API_ANALYZE] Failed: {str(e)}")
        
        session.pop("uploaded_files", None)
        session[LAST_REPORT_KEY] = results
        session["last_request_id"] = request_id
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"[API_ANALYZE] Complete: {len(results)} result(s)")
        logger.info("="*80)
        
        return jsonify({
            "success": True,
            "request_id": request_id,
            "results": results,
            "processing_time_ms": f"{processing_time:.2f}"
        })
        
    except Exception as e:
        logger.error(f"[API_ANALYZE] Error: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ---------- USER'S OWN RESULTS ENDPOINTS ----------
@app.route("/api/my-results", methods=["GET"])
@require_auth
def get_my_results():
    """Get current user's results"""
    username = getattr(request, 'authenticated_user', 'unknown')
    user_requests = get_user_requests(username)
    
    results = []
    for req_id, data in sorted(user_requests.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True):
        results.append({
            "request_id": req_id,
            "timestamp": data.get("timestamp"),
            "document_type": data.get("document_type"),
            "filename": data.get("filename"),
            "status": data.get("result", {}).get("status"),
            "category": data.get("result", {}).get("category"),
            "confidence": data.get("result", {}).get("confidence")
        })
    
    return jsonify({
        "success": True,
        "username": username,
        "count": len(results),
        "results": results
    })

@app.route("/api/my-stats", methods=["GET"])
@require_auth
def get_my_stats():
    """Get current user's statistics"""
    username = getattr(request, 'authenticated_user', 'unknown')
    stats = get_user_statistics(username)
    
    return jsonify({
        "success": True,
        "username": username,
        "statistics": stats
    })

@app.route("/api/my-report/pdf", methods=["GET"])
@require_auth
def download_my_report_pdf():
    """Download current user's aggregate report as PDF"""
    username = getattr(request, 'authenticated_user', 'unknown')
    
    try:
        stats = get_user_statistics(username)
        user_requests = get_user_requests(username)
        
        pdf_buffer = generate_user_aggregate_report(username, stats, user_requests)
        
        return send_file(
            pdf_buffer,
            download_name=f"verifai_report_{username}_{datetime.now(IST).strftime('%Y%m%d')}.pdf",
            as_attachment=True,
            mimetype="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"[MY_REPORT_PDF] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/result/<request_id>", methods=["GET"])
@require_auth
def get_result(request_id):
    """Get result by request ID (users can only see their own results)"""
    username = getattr(request, 'authenticated_user', 'unknown')
    is_admin_user = getattr(request, 'is_admin', False)
    
    request_data = get_request(request_id)
    if not request_data:
        return jsonify({"error": "Request not found"}), 404
    
    # Check ownership (admin can see all)
    if not is_admin_user and request_data.get("username") != username:
        logger.warning(f"[AUTH] User '{username}' tried to access request '{request_id}' owned by '{request_data.get('username')}'")
        return jsonify({"error": "Access denied. You can only view your own results."}), 403
    
    return jsonify({
        "success": True,
        "request_id": request_id,
        "data": request_data
    })

@app.route("/api/result/<request_id>/csv", methods=["GET"])
@require_auth
def download_result_csv(request_id):
    """Download result as CSV"""
    username = getattr(request, 'authenticated_user', 'unknown')
    is_admin_user = getattr(request, 'is_admin', False)
    
    request_data = get_request(request_id)
    if not request_data:
        return jsonify({"error": "Request not found"}), 404
    
    # Check ownership
    if not is_admin_user and request_data.get("username") != username:
        return jsonify({"error": "Access denied. You can only view your own results."}), 403

    result = request_data.get("result", {})
    checks = result.get("verificationChecks", {})

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(CSV_HEADERS)

    writer.writerow([
        request_id,
        request_data.get("timestamp", ""),
        request_data.get("username", "unknown"),
        result.get("documentName", ""),
        result.get("documentType", ""),
        result.get("status", ""),
        result.get("category", ""),
        result.get("confidence", ""),
        result.get("reasoning", ""),
        result.get("extractedText", ""),
        result.get("translatedText", ""),
        result.get("detectedLanguage", ""),
        "Yes" if checks.get("has_security_features") else "No",
        "Yes" if checks.get("format_matches_template") else "No",
        "Yes" if checks.get("text_is_legible") else "No",
        "Yes" if checks.get("no_tampering_signs") else "No",
        "Yes" if checks.get("quality_acceptable") else "No",
        "Yes" if checks.get("dates_are_valid") else "No",
        result.get("processingTimeMs", ""),
        request_data.get("saved_file_path", ""),
    ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        download_name=f"verifai_{request_id}.csv",
        as_attachment=True,
        mimetype="text/csv"
    )

@app.route("/api/result/<request_id>/pdf", methods=["GET"])
@require_auth
def download_result_pdf(request_id):
    """Download result as PDF"""
    username = getattr(request, 'authenticated_user', 'unknown')
    is_admin_user = getattr(request, 'is_admin', False)
    
    try:
        request_data = get_request(request_id)
        
        if not request_data:
            return jsonify({"error": "Request not found"}), 404
        
        # Check ownership
        if not is_admin_user and request_data.get("username") != username:
            return jsonify({"error": "Access denied. You can only view your own results."}), 403
        
        result = request_data.get("result", {})
        results_list = [result]
        
        pdf_buffer = generate_pdf_report(results_list, request_id)
        
        return send_file(
            pdf_buffer,
            download_name=f"verifai_report_{request_id}.pdf",
            as_attachment=True,
            mimetype="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"[PDF_DOWNLOAD] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/result/<request_id>/json", methods=["GET"])
@require_auth
def download_result_json(request_id):
    """Download result as JSON"""
    username = getattr(request, 'authenticated_user', 'unknown')
    is_admin_user = getattr(request, 'is_admin', False)
    
    try:
        request_data = get_request(request_id)
        
        if not request_data:
            return jsonify({"error": "Request not found"}), 404
        
        # Check ownership
        if not is_admin_user and request_data.get("username") != username:
            return jsonify({"error": "Access denied. You can only view your own results."}), 403
        
        json_data = json.dumps(request_data, indent=2)
        
        return send_file(
            io.BytesIO(json_data.encode("utf-8")),
            download_name=f"verifai_report_{request_id}.json",
            as_attachment=True,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"[JSON_DOWNLOAD] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/download-report")
@require_auth
def download_report():
    """Legacy download endpoint"""
    username = getattr(request, 'authenticated_user', 'unknown')
    request_id = session.get("last_request_id", generate_request_id())
    
    try:
        last = session.get(LAST_REPORT_KEY)
        if not last:
            report_text = f"No recent analysis found.\nDate: {datetime.now(IST)}\nUser: {username}\n"
            return send_file(
                io.BytesIO(report_text.encode()), 
                download_name="verifai_report.txt", 
                as_attachment=True, 
                mimetype="text/plain"
            )

        headers = [
            "Document Name", "Document Type", "Status", "Category", "Confidence (%)",
            "Reasoning", "Extracted Text", "Translated Text", "Language",
            "Security Features", "Format Match", "Text Legible", "No Tampering",
            "Quality OK", "Dates Valid", "Process Date/Time", "Processing Time (ms)"
        ]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        
        for row in last:
            checks = row.get("verificationChecks", {})
            writer.writerow([
                row.get("documentName", ""),
                row.get("documentType", ""),
                row.get("status", ""),
                row.get("category", ""),
                row.get("confidence", ""),
                row.get("reasoning", ""),
                row.get("extractedText", ""),
                row.get("translatedText", ""),
                row.get("detectedLanguage", ""),
                "Yes" if checks.get("has_security_features") else "No",
                "Yes" if checks.get("format_matches_template") else "No",
                "Yes" if checks.get("text_is_legible") else "No",
                "Yes" if checks.get("no_tampering_signs") else "No",
                "Yes" if checks.get("quality_acceptable") else "No",
                "Yes" if checks.get("dates_are_valid") else "No",
                row.get("processDateTime", ""),
                row.get("processingTimeMs", "")
            ])
        
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")), 
            download_name="verifai_report.csv", 
            as_attachment=True, 
            mimetype="text/csv"
        )

    except Exception as e:
        logger.error(f"[DOWNLOAD] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========================================================================
# ADMIN-ONLY ENDPOINTS
# ========================================================================

@app.route("/api/admin/users", methods=["GET"])
@require_admin
def admin_list_users():
    """List all users (admin only)"""
    users_list = []
    for username, data in USERS.items():
        user_stats = get_user_statistics(username)
        users_list.append({
            "username": username,
            "role": data.get("role"),
            "full_name": data.get("full_name"),
            "total_verifications": user_stats["total_requests"],
            "legit_count": user_stats["legit"],
            "suspicious_count": user_stats["suspicious"],
            "not_legit_count": user_stats["not_legit"]
        })
    
    return jsonify({
        "success": True,
        "count": len(users_list),
        "users": users_list
    })

@app.route("/api/admin/stats", methods=["GET"])
@require_admin
def admin_get_stats():
    """Get overall statistics (admin only)"""
    stats = get_user_statistics()  # No username = all users
    
    # Add per-user breakdown
    user_stats = {}
    for username in USERS.keys():
        user_stats[username] = get_user_statistics(username)
    
    return jsonify({
        "success": True,
        "overall_statistics": stats,
        "per_user_statistics": user_stats
    })

@app.route("/api/admin/results", methods=["GET"])
@require_admin
def admin_get_all_results():
    """Get all results (admin only)"""
    all_requests = get_all_requests()
    
    results = []
    for req_id, data in sorted(all_requests.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True):
        results.append({
            "request_id": req_id,
            "timestamp": data.get("timestamp"),
            "username": data.get("username"),
            "document_type": data.get("document_type"),
            "filename": data.get("filename"),
            "status": data.get("result", {}).get("status"),
            "category": data.get("result", {}).get("category"),
            "confidence": data.get("result", {}).get("confidence")
        })
    
    return jsonify({
        "success": True,
        "count": len(results),
        "results": results
    })

@app.route("/api/admin/user/<target_username>/results", methods=["GET"])
@require_admin
def admin_get_user_results(target_username):
    """Get specific user's results (admin only)"""
    if target_username not in USERS:
        return jsonify({"error": "User not found"}), 404
    
    user_requests = get_user_requests(target_username)
    
    results = []
    for req_id, data in sorted(user_requests.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True):
        results.append({
            "request_id": req_id,
            "timestamp": data.get("timestamp"),
            "document_type": data.get("document_type"),
            "filename": data.get("filename"),
            "status": data.get("result", {}).get("status"),
            "category": data.get("result", {}).get("category"),
            "confidence": data.get("result", {}).get("confidence")
        })
    
    return jsonify({
        "success": True,
        "username": target_username,
        "count": len(results),
        "results": results
    })

@app.route("/api/admin/user/<target_username>/report/pdf", methods=["GET"])
@require_admin
def admin_download_user_report(target_username):
    """Download specific user's aggregate report (admin only)"""
    if target_username not in USERS:
        return jsonify({"error": "User not found"}), 404
    
    try:
        stats = get_user_statistics(target_username)
        user_requests = get_user_requests(target_username)
        
        pdf_buffer = generate_user_aggregate_report(target_username, stats, user_requests)
        
        return send_file(
            pdf_buffer,
            download_name=f"verifai_report_{target_username}_{datetime.now(IST).strftime('%Y%m%d')}.pdf",
            as_attachment=True,
            mimetype="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"[ADMIN_USER_REPORT] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/report/pdf", methods=["GET"])
@require_admin
def admin_download_all_report():
    """Download overall aggregate report (admin only)"""
    try:
        stats = get_user_statistics()  # All users
        
        pdf_buffer = generate_user_aggregate_report("ALL USERS", stats, REQUEST_STORAGE)
        
        return send_file(
            pdf_buffer,
            download_name=f"verifai_overall_report_{datetime.now(IST).strftime('%Y%m%d')}.pdf",
            as_attachment=True,
            mimetype="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"[ADMIN_ALL_REPORT] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/logs/csv", methods=["GET"])
@require_admin
def admin_get_csv_logs():
    """Get CSV audit logs (admin only)"""
    try:
        if not os.path.exists(CSV_AUDIT_FILE):
            return jsonify({"error": "CSV log file not found"}), 404
        
        with open(CSV_AUDIT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        return send_file(
            io.BytesIO(content.encode("utf-8")),
            download_name="verifai_audit_log.csv",
            as_attachment=True,
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/logs/json", methods=["GET"])
@require_admin
def admin_get_json_logs():
    """Get JSON logs (admin only)"""
    try:
        if not os.path.exists(JSON_LOG_FILE):
            return jsonify({"error": "JSON log file not found"}), 404
        
        with open(JSON_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        return jsonify({
            "count": len(logs),
            "logs": logs[-100:]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/files", methods=["GET"])
@require_admin
def admin_list_files():
    """List all uploaded files (admin only)"""
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    # Try to find the owner
                    request_id = filename.rsplit(".", 1)[0]
                    request_data = get_request(request_id)
                    owner = request_data.get("username", "unknown") if request_data else "unknown"
                    
                    files.append({
                        "filename": filename,
                        "request_id": request_id,
                        "owner": owner,
                        "size": os.path.getsize(filepath),
                        "modified": datetime.fromtimestamp(os.path.getmtime(filepath), IST).isoformat()
                    })
        
        return jsonify({
            "count": len(files),
            "upload_folder": UPLOAD_FOLDER,
            "files": sorted(files, key=lambda x: x["modified"], reverse=True)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/file/<request_id>", methods=["GET"])
@require_admin
def admin_get_file(request_id):
    """Download an uploaded file (admin only)"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.startswith(request_id):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    return send_file(filepath, as_attachment=True)
        
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- HEALTH CHECK (No Auth Required) ----------
@app.route("/health")
def health():
    """Health check endpoint (no auth required)"""
    gemini_status = "configured" if GEMINI_API_KEY else "not_configured"
    
    return jsonify({
        "status": "healthy",
        "gemini_api": gemini_status,
        "gemini_model": GEMINI_MODEL,
        "max_workers": MAX_WORKERS,
        "stored_requests": len(REQUEST_STORAGE),
        "endpoints": {
            "public": ["GET /health"],
            "auth": ["POST /api/login", "POST /api/logout", "GET /api/auth/status"],
            "user": [
                "POST /api/verify",
                "POST /api/upload",
                "POST /api/analyze",
                "GET /api/my-results",
                "GET /api/my-stats",
                "GET /api/my-report/pdf",
                "GET /api/result/<id>",
                "GET /api/result/<id>/pdf",
                "GET /api/result/<id>/csv",
                "GET /api/result/<id>/json",
                "GET /download-report"
            ]
        }
    })

# ---------- MAIN PAGE ----------
@app.route("/")
def index():
    """Serve main page"""
    return send_from_directory(".", "index.html")

# ---------- MAIN ----------
if __name__ == "__main__":
    load_storage()
    
    logger.info("="*80)
    logger.info("VerifAI Server Starting (RBAC Version)...")
    logger.info(f"Gemini API: {'Configured' if GEMINI_API_KEY else 'NOT CONFIGURED'}")
    logger.info(f"Gemini Model: {GEMINI_MODEL}")
    logger.info(f"Stored Requests: {len(REQUEST_STORAGE)}")
    logger.info(f"Upload Folder: {UPLOAD_FOLDER}")
    logger.info("")
    logger.info("REGISTERED USERS:")
    for username, data in USERS.items():
        logger.info(f"  - {username}: {data.get('role')} ({data.get('full_name')})")
    logger.info("")
    logger.info("ROLE-BASED ACCESS:")
    logger.info("  - user: Can verify documents and view only their own results")
    logger.info("")
    logger.info("CURL EXAMPLES:")
    logger.info('  # Regular user verification:')
    logger.info('  curl -X POST http://localhost:5000/api/verify \\')
    logger.info('       -F "file=@document.pdf" \\')
    logger.info('       -F "document_type=PAN Card" \\')
    logger.info('       -u kyc_user1:kyc@123')
    logger.info('')
    logger.info("="*80)
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
