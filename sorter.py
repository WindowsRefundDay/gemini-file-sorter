import sys
import argparse
import mimetypes
import datetime
import re
import json
import os
import logging
import threading
import shutil
from pathlib import Path
from typing import List, Dict, Callable, Any, Generator, Optional, Tuple, Union

# --- Basic Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file *first*
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Loaded environment variables from .env file if present.")
except ImportError:
    logging.info(".env file processing skipped: python-dotenv not installed.")

# --- Optional Dependency Handling ---

# Improve tkinter detection and error handling
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog, scrolledtext
    import tkinter.ttk as ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("tkinter module not found. GUI functionality will be disabled.")

# Improve magic detection
try:
    import magic
    MAGIC_AVAILABLE = True
    logging.info("Using 'python-magic' for enhanced MIME type detection.")
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("'python-magic' library not found. Falling back to standard mimetypes. "
                    "Install with 'pip install python-magic' (or 'python-magic-bin' on Windows) for better accuracy.")

# Try to import google-generativeai with proper error handling
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    # Log error, exit handled later if AI features are attempted
    logging.error("google-generativeai package not found. This is required for the AI functionality.")
    logging.error("Please install it using pip: pip install google-generativeai")
    # Don't exit immediately, allow CLI --help etc. Exit occurs if genai is actually used.


# --- Constants ---
CONFIG_FILE = Path.home() / '.ai_file_sorter_config.json'
DEFAULT_RULE = [["True", "Uncategorized"]]  # Default rule if AI fails or no rules match
API_KEY = os.getenv('GEMINI_API_KEY', '') # Get API key from environment variable or .env


# --- Configuration Management ---

def save_config(source_dir: str = "", target_dir: str = "") -> None:
    """Save configuration (source/target directories) to JSON file."""
    config = {
        'source_dir': source_dir,
        'target_dir': target_dir
        # Note: API key is NOT saved here for security, fetched from env/input
    }
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
        logging.info(f"Saved configuration to {CONFIG_FILE}")
    except IOError as e:
        logging.warning(f"Could not save configuration to {CONFIG_FILE}: {e}")
    except Exception as e:
        logging.warning(f"An unexpected error occurred while saving configuration: {e}")


def load_config() -> dict:
    """Load configuration from JSON file."""
    global API_KEY # Allow modification of the global API_KEY variable
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
            logging.info(f"Loaded configuration from {CONFIG_FILE}")
            # Re-check environment variable for API key in case it changed
            env_api_key = os.getenv('GEMINI_API_KEY')
            if env_api_key:
                API_KEY = env_api_key
                # Configure Gemini if key is found in env during load
                if GENAI_AVAILABLE and API_KEY:
                     try:
                         genai.configure(api_key=API_KEY)
                         logging.info("Gemini API configured using environment variable key.")
                     except Exception as e:
                         logging.error(f"Failed to configure Gemini API during config load: {e}")
            return config
        except json.JSONDecodeError as e:
            logging.warning(f"Could not parse configuration file {CONFIG_FILE}: {e}")
        except Exception as e:
            logging.warning(f"An unexpected error occurred while loading configuration: {e}")
    return {'source_dir': '', 'target_dir': ''} # Return defaults if file doesn't exist or fails


def configure_gemini(api_key: str) -> bool:
    """Configure the Gemini API with the provided API key."""
    global API_KEY
    if not GENAI_AVAILABLE:
        logging.error("Cannot configure Gemini: google-generativeai package not installed.")
        return False
    if not api_key:
        logging.error("Cannot configure Gemini: API key is empty.")
        return False

    API_KEY = api_key
    try:
        genai.configure(api_key=api_key)
        logging.info("Gemini API configured successfully.")
        # No need to save config here, as API key isn't saved in the config file
        return True
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        API_KEY = '' # Clear invalid key
        return False


# --- File Analysis ---

def analyze_text_content(file_path: Path, max_lines: int = 10) -> str:
    """
    Analyze the first few lines of a text file to determine its content type.

    Args:
        file_path: Path to the text file.
        max_lines: Maximum number of lines to read.

    Returns:
        A string describing the content type (e.g., 'source_code', 'json_data').
    """
    try:
        # Use 'utf-8-sig' to handle potential BOM (Byte Order Mark)
        with file_path.open('r', encoding='utf-8-sig', errors='ignore') as f:
            first_lines = [line.strip() for i, line in enumerate(f) if i < max_lines]
            content = ' '.join(first_lines)

            # Simple content type detection based on patterns (case-insensitive)
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ['import ', 'def ', 'class ', 'const ', 'let ', 'var ', 'function']):
                return 'source_code'
            elif any(keyword in content_lower for keyword in ['select ', 'insert ', 'update ', 'delete ', 'create table']):
                return 'sql_script'
            elif content.startswith('{') or content.startswith('['):
                # Basic check for JSON/JSON-like structure
                 try:
                     # Try parsing a snippet to be more confident
                     snippet = "".join(first_lines)
                     json.loads(snippet if len(snippet) < 1000 else snippet[:1000] + ('}' if snippet.startswith('{') else ']'))
                     return 'json_data'
                 except json.JSONDecodeError:
                     # If parsing fails, could still be JSON fragment or other text
                     if content.startswith('{') or content.startswith('['):
                         return 'json_like'
                     pass # Fall through to other checks
            elif any(keyword in content_lower for keyword in ['<!doctype', '<html', '<?xml', '<svg']):
                return 'markup'
            elif any(keyword in content_lower for keyword in ['dear ', 'hi ', 'hello ', 'regards', 'sincerely']):
                return 'letter'
            else:
                return 'general_text'
    except (IOError, OSError) as e:
        logging.warning(f"Could not read file {file_path} for text analysis: {e}")
        return 'read_error'
    except UnicodeDecodeError as e:
        logging.warning(f"Could not decode file {file_path} as UTF-8 for analysis: {e}")
        return 'encoding_error'
    except Exception as e:
        logging.warning(f"Unexpected error analyzing text content of {file_path}: {e}")
        return 'unknown'
    return 'unknown' # Default if no patterns match or file is empty


def get_file_category(mime_type: str, extension: str = '') -> str:
    """
    Get a high-level category for a file based on its MIME type and extension.

    Args:
        mime_type: The MIME type string (e.g., 'text/plain').
        extension: The file extension including the dot (e.g., '.txt').

    Returns:
        High-level category name (e.g., 'code', 'document', 'image').
    """
    ext = extension.lower()
    mime = mime_type.lower() if mime_type else 'application/octet-stream' # Handle None mime_type

    # More comprehensive extension mapping
    extension_mapping = {
        # Code
        '.py': 'code', '.pyw': 'code', '.pyc': 'code', '.pyd': 'code',
        '.js': 'code', '.jsx': 'code', '.ts': 'code', '.tsx': 'code', '.mjs': 'code', '.cjs': 'code',
        '.java': 'code', '.class': 'code', '.jar': 'code',
        '.c': 'code', '.h': 'code', '.cpp': 'code', '.hpp': 'code', '.cs': 'code',
        '.go': 'code', '.rs': 'code', '.swift': 'code', '.kt': 'code', '.kts': 'code',
        '.php': 'code', '.rb': 'code', '.pl': 'code', '.pm': 'code',
        '.sh': 'script', '.bash': 'script', '.zsh': 'script', '.ps1': 'script', '.bat': 'script', '.cmd': 'script',
        '.sql': 'database', '.ddl': 'database', '.dml': 'database',
        # Web
        '.html': 'web', '.htm': 'web', '.xhtml': 'web',
        '.css': 'web', '.scss': 'web', '.less': 'web',
        '.xml': 'data', '.xsd': 'data', '.xslt': 'web', # XSLT can be web-related
        # Data
        '.json': 'data', '.yaml': 'data', '.yml': 'data', '.csv': 'data', '.tsv': 'data',
        '.ini': 'config', '.cfg': 'config', '.conf': 'config', '.toml': 'config',
        '.log': 'log',
        # Documents & Text
        '.txt': 'text', '.md': 'document', '.rst': 'document', '.rtf': 'document',
        '.pdf': 'document', '.epub': 'document', '.mobi': 'document',
        '.doc': 'document', '.docx': 'document', '.odt': 'document',
        # Spreadsheets
        '.xls': 'spreadsheet', '.xlsx': 'spreadsheet', '.ods': 'spreadsheet',
        # Presentations
        '.ppt': 'presentation', '.pptx': 'presentation', '.odp': 'presentation',
        # Archives
        '.zip': 'archive', '.rar': 'archive', '.tar': 'archive', '.gz': 'archive', '.bz2': 'archive', '.7z': 'archive', '.xz': 'archive', '.iso': 'archive',
        # Media
        '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image', '.bmp': 'image', '.tiff': 'image', '.webp': 'image', '.svg': 'image', '.ico': 'image',
        '.mp3': 'audio', '.wav': 'audio', '.ogg': 'audio', '.flac': 'audio', '.aac': 'audio', '.m4a': 'audio',
        '.mp4': 'video', '.avi': 'video', '.mkv': 'video', '.mov': 'video', '.wmv': 'video', '.flv': 'video', '.webm': 'video',
        # Fonts
        '.ttf': 'font', '.otf': 'font', '.woff': 'font', '.woff2': 'font',
        # Executables/Binaries
        '.exe': 'binary', '.dll': 'binary', '.so': 'binary', '.dylib': 'binary', '.app': 'binary', '.msi': 'installer',
        # Other common types
        '.ics': 'calendar', '.vcf': 'contact',
        '.torrent': 'torrent'
    }

    # Check extension first (often more specific)
    if ext in extension_mapping:
        return extension_mapping[ext]

    # Fallback to MIME type mapping
    # Using startswith for broader matching
    mime_mapping = {
        'text/': 'text', # General text before specific text types
        'text/html': 'web',
        'text/css': 'web',
        'text/xml': 'data', # XML generally data unless extension is web-specific
        'text/javascript': 'code',
        'text/x-python': 'code',
        'text/x-java-source': 'code',
        'text/x-csrc': 'code',
        'text/x-script': 'script',
        'text/x-shellscript': 'script',
        'text/x-sql': 'database',
        'text/csv': 'data',
        'text/markdown': 'document',
        'image/': 'image',
        'audio/': 'audio',
        'video/': 'video',
        'font/': 'font',
        'application/pdf': 'document',
        'application/msword': 'document',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
        'application/vnd.oasis.opendocument.text': 'document',
        'application/vnd.ms-excel': 'spreadsheet',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'spreadsheet',
        'application/vnd.oasis.opendocument.spreadsheet': 'spreadsheet',
        'application/vnd.ms-powerpoint': 'presentation',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'presentation',
        'application/vnd.oasis.opendocument.presentation': 'presentation',
        'application/json': 'data',
        'application/yaml': 'data',
        'application/xml': 'data',
        'application/zip': 'archive',
        'application/x-rar-compressed': 'archive',
        'application/x-tar': 'archive',
        'application/gzip': 'archive',
        'application/x-bzip2': 'archive',
        'application/x-7z-compressed': 'archive',
        'application/x-iso9660-image': 'archive',
        'application/x-executable': 'binary',
        'application/octet-stream': 'binary', # Generic binary
        'application/x-msdownload': 'binary', # .exe often reported as this
        'application/x-apple-diskimage': 'archive', # .dmg
        'application/vnd.debian.binary-package': 'installer', # .deb
        'application/x-redhat-package-manager': 'installer', # .rpm
    }

    for pattern, category in mime_mapping.items():
        # Use startswith for general types like image/, audio/, etc.
        if pattern.endswith('/') and mime.startswith(pattern):
            return category
        # Use exact match for specific types
        elif mime == pattern:
            return category

    # If extension was unknown and mime is generic binary, check common binary extensions again
    if mime == 'application/octet-stream' and ext in ['.exe', '.dll', '.so', '.dylib', '.msi']:
         return 'binary' if ext != '.msi' else 'installer'

    # If MIME is text/plain, check extension for more specific category
    if mime == 'text/plain':
        if ext in ['.log']: return 'log'
        if ext in ['.cfg', '.ini', '.conf', '.toml']: return 'config'
        if ext in ['.md', '.rst']: return 'document'
        if ext in ['.sql']: return 'database'
        if ext in ['.sh', '.bat', '.cmd', '.ps1']: return 'script'
        return 'text' # Default for text/plain

    return 'other' # Default category


def is_text_file(mime_type: str, file_path: Path) -> bool:
    """
    Check if a file is likely a text file based on MIME type and extension.

    Args:
        mime_type: The MIME type string.
        file_path: The Path object for the file (used for extension check).

    Returns:
        True if the file is likely text-based, False otherwise.
    """
    if not mime_type:
        return False

    mime_lower = mime_type.lower()

    # Primary check: MIME type starts with 'text/'
    if mime_lower.startswith('text/'):
        return True

    # Secondary check: Specific application types known to be text
    text_app_types = {
        'application/json',
        'application/xml',
        'application/yaml',
        'application/javascript',
        'application/ecmascript',
        'application/sql',
        # Add more as needed
    }
    if mime_lower in text_app_types:
        return True

    # Tertiary check: If MIME is generic binary but extension suggests text
    if mime_lower == 'application/octet-stream':
        text_extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.swift', '.kt', '.php', '.rb', '.pl', '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd', '.sql', '.md', '.rst', '.log', '.ini', '.cfg', '.conf', '.toml', '.yaml', '.yml', '.json', '.xml', '.html', '.htm', '.css', '.scss', '.less', '.svg'}
        if file_path.suffix.lower() in text_extensions:
            # Log this guess
            logging.debug(f"Guessed {file_path.name} as text based on extension despite generic MIME '{mime_type}'.")
            return True

    return False


def extract_metadata(file_path: Path) -> Dict[str, Union[str, int, float, bool, None]]:
    """
    Extract comprehensive metadata from a file path.

    Args:
        file_path: Path to the file.

    Returns:
        Dictionary with detailed metadata, using None for unavailable fields.
    """
    metadata: Dict[str, Union[str, int, float, bool, None]] = {
        'name': None, 'stem': None, 'extension': None, 'size_bytes': None, 'size_mb': None,
        'created_timestamp': None, 'modified_timestamp': None, 'accessed_timestamp': None,
        'created_date': None, 'modified_date': None, 'accessed_date': None,
        'is_hidden': None, 'is_readonly': None, 'mime_type': None, 'category': None,
        'content_type': None, 'has_year_in_name': None, 'has_date_in_name': None,
        'full_path': str(file_path.resolve()) # Add full path
    }
    try:
        # Basic path info
        metadata['name'] = file_path.name
        metadata['stem'] = file_path.stem
        metadata['extension'] = file_path.suffix.lower() if file_path.suffix else ''
        metadata['is_hidden'] = file_path.name.startswith('.')

        # File stats
        stats = file_path.stat()
        metadata['size_bytes'] = stats.st_size
        metadata['size_mb'] = round(stats.st_size / (1024 * 1024), 3) if stats.st_size is not None else 0.0
        metadata['created_timestamp'] = stats.st_ctime
        metadata['modified_timestamp'] = stats.st_mtime
        metadata['accessed_timestamp'] = stats.st_atime
        metadata['created_date'] = datetime.datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d')
        metadata['modified_date'] = datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d')
        metadata['accessed_date'] = datetime.datetime.fromtimestamp(stats.st_atime).strftime('%Y-%m-%d')
        metadata['is_readonly'] = not os.access(file_path, os.W_OK)

        # MIME type detection
        mime_type = None
        if MAGIC_AVAILABLE:
            try:
                # Use non-blocking call if available? No, magic doesn't support async easily.
                # Keep buffer size reasonable
                mime_type = magic.from_file(str(file_path), mime=True)
            except magic.MagicException as e:
                logging.warning(f"python-magic failed for {file_path.name}: {e}. Falling back.")
            except FileNotFoundError: # Handle race condition if file deleted between listing and analysis
                 logging.warning(f"File not found during MIME check: {file_path.name}")
                 return metadata # Return partial metadata
            except Exception as e: # Catch other potential magic errors
                logging.warning(f"Unexpected error using python-magic for {file_path.name}: {e}. Falling back.")

        if mime_type is None:
             # Fallback to mimetypes or default
             guessed_type, _ = mimetypes.guess_type(file_path)
             mime_type = guessed_type or 'application/octet-stream'

        metadata['mime_type'] = mime_type

        # Categorization
        metadata['category'] = get_file_category(mime_type, metadata['extension'])

        # Name analysis
        # Improved regex for year/date detection
        metadata['has_year_in_name'] = bool(re.search(r'\b(19[89]\d|20[0-2]\d)\b', file_path.stem)) # 1980-2029
        # More flexible date patterns (YYYY-MM-DD, YYYYMMDD, MM-DD-YYYY, etc.)
        metadata['has_date_in_name'] = bool(re.search(r'\b(\d{4}[-/._]?\d{2}[-/._]?\d{2}|\d{2}[-/._]?\d{2}[-/._]?\d{4})\b', file_path.stem))

        # Add text content analysis for likely text files
        if is_text_file(mime_type, file_path):
            metadata['content_type'] = analyze_text_content(file_path)
        else:
            metadata['content_type'] = 'non_text'

    except FileNotFoundError:
        logging.error(f"File not found during metadata extraction: {file_path}")
        # Return dictionary with None values as initialized
    except OSError as e:
        logging.error(f"OS error extracting metadata for {file_path}: {e}")
        # Return dictionary with None values as initialized
    except Exception as e:
        logging.error(f"Unexpected error extracting metadata for {file_path}: {e}")
        # Return dictionary with None values as initialized

    return metadata


# --- AI Rule Generation ---

def generate_sorting_rules(file_list: List[Path], user_description: str = "") -> List[List[str]]:
    """
    Generate sorting rules using the configured Gemini model.

    Args:
        file_list: List of file paths to analyze for context (if no user description).
        user_description: Optional user description to guide rule generation.

    Returns:
        List of rules [[condition_string, target_directory_string], ...],
        or DEFAULT_RULE on failure.
    """
    if not API_KEY:
        logging.error("Cannot generate rules: Gemini API key not configured.")
        # Optionally raise an error or return default rule immediately
        # raise ValueError("API key not configured.")
        return DEFAULT_RULE[:] # Return a copy

    if not GENAI_AVAILABLE:
         logging.error("Cannot generate rules: google-generativeai package not available.")
         return DEFAULT_RULE[:] # Return a copy

    try:
        # Ensure the model name is valid, consider making configurable?
        # Using a known reliable and fast model for this task.
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        return DEFAULT_RULE[:] # Return a copy

    # Construct the prompt
    metadata_keys_description = """
    - name (str): Full file name (e.g., 'report_2023-10-26.pdf').
    - stem (str): File name without extension (e.g., 'report_2023-10-26').
    - extension (str): File extension with dot, lowercase (e.g., '.pdf').
    - size_bytes (int): File size in bytes.
    - size_mb (float): File size in megabytes.
    - created_date (str): Creation date ('YYYY-MM-DD').
    - modified_date (str): Last modified date ('YYYY-MM-DD').
    - accessed_date (str): Last accessed date ('YYYY-MM-DD').
    - is_hidden (bool): True if the file name starts with a dot.
    - is_readonly (bool): True if the file appears to be read-only.
    - mime_type (str): Detailed MIME type (e.g., 'application/pdf').
    - category (str): High-level category (e.g., 'document', 'image', 'code', 'archive', 'data', 'web', 'text', 'spreadsheet', 'presentation', 'binary', 'other').
    - content_type (str | None): For text files: 'source_code', 'sql_script', 'json_data', 'json_like', 'markup', 'letter', 'general_text', 'read_error', 'encoding_error', 'unknown'. For non-text: 'non_text'.
    - has_year_in_name (bool): True if filename likely contains a year (1980-2029).
    - has_date_in_name (bool): True if filename likely contains a date pattern (YYYYMMDD variations).
    - full_path (str): The absolute path to the file.
    """

    if user_description:
        prompt = f"""
        Analyze the following user request and generate file sorting rules.
        The goal is to categorize files based on the user's intent.

        User Request: "{user_description}"

        Available file metadata fields for conditions (use `metadata['field_name']`):
        {metadata_keys_description}

        Output requirements:
        1.  Generate a Python list of lists, where each inner list is `["condition_string", "target_folder_name"]`.
        2.  The `condition_string` MUST be a valid Python boolean expression evaluating the `metadata` dictionary. Use standard Python operators (`==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not in`, `and`, `or`, `not`). Use methods like `.startswith()`, `.endswith()`, `.lower()` on string metadata fields.
        3.  The `target_folder_name` is a simple string for the destination subfolder (e.g., "Invoices", "Python Code", "Archived Projects/Old Reports"). It can contain slashes for subdirectories relative to the main target directory.
        4.  Create specific rules based on the user request. If the request is vague, try to create sensible defaults based on common file types.
        5.  Include a fallback rule `["True", "Uncategorized"]` at the end ONLY if no other rules seem appropriate or if the request is extremely broad. If specific rules cover most cases, you might not need it.
        6.  OUTPUT ONLY THE PYTHON LIST AS VALID JSON. No explanations, comments, or surrounding text.

        Example Output Format:
        [
            ["metadata['category'] == 'document' and 'invoice' in metadata['name'].lower()", "Financial/Invoices"],
            ["metadata['category'] == 'code' and metadata['extension'] == '.py'", "Code/Python"],
            ["metadata['extension'] in ['.zip', '.rar', '.7z']", "Archives"],
            ["metadata['category'] == 'image'", "Media/Images"],
            ["int(metadata['modified_date'].split('-')[0]) < 2020", "Archived/Old Files"],
            ["True", "Miscellaneous"]
        ]
        """
    else:
        # Analyze a sample of files for automatic rule generation
        sample_size = min(len(file_list), 30) # Analyze up to 30 files
        logging.info(f"No user description provided. Analyzing {sample_size} files for automatic rule generation.")
        if not file_list:
             logging.warning("File list is empty. Cannot generate context-based rules.")
             return DEFAULT_RULE[:] # Return a copy
        
        sample_metadata = [extract_metadata(f) for f in file_list[:sample_size]]
        # Filter out files where metadata extraction failed completely (all None)
        sample_metadata = [m for m in sample_metadata if m.get('name')]

        if not sample_metadata:
            logging.warning("Failed to extract metadata from any sample files. Using default rule.")
            return DEFAULT_RULE[:] # Return a copy

        # Simple context summarization
        categories = sorted(list(set(m['category'] for m in sample_metadata if m.get('category'))))
        extensions = sorted(list(set(m['extension'] for m in sample_metadata if m.get('extension'))))
        content_types = sorted(list(set(m['content_type'] for m in sample_metadata if m.get('content_type') and m.get('content_type') != 'non_text')))
        uses_dates = any(m['has_date_in_name'] for m in sample_metadata)
        uses_years = any(m['has_year_in_name'] for m in sample_metadata)

        context = {
            'file_count_analyzed': len(sample_metadata),
            'dominant_categories': categories,
            'present_extensions': extensions[:15], # Limit extensions listed
            'text_content_types': content_types,
            'uses_dates_in_names': uses_dates,
            'uses_years_in_names': uses_years,
            'sample_filenames': [m['name'] for m in sample_metadata[:5]] # Show a few examples
        }

        prompt = f"""
        Analyze the following file context summary and generate intelligent file sorting rules.
        The goal is to create a sensible folder structure based on the types of files present.

        File Context Summary:
        {json.dumps(context, indent=2)}

        Available file metadata fields for conditions (use `metadata['field_name']`):
        {metadata_keys_description}

        Output requirements:
        1.  Generate a Python list of lists, where each inner list is `["condition_string", "target_folder_name"]`.
        2.  The `condition_string` MUST be a valid Python boolean expression evaluating the `metadata` dictionary. Use standard Python operators (`==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not in`, `and`, `or`, `not`). Use methods like `.startswith()`, `.endswith()`, `.lower()` on string metadata fields.
        3.  The `target_folder_name` is a simple string for the destination subfolder (e.g., "Source Code", "Documents/Reports", "Media/Images"). It can contain slashes for subdirectories relative to the main target directory.
        4.  Prioritize rules based on the `category` and `content_type`.
        5.  Consider using date/year information if present (e.g., group by year `metadata['modified_date'].split('-')[0]`).
        6.  Group common file types logically (e.g., all images together, all documents together).
        7.  Include a fallback rule `["True", "Uncategorized"]` at the very end to catch anything not matched by specific rules.
        8.  OUTPUT ONLY THE PYTHON LIST AS VALID JSON. No explanations, comments, or surrounding text.

        Example Output Format:
        [
            ["metadata['category'] == 'code'", "Development/Code"],
            ["metadata['category'] == 'document'", "Documents"],
            ["metadata['category'] == 'image' or metadata['category'] == 'video'", "Media"],
            ["metadata['category'] == 'archive'", "Archives"],
            ["metadata['has_year_in_name'] and int(metadata['modified_date'].split('-')[0]) < 2021", "Archived/" + metadata['modified_date'].split('-')[0]],
            ["True", "Other Files"]
        ]
        """

    logging.info("Generating sorting rules via Gemini API...")
    try:
        # Increased timeout, decreased retries for potentially long generation
        # Configure safety settings to be less restrictive for this use case if needed
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]
        # response = model.generate_content(prompt, request_options={"timeout": 120, "retry": 2}, safety_settings=safety_settings)

        response = model.generate_content(prompt, request_options={"timeout": 120})

        # More robust response parsing
        if not response.candidates:
            logging.error("AI response issue: No candidates found.")
            return DEFAULT_RULE[:] # Return a copy
        
        # Handle potential content filtering or other issues
        if response.candidates[0].finish_reason != 1: # 1 = STOP
             logging.error(f"AI generation finished unexpectedly. Reason: {response.candidates[0].finish_reason}. Safety Ratings: {response.candidates[0].safety_ratings}")
             # Log the prompt feedback if available
             if response.prompt_feedback:
                 logging.error(f"Prompt Feedback: {response.prompt_feedback}")
             return DEFAULT_RULE[:] # Return a copy


        response_text = response.candidates[0].content.parts[0].text.strip()

        # Find the JSON block more reliably
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1

        if 0 <= json_start < json_end:
            json_str = response_text[json_start:json_end]
            try:
                rules = json.loads(json_str)
                # Basic validation of rule structure
                if isinstance(rules, list) and all(isinstance(rule, list) and len(rule) == 2 and isinstance(rule[0], str) and isinstance(rule[1], str) for rule in rules):
                    logging.info(f"Successfully generated {len(rules)} sorting rules.")
                    # Log the generated rules at DEBUG level
                    logging.debug(f"Generated rules: {rules}")
                    return rules
                else:
                    logging.error(f"Generated JSON is not in the expected format (List[List[str, str]]): {json_str}")
                    return DEFAULT_RULE[:] # Return a copy
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from AI response: {e}\nResponse snippet: {json_str[:500]}")
                return DEFAULT_RULE[:] # Return a copy
        else:
            logging.error(f"Could not find valid JSON list ('[...]') in AI response.\nResponse text: {response_text}")
            return DEFAULT_RULE[:] # Return a copy

    except Exception as e:
        # Catch potential API errors, network issues, etc.
        logging.error(f"Error during AI rule generation or processing: {e}", exc_info=True)
        return DEFAULT_RULE[:] # Return a copy


# --- File System Operations ---

def find_files(root_dir: Path, ignored_dirs: Optional[set] = None) -> Generator[Path, None, None]:
    """
    Recursively find all files in the given directory, skipping ignored ones.

    Args:
        root_dir: Directory to search for files.
        ignored_dirs: A set of absolute directory paths to ignore during traversal.

    Yields:
        Each file Path found that is not within an ignored directory.
    """
    if ignored_dirs is None:
        ignored_dirs = set()

    # Ensure root_dir is absolute for comparison
    try:
        abs_root_dir = root_dir.resolve(strict=True) # strict=True raises FileNotFoundError if path doesn't exist
    except FileNotFoundError:
        logging.error(f"find_files: Source directory not found: {root_dir}")
        return
    except OSError as e:
         logging.error(f"find_files: Cannot resolve path {root_dir}: {e}")
         return

    # Check if the current directory itself should be ignored
    if abs_root_dir in ignored_dirs:
        logging.debug(f"Skipping ignored directory: {abs_root_dir}")
        return
        
    # Add common system/app directories to ignore by default (add more as needed)
    # These checks are basic; a more robust solution might use patterns
    default_ignores = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', '.env'}
    if root_dir.name in default_ignores or root_dir.name.startswith('~$'): # Skip temp Word files etc.
         logging.info(f"Skipping common ignored directory: {root_dir}")
         return

    try:
        for item in root_dir.iterdir():
            try:
                # Skip hidden files/directories (unless root itself is hidden)
                if item.name.startswith('.') and not root_dir.name.startswith('.'):
                     logging.debug(f"Skipping hidden item: {item}")
                     continue
                     
                # Skip common system/temp files
                if item.name.lower() in ['desktop.ini', 'thumbs.db', '.ds_store']:
                    logging.debug(f"Skipping system file: {item}")
                    continue

                if item.is_file():
                    yield item
                elif item.is_dir():
                    # Recursively process subdirectories, passing ignored_dirs down
                    yield from find_files(item, ignored_dirs)
            except (PermissionError, OSError) as e:
                logging.warning(f"Could not access item {item}: {e}")
            except FileNotFoundError:
                 logging.warning(f"Item disappeared during scan: {item}") # Handle race condition
    except (PermissionError, OSError) as e:
        logging.warning(f"Could not read directory {root_dir}: {e}")
    except FileNotFoundError:
         logging.warning(f"Directory disappeared during scan: {root_dir}")


def move_file(file_path: Path, target_dir: Path, dry_run: bool = False) -> bool:
    """
    Move a file to the target directory, handling name collisions.

    Args:
        file_path: Path to the file to move.
        target_dir: Directory to move the file into.
        dry_run: If True, only log the action without moving.

    Returns:
        True if the file was (or would be) moved, False otherwise.
    """
    try:
        # Ensure target directory exists
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        elif not target_dir.exists():
             logging.info(f"[Dry Run] Would create directory: {target_dir}")


        target_path = target_dir / file_path.name

        # Handle potential name collisions
        counter = 1
        original_stem = file_path.stem
        original_suffix = file_path.suffix

        while target_path.exists():
            # Prevent moving onto itself if source/target overlap in complex ways
            if target_path.resolve() == file_path.resolve():
                 logging.warning(f"Skipping move: Source and target path are identical ({file_path})")
                 return False

            # Smart counter append: remove existing counter like _1, _2 before adding new one
            match = re.match(r'(.*?)_(\d+)$', original_stem)
            base_stem = match.group(1) if match else original_stem

            new_name = f"{base_stem}_{counter}{original_suffix}"
            target_path = target_dir / new_name
            counter += 1
            if counter > 100: # Safety break for potential infinite loop
                logging.error(f"Could not find unique name for {file_path.name} in {target_dir} after 100 attempts. Skipping.")
                return False

        if dry_run:
            if target_path.name != file_path.name:
                 logging.info(f"[Dry Run] Would move {file_path.name} to {target_dir / target_path.name} (renamed due to collision)")
            else:
                 logging.info(f"[Dry Run] Would move {file_path.name} to {target_path}")
            return True
        else:
            # Use shutil.move for better cross-device compatibility
            shutil.move(str(file_path), str(target_path))
            if target_path.name != file_path.name:
                 logging.info(f"Moved and renamed {file_path.name} to {target_path}")
            else:
                 logging.info(f"Moved {file_path.name} to {target_path}")
            return True

    except (IOError, OSError, shutil.Error) as e:
        logging.error(f"Error moving file {file_path} to {target_dir}: {e}")
    except Exception as e:
         logging.error(f"Unexpected error moving file {file_path} to {target_dir}: {e}", exc_info=True)
    return False


# --- Main Sorting Logic ---

def sort_files(
    source_dir: Path,
    target_root: Path,
    user_description: str = "",
    dry_run: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None
) -> Tuple[int, int]:
    """
    Sort files from source directory to target directory using AI-generated rules.

    Args:
        source_dir: Directory containing files to sort.
        target_root: Root directory to sort files into subfolders.
        user_description: Optional description to guide AI sorting rule generation.
        dry_run: If True, simulate sorting without moving files.
        progress_callback: Optional function to call with progress percentage (0.0 to 100.0).
        status_callback: Optional function to call with status update strings.

    Returns:
        Tuple of (files_processed, files_moved_or_simulated).
    """
    def update_status(message: str):
        logging.info(message)
        if status_callback:
            status_callback(message)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found or is not a directory: {source_dir}")

    if not target_root.exists():
        update_status(f"Target directory {target_root} does not exist.")
        if dry_run:
            update_status("[Dry Run] Would create target root directory.")
        else:
            try:
                target_root.mkdir(parents=True, exist_ok=True)
                update_status(f"Created target directory: {target_root}")
            except (OSError, PermissionError) as e:
                raise OSError(f"Could not create target directory {target_root}: {e}") from e
    elif not target_root.is_dir():
         raise NotADirectoryError(f"Target path exists but is not a directory: {target_root}")

    # --- Phase 1: Find Files ---
    update_status(f"Scanning source directory: {source_dir}")
    try:
        # Find files, ignoring the target directory if it's inside the source
        ignored_dirs = {target_root.resolve()} if target_root.resolve().is_relative_to(source_dir.resolve()) else set()
        file_list = list(find_files(source_dir, ignored_dirs=ignored_dirs))
    except Exception as e:
        logging.error(f"Error finding files in {source_dir}: {e}", exc_info=True)
        raise # Re-raise after logging

    if not file_list:
        update_status("No files found in source directory (or all ignored/inaccessible).")
        if progress_callback: progress_callback(100.0) # Mark as complete
        return 0, 0

    update_status(f"Found {len(file_list)} files to process.")

    # --- Phase 2: Generate Rules ---
    update_status("Generating sorting rules using AI...")
    rules = generate_sorting_rules(file_list, user_description)
    if not rules or rules == DEFAULT_RULE:
        update_status("Using default/fallback sorting rules.")
    else:
        update_status(f"Generated {len(rules)} custom sorting rules.")

    # --- Phase 3: Compile Rules (Optimization) ---
    compiled_rules: List[Tuple[Any, str]] = []
    update_status("Compiling rules...")
    for i, (condition_str, target_dir_name) in enumerate(rules):
        try:
            compiled_condition = compile(condition_str, f"<rule_{i}>", "eval")
            compiled_rules.append((compiled_condition, target_dir_name))
        except SyntaxError as e:
            logging.error(f"Syntax error in generated rule condition #{i+1}: '{condition_str}'. Error: {e}. Skipping rule.")
        except Exception as e:
            logging.error(f"Unexpected error compiling rule condition #{i+1}: '{condition_str}'. Error: {e}. Skipping rule.", exc_info=True)

    if not compiled_rules:
        logging.error("No valid rules could be compiled. Aborting sort.")
        raise ValueError("Failed to compile any sorting rules.")

    # --- Phase 4: Process Files ---
    files_processed = 0
    files_moved_or_simulated = 0
    total_files = len(file_list)
    update_status(f"Starting file processing ({'Dry Run' if dry_run else 'Move Mode'})...")

    for i, file_path in enumerate(file_list):
        files_processed += 1
        update_status(f"Processing file {i+1}/{total_files}: {file_path.name}")

        if not file_path.exists():
            logging.warning(f"File disappeared before processing: {file_path}. Skipping.")
            continue

        metadata = extract_metadata(file_path)
        if not metadata.get('name'): # Check if metadata extraction failed
            logging.warning(f"Skipping file due to metadata extraction failure: {file_path.name}")
            continue

        target_sub_dir_name = None

        # Apply compiled rules
        for condition_code, target_name in compiled_rules:
            try:
                # Evaluate the compiled condition in a restricted but functional namespace
                # Provide access to common builtins used in conditions (int, str, float, bool, len)
                # and the metadata dict itself.
                allowed_globals = {"__builtins__": {"str": str, "int": int, "float": float, "bool": bool, "len": len, "print": None, "True": True, "False": False, "None": None}}
                allowed_locals = {"metadata": metadata}

                if eval(condition_code, allowed_globals, allowed_locals):
                    target_sub_dir_name = target_name
                    logging.debug(f"File '{file_path.name}' matched rule condition '{condition_code.co_filename}' -> target '{target_name}'")
                    break # First matching rule wins
            except NameError as e:
                 logging.error(f"Error evaluating condition '{condition_code.co_filename}' for {file_path.name}: Name not defined ({e}). Check rule syntax and available metadata keys. Skipping rule for this file.")
                 # Continue to next rule
            except TypeError as e:
                 logging.error(f"Error evaluating condition '{condition_code.co_filename}' for {file_path.name}: Type mismatch ({e}). Check rule logic against metadata types (e.g., comparing string to int?). Skipping rule for this file.")
                 # Continue to next rule
            except KeyError as e:
                 logging.error(f"Error evaluating condition '{condition_code.co_filename}' for {file_path.name}: Metadata key not found ({e}). Check rule syntax. Skipping rule for this file.")
                 # Continue to next rule
            except Exception as e:
                logging.error(f"Error evaluating condition '{condition_code.co_filename}' for file {file_path.name}: {e}", exc_info=True)
                # Decide whether to stop or just skip this rule for this file
                # For robustness, let's continue to the next rule
                continue

        if target_sub_dir_name is None:
            # This should theoretically not happen if the last rule is "True",
            # but handle defensively.
            logging.warning(f"File {file_path.name} did not match any rule, including fallback. Placing in 'Uncategorized'.")
            target_sub_dir_name = "Uncategorized"

        # Sanitize the target directory name (replace invalid chars) - Optional but recommended
        # Basic sanitization: remove chars problematic for filenames/paths
        sanitized_target_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', target_sub_dir_name)
        if sanitized_target_name != target_sub_dir_name:
             logging.debug(f"Sanitized target dir name from '{target_sub_dir_name}' to '{sanitized_target_name}'")

        # Construct the full target path
        full_target_dir = target_root / sanitized_target_name

        # Check if the file is already in the correct target directory
        try:
            # Use resolve() to compare canonical paths, handling symlinks etc.
            if file_path.parent.resolve() == full_target_dir.resolve():
                logging.debug(f"File {file_path.name} is already in the target directory {full_target_dir}. Skipping move.")
                continue
        except (OSError, FileNotFoundError) as e:
             logging.warning(f"Could not resolve path for comparison {file_path.parent} or {full_target_dir}: {e}")
             # Proceed with the move attempt cautiously


        # Move the file (or log if dry run)
        if move_file(file_path, full_target_dir, dry_run):
            files_moved_or_simulated += 1

        # Update progress
        if progress_callback:
            progress = (i + 1) / total_files * 100.0
            progress_callback(progress)

    update_status(f"Processing complete. Processed: {files_processed}. {'Moved' if not dry_run else 'Simulated Moves'}: {files_moved_or_simulated}.")
    if dry_run:
         update_status("Dry run finished. No files were actually moved.")

    return files_processed, files_moved_or_simulated


# --- GUI Implementation ---

def create_gui() -> None:
    """Create and run the tkinter GUI."""
    if not TKINTER_AVAILABLE:
        logging.error("Tkinter is not available. Cannot create GUI.")
        print("Error: Tkinter is required for the GUI but is not installed or configured correctly.")
        print_tkinter_install_help()
        sys.exit(1)

    logging.info("Launching GUI interface...")

    window = tk.Tk()
    window.title("AI File Sorter")
    window.geometry("650x600")  # Adjusted size for new elements
    window.minsize(550, 500) # Set a minimum size
    window.resizable(True, True)

    # --- Style ---
    style = ttk.Style()
    style.theme_use('clam') # Or 'alt', 'default', 'classic'

    # --- Variables ---
    config = load_config()
    api_key_var = tk.StringVar(value=API_KEY) # Use global API_KEY loaded earlier
    source_dir_var = tk.StringVar(value=config.get('source_dir', ''))
    target_dir_var = tk.StringVar(value=config.get('target_dir', ''))
    status_var = tk.StringVar(value="Ready")
    progress_var = tk.DoubleVar(value=0.0)
    dry_run_var = tk.BooleanVar(value=False)

    # --- Functions ---
    def update_status(message: str):
        """Update status bar and log."""
        logging.info(message)
        status_var.set(message)
        window.update_idletasks() # Force GUI update

    def update_progress(value: float):
        """Update progress bar."""
        progress_var.set(value)
        window.update_idletasks() # Force GUI update

    def browse_directory(dir_var: tk.StringVar, title: str):
        """Open directory dialog and update variable."""
        directory = filedialog.askdirectory(title=title)
        if directory:
            dir_var.set(directory)
            save_config(source_dir_var.get(), target_dir_var.get())

    def save_api_key_action():
        """Validate and save API key."""
        key = api_key_var.get().strip()
        if configure_gemini(key):
            messagebox.showinfo("Success", "Gemini API configured for this session.")
        else:
            messagebox.showerror("Error", "Failed to configure Gemini API. Check console logs and API key.")

    def run_sort_threaded():
        """Run the sorting logic in a separate thread."""
        # Disable button during sort
        sort_button.config(state=tk.DISABLED)
        dry_run_checkbox.config(state=tk.DISABLED)

        # Get values from GUI elements
        source_dir = source_dir_var.get().strip()
        target_dir = target_dir_var.get().strip()
        user_description = description_entry.get("1.0", "end-1c").strip()
        is_dry_run = dry_run_var.get()

        # --- Input Validation ---
        if not API_KEY: # Check the globally configured key
            messagebox.showerror("Error", "Gemini API key not configured. Please enter and save a valid key.")
            sort_button.config(state=tk.NORMAL)
            dry_run_checkbox.config(state=tk.NORMAL)
            return

        if not source_dir or not Path(source_dir).is_dir():
            messagebox.showerror("Error", "Invalid Source Directory. Please select an existing directory.")
            sort_button.config(state=tk.NORMAL)
            dry_run_checkbox.config(state=tk.NORMAL)
            return

        if not target_dir:
             messagebox.showerror("Error", "Target Directory cannot be empty. Please select a directory.")
             sort_button.config(state=tk.NORMAL)
             dry_run_checkbox.config(state=tk.NORMAL)
             return

        source_path = Path(source_dir)
        target_path = Path(target_dir)

        if not source_path.is_dir(): # Re-check just before starting
             messagebox.showerror("Error", f"Source directory not found: {source_path}")
             sort_button.config(state=tk.NORMAL)
             dry_run_checkbox.config(state=tk.NORMAL)
             return

        # Prevent sorting into itself directly (basic check)
        try:
            if source_path.resolve() == target_path.resolve():
                messagebox.showerror("Error", "Source and Target directories cannot be the same.")
                sort_button.config(state=tk.NORMAL)
                dry_run_checkbox.config(state=tk.NORMAL)
                return
            # More robust check: target is inside source
            if target_path.resolve().is_relative_to(source_path.resolve()):
                 if not messagebox.askyesno("Warning", f"The target directory ({target_path.name}) appears to be inside the source directory.\nThis is generally safe (target folder will be ignored during scan), but proceed with caution.\n\nContinue sorting?"):
                     sort_button.config(state=tk.NORMAL)
                     dry_run_checkbox.config(state=tk.NORMAL)
                     update_status("Sort cancelled by user.")
                     return
        except (OSError, FileNotFoundError) as e:
            messagebox.showerror("Error", f"Could not verify directory paths: {e}")
            sort_button.config(state=tk.NORMAL)
            dry_run_checkbox.config(state=tk.NORMAL)
            return


        # --- Execute Sorting ---
        try:
            update_status("Starting sort process...")
            update_progress(0.0) # Reset progress

            files_processed, files_moved = sort_files(
                source_path, target_path, user_description, is_dry_run,
                progress_callback=update_progress, # Pass GUI update functions
                status_callback=update_status
            )

            final_msg = f"Sort finished! Processed: {files_processed}, {'Moved' if not is_dry_run else 'Simulated'}: {files_moved}."
            update_status(final_msg)
            messagebox.showinfo("Success", final_msg)

        except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as e:
            error_msg = f"Sorting Error: {e}"
            update_status(error_msg)
            messagebox.showerror("Error", error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logging.error(error_msg, exc_info=True) # Log full traceback
            update_status(error_msg)
            messagebox.showerror("Error", error_msg)
        finally:
            # Re-enable button regardless of outcome
            sort_button.config(state=tk.NORMAL)
            dry_run_checkbox.config(state=tk.NORMAL)
            update_progress(0.0) # Reset progress bar


    def start_sort():
        """Starts the sorting process in a new thread."""
        # Create and start the thread
        sort_thread = threading.Thread(target=run_sort_threaded, daemon=True) # Daemon allows exit even if thread runs
        sort_thread.start()

    # --- GUI Layout ---

    # Main frame
    main_frame = ttk.Frame(window, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # API Key Frame
    api_frame = ttk.LabelFrame(main_frame, text="Gemini API Configuration", padding="10")
    api_frame.pack(fill=tk.X, pady=(0, 10))
    api_frame.columnconfigure(1, weight=1)

    ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
    api_key_entry = ttk.Entry(api_frame, textvariable=api_key_var, width=45, show="*") # Use show="*"
    api_key_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    api_save_button = ttk.Button(api_frame, text="Configure API", command=save_api_key_action)
    api_save_button.grid(row=0, column=2, padx=(5, 0), pady=5)

    # Directory Frame
    dir_frame = ttk.LabelFrame(main_frame, text="Directory Selection", padding="10")
    dir_frame.pack(fill=tk.X, pady=(0, 10))
    dir_frame.columnconfigure(1, weight=1)

    ttk.Label(dir_frame, text="Source:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
    source_entry = ttk.Entry(dir_frame, textvariable=source_dir_var, width=40)
    source_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    source_button = ttk.Button(dir_frame, text="Browse...", command=lambda: browse_directory(source_dir_var, "Select Source Directory"))
    source_button.grid(row=0, column=2, padx=(5, 0), pady=5)

    ttk.Label(dir_frame, text="Target:").grid(row=1, column=0, padx=(0, 5), pady=5, sticky="w")
    target_entry = ttk.Entry(dir_frame, textvariable=target_dir_var, width=40)
    target_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    target_button = ttk.Button(dir_frame, text="Browse...", command=lambda: browse_directory(target_dir_var, "Select Target Directory"))
    target_button.grid(row=1, column=2, padx=(5, 0), pady=5)

    # Sorting Configuration Frame
    config_frame = ttk.LabelFrame(main_frame, text="Sorting Configuration", padding="10")
    config_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    config_frame.rowconfigure(1, weight=1) # Allow text box to expand
    config_frame.columnconfigure(0, weight=1) # Allow text box to expand

    ttk.Label(config_frame, text="Optional Sorting Description (Leave blank for automatic rules):").grid(row=0, column=0, columnspan=2, padx=0, pady=(0,5), sticky="w")
    # Use ScrolledText for potentially long descriptions
    description_entry = scrolledtext.ScrolledText(config_frame, height=6, width=50, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
    description_entry.grid(row=1, column=0, columnspan=2, padx=0, pady=5, sticky="nsew")

    # Dry Run Checkbox
    dry_run_checkbox = ttk.Checkbutton(config_frame, text="Dry Run (Simulate sorting, don't move files)", variable=dry_run_var)
    dry_run_checkbox.grid(row=2, column=0, columnspan=2, padx=0, pady=(10, 5), sticky="w")

    # Action Frame (Sort Button)
    action_frame = ttk.Frame(main_frame)
    action_frame.pack(fill=tk.X, pady=(5, 0))
    action_frame.columnconfigure(0, weight=1) # Center button

    sort_button = ttk.Button(action_frame, text="Start Sorting", command=start_sort, style='Accent.TButton') # Use Accent style if available
    sort_button.grid(row=0, column=0, pady=5)
    try: # Apply accent style if supported
         style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'), foreground='white', background='#0078D4') # Example blue
    except tk.TclError:
         logging.warning("Accent button style not available on this system.")


    # --- Status Bar & Progress Bar ---
    status_frame = ttk.Frame(window, relief=tk.SUNKEN, padding=2)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)

    status_label = ttk.Label(status_frame, textvariable=status_var, anchor="w")
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    progress_bar = ttk.Progressbar(status_frame, variable=progress_var, maximum=100.0, length=150)
    progress_bar.pack(side=tk.RIGHT, padx=5)

    # --- Start GUI ---
    window.mainloop()

# --- CLI Implementation ---

def print_tkinter_install_help():
    """Prints helpful messages for installing Tkinter."""
    print("\n---")
    print("Note: To use the graphical interface (GUI), the 'tkinter' module is required.")
    print("It seems tkinter is not installed or accessible in your current Python environment.")
    print("\nInstallation Instructions:")
    print("  Windows:")
    print("    - Tkinter is usually included with the standard Python installer from python.org.")
    print("    - If missing, re-run the Python installer and ensure 'tcl/tk and IDLE' is selected.")
    print("  macOS:")
    print("    - Using Homebrew (Recommended):")
    print("      `brew install python-tk`")
    print("      (Ensure you are using the Homebrew-installed Python: `which python3`)")
    print("    - Using pyenv:")
    print("      `brew install tcl-tk`")
    print("      Then reinstall your Python version with the Tcl/Tk flags, e.g.:")
    print("      `env PYTHON_CONFIGURE_OPTS=\"--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'\" pyenv install 3.11.4` (replace version)")
    print("  Linux (Debian/Ubuntu):")
    print("    `sudo apt-get update && sudo apt-get install python3-tk`")
    print("  Linux (Fedora):")
    print("    `sudo dnf install python3-tkinter`")
    print("\nAfter installation, please restart the script.")
    print("---\n")


def run_cli():
    """Run the command-line interface version of the file sorter."""
    parser = argparse.ArgumentParser(
        description="AI File Sorter: Sorts files into categorized subdirectories using Google Gemini.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve newline formatting in help
        )
    parser.add_argument(
        "source",
        help="Source directory containing files to sort."
        )
    parser.add_argument(
        "target",
        help="Target root directory where categorized subdirectories will be created."
        )
    parser.add_argument(
        "--api-key",
        help="Gemini API Key. Overrides GEMINI_API_KEY environment variable if provided."
        )
    parser.add_argument(
        "--description", "-d",
        default="",
        help="Natural language description of how you want files sorted.\n(e.g., 'Group photos by year', 'Separate invoices and reports').\nIf omitted, AI will generate rules based on file types found."
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the sorting process without actually moving any files. Logs planned actions."
        )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (Default: INFO)."
        )
    parser.add_argument(
        "--log-file",
        help="Optional file path to write logs to, in addition to the console."
        )

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.getLogger().setLevel(log_level)
    
    # Clear existing handlers (like the basicConfig one) to avoid duplicates if run multiple times
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        
    # Console Handler (always add)
    console_handler = logging.StreamHandler(sys.stderr) # Use stderr for logs
    console_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(console_handler)

    # File Handler (optional)
    if args.log_file:
        try:
            log_path = Path(args.log_file).resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True) # Create dir if needed
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            logging.getLogger().addHandler(file_handler)
            logging.info(f"Logging to file: {log_path}")
        except Exception as e:
            logging.error(f"Failed to configure logging to file {args.log_file}: {e}")


    # --- Configure API Key ---
    cli_api_key = args.api_key
    effective_api_key = cli_api_key or API_KEY # CLI arg takes precedence over env/global

    if not effective_api_key:
        logging.critical("Error: No Gemini API key provided.")
        logging.critical("Please provide one using --api-key argument or by setting the GEMINI_API_KEY environment variable.")
        sys.exit(1)

    if not configure_gemini(effective_api_key):
        logging.critical("Failed to configure the Gemini API with the provided key. Exiting.")
        sys.exit(1)

    # --- Validate Paths ---
    source_dir = Path(args.source)
    target_dir = Path(args.target)

    try:
        source_dir = source_dir.resolve(strict=True) # Check existence and get absolute path
    except FileNotFoundError:
        logging.critical(f"Error: Source directory not found: {args.source}")
        sys.exit(1)
    except OSError as e:
        logging.critical(f"Error accessing source directory {args.source}: {e}")
        sys.exit(1)

    if not source_dir.is_dir():
        logging.critical(f"Error: Source path is not a directory: {source_dir}")
        sys.exit(1)

    # Target directory doesn't strictly need to exist beforehand,
    # but we should resolve its path and check for conflicts.
    try:
        target_dir = target_dir.resolve() # Get absolute path, creates parent if needed later
    except OSError as e:
        logging.critical(f"Error resolving target directory path {args.target}: {e}")
        sys.exit(1)

    if target_dir.exists() and not target_dir.is_dir():
        logging.critical(f"Error: Target path exists but is not a directory: {target_dir}")
        sys.exit(1)

    # Check for dangerous overlap (target == source or target inside source)
    if source_dir == target_dir:
        logging.critical("Error: Source and Target directories cannot be the same.")
        sys.exit(1)
        
    if target_dir.is_relative_to(source_dir):
         logging.warning(f"Warning: Target directory ({target_dir}) is inside the Source directory ({source_dir}).")
         logging.warning("The target folder itself will be skipped during the scan, but ensure this is intended.")
         # Optionally add a confirmation prompt here for interactive CLI use
         # input("Press Enter to continue or Ctrl+C to cancel...")


    logging.info(f"Source directory: {source_dir}")
    logging.info(f"Target directory: {target_dir}")
    if args.description:
        logging.info(f"Using description: \"{args.description}\"")
    else:
        logging.info("No description provided, using automatic rule generation.")
    if args.dry_run:
        logging.info("--- DRY RUN MODE ENABLED ---")


    # --- Execute Sorting ---
    try:
        files_processed, files_moved = sort_files(
            source_dir,
            target_dir,
            args.description,
            args.dry_run,
            status_callback=logging.info # Use logger for status in CLI
            )
        logging.info("=" * 30)
        logging.info(f"Sorting Finished. Processed: {files_processed}. {'Moved' if not args.dry_run else 'Simulated Moves'}: {files_moved}.")
        if args.dry_run:
            logging.info("--- DRY RUN COMPLETE - NO FILES MOVED ---")
        logging.info("=" * 30)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as e:
        logging.critical(f"Sorting failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred during sorting: {e}", exc_info=True)
        sys.exit(1)

# --- Main Execution ---

if __name__ == "__main__":
    # Load configuration early (sets API key from env if available)
    load_config()

    # Determine mode (CLI vs GUI)
    # Use CLI if any arguments are passed (besides script name) OR if tkinter is unavailable
    if len(sys.argv) > 1 or not TKINTER_AVAILABLE:
        if not TKINTER_AVAILABLE and len(sys.argv) <= 1:
            # Only show detailed help if GUI was expected but unavailable
            logging.warning("Tkinter module not found or failed to import. GUI is disabled.")
            print_tkinter_install_help()
            print("Running in Command-Line Interface (CLI) mode.")
            print("Use '--help' argument for usage instructions.")
            print("\nExample CLI command:")
            print(f"  python {os.path.basename(__file__)} ./my_files ./sorted_files --description \"Organize by file type\"")
            # We don't exit here, let run_cli() handle potential arg errors

        run_cli()
    else:
        # Launch GUI
        create_gui()