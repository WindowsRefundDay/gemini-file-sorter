# Gemini File Sorter 📂🤖

A Python script that intelligently sorts files into organized subdirectories using the Google Gemini AI. It analyzes file metadata and content (where applicable) to generate custom sorting rules, either automatically or based on your instructions.

## ✨ Key Features

*   **AI-Powered Rule Generation:** Leverages Google Gemini to create effective sorting rules.
*   **Flexible Sorting Logic:**
    *   **Automatic Mode:** Generates rules based on the types of files found in the source directory.
    *   **Custom Mode:** Provide a natural language description (e.g., "Put invoices in Finance/2024", "Group photos by year") for tailored sorting.
*   **Comprehensive Metadata Analysis:** Considers filename, extension, MIME type, creation/modification dates, size, and even performs basic content analysis for text-based files.
*   **User-Friendly GUI (Optional):** Simple Tkinter interface for easy configuration and operation (requires `tkinter`).
*   **Robust Command-Line Interface (CLI):** Full functionality available via CLI arguments for automation or use on systems without a GUI.
*   **Dry Run Mode:** Simulate the sorting process and see what *would* happen without actually moving any files – perfect for testing rules!
*   **Configuration Persistence:** Remembers source/target directories between sessions (saved in `~/.ai_file_sorter_config.json`).
*   **Enhanced MIME Type Detection:** Uses `python-magic` (if available) for more accurate file type identification.
*   **Collision Handling:** Automatically renames files if a file with the same name already exists in the target directory.
*   **Logging:** Provides informative console output and optional logging to a file for debugging.
*   **Responsive GUI:** Sorting operations run in a background thread to prevent the GUI from freezing.

## ⚙️ Requirements

*   **Python:** 3.8 or higher recommended.
*   **Libraries:**
    *   `google-generativeai`: For interacting with the Gemini API.
    *   `python-dotenv`: For loading API keys from a `.env` file.
    *   `python-magic`: For improved file type detection (see Installation notes).
    *   `tkinter`: **Required only for the GUI mode.** (Often included with Python, but may need separate installation - see Troubleshooting).
*   **Google Gemini API Key:** You need an API key from Google AI Studio.
    *   Get one here: [https://aistudio.google.com/](https://aistudio.google.com/)

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/windowsrefundday/gemini-file-sorter.git
    cd gemini-file-sorter
    ```

2.  **Install required Python packages:**
    *(It's recommended to use a virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on `python-magic`:**
        *   **Windows:** You might need `pip install python-magic-bin` instead.
        *   **Linux (Debian/Ubuntu):** You may need `sudo apt-get install libmagic1`.
        *   **macOS (Homebrew):** You may need `brew install libmagic`.
        *   If `python-magic` cannot be installed or fails, the script will fall back to standard MIME type detection, which might be less accurate for some files.

3.  **Configure your API Key:** See the "Configuration" section below.

## 🔑 Configuration (Gemini API Key)

You need to provide your Gemini API key to the script. It checks for the key in this order of priority:

1.  **GUI Input:** Entered directly into the API Key field in the GUI (valid for the current session only).
2.  **CLI Argument:** Passed via the `--api-key` flag when running from the command line.
3.  **Environment Variable:** Set the `GEMINI_API_KEY` environment variable in your system.
    *   **Linux/macOS:** `export GEMINI_API_KEY="YOUR_API_KEY_HERE"`
    *   **Windows (Command Prompt):** `set GEMINI_API_KEY=YOUR_API_KEY_HERE`
    *   **Windows (PowerShell):** `$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"`
4.  **`.env` File:** Create a file named `.env` in the *same directory* as the script (`sorter.py`) and add the following line:
    ```dotenv
    GEMINI_API_KEY=YOUR_API_KEY_HERE
    ```
    *(The `.env` file method is often the easiest for local development.)*

**Note:** Directory paths selected via the GUI are saved automatically to `~/.ai_file_sorter_config.json` for persistence across sessions. The API key is *not* saved in this file for security reasons.

## ▶️ Usage

### Graphical User Interface (GUI)

*(Requires `tkinter` to be installed and working)*

1.  **Run the script without arguments:**
    ```bash
    python sorter.py
    ```
2.  **Configure API Key:** If not set via environment variable or `.env`, enter your Gemini API key in the "API Key" field and click "Configure API".
3.  **Select Directories:** Use the "Browse..." buttons to choose the "Source" directory (containing files to sort) and the "Target" directory (where categorized subfolders will be created).
4.  **(Optional) Provide Sorting Description:** Enter instructions in the text box if you want specific sorting logic (e.g., "Group documents by year", "Separate work and personal photos"). Leave blank for automatic rule generation based on file types.
5.  **(Optional) Enable Dry Run:** Check the "Dry Run" box to simulate the sort without moving files. Check the console output or log file to see the planned actions.
6.  **Click "Start Sorting".** Progress and status updates will appear at the bottom.

### Command-Line Interface (CLI)

1.  **Run the script with arguments:**
    ```bash
    python sorter.py <source_directory> <target_directory> [options]
    ```

2.  **Required Arguments:**
    *   `<source_directory>`: Path to the folder containing files you want to sort.
    *   `<target_directory>`: Path to the folder where sorted subdirectories will be created.

3.  **Optional Arguments:**
    *   `--api-key YOUR_API_KEY`: Specify your API key directly (overrides environment/`.env`).
    *   `--description "Your sorting instructions"` or `-d "..."`: Provide natural language sorting rules.
    *   `--dry-run`: Perform a simulation without moving files.
    *   `--log-level LEVEL`: Set console logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.
    *   `--log-file FILEPATH`: Write logs to the specified file in addition to the console.

4.  **Examples:**
    *   **Automatic sorting:**
        ```bash
        python sorter.py ./Downloads ./SortedDownloads
        ```
    *   **Custom sorting with description:**
        ```bash
        python sorter.py /path/to/docs /path/to/organized -d "Put PDFs in 'Manuals', put images > 5MB in 'Large Photos'"
        ```
    *   **Dry run:**
        ```bash
        python sorter.py ./MessyFolder ./CleanFolder --dry-run
        ```
    *   **Debug logging to file:**
        ```bash
        python sorter.py ./Input ./Output --log-level DEBUG --log-file sorter.log
        ```

## ☂️ Troubleshooting

*   **`tkinter not found` / `No module named _tkinter` Error:**
    This means the GUI library is missing or not linked correctly in your Python installation.
    *   **Windows:** Tkinter is usually included. Try reinstalling Python from [python.org](https://python.org/), ensuring the "tcl/tk and IDLE" component is selected during installation.
    *   **macOS (Homebrew Python):** `brew install python-tk`
    *   **macOS (System Python / pyenv):** Installation can be complex. See Python documentation or pyenv guides for installing with Tcl/Tk support. Often involves `brew install tcl-tk` first.
    *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install python3-tk`
    *   **Linux (Fedora):** `sudo dnf install python3-tkinter`
    *   **Alternative:** Use the CLI mode, which does not require `tkinter`.
*   **API Key Errors:** Ensure your key is correct, has billing enabled (if required by Google), and is provided correctly via one of the configuration methods. Check console logs for specific error messages from the Gemini API.
*   **`python-magic` Errors:** If you have issues installing or running `python-magic`, ensure the underlying `libmagic` library is installed correctly on your system (see Installation). If issues persist, the script will still function but with potentially less accurate file typing.
*   **Permission Errors:** Ensure the script has read permissions for the source directory and write permissions for the target directory.
*   **Other Issues:** Check the console output or the log file (if specified using `--log-file`) for detailed error messages. Use `--log-level DEBUG` for maximum detail.

## 📝 TODO List

*   **Local Model Support:**
    *   Add support for Gemma local models for offline sorting
    *   Integrate with Ollama for local model inference
    *   Support for GGUF models via llama.cpp
    *   Allow users to choose between cloud and local models

*   **Enhanced Sorting Features:**
    *   Add support for custom sorting rule templates
    *   Implement recursive sorting rules (sort within sorted folders)
    *   Add file deduplication based on content hash
    *   Support for sorting based on file content (e.g., image recognition, document topics)
    *   Add batch processing for large directories
    *   Implement undo/redo functionality for file moves

*   **User Interface Improvements:**
    *   Add dark mode support
    *   Create a web interface alternative to tkinter
    *   Add file preview functionality
    *   Show sorting statistics and visualizations
    *   Add drag-and-drop support
    *   Implement real-time directory monitoring for automatic sorting

*   **Performance & Scalability:**
    *   Add parallel processing for large file sets
    *   Implement caching for file metadata
    *   Add support for network drives and cloud storage
    *   Optimize memory usage for large directories
    *   Add progress bars for long-running operations

*   **Security & Reliability:**
    *   Add file operation logging and audit trail
    *   Implement backup functionality before moving files
    *   Add checksum verification for moved files
    *   Implement file permission preservation
    *   Add support for encrypted directories

*   **Integration & Extensibility:**
    *   Create plugin system for custom sorting rules
    *   Add support for cloud storage services (Google Drive, Dropbox)
    *   Implement file tagging system
    *   Add support for external metadata databases
    *   Create API for external tool integration

*   **Documentation & Testing:**
    *   Add comprehensive test suite
    *   Create video tutorials
    *   Add internationalization support
    *   Create detailed API documentation
    *   Add benchmarking tools

## 🤝 Contributing

Contributions are welcome!

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure requirements are updated if necessary (`requirements.txt`).
5.  Commit your changes (`git commit -am 'Add some feature'`).
6.  Push to the branch (`git push origin feature/your-feature-name`).
7.  Create a new Pull Request.

## 🔒 Security Note

Your Gemini API key is sensitive.
*   **Never commit your API key directly into the code.**
*   Use environment variables or a `.env` file (which should be added to your `.gitignore`) to handle keys securely.

## 📄 License

This project is open source licensed under the [MIT License](LICENSE).
