# 🧠 SmartDoc Analyzer

SmartDoc Analyzer is a next-generation document intelligence tool that allows you to securely analyze and interact with various file types, including PDFs, images, spreadsheets, text files, JSON, Word documents, and PowerPoint presentations. With its advanced AI-powered conversational interface, you can ask questions about your documents and get meaningful insights in real-time.

---

## 🚀 Features

- **Multi-File Support**: Analyze multiple file types, including:
  - PDFs
  - Images (`.png`, `.jpg`, `.jpeg`)
  - Spreadsheets (`.csv`, `.xlsx`, `.xls`)
  - Text files (`.txt`)
  - JSON files
  - Word documents (`.docx`)
  - PowerPoint presentations (`.pptx`)
- **AI-Powered Chat Interface**: Ask questions about your documents and receive intelligent responses.
- **Customizable Prompts**: Tailored prompts for JSON analysis, including UUID detection, nested object exploration, and more.
- **Secure and Efficient**: Processes files locally and ensures data privacy.
- **Parallel Processing**: Uses multithreading to process multiple files simultaneously for faster results.
- **Interactive UI**: Built with Streamlit, featuring a modern and user-friendly interface.
- **Source References**: Provides references to the source documents for every response.

---

## 🛠️ Setup

### Prerequisites

1. **Python**: Ensure you have Python 3.8 or higher installed.
2. **Dependencies**: Install the required Python packages.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/smartdoc-analyzer.git
   cd smartdoc-analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   VISION_AGENT_API_KEY=your_vision_agent_api_key
   ```

4. Run the application:
   ```bash
   streamlit run SMARTDOC.py
   ```

---

## 📋 Usage

### 1. Upload Documents
- Use the file uploader in the sidebar to upload your documents.
- Supported file types: `.pdf`, `.csv`, `.xlsx`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.json`, `.docx`, `.pptx`.

### 2. Process Documents
- Click the **🚀 Process Documents** button to analyze the uploaded files.
- The app will extract content from the documents and prepare it for interaction.

### 3. Ask Questions
- Use the chat interface to ask questions about your documents.
- Example questions:
  - "What are the key points in the PDF?"
  - "Summarize the data in the spreadsheet."
  - "What does the JSON file contain?"

### 4. View Source References
- Expand the **🔍 View Source References** section to see the source documents for the responses.

---

## 🖥️ Supported File Types and Processing Details

### 1. **PDF Files**
- Extracts text using `Agentic Document Extracter` from Landing.ai.
- Processes text into chunks for efficient querying.

### 2. **Images**
- Uses OCR (Optical Character Recognition) to extract text from images.

### 3. **Spreadsheets**
- Reads `.csv` files using `pandas.read_csv`.
- Reads `.xlsx` and `.xls` files using `pandas.read_excel`.

### 4. **Text Files**
- Reads plain text files and splits them into chunks.

### 5. **JSON Files**
- Parses JSON data and supports both list and dictionary structures.
- Detects UUIDs, nested objects, and numeric/string IDs.

### 6. **Word Documents**
- Extracts text from `.docx` files using `python-docx`.

### 7. **PowerPoint Presentations**
- Extracts text from slides in `.pptx` files using `python-pptx`.

---

## 🧩 Architecture

### Key Components

1. **File Processing Pipeline**:
   - Detects file type and processes files accordingly.
   - Uses temporary files for efficient handling.

2. **Chunking**:
   - Splits large text into smaller chunks using `RecursiveCharacterTextSplitter`.

3. **Vectorstore**:
   - Embeds document chunks using `HuggingFaceEmbeddings`.
   - Stores embeddings in a FAISS vectorstore for fast retrieval.

4. **Conversational Chain**:
   - Powered by `ChatOpenAI` for intelligent responses.
   - Uses `ConversationBufferMemory` to maintain chat history.

5. **Streamlit UI**:
   - Provides an interactive interface for uploading files, processing documents, and chatting.

---


## 🐛 Debugging and Troubleshooting

### Common Issues

1. **Encoding Errors**:
   - If you encounter encoding issues with JSON or text files, ensure the file is UTF-8 encoded.

2. **Missing Dependencies**:
   - Ensure all required Python packages are installed:
     ```bash
     pip install -r requirements.txt
     ```

3. **API Key Errors**:
   - Ensure your `.env` file contains valid API keys.

### Debug Logs
- Debug logs are commented in the code to help identify issues during file processing.
- Check the terminal output for detailed error messages.


---

## 🤝 Contributing

We welcome contributions! If you'd like to contribute, please fork the repository and submit a pull request.

---




