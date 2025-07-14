# FePTP: Iron-containing Phase Transformation Pathway Extractor

**FePTP**  is an open-source, end-to-end automated information extraction pipeline designed to assist researchers in identifying and extract phase transformation pathways involving iron-containing phases. It combines the power of large language models (LLMs) with fine-tuned small models for accurate, hallucination-resistant extraction from scientific literature.

---
## 🔍 Overview

FePTP integrates:

- **Zero-shot & in-context learning** using LLMs for flexible, domain-adaptive extraction.
    
- **Domain-adapted smaller models** for efficient topic filtering and attribute tagging.
    
- **Curator modules** to reduce factual errors:
    
- **Entity resolution & conflict handling** for robust structuring of transformation data.
    
This pipeline has been used to generate a curated dataset of **Fe-containing Phase Transformation Pathways (FePTP)** from experimentally or field-observed records.

---

## 🗂️ Project Structure

```
FePTP/
│
├── config/               # Configuration files
├── data/                   # Raw and processed data files
├── feptp_pipeline/    
│   ├── topic_filtering/    # Filtering documents by topic    
│   ├── IE/                # Information extraction and curation
│   └── paper2vector/          #  Paper-to-vector pipeline for semantic search
├── logs/                   # Log
├── model/                # Model checkpoints and fine-tuned weights
├── output/                # OCR results
├── prompt/               # Prompt templates for LLMs
├── script/                 # Executable scripts
├── docker-compose.yml     # Docker-based deployment setup for Weaviate
├── setup.py
├── requirements.txt
└── README.md
```
---

## 🚀 Getting Started

### 1. Clone and install
First, clone the repository and set up a virtual environment:
```bash
git clone https://github.com/yourusername/FePTP.git 
cd FePTP
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Then, install the required dependencies and the project as a package:
```bash
pip install .
```
### 2. Create a Weaviate instance
We provide a pre-configured `docker-compose.yml` file in the project root.  
To start a local Weaviate instance:
```bash
docker-compose up -d  # Wait until everything is downloaded
python script/create_db_weaviate.py
```
This will:
- Launch Weaviate with default ports (`8080` for REST, `50051` for gRPC)
- Prepare it for receiving vectorized documents.
See more at [Weaviate](https://weaviate.io/).
### 3. [Optional] Download Required OCR Models (if PDF processing is needed)
This project uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for table and layout extraction. Due to licensing and file size constraints, the following models **must be downloaded manually** and placed in the appropriate folder. Download from [Model List](https://paddlepaddle.github.io/PaddleOCR/main/en/version2.x/ppstructure/models_list.html).
#### Required Models

| Model Type       | Download URL                      | Target Directory (relative to project root) |
| ---------------- | --------------------------------- | ------------------------------------------- |
| Layout Model     | ppyolov2_r50vd_dcn_365e_publaynet | `adds/paddleocr/layout/`                    |
| Table Structure  | en_ppstructure_mobile_v2.0_SLANet | `adds/paddleocr/table/`                     |
| Text Detection   | en_ppocr_mobile_v2.0_table_det    | `adds/paddleocr/det/en/`                    |
| Text Recognition | en_ppocr_mobile_v2.0_table_rec    | `adds/paddleocr/rec/en/`                    |
#### How to Set Up
1. Download each `.tar` file using the links above.
    
2. Extract each archive into its respective directory. 
    
3. After extraction, ensure the directory structure matches what the script expects.

### 4. Run on your own data
You can use FePTP’s pipeline on your own collection of scientific papers. Follow the steps below to prepare your data and customize the extraction logic.
#### 📦 Prepare Your Data in MongoDB
You can use [**scicrawler**](https://github.com/Laaery/scicrawler.git) to automatically collect and parse scientific papers and store them in MongoDB.
Each document inserted should follow this structure:
```json
{"doi": "10.1234/example.5678",
"title": "Title of Your Paper",
"abstract": "Abstract text here...",
"full_text": "The full text of the paper, as plain text or structured string",
"full_text_type": "plain text"}
```
Optional fields:
- `"table"`: list of extracted tables
- `"si"`: list of local paths to supplementary material files (e.g., `["si/file1.pdf", "si/file2.docx"]`)
👉 After collection, your documents will be stored in your specified MongoDB database and collection, ready for vectorization by the FePTP pipeline.
> 📘 **Tip**: You can also insert your own structured data manually using any MongoDB client or script, as long as the schema above is followed.
#### 🧠 Configure Your LLM
To configure your LLM:
1. **Set your API key**  
    If you're using OpenAI or another cloud-hosted LLM provider, you need to set your API key as an environment variable. For example, 
    ```bash
    export OPENAI_API_KEY=your-key-here
    ```
	We use service from OpenAI and Google Vertex AI.
2. **Choose your model**  
    Support multiple LLM providers through [LangChain](https://www.langchain.com/).
3. **Modify client settings**  
    You can modify model backend, temperature, max tokens, etc., in ie_core module or a config file.
---
#### ✏️ Modify Prompt Templates
Prompt templates define how instructions and examples are formatted for the LLM. To customize them:
1. Go to the `prompt/` directory (or wherever your prompts are stored).
    
2. Customize the prompt freely according to your requirements
#### 🔍 Run the Pipeline
To run the FePTP pipeline on your MongoDB collection:
```bash
python script/text_vectorization/run_paper2vector.py 
    --db_name your_db_name \
    --collection_name your_collection_name   
```
This will:
- Connect to your MongoDB instance
- Retrieve documents from the specified collection
- Vectorize the text
- Store the vectors in Weaviate for semantic search

To extract content from a specific paper based on your query, you can use the `run_auto_extract.py` script:
```bash
python script/ie/run_auto_extract.py --doi xxx/xxxxxxxx --query "xxxx"
```
---

## 🔗 Citation

If you use **FePTP** or the FePTP dataset in your research, please cite:

bibtex