# LAISA - Housing Assistant Chatbot

LAISA (Legal AI Support Assistant) is a Streamlit-based chatbot application designed to help users navigate housing-related laws and regulations. Built by SIIP Group, it uses OpenAI's language models with Pinecone vector search to provide informative responses about housing rights, responsibilities, and regulations.

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. **Install dependencies**
   ```bash
   cd beta_streamlit-main
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```

### Testing Your Setup

test your configuration:

```bash
python setup_test.py
```

### Running the Application

1. **Setup Test (Recommended first step)**
   ```bash
   python setup_test.py
   ```

2. **Main Application**
   ```bash
   streamlit run app.py
   ```


3**Using tox (recommended for development)**
   ```bash
   tox -e laisa
   ```



### Running Tests

```bash
tox -e test_package
```
ntact the SIIP Group team or create an issue in the repository.