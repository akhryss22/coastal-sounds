# Coastal Sounds: AI-Powered Traditional Music Preservation

**Preserving Filipino coastal musical heritage through artificial intelligence**

A web application that bridges ancestral musical traditions with modern AI technology, designed to serve rural Filipino coastal communities by providing tools for music preservation and generation.

**Live Demo** https://coastalsounds.streamlit.app/

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/coastal-sounds.git
   cd coastal-sounds
   ```

2. Create virtual environment
   ```bash
   python -m venv coastal_sounds_env
   source coastal_sounds_env/bin/activate  # Windows: coastal_sounds_env\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application
   ```bash
   streamlit run coastal_sounds_app.py
   ```

5. Open your browser to `http://localhost:8501`

## Technology Stack

### AI Models
- **facebook/musicgen-melody**: Meta's music generation model
- **MIT/ast-finetuned-audioset-10-10-0.4593**: Audio classification for content analysis

### Core Technologies
- **Frontend**: Streamlit
- **Audio Processing**: librosa, torchaudio, scipy
- **Machine Learning**: PyTorch, Transformers
- **Data Handling**: NumPy, Pandas

## Usage Guide

1. **Upload & Enhance**: Upload traditional recordings, select enhancement options, and add cultural metadata
2. **Generate Music**: Create AI compositions using cultural prompts and traditional instrument specifications
3. **Browse Archive**: Search and explore preserved music with cultural context
4. **Discover Heritage**: Explore music by region and learn about traditional instruments

## Cultural Guidelines

- **Respect Sacred Music**: Ensure proper permissions for ceremonial pieces
- **Community Consent**: Always credit original performers and communities
- **Cultural Authenticity**: Maintain respect for traditional meanings and contexts
- **Educational Purpose**: Use AI generation responsibly for cultural preservation

## License

This project is licensed under the MIT License. Traditional music and cultural content shared on this platform remain the intellectual property of their respective communities.

---

*Made with care for Filipino coastal communities and cultural preservation worldwide.*
