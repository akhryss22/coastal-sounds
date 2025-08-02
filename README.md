# Coastal Sounds: AI-Powered Traditional Music Preservation

**Preserving Filipino coastal musical heritage through artificial intelligence**

A Streamlit web application that bridges ancestral musical traditions with modern AI technology, designed to serve rural Filipino coastal communities by providing tools for music preservation and generation.

## Project Overview

Coastal Sounds addresses the critical need to preserve rapidly disappearing traditional Filipino coastal music. As younger generations migrate to urban areas, centuries-old musical traditions risk being lost forever. Our platform provides accessible tools for communities to document, enhance, and reimagine their cultural heritage using AI technology.

### Core Mission
- Preserve traditional music before it disappears
- Empower coastal communities through accessible technology  
- Connect Filipino diaspora communities with their cultural roots
- Educate younger generations about their musical heritage

## Key Features

### Music Enhancement & Processing
- Upload traditional recordings in multiple audio formats
- AI-powered enhancement to improve audio quality
- Generate instrumental backing tracks for vocals
- Extend short recordings into full compositions
- Automatic classification and tagging of musical content

### AI Music Generation
- Create new compositions inspired by Filipino coastal music
- Specialized prompts for traditional instruments (kulintang, bamboo flute, gongs)
- Generate music specific to different coastal regions
- Create educational versions for learning traditional music

### Community Archive
- Searchable database of original and AI-enhanced recordings
- Integration of cultural context, stories, and meanings
- Comprehensive metadata by region, instrument, and cultural significance
- Collaborative platform for cultural knowledge sharing

### Music Discovery
- Browse music by coastal regions and island communities
- Organized by folk songs, ceremonial music, work songs, lullabies, and festivals
- Educational content about traditional instruments and cultural significance

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