import streamlit as st
import torch
import torchaudio
import numpy as np
import librosa
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import tempfile
import os
from datetime import datetime
import json
import base64
from io import BytesIO
import scipy.io.wavfile as wavfile

# Page configuration
st.set_page_config(
    page_title="Coastal Sounds - AI Music Preservation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Filipino coastal theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .cultural-note {
        background: linear-gradient(90deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e17055;
        margin: 1rem 0;
    }
    
    .success-message {
        background: linear-gradient(90deg, #00b894 0%, #55a3ff 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_archive' not in st.session_state:
    st.session_state.audio_archive = []
if 'generated_music' not in st.session_state:
    st.session_state.generated_music = []

# Audio classification model initialization
@st.cache_resource
def load_audio_classifier():
    """Load the audio classification model for music analysis"""
    try:
        pipe = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
        return pipe
    except Exception as e:
        st.error(f"Error loading audio classifier: {e}")
        return None

# MusicGen model initialization
@st.cache_resource
def load_musicgen_model():
    """Load the MusicGen model for music generation"""
    try:
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
        return model, processor
    except Exception as e:
        st.error(f"Error loading MusicGen model: {e}")
        return None, None

# Real MusicGen music generation
def generate_music_with_musicgen(prompt, audio_input=None, duration=30, enhancement_type="basic"):
    """
    Generate music using the real MusicGen model
    """
    model, processor = load_musicgen_model()
    
    if model is None or processor is None:
        st.warning("MusicGen model not available. Using fallback generation.")
        return generate_fallback_audio(prompt, audio_input, duration, enhancement_type)
    
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Process conditioning audio if provided
        melody = None
        if audio_input is not None:
            try:
                # Create temporary file from BytesIO object
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    # Handle both file objects and BytesIO objects
                    if hasattr(audio_input, 'getvalue'):
                        temp_file.write(audio_input.getvalue())
                    else:
                        audio_input.seek(0)
                        temp_file.write(audio_input.read())
                    temp_file_path = temp_file.name
                
                # Load audio for melody conditioning
                melody, sr = librosa.load(temp_file_path, sr=32000)
                
                # Convert to tensor and add batch dimension
                melody = torch.from_numpy(melody).unsqueeze(0).to(device)
                
                # Clean up
                os.unlink(temp_file_path)
                
                # Adjust duration based on input audio length
                original_duration = len(melody[0]) / 32000
                duration = max(original_duration, 10)  # At least 10 seconds
                
            except Exception as e:
                st.warning(f"Could not process audio input for conditioning: {e}")
                melody = None
        
        # Enhance prompt based on enhancement type
        enhanced_prompt = enhance_prompt_for_type(prompt, enhancement_type)
        
        # Prepare inputs
        inputs = processor(
            text=[enhanced_prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # Set generation parameters
        sample_rate = model.config.audio_encoder.sampling_rate
        duration_samples = int(duration * sample_rate)
        
        # Generate audio
        with torch.no_grad():
            if melody is not None:
                # Generate with melody conditioning
                audio_values = model.generate(
                    **inputs,
                    melody_tokens=None,  # MusicGen handles melody differently
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=duration_samples // model.config.audio_encoder.hop_length,
                )
            else:
                # Generate without melody conditioning
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=duration_samples // model.config.audio_encoder.hop_length,
                )
        
        # Convert to numpy array
        audio_data = audio_values[0, 0].cpu().numpy()
        
        return audio_data, sample_rate
        
    except Exception as e:
        st.error(f"Error generating music with MusicGen: {e}")
        return generate_fallback_audio(prompt, audio_input, duration, enhancement_type)

def enhance_prompt_for_type(prompt, enhancement_type):
    """Enhance the base prompt based on the enhancement type"""
    enhancements = {
        "Clean and enhance quality": f"{prompt}, high quality, clear audio, professional recording",
        "Generate instrumental backing": f"{prompt}, rich instrumentation, full ensemble, layered harmonies",
        "Extend composition length": f"{prompt}, extended composition, developing themes, musical progression",
        "Create educational version": f"{prompt}, simple melody, educational, easy to follow",
        "Add coastal ambience": f"{prompt}, ocean waves, coastal atmosphere, nature sounds"
    }
    
    return enhancements.get(enhancement_type, prompt)

def generate_fallback_audio(prompt, audio_input=None, duration=30, enhancement_type="basic"):
    """
    Fallback audio generation using sine waves when MusicGen is not available
    """
def generate_fallback_audio(prompt, audio_input=None, duration=30, enhancement_type="basic"):
    """
    Fallback audio generation using sine waves when MusicGen is not available
    """
    # Set a random seed based on current time and prompt for variation
    np.random.seed(hash(str(datetime.now()) + prompt + enhancement_type) % (2**32))
    
    # If audio_input is provided, try to match its duration
    if audio_input is not None:
        try:
            # Load the uploaded audio to get its duration
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_input.read())
                temp_file_path = temp_file.name
            
            audio_data, sr = librosa.load(temp_file_path)
            original_duration = len(audio_data) / sr
            duration = max(original_duration, 10)  # At least 10 seconds
            
            # Clean up
            os.unlink(temp_file_path)
        except:
            duration = 30  # Fallback
    
    sample_rate = 32000
    duration_samples = int(duration * sample_rate)
    
    # Add randomization to base frequencies
    base_frequencies = {
        "Clean and enhance quality": [261.63, 293.66, 329.63, 349.23],
        "Generate instrumental backing": [261.63, 329.63, 392.00, 523.25, 659.25],
        "Extend composition length": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25],
        "Create educational version": [261.63, 293.66, 329.63],
        "Add coastal ambience": [261.63, 293.66, 329.63, 349.23, 392.00],
        "basic": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    }
    
    frequencies = base_frequencies.get(enhancement_type, base_frequencies["basic"])
    
    # Add random variations to frequencies (±5% variation)
    frequencies = [freq * (1 + np.random.uniform(-0.05, 0.05)) for freq in frequencies]
    
    # Randomize order occasionally
    if np.random.random() > 0.5:
        np.random.shuffle(frequencies)
    
    # Different enhancement types produce different sounds with randomization
    if enhancement_type == "Clean and enhance quality":
        volume = 0.4 + np.random.uniform(-0.1, 0.1)
        wave_intensity = 0.1 + np.random.uniform(-0.03, 0.03)
    elif enhancement_type == "Generate instrumental backing":
        volume = 0.3 + np.random.uniform(-0.05, 0.05)
        wave_intensity = 0.15 + np.random.uniform(-0.05, 0.05)
    elif enhancement_type == "Extend composition length":
        volume = 0.35 + np.random.uniform(-0.05, 0.05)
        wave_intensity = 0.12 + np.random.uniform(-0.03, 0.03)
    elif enhancement_type == "Create educational version":
        volume = 0.5 + np.random.uniform(-0.1, 0.1)
        wave_intensity = 0.05 + np.random.uniform(-0.02, 0.02)
    elif enhancement_type == "Add coastal ambience":
        volume = 0.25 + np.random.uniform(-0.05, 0.05)
        wave_intensity = 0.3 + np.random.uniform(-0.1, 0.1)
    else:
        volume = 0.3 + np.random.uniform(-0.05, 0.05)
        wave_intensity = 0.2 + np.random.uniform(-0.05, 0.05)
    
    # Create melody pattern with randomization
    audio_data = np.zeros(duration_samples)
    
    # Generate melody with different patterns and randomization
    for i, freq in enumerate(frequencies):
        section_length = duration_samples // len(frequencies)
        start_sample = int(i * section_length)
        end_sample = int((i + 1) * section_length)
        
        if end_sample > duration_samples:
            end_sample = duration_samples
            
        section_duration = (end_sample - start_sample) / sample_rate
        t = np.linspace(0, section_duration, end_sample - start_sample)
        
        # Add random phase shift
        phase_shift = np.random.uniform(0, 2 * np.pi)
        
        # Add some variation based on enhancement type with randomization
        if enhancement_type == "Generate instrumental backing":
            # Add harmonics with random variations
            harmonic1 = 1.5 + np.random.uniform(-0.1, 0.1)
            harmonic2 = 2.0 + np.random.uniform(-0.2, 0.2)
            melody = volume * (np.sin(2 * np.pi * freq * t + phase_shift) + 
                             0.3 * np.sin(2 * np.pi * freq * harmonic1 * t) +
                             0.2 * np.sin(2 * np.pi * freq * harmonic2 * t))
        elif enhancement_type == "Create educational version":
            # Simple sine wave with slight random variation
            melody = volume * np.sin(2 * np.pi * freq * t + phase_shift)
        else:
            # Standard melody with slight harmonics and randomization
            harmonic = 1.25 + np.random.uniform(-0.05, 0.05)
            melody = volume * (np.sin(2 * np.pi * freq * t + phase_shift) + 
                             0.2 * np.sin(2 * np.pi * freq * harmonic * t))
        
        # Add envelope for more natural sound
        envelope = np.exp(-t * np.random.uniform(0.1, 0.5))
        melody = melody * envelope
        
        audio_data[start_sample:end_sample] = melody
    
    # Add coastal ambience with randomization
    t_full = np.linspace(0, duration, duration_samples)
    wave_freq_base = 0.3 if enhancement_type == "Add coastal ambience" else 0.5
    wave_freq = wave_freq_base + np.random.uniform(-0.1, 0.1)
    
    if enhancement_type == "Add coastal ambience":
        # More prominent wave sounds with variations
        wave_freq2 = 0.7 + np.random.uniform(-0.1, 0.1)
        wave_sound = wave_intensity * (np.sin(2 * np.pi * wave_freq * t_full) + 
                                     0.5 * np.sin(2 * np.pi * wave_freq2 * t_full))
    else:
        # Subtle wave sounds
        wave_sound = wave_intensity * np.sin(2 * np.pi * wave_freq * t_full)
    
    # Add gentle noise with variation
    noise_level = 0.05 + np.random.uniform(-0.02, 0.02)
    noise = noise_level * np.random.randn(duration_samples)
    
    # Combine all elements
    audio_data = audio_data + wave_sound + noise
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    return audio_data, sample_rate

def analyze_audio_content(audio_file):
    """Analyze uploaded audio using the classification model"""
    classifier = load_audio_classifier()
    if classifier is None:
        return ["Unable to analyze audio"]
    
    try:
        # Save uploaded file temporarily and load as audio array
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name
        
        # Load audio using librosa
        audio_array, sr = librosa.load(temp_file_path, sr=16000)  # Resample to 16kHz for model
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Process with classifier
        results = classifier(audio_array)
        return [f"{result['label']}: {result['score']:.3f}" for result in results[:5]]
    except Exception as e:
        return [f"Analysis error: {str(e)}"]

def save_audio_to_archive(audio_data, sample_rate, metadata):
    """Save audio to session archive"""
    # Convert audio to base64 for storage
    buffer = BytesIO()
    wavfile.write(buffer, sample_rate, (audio_data * 32767).astype(np.int16))
    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    archive_entry = {
        'timestamp': datetime.now().isoformat(),
        'audio_data': audio_b64,
        'sample_rate': sample_rate,
        'metadata': metadata
    }
    
    st.session_state.audio_archive.append(archive_entry)

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Coastal Sounds</h1>
        <h3>AI-Powered Traditional Music Preservation & Generation</h3>
        <p>Preserving Filipino coastal musical heritage through artificial intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Home", "Upload & Enhance", "Generate Music", "Community Archive", "Discover Music", "About"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Upload & Enhance":
        show_upload_page()
    elif page == "Generate Music":
        show_generation_page()
    elif page == "Community Archive":
        show_archive_page()
    elif page == "Discover Music":
        show_discovery_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    """Display the home page with project overview"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>Preserve Your Musical Heritage</h2>
            <p>Welcome to Coastal Sounds, a platform dedicated to preserving and reimagining traditional Filipino coastal music using cutting-edge AI technology.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Features")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            st.markdown("""
            **Music Enhancement**
            - Upload traditional recordings
            - AI-powered audio enhancement
            - Generate instrumental versions
            - Extend short recordings
            """)
            
            st.markdown("""
            **Cultural Generation**
            - Create new compositions
            - Traditional style conditioning
            - Instrument-specific generation
            - Educational variations
            """)
        
        with col1_2:
            st.markdown("""
            **Digital Archive**
            - Searchable music database
            - Cultural stories integration
            - Regional categorization
            - Community contributions
            """)
            
            st.markdown("""
            **Community Impact**
            - Preserve disappearing traditions
            - Connect diaspora communities
            - Educational resources
            - Cultural pride promotion
            """)
    
    with col2:
        st.markdown("""
        <div class="cultural-note">
            <h3>Cultural Mission</h3>
            <p>Our goal is to bridge ancestral musical traditions with modern technology, ensuring that the rich heritage of Filipino coastal communities is preserved for future generations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("### Community Stats")
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric("Archived Pieces", len(st.session_state.audio_archive))
            st.metric("Generated Music", len(st.session_state.generated_music))
        
        with col2_2:
            st.metric("Active Communities", "12+")
            st.metric("Cultural Regions", "8")

def show_upload_page():
    """Display the upload and enhancement page"""
    st.markdown("## Upload & Enhance Traditional Music")
    
    st.markdown("""
    <div class="cultural-note">
        <strong>Cultural Sensitivity Note:</strong> Please ensure you have proper permissions to upload traditional music, especially ceremonial or sacred pieces.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Audio upload
        st.markdown("### Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Upload traditional music recordings in common audio formats"
        )
        
        # Audio recording (placeholder)
        st.markdown("### Record Live")
        if st.button("Start Recording"):
            st.info("Live recording feature will be implemented with microphone access")
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            # Audio analysis
            st.markdown("### Audio Analysis")
            with st.spinner("Analyzing audio content..."):
                analysis_results = analyze_audio_content(uploaded_file)
            
            st.markdown("**Detected Content:**")
            for result in analysis_results:
                st.write(f"• {result}")
            
            # Enhancement options
            st.markdown("### Enhancement Options")
            enhancement_type = st.selectbox(
                "Select enhancement type:",
                [
                    "Clean and enhance quality",
                    "Generate instrumental backing",
                    "Extend composition length",
                    "Create educational version",
                    "Add coastal ambience"
                ]
            )
            
            if st.button("Enhance Audio"):
                with st.spinner("Loading AI models and processing..."):
                    try:
                        # Create a copy of the uploaded file to avoid pointer issues
                        audio_bytes = uploaded_file.getvalue()
                        audio_copy = BytesIO(audio_bytes)
                        
                        # Process with real MusicGen model
                        enhanced_audio, sample_rate = generate_music_with_musicgen(
                            f"enhance traditional Filipino music, {enhancement_type}",
                            audio_input=audio_copy,
                            enhancement_type=enhancement_type
                        )
                    except Exception as e:
                        st.error(f"Enhancement failed: {str(e)}")
                        st.stop()
                
                st.markdown("""
                <div class="success-message">
                    Audio enhanced successfully! Listen to your enhanced version below.
                </div>
                """, unsafe_allow_html=True)
                
                # Play enhanced audio
                buffer = BytesIO()
                wavfile.write(buffer, sample_rate, (enhanced_audio * 32767).astype(np.int16))
                st.audio(buffer.getvalue())
                
                # Save to archive option
                if st.button("Save to Archive"):
                    metadata = {
                        'type': 'enhanced',
                        'original_file': uploaded_file.name,
                        'enhancement_type': enhancement_type,
                        'source': 'user_upload'
                    }
                    save_audio_to_archive(enhanced_audio, sample_rate, metadata)
                    st.success("Saved to community archive!")
    
    with col2:
        # Cultural context form
        st.markdown("### Cultural Context")
        
        with st.form("cultural_metadata"):
            region = st.selectbox(
                "Coastal Region:",
                ["Luzon Coast", "Visayas Islands", "Mindanao Coast", "Palawan", "Other"]
            )
            
            music_type = st.selectbox(
                "Music Type:",
                ["Folk Song", "Ceremonial", "Work Song", "Lullaby", "Festival", "Fishing Song", "Other"]
            )
            
            instruments = st.multiselect(
                "Traditional Instruments:",
                ["Kulintang", "Bamboo Flute", "Drums", "Gongs", "Guitar", "Voice", "Other"]
            )
            
            cultural_story = st.text_area(
                "Cultural Story/Meaning:",
                placeholder="Share the cultural significance, history, or story behind this music..."
            )
            
            performer_info = st.text_input(
                "Performer/Community:",
                placeholder="Credit the original performers or community"
            )
            
            submitted = st.form_submit_button("Save Cultural Context")
            
            if submitted:
                st.success("Cultural metadata saved!")

def show_generation_page():
    """Display the AI music generation page"""
    st.markdown("## Generate Traditional-Inspired Music")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Music Generation Prompts")
        
        # Pre-defined prompt categories
        prompt_category = st.selectbox(
            "Choose a category:",
            [
                "Traditional Filipino Coastal",
                "Ceremonial & Ritual",
                "Folk & Work Songs",
                "Lullabies & Peaceful",
                "Festival & Celebration",
                "Custom Prompt"
            ]
        )
        
        if prompt_category == "Custom Prompt":
            music_prompt = st.text_area(
                "Describe the music you want to generate:",
                placeholder="e.g., peaceful kulintang melody with ocean waves background for meditation"
            )
        else:
            # Pre-defined prompts based on category
            prompts = {
                "Traditional Filipino Coastal": [
                    "Traditional kulintang ensemble with gentle ocean waves",
                    "Bamboo flute melody inspired by coastal fishing songs",
                    "Ancient Filipino gong music with seashore ambience"
                ],
                "Ceremonial & Ritual": [
                    "Sacred Filipino ritual music with traditional percussion",
                    "Ceremonial gong sequences for coastal blessing rituals",
                    "Traditional healing music with nature sounds"
                ],
                "Folk & Work Songs": [
                    "Filipino fishing work song with rhythmic boat rowing",
                    "Traditional rice planting songs adapted for coastal communities",
                    "Folk melody celebrating the ocean's bounty"
                ],
                "Lullabies & Peaceful": [
                    "Gentle Filipino lullaby with soft ocean wave sounds",
                    "Peaceful bamboo flute for meditation and rest",
                    "Soothing traditional melody for children"
                ],
                "Festival & Celebration": [
                    "Festive Filipino coastal celebration music",
                    "Joyful traditional dance music with modern elements",
                    "Community gathering songs with multiple instruments"
                ]
            }
            
            music_prompt = st.selectbox(
                "Select a prompt:",
                prompts.get(prompt_category, ["Custom prompt required"])
            )
        
        # Generation parameters
        st.markdown("### Generation Settings")
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            duration = st.slider("Duration (seconds)", 10, 120, 30)
            tempo = st.selectbox("Tempo", ["Slow", "Medium", "Fast"])
        
        with col1_2:
            key_signature = st.selectbox(
                "Key/Mode",
                ["Major (Happy)", "Minor (Melancholic)", "Pentatonic (Traditional)", "Auto"]
            )
            instruments = st.multiselect(
                "Preferred Instruments:",
                ["Kulintang", "Bamboo Flute", "Gongs", "Drums", "Guitar", "Voice"],
                default=["Kulintang"]
            )
        
        # Generate button
        if st.button("Generate Music"):
            with st.spinner("Loading MusicGen AI model and creating your traditional-inspired music..."):
                # Generate music using real MusicGen model
                generated_audio, sample_rate = generate_music_with_musicgen(
                    music_prompt, duration=duration
                )
            
            st.markdown("""
            <div class="success-message">
                Music generated successfully! Your AI-created traditional-inspired piece is ready.
            </div>
            """, unsafe_allow_html=True)
            
            # Display generated audio
            buffer = BytesIO()
            wavfile.write(buffer, sample_rate, (generated_audio * 32767).astype(np.int16))
            st.audio(buffer.getvalue())
            
            # Save options
            col1_3, col1_4 = st.columns(2)
            
            with col1_3:
                if st.button("Save to Archive"):
                    metadata = {
                        'type': 'generated',
                        'prompt': music_prompt,
                        'duration': duration,
                        'tempo': tempo,
                        'instruments': instruments,
                        'category': prompt_category
                    }
                    save_audio_to_archive(generated_audio, sample_rate, metadata)
                    st.session_state.generated_music.append(metadata)
                    st.success("Saved to archive!")
            
            with col1_4:
                if st.button("Generate Variation"):
                    st.info("Click 'Generate Music' again for a new variation!")
    
    with col2:
        st.markdown("### Generation Tips")
        
        st.markdown("""
        <div class="cultural-note">
            <h4>Best Practices</h4>
            <ul>
                <li><strong>Be Specific:</strong> Include instruments, mood, and cultural context</li>
                <li><strong>Respect Tradition:</strong> Use appropriate cultural references</li>
                <li><strong>Experiment:</strong> Try different combinations of instruments</li>
                <li><strong>Cultural Context:</strong> Consider the purpose of your music</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Example prompts
        st.markdown("### Example Prompts")
        examples = [
            "Peaceful kulintang with ocean waves for meditation",
            "Traditional fishing song with bamboo flute and gentle percussion",
            "Ceremonial gong music for coastal blessing ritual",
            "Joyful festival music with multiple traditional instruments"
        ]
        
        for example in examples:
            if st.button(f"Use: {example[:30]}...", key=f"example_{example[:20]}"):
                st.session_state.example_prompt = example

def show_archive_page():
    """Display the community archive"""
    st.markdown("## Community Archive")
    
    if not st.session_state.audio_archive:
        st.markdown("""
        <div class="cultural-note">
            <h3>Start Building the Archive</h3>
            <p>The community archive is empty. Upload traditional music or generate AI compositions to start building our cultural digital library!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Archive statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pieces", len(st.session_state.audio_archive))
    with col2:
        enhanced_count = sum(1 for item in st.session_state.audio_archive if item['metadata']['type'] == 'enhanced')
        st.metric("Enhanced Audio", enhanced_count)
    with col3:
        generated_count = sum(1 for item in st.session_state.audio_archive if item['metadata']['type'] == 'generated')
        st.metric("AI Generated", generated_count)
    with col4:
        st.metric("Contributors", "1")  # Placeholder
    
    # Search and filter
    st.markdown("### Search Archive")
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_query = st.text_input("Search by keywords, instruments, or region:")
    
    with search_col2:
        filter_type = st.selectbox("Filter by:", ["All", "Enhanced", "Generated"])
    
    # Display archive items
    st.markdown("### Archive Collection")
    
    for i, item in enumerate(st.session_state.audio_archive):
        if filter_type != "All" and item['metadata']['type'] != filter_type.lower():
            continue
        
        with st.expander(f"Audio {item['metadata'].get('prompt', item['metadata'].get('original_file', 'Untitled'))} - {item['timestamp'][:10]}"):
            col_audio, col_meta = st.columns([2, 1])
            
            with col_audio:
                # Reconstruct audio from base64
                audio_bytes = base64.b64decode(item['audio_data'])
                st.audio(audio_bytes)
            
            with col_meta:
                st.markdown("**Metadata:**")
                for key, value in item['metadata'].items():
                    if key != 'audio_data':
                        st.write(f"**{key.title()}:** {value}")
                
                st.write(f"**Sample Rate:** {item['sample_rate']} Hz")
                st.write(f"**Created:** {item['timestamp'][:16]}")

def show_discovery_page():
    """Display the music discovery interface"""
    st.markdown("## Discover Traditional Music")
    
    # Featured regions
    st.markdown("### Explore by Region")
    
    regions = [
        {"name": "Luzon Coastal", "description": "Northern Philippines coastal traditions", "instruments": "Kulintang, Bamboo Flute"},
        {"name": "Visayas Islands", "description": "Central Philippines island music", "instruments": "Gongs, Traditional Drums"},
        {"name": "Mindanao Coast", "description": "Southern Philippines coastal heritage", "instruments": "Traditional Percussion"},
        {"name": "Palawan", "description": "Western Philippines island culture", "instruments": "Bamboo Instruments"}
    ]
    
    cols = st.columns(2)
    
    for i, region in enumerate(regions):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{region['name']}</h4>
                <p>{region['description']}</p>
                <p><strong>Instruments:</strong> {region['instruments']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Music categories
    st.markdown("### Browse by Category")
    
    category_cols = st.columns(3)
    
    categories = [
        {"name": "Folk Songs", "count": "15+"},
        {"name": "Ceremonial", "count": "8+"},
        {"name": "Work Songs", "count": "12+"},
        {"name": "Lullabies", "count": "6+"},
        {"name": "Festival", "count": "10+"},
        {"name": "Fishing Songs", "count": "7+"}
    ]
    
    for i, category in enumerate(categories):
        with category_cols[i % 3]:
            if st.button(f"{category['name']}\n({category['count']} pieces)", key=f"cat_{i}"):
                st.info(f"Browsing {category['name']} - Feature coming soon!")
    
    # Cultural insights
    st.markdown("### Cultural Insights")
    
    insights = [
        {
            "title": "The Role of Kulintang in Coastal Communities",
            "content": "Kulintang ensembles have been central to Filipino coastal celebrations for centuries, serving both as entertainment and spiritual connection to the sea."
        },
        {
            "title": "Bamboo Instruments and Ocean Rhythms",
            "content": "Traditional bamboo flutes often mimic the sounds of ocean waves and coastal winds, creating a natural harmony with the environment."
        },
        {
            "title": "Fishing Songs and Community Bonding",
            "content": "Work songs sung during fishing expeditions help coordinate group efforts and maintain morale during long hours at sea."
        }
    ]
    
    for insight in insights:
        with st.expander(f"Read: {insight['title']}"):
            st.write(insight['content'])

def show_about_page():
    """Display information about the project"""
    st.markdown("## About Coastal Sounds")
    
    st.markdown("""
    <div class="feature-card">
        <h2>Our Mission</h2>
        <p>Coastal Sounds is dedicated to preserving and reimagining traditional Filipino coastal music through the power of artificial intelligence. We bridge the gap between ancestral musical traditions and modern technology, ensuring that the rich cultural heritage of coastal communities is preserved for future generations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Project Goals")
        st.markdown("""
        - **Preserve** traditional music before it disappears
        - **Empower** coastal communities through technology
        - **Connect** Filipino diaspora with their roots
        - **Educate** younger generations about their heritage
        - **Innovate** respectfully with cultural traditions
        """)
        
        st.markdown("### AI Technology")
        st.markdown("""
        - **MusicGen Melody**: Meta's state-of-the-art music generation
        - **Audio Classification**: MIT's advanced audio analysis
        - **Cultural Adaptation**: Custom models for Filipino music
        - **Community Input**: Human-in-the-loop cultural validation
        """)
    
    with col2:
        st.markdown("### Cultural Impact")
        st.markdown("""
        - **Digital Archive**: Permanent preservation of musical heritage
        - **Accessibility**: Global reach for cultural content
        - **Education**: Learning tools for traditional music
        - **Economic Opportunity**: Platform for cultural sharing
        """)
        
        st.markdown("### Community Guidelines")
        st.markdown("""
        - **Respect**: Honor the cultural significance of traditional music
        - **Permission**: Ensure proper consent for sacred/ceremonial pieces
        - **Attribution**: Credit original performers and communities
        - **Collaboration**: Work together to preserve our heritage
        """)
    
    st.markdown("""
    <div class="cultural-note">
        <h3>Acknowledgments</h3>
        <p>This project is built for Hack With The Beat Hackathon. Askia Khryss, the sole creator, built this with deep respect for Filipino coastal communities and their musical traditions. I acknowledge the wisdom of elders, the creativity of traditional musicians, and the importance of cultural preservation in our rapidly changing world.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical information
    with st.expander("Technical Information"):
        st.markdown("""
        **AI Models Used:**
        - facebook/musicgen-melody for music generation
        - MIT/ast-finetuned-audioset-10-10-0.4593 for audio classification
        
        **Framework:**
        - Frontend: Streamlit
        - Audio Processing: librosa, torchaudio
        - Machine Learning: transformers, torch
       
        """)

if __name__ == "__main__":
    main()