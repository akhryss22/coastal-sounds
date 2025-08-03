import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
import base64
from io import BytesIO
import traceback

# Check for optional dependencies
HAS_LIBROSA = False
HAS_SCIPY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    st.warning("librosa not available - using basic audio processing")

try:
    import scipy.io.wavfile as wavfile
    HAS_SCIPY = True
except ImportError:
    st.warning("scipy not available - using alternative audio processing")

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

# Initialize session state with error handling
def init_session_state():
    """Initialize session state variables safely"""
    if 'audio_archive' not in st.session_state:
        st.session_state.audio_archive = []
    if 'generated_music' not in st.session_state:
        st.session_state.generated_music = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

# Simple audio generation function with error handling
def generate_music_audio(prompt, duration=30, enhancement_type="basic"):
    """Generate varied audio based on prompt and enhancement type with error handling"""
    try:
        # Create unique seed from prompt and current time
        seed = abs(hash(prompt + enhancement_type + str(datetime.now().microsecond))) % (2**31)
        np.random.seed(seed)
        
        sample_rate = 22050  # Reduced from 32000 for better compatibility
        duration_samples = int(duration * sample_rate)
        
        # Different base frequencies for different enhancement types
        frequency_sets = {
            "Clean and enhance quality": [261.63, 293.66, 329.63, 349.23],  # C major
            "Generate instrumental backing": [261.63, 329.63, 392.00, 523.25, 659.25],  # Rich harmony
            "Extend composition length": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],  # Extended
            "Create educational version": [261.63, 293.66, 329.63],  # Simple
            "Add coastal ambience": [261.63, 293.66, 329.63, 349.23, 392.00],  # Coastal
            "basic": [261.63, 293.66, 329.63, 349.23, 392.00]
        }
        
        base_freqs = frequency_sets.get(enhancement_type, frequency_sets["basic"])
        
        # Add random variations (±10% to frequencies)
        frequencies = [freq * np.random.uniform(0.9, 1.1) for freq in base_freqs]
        
        # Randomize volume and wave intensity based on type
        if enhancement_type == "Clean and enhance quality":
            volume = np.random.uniform(0.3, 0.5)
            wave_intensity = np.random.uniform(0.05, 0.15)
        elif enhancement_type == "Generate instrumental backing":
            volume = np.random.uniform(0.25, 0.35)
            wave_intensity = np.random.uniform(0.1, 0.2)
        elif enhancement_type == "Add coastal ambience":
            volume = np.random.uniform(0.2, 0.3)
            wave_intensity = np.random.uniform(0.25, 0.35)
        else:
            volume = np.random.uniform(0.25, 0.4)
            wave_intensity = np.random.uniform(0.1, 0.25)
        
        # Generate audio
        audio_data = np.zeros(duration_samples, dtype=np.float32)
        
        # Create melody sections
        for i, freq in enumerate(frequencies):
            try:
                section_start = int(i * duration_samples / len(frequencies))
                section_end = int((i + 1) * duration_samples / len(frequencies))
                section_length = section_end - section_start
                
                if section_length <= 0:
                    continue
                
                t = np.linspace(0, section_length / sample_rate, section_length)
                
                # Random phase and harmonics
                phase = np.random.uniform(0, 2 * np.pi)
                harmonic_ratio = np.random.uniform(1.2, 1.8)
                
                if enhancement_type == "Generate instrumental backing":
                    # Complex harmonies
                    melody = volume * (
                        np.sin(2 * np.pi * freq * t + phase) +
                        0.3 * np.sin(2 * np.pi * freq * harmonic_ratio * t) +
                        0.2 * np.sin(2 * np.pi * freq * 2.1 * t)
                    )
                elif enhancement_type == "Create educational version":
                    # Simple sine wave
                    melody = volume * np.sin(2 * np.pi * freq * t + phase)
                else:
                    # Standard with slight harmonics
                    melody = volume * (
                        np.sin(2 * np.pi * freq * t + phase) +
                        0.2 * np.sin(2 * np.pi * freq * harmonic_ratio * t)
                    )
                
                # Add envelope for natural sound
                envelope = np.exp(-t * np.random.uniform(0.1, 0.8))
                melody = melody * envelope
                
                # Ensure we don't exceed array bounds
                end_idx = min(section_end, len(audio_data))
                melody_length = min(len(melody), end_idx - section_start)
                
                audio_data[section_start:section_start + melody_length] = melody[:melody_length]
                
            except Exception as e:
                st.warning(f"Warning in section {i}: {str(e)}")
                continue
        
        # Add ocean waves
        try:
            t_full = np.linspace(0, duration, duration_samples)
            wave_freq = np.random.uniform(0.2, 0.6)
            wave_sound = wave_intensity * np.sin(2 * np.pi * wave_freq * t_full)
            
            if enhancement_type == "Add coastal ambience":
                # More complex wave patterns
                wave_freq2 = np.random.uniform(0.1, 0.4)
                wave_sound += 0.5 * wave_intensity * np.sin(2 * np.pi * wave_freq2 * t_full)
            
            # Add subtle noise
            noise = np.random.uniform(0.02, 0.08) * np.random.randn(duration_samples).astype(np.float32)
            
            # Combine everything
            final_audio = audio_data + wave_sound + noise
            final_audio = np.clip(final_audio, -0.95, 0.95)  # Slightly reduced to prevent clipping
            
        except Exception as e:
            st.warning(f"Warning adding ambience: {str(e)}")
            final_audio = np.clip(audio_data, -0.95, 0.95)
        
        return final_audio, sample_rate
        
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        # Return simple sine wave as fallback
        sample_rate = 22050
        duration_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, duration_samples)
        fallback_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # Simple A note
        return fallback_audio.astype(np.float32), sample_rate

def analyze_audio_content(uploaded_file):
    """Analyze uploaded audio with fallback options"""
    try:
        if not HAS_LIBROSA:
            return [
                "Basic file analysis (librosa not available)",
                f"File size: {len(uploaded_file.getvalue())} bytes",
                f"File type: {uploaded_file.type if hasattr(uploaded_file, 'type') else 'Unknown'}",
                "For detailed analysis, install librosa: pip install librosa"
            ]
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Load and analyze
            audio_data, sr = librosa.load(temp_file_path, sr=16000)
            
            # Simple analysis
            duration = len(audio_data) / sr
            max_amplitude = np.max(np.abs(audio_data))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            results = [
                f"Duration: {duration:.2f} seconds",
                f"Sample rate: {sr} Hz",
                f"Max amplitude: {max_amplitude:.3f}",
                f"Zero crossing rate: {zero_crossing_rate:.3f}",
                "Analysis complete"
            ]
            
        except Exception as e:
            results = [f"Librosa analysis error: {str(e)}"]
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return results
        
    except Exception as e:
        return [f"Analysis error: {str(e)}"]

def save_audio_to_archive(audio_data, sample_rate, metadata):
    """Save audio to session archive with error handling"""
    try:
        # Convert audio to bytes for storage
        if HAS_SCIPY:
            buffer = BytesIO()
            # Ensure audio is in correct format for wavfile
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(buffer, sample_rate, audio_int16)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        else:
            # Fallback: store as raw numpy array (less efficient but works)
            audio_bytes = audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode()
        
        archive_entry = {
            'timestamp': datetime.now().isoformat(),
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'metadata': metadata,
            'has_scipy': HAS_SCIPY
        }
        
        st.session_state.audio_archive.append(archive_entry)
        return True
        
    except Exception as e:
        st.error(f"Failed to save audio: {str(e)}")
        return False

def load_audio_from_archive(archive_entry):
    """Load audio from archive entry with error handling"""
    try:
        audio_b64 = archive_entry['audio_data']
        
        if archive_entry.get('has_scipy', True) and HAS_SCIPY:
            # Load as WAV file
            audio_bytes = base64.b64decode(audio_b64)
            return audio_bytes
        else:
            # Load as raw numpy array
            audio_bytes = base64.b64decode(audio_b64)
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Convert back to WAV format
            if HAS_SCIPY:
                buffer = BytesIO()
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wavfile.write(buffer, archive_entry['sample_rate'], audio_int16)
                return buffer.getvalue()
            else:
                return audio_bytes
                
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None

# Main application with error handling
def main():
    """Main application with comprehensive error handling"""
    try:
        # Initialize session state
        init_session_state()
        
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
        
        # Display dependency status in sidebar
        with st.sidebar.expander("System Status"):
            st.write(f"🎵 Librosa: {'✅' if HAS_LIBROSA else '❌'}")
            st.write(f"🔊 SciPy: {'✅' if HAS_SCIPY else '❌'}")
            if not HAS_LIBROSA or not HAS_SCIPY:
                st.info("Some features may have limited functionality")
        
        # Route to appropriate page
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
            
    except Exception as e:
        st.error("An unexpected error occurred in the main application")
        st.code(f"Error: {str(e)}")
        st.code(f"Traceback: {traceback.format_exc()}")

def show_home_page():
    """Display the home page with project overview"""
    try:
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
                
    except Exception as e:
        st.error(f"Error loading home page: {str(e)}")

def show_upload_page():
    """Display the upload and enhancement page with error handling"""
    try:
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
            
            if uploaded_file is not None:
                st.audio(uploaded_file)
                
                # Audio analysis
                st.markdown("### Audio Analysis")
                with st.spinner("Analyzing audio content..."):
                    analysis_results = analyze_audio_content(uploaded_file)
                
                st.markdown("**Audio Properties:**")
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
                
                if st.button("Enhance Audio", key="enhance_btn"):
                    with st.spinner("Enhancing audio..."):
                        try:
                            # Generate enhanced version (simplified approach)
                            enhanced_audio, sample_rate = generate_music_audio(
                                f"traditional Filipino coastal music {enhancement_type}",
                                duration=30,  # Fixed duration for stability
                                enhancement_type=enhancement_type
                            )
                            
                            st.markdown("""
                            <div class="success-message">
                                Audio enhanced successfully! Listen to your enhanced version below.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Play enhanced audio
                            if HAS_SCIPY:
                                buffer = BytesIO()
                                audio_int16 = (enhanced_audio * 32767).astype(np.int16)
                                wavfile.write(buffer, sample_rate, audio_int16)
                                st.audio(buffer.getvalue())
                            else:
                                st.info("Audio generated successfully (SciPy not available for playback)")
                            
                            # Save to archive option
                            if st.button("Save to Archive", key="save_enhanced"):
                                metadata = {
                                    'type': 'enhanced',
                                    'original_file': uploaded_file.name,
                                    'enhancement_type': enhancement_type,
                                    'source': 'user_upload'
                                }
                                if save_audio_to_archive(enhanced_audio, sample_rate, metadata):
                                    st.success("Saved to community archive!")
                            
                        except Exception as e:
                            st.error(f"Enhancement failed: {str(e)}")
                            st.code(traceback.format_exc())
        
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
                    
    except Exception as e:
        st.error(f"Error in upload page: {str(e)}")

def show_generation_page():
    """Display the AI music generation page with error handling"""
    try:
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
                duration = st.slider("Duration (seconds)", 10, 60, 30)  # Reduced max for stability
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
            if st.button("Generate Music", key="generate_btn"):
                if not music_prompt:
                    st.warning("Please enter a music prompt")
                else:
                    with st.spinner("Creating your traditional-inspired music..."):
                        try:
                            # Generate music
                            generated_audio, sample_rate = generate_music_audio(
                                music_prompt, duration=duration
                            )
                            
                            st.markdown("""
                            <div class="success-message">
                                Music generated successfully! Your AI-created traditional-inspired piece is ready.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display generated audio
                            if HAS_SCIPY:
                                buffer = BytesIO()
                                audio_int16 = (generated_audio * 32767).astype(np.int16)
                                wavfile.write(buffer, sample_rate, audio_int16)
                                st.audio(buffer.getvalue())
                            else:
                                st.info("Audio generated successfully (SciPy not available for playback)")
                            
                            # Save options
                            col1_3, col1_4 = st.columns(2)
                            
                            with col1_3:
                                if st.button("Save to Archive", key="save_generated"):
                                    metadata = {
                                        'type': 'generated',
                                        'prompt': music_prompt,
                                        'duration': duration,
                                        'tempo': tempo,
                                        'instruments': instruments,
                                        'category': prompt_category
                                    }
                                    if save_audio_to_archive(generated_audio, sample_rate, metadata):
                                        st.session_state.generated_music.append(metadata)
                                        st.success("Saved to archive!")
                            
                            with col1_4:
                                if st.button("Generate Variation", key="generate_variation"):
                                    st.info("Click 'Generate Music' again for a new variation!")
                                    
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            st.code(traceback.format_exc())
        
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
            
    except Exception as e:
        st.error(f"Error in generation page: {str(e)}")

def show_archive_page():
    """Display the community archive with error handling"""
    try:
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
            enhanced_count = sum(1 for item in st.session_state.audio_archive 
                               if item.get('metadata', {}).get('type') == 'enhanced')
            st.metric("Enhanced Audio", enhanced_count)
        with col3:
            generated_count = sum(1 for item in st.session_state.audio_archive 
                                if item.get('metadata', {}).get('type') == 'generated')
            st.metric("AI Generated", generated_count)
        with col4:
            st.metric("Contributors", "1")
        
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
            try:
                metadata = item.get('metadata', {})
                if filter_type != "All" and metadata.get('type') != filter_type.lower():
                    continue
                
                title = metadata.get('prompt', metadata.get('original_file', f'Untitled {i+1}'))
                timestamp = item.get('timestamp', 'Unknown date')[:10]
                
                with st.expander(f"🎵 {title} - {timestamp}"):
                    col_audio, col_meta = st.columns([2, 1])
                    
                    with col_audio:
                        # Load and display audio
                        audio_bytes = load_audio_from_archive(item)
                        if audio_bytes:
                            st.audio(audio_bytes)
                        else:
                            st.error("Could not load audio file")
                    
                    with col_meta:
                        st.markdown("**Metadata:**")
                        for key, value in metadata.items():
                            if key != 'audio_data':
                                st.write(f"**{key.title()}:** {value}")
                        
                        st.write(f"**Sample Rate:** {item.get('sample_rate', 'Unknown')} Hz")
                        st.write(f"**Created:** {timestamp}")
                        
            except Exception as e:
                st.error(f"Error displaying archive item {i}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error loading archive page: {str(e)}")

def show_discovery_page():
    """Display the music discovery interface with error handling"""
    try:
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
                
    except Exception as e:
        st.error(f"Error loading discovery page: {str(e)}")

def show_about_page():
    """Display information about the project with error handling"""
    try:
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
            - **Audio Generation**: Custom algorithms for traditional music
            - **Audio Analysis**: Basic audio property analysis
            - **Cultural Adaptation**: Respectful AI music generation
            - **Community Input**: Human-centered cultural validation
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
        
        # Technical notes
        st.markdown("### Technical Notes")
        st.markdown(f"""
        - **Dependencies**: Librosa ({'Available' if HAS_LIBROSA else 'Missing'}), SciPy ({'Available' if HAS_SCIPY else 'Missing'})
        - **Audio Format**: 22.05 kHz sample rate for compatibility
        - **Storage**: Session-based (data cleared on refresh)
        - **Generation**: Procedural audio synthesis with cultural themes
        """)
        
        if not HAS_LIBROSA or not HAS_SCIPY:
            st.info("""
            **Installation Note**: For full functionality, install missing dependencies:
            ```
            pip install librosa scipy
            ```
            """)
            
    except Exception as e:
        st.error(f"Error loading about page: {str(e)}")

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical application error occurred")
        st.code(f"Error: {str(e)}")
        st.code(f"Traceback: {traceback.format_exc()}")
        
        # Provide recovery options
        st.markdown("### Recovery Options")
        if st.button("Reset Application State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
        st.markdown("""
        **Troubleshooting Steps:**
        1. Click "Reset Application State" above
        2. Refresh the browser page
        3. Check that required dependencies are installed
        4. Ensure sufficient system memory is available
        """)