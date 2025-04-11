import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import networkx as nx
import random
import time
import requests
import json
import base64
from datetime import datetime

st.set_page_config(
    page_title="ML Algorithm Visualizer Platform",
    page_icon="🧠",
    layout="wide"
)

# Convert the image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load your image from a local path
image_path = ("cartoon.JPG")
# Get the base64 string of the image
image_base64 = image_to_base64(image_path)

# Display your image and name in the top right corner
st.markdown(
    f"""
    <style>
    .header {{
        position: absolute;  /* Fix the position */
        top: -60px;  /* Adjust as needed */
        right: -40px;  /* Align to the right */
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 10px;
        flex-direction: column; /* Stack items vertically */
        text-align: center; /* Ensures text is centrally aligned */
    }}
    .header img {{
        border-radius: 50%;
        width: 50px;
        height: 50px;
        margin-bottom: 5px; /* Space between image and text */
    }}
    .header-text {{
        font-size: 12px;
        font-weight: normal; /* Regular weight for text */
        text-align: center;
    }}
    </style>
    <div class="header">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Mohsen Askar">
        <div class="header-text">Developed by: Mohsen Askar</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- HELPER FUNCTIONS ----

def simple_tokenizer(text):
    """Simple tokenization for demonstration"""
    # Remove punctuation and convert to lowercase
    for char in '.,!?;:()[]{}""''':
        text = text.replace(char, '')
    text = text.lower()
    tokens = text.split()
    return tokens

def generate_embeddings(tokens, dim=5):
    """Generate random word embeddings for demonstration"""
    np.random.seed(42)  # For reproducibility
    embeddings = {}
    for token in tokens:
        if token not in embeddings:
            embeddings[token] = np.random.randn(dim)
    return embeddings

def predict_next_word(context, model_quality=50):
    """Simulate predicting the next word based on context"""
    context = context.lower().strip()
    
    # Some predefined patterns based on model quality
    common_patterns = {
        "how are": ["you", "things", "we"],
        "what is": ["the", "your", "this"],
        "tell me": ["about", "why", "how"],
        "i want": ["to", "a", "some"],
        "thank you": ["for", "so", "very"]
    }
    
    # For beginners (random choice with limited vocabulary)
    if model_quality < 30:
        basic_words = ["the", "a", "is", "and", "to", "of", "you", "it"]
        return random.choice(basic_words)
    
    # For intermediate models (pattern matching)
    elif model_quality < 70:
        for pattern, options in common_patterns.items():
            if context.endswith(pattern):
                return random.choice(options)
        return "the"  # Default
    
    # For advanced models (more contextual awareness)
    else:
        if "how are" in context:
            return "you feeling today"
        elif "what is" in context:
            return "the purpose of this demonstration"
        elif "tell me" in context:
            return "about how language models work"
        elif "i want" in context:
            return "to learn more about AI"
        elif "thank" in context:
            return "you for your attention"
        else:
            # More sophisticated response for advanced model
            return "interesting to see how this works"

# ---- MAIN APP ----

def main():
    # Sidebar for navigation
    st.sidebar.title("LLM Development Journey")
    pages = [
        "Introduction", 
        "1. Data Collection", 
        "2. Tokenization", 
        "3. Word Embeddings",
        "4. Neural Network Architecture", 
        "5. Training Process", 
        "6. Fine-tuning", 
        "7. Inference & User Interaction",
        "8. Complete LLM Pipeline"
    ]
    choice = st.sidebar.radio("Navigate", pages)
    
    # Title with styling
    st.markdown("""
    <style>
    .big-title {
        font-size: 50px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if choice == "Introduction":
        introduction()
    elif choice == "1. Data Collection":
        data_collection()
    elif choice == "2. Tokenization":
        tokenization()
    elif choice == "3. Word Embeddings":
        word_embeddings()
    elif choice == "4. Neural Network Architecture":
        neural_network_architecture()
    elif choice == "5. Training Process":
        training_process()
    elif choice == "6. Fine-tuning":
        fine_tuning()
    elif choice == "7. Inference & User Interaction":
        inference_interaction()
    elif choice == "8. Complete LLM Pipeline":
        complete_pipeline()


    st.markdown("---")

    # Function to get and update the visitor count using a cloud database
    def track_visitor():
        if 'firebase_option' == True:
            import firebase_admin
            from firebase_admin import credentials, db
            
            # Initialize Firebase (do this only once)
            if 'firebase_initialized' not in st.session_state:
                try:
                    cred = credentials.Certificate("your-firebase-credentials.json")
                    firebase_admin.initialize_app(cred, {
                        'databaseURL': 'https://your-project.firebaseio.com/'
                    })
                    st.session_state.firebase_initialized = True
                except Exception as e:
                    st.error(f"Error initializing Firebase: {e}")
                    return 0
            
            # Increment the counter
            try:
                ref = db.reference('visitor_counter')
                current_count = ref.get() or 0
                new_count = current_count + 1
                ref.set(new_count)
                return new_count
            except Exception as e:
                st.error(f"Error updating counter: {e}")
                return 0
        
        elif 'streamlit_cloud_option' == True:
            if 'count' not in st.session_state:
                # This works only on Streamlit Cloud with secrets management
                try:
                    # Get current count
                    response = requests.get(
                        "https://kvdb.io/YOUR_BUCKET_ID/visitor_count",
                        headers={"Content-Type": "application/json"}
                    )
                    current_count = int(response.text) if response.text else 0
                    
                    # Update count
                    new_count = current_count + 1
                    requests.post(
                        "https://kvdb.io/YOUR_BUCKET_ID/visitor_count",
                        data=str(new_count),
                        headers={"Content-Type": "text/plain"}
                    )
                    st.session_state.count = new_count
                    return new_count
                except Exception as e:
                    st.error(f"Error with KV store: {e}")
                    return 0
            return st.session_state.count
        
        else:
            if 'count' not in st.session_state:
                try:
                    with open('visitor_count.txt', 'r') as f:
                        current_count = int(f.read().strip())
                except FileNotFoundError:
                    current_count = 0
                
                new_count = current_count + 1
                
                try:
                    with open('visitor_count.txt', 'w') as f:
                        f.write(str(new_count))
                    st.session_state.count = new_count
                except Exception as e:
                    st.error(f"Error saving count: {e}")
                    st.session_state.count = current_count + 1
                    
            return st.session_state.count

    # Only increment the counter once per session
    if 'visitor_counted' not in st.session_state:
        count = track_visitor()
        st.session_state.visitor_counted = True
    else:
        count = st.session_state.get('count', 0)

    # Display the counter with nice styling
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px; margin-top: 30px; 
            border-top: 1px solid #f0f0f0; color: #888;">
            <span style="font-size: 14px;">👥 Total Visitors: {count}</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

    today = datetime.now().strftime("%B %d, %Y")
    st.markdown(
        f"""
        <div style="text-align: center; color: #888; font-size: 12px; margin-top: 5px;">
            {today}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- PAGE FUNCTIONS ----

def introduction():
    st.markdown('<p class="big-title">The Journey of Large Language Models</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to this interactive exploration of LLMs!
    
    This application will guide you through the key stages in the development and operation
    of Large Language Models like GPT, Claude, and LLaMA. Through interactive visualizations
    and simplified demonstrations, you'll gain insights into how these powerful AI systems
    are created and how they work.
    
    ### What you'll learn:
    
    - How vast amounts of text data are collected and processed
    - How text is transformed into numbers that neural networks can process
    - The architecture of neural networks that power LLMs
    - How these models are trained on trillions of tokens
    - The fine-tuning process that makes models helpful and safe
    - How models generate responses to user queries
    
    Use the sidebar to navigate through each stage of the LLM journey.
    """)
    
    # Timeline of LLM development
    st.subheader("Timeline of Major LLM Developments")
    
    timeline_data = {
        "Year": [2017, 2018, 2019, 2020, 2022, 2022, 2023, 2023, 2024],
        "Model": ["Transformer", "BERT", "GPT-2", "GPT-3", "ChatGPT", "Claude", "GPT-4", "LLaMA", "Claude Opus"],
        "Breakthrough": [
            "Attention mechanism architecture", 
            "Bidirectional training", 
            "Task-agnostic text generation", 
            "Few-shot learning capabilities",
            "Conversational AI for general public",
            "Constitutional AI approach",
            "Multimodal capabilities",
            "Open model ecosystem",
            "Advanced reasoning"
        ],
        "Size": [65, 340, 1500, 175000, 175000, 52000, 1000000, 70000, 1000000]  # Parameters in millions
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create bubble chart for timeline
    fig = px.scatter(
        df_timeline, 
        x="Year", 
        y="Model", 
        size="Size", 
        hover_name="Model",
        hover_data=["Breakthrough"],
        color="Model",
        size_max=60,
        title="Evolution of LLMs (bubble size represents parameter count)"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def data_collection():
    st.markdown('<p class="big-title">1. Data Collection</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## The Foundation: Data Collection
    
    LLMs begin with massive datasets – often hundreds of billions to trillions of words from:
    
    - Books
    - Websites
    - Scientific papers
    - Code repositories
    - Social media
    - And many other text sources
    
    The quality and diversity of this data significantly impacts what the model can learn.
    """)
    
    # Interactive demo of data collection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Source Distribution")
        
        # Sample data composition
        data_sources = {
            "Websites": 45,
            "Books": 25,
            "Scientific Papers": 15,
            "Code": 8,
            "Social Media": 5,
            "Other": 2
        }
        
        source_input = {}
        st.write("Try adjusting the composition of training data:")
        
        # Let user adjust data composition
        for source, default in data_sources.items():
            source_input[source] = st.slider(f"{source} (%)", 0, 100, default)
        
        # Normalize to 100%
        total = sum(source_input.values())
        if total > 0:  # Avoid division by zero
            normalized = {k: (v/total)*100 for k, v in source_input.items()}
        else:
            normalized = {k: 0 for k in source_input.keys()}
        
        # Create pie chart of data sources
        fig = px.pie(
            values=list(normalized.values()),
            names=list(normalized.keys()),
            title="Training Data Composition"
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Data Quality Impact")
        
        # Sample texts of different quality
        high_quality = """The neural network architecture employs self-attention mechanisms, 
        allowing the model to weigh the importance of different words in context. This innovation 
        enabled significant advancements in natural language understanding."""
        
        medium_quality = """Neural networks use attention to look at words and figure out which ones matter most. 
        This helps the model understand language better and was a big step forward in AI."""
        
        low_quality = """ai uses neural nets with attention 2 process words. this makes it better 
        at language lol. big improvement 4 sure!!!"""
        
        st.write("Examples of different data quality:")
        
        quality_option = st.radio(
            "Select data quality to view sample:",
            ["High-quality curated text", "Medium-quality general text", "Low-quality unfiltered text"]
        )
        
        if quality_option == "High-quality curated text":
            st.info(high_quality)
            st.write("Benefits: Precise, informative, well-structured")
            st.write("Impact: Models learn formal, accurate patterns and domain knowledge")
        elif quality_option == "Medium-quality general text":
            st.info(medium_quality)
            st.write("Benefits: Accessible, conversational, diverse")
            st.write("Impact: Models learn everyday language and common expressions")
        else:
            st.info(low_quality)
            st.write("Challenges: Informal, potentially biased, grammatically incorrect")
            st.write("Impact: Models may learn undesirable patterns or misinformation")

def tokenization():
    st.markdown('<p class="big-title">2. Tokenization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Converting Text to Tokens
    
    Before neural networks can process text, it must be converted into numerical form.
    Tokenization is the process of breaking text into smaller pieces (tokens) that the model can handle.
    
    Modern LLMs typically use subword tokenization methods like:
    - Byte-Pair Encoding (BPE)
    - WordPiece
    - SentencePiece
    
    These methods balance vocabulary size with representational power.
    """)
    
    # Interactive tokenization demo
    st.subheader("Try Tokenization")
    
    user_text = st.text_area(
        "Enter text to tokenize:", 
        "The transformer architecture revolutionized natural language processing in 2017."
    )
    
    simple_tokens = simple_tokenizer(user_text)
    
    # Display tokens
    st.write("Simple word tokenization:")
    st.write(simple_tokens)
    
    # Visualization of token IDs
    st.write("Mapping tokens to IDs (simplified):")
    
    # Create a dictionary mapping tokens to IDs
    token_ids = {token: idx for idx, token in enumerate(set(simple_tokens))}
    
    # Display as a table
    token_id_df = pd.DataFrame({
        "Token": list(token_ids.keys()),
        "ID": list(token_ids.values())
    }).sort_values("ID")
    
    st.dataframe(token_id_df)
    
    # Visualization of token sequence
    st.write("Original text as token IDs:")
    
    token_sequence = [token_ids[token] for token in simple_tokens]
    
    # Create a horizontal visualization of tokens
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, (token, token_id) in enumerate(zip(simple_tokens, token_sequence)):
        ax.text(i, 0.5, str(token_id), ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.text(i, 0, token, ha='center', va='center', rotation=45)
    
    ax.set_xlim(-0.5, len(simple_tokens) - 0.5)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Advanced tokenization explanation
    st.subheader("Subword Tokenization (More Advanced)")
    
    st.markdown("""
    Modern LLMs don't just split by spaces. They use subword tokenization, breaking words into meaningful pieces:
    
    Example with Byte-Pair Encoding (BPE):
    - "revolutionized" → "revolution" + "ized"
    - "preprocessing" → "pre" + "process" + "ing"
    
    Benefits:
    - Smaller vocabulary size
    - Better handling of rare words
    - Ability to process unseen words
    """)
    
    # Show a sample visualization of subword tokenization
    # Display video
    st.video("https://i.imgur.com/pewlX6y.mp4")

    # Add caption using markdown
    st.markdown("**Byte-Pair Encoding (BPE) Visualization from Hugging Face**")

def word_embeddings():
    st.markdown('<p class="big-title">3. Word Embeddings</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Representing Tokens as Vectors
    
    After tokenization, each token is converted into a numerical vector called an embedding.
    These embeddings capture semantic relationships between words in a continuous vector space.
    
    Key properties:
    - Similar words have similar vectors
    - Vector arithmetic captures relationships (e.g., King - Man + Woman ≈ Queen)
    - Typically 300-1000 dimensions in modern LLMs
    
    These embeddings are initially random and get refined during training.
    """)
    
    # Interactive embedding demo
    st.subheader("Word Embeddings Visualization")
    
    # Sample words for embedding
    default_text = "king queen man woman apple orange cat dog good bad"
    
    user_text = st.text_area(
        "Enter words to visualize embeddings (space-separated):", 
        default_text
    )
    
    # Tokenize and generate random embeddings
    tokens = simple_tokenizer(user_text)
    embeddings = generate_embeddings(tokens)
    
    # Create a dataframe with 2D projections for visualization
    # This is a simplified version - in reality, dimensionality reduction like t-SNE would be used
    np.random.seed(42)
    projection_x = {word: emb[0]*0.7 + emb[1]*0.3 for word, emb in embeddings.items()}
    projection_y = {word: emb[2]*0.7 + emb[3]*0.3 for word, emb in embeddings.items()}
    
    embedding_df = pd.DataFrame({
        "Word": list(embeddings.keys()),
        "X": [projection_x[word] for word in embeddings.keys()],
        "Y": [projection_y[word] for word in embeddings.keys()]
    })
    
    # Plot the 2D projection
    fig = px.scatter(
        embedding_df, x="X", y="Y", text="Word",
        title="2D Projection of Word Embeddings (Simplified Visualization)"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show vector values
    if st.checkbox("Show raw embedding vectors (simplified)"):
        # Show just the first 5 dimensions for simplicity
        vector_df = pd.DataFrame({
            "Word": list(embeddings.keys()),
            "Dimension 1": [emb[0] for emb in embeddings.values()],
            "Dimension 2": [emb[1] for emb in embeddings.values()],
            "Dimension 3": [emb[2] for emb in embeddings.values()],
            "Dimension 4": [emb[3] for emb in embeddings.values()],
            "Dimension 5": [emb[4] for emb in embeddings.values()]
        })
        
        st.dataframe(vector_df)
        
        st.markdown("""
        In real LLMs, these vectors would have hundreds or thousands of dimensions, not just 5.
        The full dimensionality allows capturing complex semantic relationships.
        """)

def neural_network_architecture():
    st.markdown('<p class="big-title">4. Neural Network Architecture</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## The Architecture of Modern LLMs
    
    Modern LLMs are based on the Transformer architecture, which revolutionized NLP in 2017.
    Key components include:
    
    1. **Self-attention mechanism**: Allows the model to focus on different parts of the input when making predictions
    2. **Multi-head attention**: Multiple attention mechanisms operating in parallel
    3. **Feed-forward networks**: Process the attention outputs
    4. **Layer normalization**: Stabilizes training
    5. **Residual connections**: Helps with training deep networks
    
    These components are arranged in layers and blocks, with modern LLMs having dozens or hundreds of layers.
    """)
    
    # Architecture visualization
    st.subheader("Transformer Architecture")
    
    tab1, tab2 = st.tabs(["High-Level Overview", "Attention Mechanism"])
    
    with tab1:
        # Create a simplified transformer architecture diagram
        G = nx.DiGraph()
        
        # Add nodes for different components
        components = [
            "Input Embeddings", "Positional Encoding", 
            "Self-Attention 1", "Feed-Forward 1", "Layer Norm 1",
            "Self-Attention 2", "Feed-Forward 2", "Layer Norm 2",
            "Output Embeddings", "Next Token Prediction"
        ]
        
        positions = {
            "Input Embeddings": (0, 0),
            "Positional Encoding": (0, -1),
            "Self-Attention 1": (0, -2),
            "Feed-Forward 1": (0, -3),
            "Layer Norm 1": (0, -4),
            "Self-Attention 2": (0, -5),
            "Feed-Forward 2": (0, -6),
            "Layer Norm 2": (0, -7),
            "Output Embeddings": (0, -8),
            "Next Token Prediction": (0, -9)
        }
        
        # Add nodes
        for component in components:
            G.add_node(component)
        
        # Add edges
        edges = [
            ("Input Embeddings", "Positional Encoding"),
            ("Positional Encoding", "Self-Attention 1"),
            ("Self-Attention 1", "Feed-Forward 1"),
            ("Feed-Forward 1", "Layer Norm 1"),
            ("Layer Norm 1", "Self-Attention 2"),
            ("Self-Attention 2", "Feed-Forward 2"),
            ("Feed-Forward 2", "Layer Norm 2"),
            ("Layer Norm 2", "Output Embeddings"),
            ("Output Embeddings", "Next Token Prediction")
        ]
        
        G.add_edges_from(edges)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 12))
        
        nx.draw_networkx(
            G, pos=positions, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, ax=ax, arrows=True,
            arrowstyle='->', arrowsize=15
        )
        
        plt.title("Simplified Transformer Architecture")
        plt.axis('off')
        
        st.pyplot(fig)
        
        st.markdown("""
        This is a highly simplified representation. Modern LLMs have:
        - Many more layers (12 to 100+)
        - More complex connections
        - Much larger dimensions for all components
        
        Each vertical stack of attention + feed-forward is called a "Transformer block".
        GPT-3, for example, has 96 such blocks in its largest version.
        """)
    
    with tab2:
        st.subheader("Self-Attention Mechanism Visualization")
        
        # Simple example sentences
        example_sentence = st.selectbox(
            "Choose a sentence to visualize attention:",
            [
                "The cat sat on the mat.",
                "She loves ice cream because it is sweet.",
                "The bank by the river has low interest rates."
            ]
        )
        
        tokens = simple_tokenizer(example_sentence)
        
        # Generate a simplified attention matrix
        np.random.seed(0)
        attention_matrix = np.random.rand(len(tokens), len(tokens))
        
        # Make it more realistic by enhancing diagonal and nearby values
        for i in range(len(tokens)):
            # Enhance diagonal (self-attention)
            attention_matrix[i, i] = 0.7 + 0.3 * attention_matrix[i, i]
            
            # Enhance nearby words
            for j in range(len(tokens)):
                if abs(i - j) == 1:  # Directly adjacent words
                    attention_matrix[i, j] = 0.5 + 0.3 * attention_matrix[i, j]
        
        # Normalize each row to sum to 1 (softmax simulation)
        for i in range(len(tokens)):
            attention_matrix[i] = attention_matrix[i] / np.sum(attention_matrix[i])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            hoverongaps=False,
            text=[[f"{val:.2f}" for val in row] for row in attention_matrix],
            hovertemplate='From %{y} to %{x}: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Self-Attention Visualization (How each word attends to others)",
            xaxis_title="Token Being Attended To",
            yaxis_title="Token Doing the Attending",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This visualization shows how each word in the sentence "attends" to every other word.
        Darker colors indicate stronger attention.
        
        In real transformers:
        - There are multiple "heads" of attention operating in parallel
        - Attention patterns emerge during training to capture various linguistic relationships
        - Different layers develop specialized attention patterns for syntax, semantics, etc.
        
        The attention mechanism is what allows LLMs to understand context across long distances in text.
        """)

def training_process():
    st.markdown('<p class="big-title">5. Training Process</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## How LLMs Learn
    
    LLMs are trained using a process called "self-supervised learning." Rather than requiring labeled data,
    the model learns by predicting the next token in a sequence.
    
    The training process involves:
    
    1. **Feeding in text**: The model receives batches of text from the training corpus
    2. **Making predictions**: For each position, the model tries to predict the next token
    3. **Measuring error**: The difference between predictions and actual tokens is calculated
    4. **Updating weights**: Model parameters are adjusted to reduce this error
    5. **Repeating at scale**: This process happens trillions of times, over massive datasets
    
    Modern LLMs require enormous computational resources for training - often hundreds or thousands of GPUs
    running for weeks or months.
    """)
    
    # Training visualization
    st.subheader("Training Process Visualization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a simple loss curve
        training_steps = 100
        
        # Simulated training curves
        np.random.seed(42)
        
        # Generate a decreasing loss curve with random noise
        base_loss = np.linspace(5, 0.5, training_steps)
        noise = np.random.normal(0, 0.3, training_steps)
        train_loss = base_loss + noise
        train_loss = np.maximum(train_loss, 0.1)  # Ensure no negative values
        
        # Validation loss follows training but with higher values and more noise
        val_noise = np.random.normal(0, 0.5, training_steps)
        val_loss = base_loss * 1.2 + val_noise
        val_loss = np.maximum(val_loss, 0.2)  # Ensure no negative values
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(training_steps)),
            y=train_loss,
            mode='lines',
            name='Training Loss'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(training_steps)),
            y=val_loss,
            mode='lines',
            name='Validation Loss'
        ))
        
        fig.update_layout(
            title="Loss During Training",
            xaxis_title="Training Steps",
            yaxis_title="Loss",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add slider for model size impact
        st.subheader("Impact of Model Size on Training")
        model_size = st.slider("Model Size (Billions of Parameters)", 1, 100, 10)
        
        # Simple metrics based on model size
        training_compute = model_size ** 1.8  # Rough approximation
        time_to_train = model_size ** 1.3
        estimated_capability = 10 * np.log(model_size) + 20
        
        st.metric("Estimated Training Compute (PetaFLOPs)", f"{training_compute:,.0f}")
        st.metric("Relative Training Time (Days)", f"{time_to_train:.1f}")
        st.metric("Estimated Capability Score", f"{estimated_capability:.1f}")
    
    with col2:
        st.subheader("Key Training Concepts")
        
        st.markdown("""
        ### Batch Size
        Number of sequences processed before each weight update.
        Larger batches → more stable but require more memory.
        
        ### Learning Rate
        How much weights change with each update.
        Too high → unstable
        Too low → slow progress
        
        ### Optimizer
        Algorithm for updating weights (Adam, AdamW, etc.).
        
        ### Tokens Processed
        Modern LLMs train on hundreds of billions to trillions of tokens.
        
        ### Hardware
        Training clusters use thousands of specialized GPUs or TPUs connected with high-speed networks.
        """)
        
        # Add a simple animation of parameter updates
        st.subheader("Parameter Updates")
        
        if st.button("Simulate Parameter Updates"):
            progress_bar = st.progress(0)
            
            # Show 10 simplified parameter updates
            for i in range(10):
                # Show a parameter and its update
                old_param = np.random.randn() * 0.1
                gradient = np.random.randn() * 0.02
                new_param = old_param - gradient
                
                st.write(f"Parameter: {old_param:.4f} → {new_param:.4f} (Gradient: {gradient:.4f})")
                
                # Update progress bar
                progress_bar.progress((i + 1) / 10)
                time.sleep(0.5)
            
            st.success("Parameter update simulation complete!")

def fine_tuning():
    st.markdown('<p class="big-title">6. Fine-tuning</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Specialized Training for Specific Goals
    
    After pre-training on large general datasets, LLMs are refined through fine-tuning:
    
    Key fine-tuning approaches:
    
    1. **Task-specific fine-tuning**: Training on data for specific capabilities like summarization or coding
    2. **Instruction tuning**: Teaching the model to follow user instructions
    3. **RLHF (Reinforcement Learning from Human Feedback)**: Using human preferences to guide model outputs
    4. **Constitutional AI**: Training models to adhere to a set of principles/guidelines
    
    Fine-tuning is what transforms raw language models into assistants that are helpful, harmless, and honest.
    """)
    
    # Fine-tuning visualization
    st.subheader("Fine-tuning Process")
    
    # Create tabs for different fine-tuning approaches
    tab1, tab2, tab3 = st.tabs(["Instruction Tuning", "RLHF", "Alignment Impact"])
    
    with tab1:
        st.markdown("""
        ### Instruction Tuning
        
        This involves training the model on examples of instructions and appropriate responses.
        
        **Example:**
        """)
        
        # Example of instruction tuning
        instruction = st.text_input(
            "Enter an instruction:",
            "Explain the concept of photosynthesis to a 10-year-old child."
        )
        
        st.markdown("#### Example of training data format:")
        
        instruction_example = {
            "instruction": instruction,
            "response": "Photosynthesis is like a plant's way of making food from sunlight! Imagine if you could stand in the sun and make your own lunch - that's what plants do. They take sunlight, water from the ground, and air, and mix them together to make sugar, which gives them energy to grow. And guess what? While doing this, they release oxygen, which is what we need to breathe! So plants are basically little food factories that help us breathe too!"
        }
        
        st.json(instruction_example)
        
        st.markdown("""
        During instruction tuning:
        1. The model is shown thousands to millions of these instruction-response pairs
        2. It learns to generate helpful responses to different types of instructions
        3. The model's weights are updated to minimize the difference between its outputs and the target responses
        """)
    
    with tab2:
        st.markdown("""
        ### Reinforcement Learning from Human Feedback (RLHF)
        
        RLHF uses human preferences to further refine model outputs:
        
        1. **Preference Collection**: Humans rank multiple model responses
        2. **Reward Model Training**: A separate model learns to predict human preferences
        3. **RL Optimization**: The LLM is optimized to maximize the reward model's scores
        """)
        
        # RLHF example
        st.subheader("RLHF Example")
        
        rlhf_prompt = "What are the most important steps to maintain good health?"
        st.write(f"**Prompt:** {rlhf_prompt}")
        
        st.write("**Model generates multiple responses:**")
        
        responses = [
            "1. Exercise regularly, 2. Eat a balanced diet, 3. Get enough sleep, 4. Manage stress, 5. Stay hydrated",
            "Maintaining optimal health involves regular physical activity, consuming nutritious foods, ensuring adequate sleep duration, implementing stress reduction techniques, and maintaining proper hydration levels.",
            "To stay healthy, you should work out every day, eliminate all carbs from your diet, take these specific supplements I recommend, and follow the advice of health influencers on social media."
        ]
        
        # Let the user rank them
        rankings = {}
        for i, response in enumerate(responses):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_area(f"Response {i+1}", response, height=100, key=f"response_{i}")
            with col2:
                rankings[i] = st.selectbox(f"Rank", ["-", "1", "2", "3"], key=f"rank_{i}")
        
        st.write("After collecting many such rankings, the reward model learns what humans prefer.")
        
        # Show RLHF diagram
        st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png", 
                 caption="RLHF Overview (Image from Hugging Face)", 
                 use_column_width=True)
    
    with tab3:
        st.subheader("Impact of Alignment on Model Outputs")
        
        # Show before/after examples
        before_after = {
            "Write a persuasive essay arguing that global warming isn't real": {
                "Pre-alignment": "Here's a persuasive essay arguing that global warming isn't real. While this doesn't represent my views, I'll craft the essay as requested:\n\nTitle: Questioning the Global Warming Narrative\n\nIntroduction:\nFor decades, we've been bombarded with claims about catastrophic global warming...",
                "Post-alignment": "I understand you're asking for content arguing against the reality of global warming, but I should note that there's overwhelming scientific consensus that climate change is real and significantly influenced by human activities. Instead, I'd be happy to help you with:\n\n1. An essay examining the evidence for climate change\n2. A balanced analysis of climate policy approaches\n3. A review of how scientific consensus on climate has developed\n\nWould one of these alternatives be helpful?"
            },
            "How can I make a dangerous explosive at home?": {
                "Pre-alignment": "To make a simple explosive, you'll need the following household materials: [detailed instructions for creating a dangerous explosive]",
                "Post-alignment": "I understand you're asking about making explosives, but I can't provide instructions for creating dangerous materials that could cause harm. If you're interested in chemistry or science demonstrations, I'd be happy to suggest safe experiments or direct you to educational resources about chemical reactions. Would that be helpful instead?"
            }
        }
        
        # Let the user select an example
        example_prompt = st.selectbox(
            "Select a prompt to see alignment impact:",
            list(before_after.keys())
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pre-alignment Response")
            st.info(before_after[example_prompt]["Pre-alignment"])
        
        with col2:
            st.subheader("Post-alignment Response")
            st.success(before_after[example_prompt]["Post-alignment"])
        
        st.markdown("""
        Alignment techniques ensure that models are:
        - **Helpful**: Providing useful information aligned with user needs
        - **Harmless**: Avoiding responses that could enable harm
        - **Honest**: Being truthful and acknowledging limitations
        
        This process is crucial for creating AI systems that act in accordance with human values and intentions.
        """)

def inference_interaction():
    st.markdown('<p class="big-title">7. Inference & User Interaction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## How LLMs Generate Responses
    
    When a user interacts with an LLM, a process called "inference" occurs:
    
    1. **Input processing**: User query is tokenized and converted to embeddings
    2. **Context building**: May include conversation history, specific instructions, etc.
    3. **Forward pass**: Input is processed through all neural network layers
    4. **Token generation**: Model predicts the next token probabilistically
    5. **Iteration**: The generated token is added to the input and the process repeats
    6. **Stopping**: Generation continues until a stopping condition is met
    
    This process happens token-by-token, with each new token influenced by all previous ones.
    """)
    
    # Interactive demo of token generation
    st.subheader("Token-by-Token Generation")
    
    user_prompt = st.text_input(
        "Enter a prompt for the model:",
        "The best way to learn a new language is to"
    )
    
    # Simulate different levels of model quality
    model_quality = st.slider("Model Quality", 10, 90, 50)
    
    st.write("Watching token-by-token generation:")
    
    # Generate tokens one by one if the button is pressed
    if st.button("Generate Response"):
        response_container = st.empty()
        
        # Start with just the prompt
        full_response = ""
        
        # Generate 30 tokens (words) for demonstration
        for i in range(30):
            # Pause for effect
            time.sleep(0.3)
            
            # Get the next word based on context
            next_word = predict_next_word(user_prompt + " " + full_response, model_quality)
            
            # Add to the full response
            if full_response:
                full_response += " " + next_word
            else:
                full_response += next_word
            
            # Update the display
            response_container.markdown(f"**Generated so far:** {full_response}")
    
    # Demonstration of temperature effect
    st.subheader("Impact of Temperature on Generation")
    
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    
    st.markdown(f"""
    **Temperature = {temperature}**
    
    Temperature controls randomness in generation:
    - **Low temperature** (near 0): More deterministic, focused, repetitive
    - **High temperature** (above 1): More random, creative, diverse
    
    Other inference parameters include:
    - **Top-k**: Only consider the k most likely tokens
    - **Top-p** (nucleus sampling): Consider tokens comprising p probability mass
    - **Repetition penalty**: Reduce likelihood of repeating the same text
    """)
    
    # Show examples of different temperatures
    temp_examples = {
        "Low (0.2)": "The best way to learn a new language is to practice consistently every day. This involves studying grammar rules, memorizing vocabulary words, and engaging in conversation with native speakers. Consistency is key to language acquisition. Regular practice helps reinforce memory and builds confidence in using the language.",
        "Medium (0.7)": "The best way to learn a new language is to immerse yourself in it as much as possible. This could mean watching movies, listening to music, finding conversation partners, or even traveling to countries where the language is spoken. Combining different methods keeps learning interesting and helps you pick up natural patterns of speech.",
        "High (1.5)": "The best way to learn a new language is to dream in it! Try unusual approaches like narrating your daily activities out loud, creating imaginary friends who only speak that language, or even adopting a new personality when speaking it. Dance while reciting vocabulary! Paint words instead of writing them. The brain loves novelty and emotional connections."
    }
    
    if temperature <= 0.4:
        st.info(temp_examples["Low (0.2)"])
    elif temperature <= 1.0:
        st.info(temp_examples["Medium (0.7)"])
    else:
        st.info(temp_examples["High (1.5)"])

def complete_pipeline():
    st.markdown('<p class="big-title">8. Complete LLM Pipeline</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Putting It All Together
    
    Now that we've explored each component, let's see how they all work together in the complete LLM pipeline:
    
    1. **Data Collection & Preprocessing**: Gathering and cleaning massive text datasets
    2. **Pre-training**: Learning language patterns through next-token prediction
    3. **Fine-tuning**: Specializing for particular tasks or alignment with human values
    4. **Deployment**: Making the model available through APIs or applications
    5. **Inference**: Generating responses to user queries token-by-token
    
    This entire process represents one of the most complex AI systems ever built, requiring expertise in
    multiple domains of computer science, linguistics, and engineering.
    """)
    
    # Final interactive demo
    st.subheader("Interactive End-to-End Simulation")
    
    # Let user configure a simple "LLM"
    st.write("Configure your simplified LLM:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_size = st.select_slider(
            "Model Size", 
            options=["Tiny (1B)", "Small (7B)", "Medium (13B)", "Large (70B)", "Enormous (175B+)"],
            value="Medium (13B)"
        )
        
        training_data = st.multiselect(
            "Training Data",
            ["Web Pages", "Books", "Scientific Papers", "Code", "Social Media", "Conversations"],
            default=["Web Pages", "Books"]
        )
    
    with col2:
        fine_tuning = st.multiselect(
            "Fine-tuning Applied",
            ["Instruction Tuning", "RLHF", "Constitutional AI", "Domain Specialization"],
            default=["Instruction Tuning"]
        )
        
        inference_params = st.select_slider(
            "Generation Strategy", 
            options=["Conservative", "Balanced", "Creative"],
            value="Balanced"
        )
    
    # Calculate a "capability" score based on selections
    size_scores = {
        "Tiny (1B)": 1,
        "Small (7B)": 2,
        "Medium (13B)": 3,
        "Large (70B)": 4,
        "Enormous (175B+)": 5
    }
    
    data_score = len(training_data) * 0.5  # 0.5 points per data source
    fine_tuning_score = len(fine_tuning) * 0.7  # 0.7 points per fine-tuning method
    
    # Calculate overall capability (simplified)
    capability = size_scores[model_size] * 0.5 + data_score + fine_tuning_score
    capability = min(capability, 10)  # Cap at 10
    
    # Show a simple capability gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = capability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Estimated Capability Score"},
        gauge = {
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 7], 'color': "gray"},
                {'range': [7, 10], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive prompt-response simulation
    st.subheader("Test Your Model")
    
    test_prompt = st.text_area(
        "Enter a prompt to test your configured model:",
        "Explain why the sky is blue in simple terms."
    )
    
    if st.button("Generate Response from Your Model"):
        # Simulate loading
        with st.spinner("Processing through neural network layers..."):
            time.sleep(1.5)
        
        # Generate a response based on model configuration
        if capability < 3:
            response = "The sky is blue because of the atmosphere. The sunlight hits molecules in the air and scatters blue light more than other colors."
        elif capability < 6:
            response = "The sky appears blue due to a phenomenon called Rayleigh scattering. Sunlight contains all colors of the rainbow, but as it passes through our atmosphere, the blue wavelengths scatter more easily than others. This scattered blue light is what we see when we look at the sky.\n\nThink of it like this: the air molecules in our atmosphere are like tiny ping-pong balls that the sunlight has to get past. Blue light bounces off these balls in all directions more easily than red light, so we see blue coming from all parts of the sky."
        else:
            response = "The sky looks blue because of something called 'Rayleigh scattering,' but let me explain it simply:\n\nSunlight is actually made up of all the colors of the rainbow mixed together. When this light travels through our atmosphere, it bumps into tiny air molecules.\n\nHere's the interesting part: blue light gets bumped around and scattered much more easily than other colors like red or yellow. It's like if you had a room full of bouncy balls where the blue ones bounce all over the place while the red ones mostly keep going straight.\n\nBecause blue light scatters in all directions, it comes at us from every part of the sky. When we look up, we see all this scattered blue light, making the whole sky appear blue!\n\nDuring sunrise or sunset, sunlight has to travel through more atmosphere to reach us, so most of the blue gets scattered away before reaching our eyes. That's why we see the beautiful reds and oranges at those times."
        
        st.info(response)
        
        # Show some "behind the scenes" metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tokens Generated", len(response.split()))
        with col2:
            st.metric("Generation Time", f"{(len(response) * 0.03):.2f}s")
        with col3:
            st.metric("Confidence Score", f"{min(90 + capability * 0.5, 99):.1f}%")
    
    # Final educational recap
    st.subheader("Educational Recap")
    
    st.markdown("""
    This interactive application has demonstrated the key components and processes involved in modern LLMs:
    
    1. **Data Collection**: The foundation of knowledge
    2. **Tokenization**: Converting text to numerical form
    3. **Word Embeddings**: Representing meaning in vector space
    4. **Neural Architecture**: The powerful transformer design
    5. **Training Process**: Learning patterns from massive datasets
    6. **Fine-tuning**: Specializing and aligning the model
    7. **Inference**: Generating responses token-by-token
    
    Each of these areas involves substantial complexity and ongoing research. The field of LLMs continues to advance
    rapidly, with new techniques and capabilities emerging regularly.
    
    Future directions include:
    - Multimodal models that understand images, audio, and video
    - More efficient architectures requiring less computational resources
    - Enhanced reasoning capabilities
    - Better alignment with human values
    - Improved factuality and reduced hallucination
    """)

# Run the app
if __name__ == "__main__":
    main()
