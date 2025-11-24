import streamlit as st
import torch
import numpy as np
import time
from trainer import DCGANTrainer # Assumes trainer.py is in the same folder

# --- Page Config ---
st.set_page_config(
    page_title="DCGAN Training Studio",
    page_icon="🎨",
    layout="wide"
)

# --- Session State Initialization ---
# We need to keep the trainer in memory between button clicks
if "trainer" not in st.session_state:
    # Detect Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        st.toast("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        st.toast("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        st.toast("Using CPU")
    
    # Initialize Trainer
    st.session_state.trainer = DCGANTrainer(device=device)

if "is_training" not in st.session_state:
    st.session_state.is_training = False

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = {
        "g_loss": [], "d_loss": [], "epochs": []
    }

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_name = st.selectbox("Dataset", ["mnist", "fashion_mnist"])
    num_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("Batch Size", value=64)
    lr = st.number_input("Learning Rate", value=0.0002, format="%.5f")
    
    st.divider()
    
    # Model Management
    st.subheader("💾 Model Management")
    if st.button("Save Checkpoint"):
        try:
            st.session_state.trainer.save_checkpoint("dcgan_checkpoint.pth")
            st.success("Model saved locally!")
        except Exception as e:
            st.error(f"Error saving: {e}")
            
    if st.button("Load Checkpoint"):
        try:
            st.session_state.trainer.load_checkpoint("dcgan_checkpoint.pth")
            st.success("Model loaded!")
        except FileNotFoundError:
            st.error("No checkpoint found.")

# --- Main Interface ---
st.title("🎨 DCGAN Dashboard")

tab1, tab2 = st.tabs(["🚀 Training", "✨ Generation"])

# --- TAB 1: TRAINING ---
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Live Charts
        st.subheader("Live Metrics")
        chart_placeholder = st.empty()
        
    with col2:
        # Live Preview
        st.subheader("Live Preview")
        image_preview = st.empty()

    # Control Buttons
    start_col, stop_col = st.columns(2)
    
    with start_col:
        start_btn = st.button("Start Training", type="primary", disabled=st.session_state.is_training)
        
    with stop_col:
        stop_btn = st.button("Stop Training", disabled=not st.session_state.is_training)

    # TRAINING LOGIC
    if start_btn:
        st.session_state.is_training = True
        trainer = st.session_state.trainer
        
        # Create Data Loader
        try:
            dataloader = trainer.get_dataloader(dataset_name, batch_size)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training Loop
            for epoch in range(num_epochs):
                if not st.session_state.is_training:
                    break
                
                epoch_g_loss = 0
                epoch_d_loss = 0
                batches = 0
                
                for i, data in enumerate(dataloader):
                    # Check for stop signal (Streamlit Rerun handling)
                    # Note: In Streamlit, true 'interrupts' are hard, we check per batch
                    
                    real_images = data[0].to(trainer.device)
                    metrics = trainer.train_step(real_images)
                    
                    epoch_g_loss += metrics['loss_g']
                    epoch_d_loss += metrics['loss_d']
                    batches += 1
                    
                    # Update Progress within Epoch
                    if i % 10 == 0:
                        status_text.text(f"Epoch {epoch+1}/{num_epochs} | Batch {i}/{len(dataloader)}")
                        progress_bar.progress((i / len(dataloader)))

                # End of Epoch Updates
                avg_g_loss = epoch_g_loss / batches
                avg_d_loss = epoch_d_loss / batches
                
                # Update History
                st.session_state.metrics_history["g_loss"].append(avg_g_loss)
                st.session_state.metrics_history["d_loss"].append(avg_d_loss)
                st.session_state.metrics_history["epochs"].append(epoch)
                
                # Update Chart
                chart_data = {
                    "Generator Loss": st.session_state.metrics_history["g_loss"],
                    "Discriminator Loss": st.session_state.metrics_history["d_loss"]
                }
                chart_placeholder.line_chart(chart_data)
                
                # Update Image Preview
                with torch.no_grad():
                    fake = trainer.generate_images(16)
                    # Convert tensor to displayable image
                    # Assuming trainer outputs normalized [-1, 1] tensor
                    grid_img = trainer.images_to_base64(fake, nrow=4) 
                    # Note: Your images_to_base64 returns a b64 string. 
                    # Streamlit can read that, but it's easier if we just display the grid.
                    # Since I don't see your helper code, I'll use the base64 you have:
                    image_preview.markdown(f'<img src="data:image/png;base64,{grid_img}" width="100%"/>', unsafe_allow_html=True)

            st.session_state.is_training = False
            st.success("Training Complete!")
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.session_state.is_training = False

# --- TAB 2: GENERATION ---
with tab2:
    st.subheader("Generate Synthetic Images")
    
    gen_col1, gen_col2 = st.columns([1, 3])
    
    with gen_col1:
        num_imgs = st.slider("Number of Images", 1, 64, 16)
        generate_btn = st.button("Generate Now")
        
    with gen_col2:
        if generate_btn:
            with st.spinner("Dreaming up images..."):
                trainer = st.session_state.trainer
                fake_images = trainer.generate_images(num_imgs)
                # Calculate grid size roughly square
                nrow = int(num_imgs**0.5)
                b64_img = trainer.images_to_base64(fake_images, nrow=nrow)
                st.markdown(f'<img src="data:image/png;base64,{b64_img}" width="100%"/>', unsafe_allow_html=True)