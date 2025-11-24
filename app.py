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
# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_name = st.selectbox("Dataset", ["mnist", "fashion_mnist"])
    num_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("Batch Size", value=64)
    lr = st.number_input("Learning Rate", value=0.0002, format="%.5f")
    
    st.divider()
    
    # Section 1: Save/Load (Manual)
    st.subheader("💾 Manual Checkpoints")
    if st.button("Save Current State"):
        try:
            st.session_state.trainer.save_checkpoint("manual_checkpoint.pth")
            st.success("Saved as manual_checkpoint.pth")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    
    # Section 2: Load Pre-Trained Models (The Fix)
    st.subheader("📂 Load Pre-Trained Model")
    
    import os
    # Find all .pth files in the current directory
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if not model_files:
        st.warning("No .pth models found in repo root.")
    else:
        # THIS was missing in your code:
        selected_model = st.selectbox("Select a model file", model_files)
        
        if st.button("Load Selected Model"):
            try:
                # 1. Get reference to the internal networks
                trainer = st.session_state.trainer
                
                # Check for standard naming conventions
                if hasattr(trainer, 'generator'):
                    net_g = trainer.generator
                elif hasattr(trainer, 'netG'):
                    net_g = trainer.netG
                else:
                    st.error("Could not find 'generator' or 'netG' in trainer.")
                    st.stop()

                # 2. Load the file
                # map_location='cpu' is CRITICAL for cloud deployment
                checkpoint = torch.load(selected_model, map_location=torch.device('cpu'))

                # 3. Smart State Dict Loading
                if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                    state_dict = checkpoint['generator_state_dict']
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Assume the checkpoint IS the state dict
                    state_dict = checkpoint

                # 4. FIX KEY MISMATCHES (Strict Mode Fix)
                try:
                    net_g.load_state_dict(state_dict, strict=True)
                except RuntimeError as e:
                    st.warning(f"Strict loading failed, trying loose loading...")
                    net_g.load_state_dict(state_dict, strict=False)

                # 5. Force Eval Mode
                net_g.eval()
                
                st.success(f"Successfully loaded {selected_model}!")
                st.balloons()
                
                # Debug Check
                with torch.no_grad():
                    dummy_noise = torch.randn(1, 100, 1, 1, device=trainer.device)
                    test_out = net_g(dummy_noise)
                    st.info(f"Diagnostic: Output range is {test_out.min().item():.2f} to {test_out.max().item():.2f}")
                
            except Exception as e:
                st.error(f"CRITICAL LOAD ERROR: {e}")

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