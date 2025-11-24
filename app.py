import streamlit as st
import torch
import numpy as np
import os
from trainer import DCGANTrainer  # Assumes trainer.py is in the same folder

# --- Page Config ---
st.set_page_config(
    page_title="DCGAN Training Studio",
    page_icon="🎨",
    layout="wide"
)

# --- Session State Initialization ---
if "trainer" not in st.session_state:
    # Detect Device
    # Note: On Streamlit Cloud (Linux), this will default to CPU usually.
    # If you have a GPU instance, it might pick up CUDA.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.toast("Using NVIDIA GPU (CUDA)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        st.toast("Using Apple Silicon GPU (MPS)")
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
    
    # Section 1: Save/Load (Manual)
    st.subheader("💾 Manual Checkpoints")
    if st.button("Save Current State"):
        try:
            st.session_state.trainer.save_checkpoint("manual_checkpoint.pth")
            st.success("Saved as manual_checkpoint.pth")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    
    # Section 2: Load Pre-Trained Models
    st.subheader("📂 Load Pre-Trained Model")
    
    # Find all .pth files in the current directory
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if not model_files:
        st.warning("No .pth models found in repo root.")
    else:
        selected_model = st.selectbox("Select a model file", model_files)
        
        if st.button("Load Selected Model"):
            try:
                # 1. Setup
                trainer = st.session_state.trainer
                # flexible attribute access
                if hasattr(trainer, 'generator'):
                    net_g = trainer.generator
                elif hasattr(trainer, 'netG'):
                    net_g = trainer.netG
                else:
                    st.error("❌ Could not find generator network in trainer!")
                    st.stop()

                # 2. Load File
                checkpoint = torch.load(selected_model, map_location=torch.device('cpu'))
                
                # 3. UNWRAP THE WEIGHTS (This is the fix)
                state_dict = checkpoint
                
                # Check for the specific key found in your file dump
                if isinstance(checkpoint, dict) and 'netG_state_dict' in checkpoint:
                    state_dict = checkpoint['netG_state_dict']
                    st.toast("Found 'netG_state_dict' key! Unwrapping...")
                
                # Fallback for other common names just in case
                elif isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                    state_dict = checkpoint['generator_state_dict']

                # 4. Clean Keys (Strip 'module.' if present)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '') # Fix DataParallel naming
                    new_state_dict[name] = v
                state_dict = new_state_dict

                # 5. Load
                net_g.load_state_dict(state_dict, strict=False)
                
                # 6. Validate
                net_g.eval()
                st.success(f"✅ Successfully loaded {selected_model}")
                st.balloons()

                # 7. Diagnostic Check
                with torch.no_grad():
                    dummy = torch.randn(1, 100, 1, 1, device=trainer.device)
                    out = net_g(dummy)
                    min_v, max_v = out.min().item(), out.max().item()
                    st.info(f"Diagnostic: Output pixel range {min_v:.2f} to {max_v:.2f}")

            except Exception as e:
                st.error(f"Error loading model: {e}")

# --- Main Interface ---
st.title("🎨 DCGAN Dashboard")

# Create the tabs (This was the missing part causing your NameError!)
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
                    # Use existing helper or generic grid
                    try:
                        grid_img = trainer.images_to_base64(fake, nrow=4)
                        image_preview.markdown(f'<img src="data:image/png;base64,{grid_img}" width="100%"/>', unsafe_allow_html=True)
                    except Exception:
                        st.warning("Could not render image preview.")

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
                
                try:
                    fake_images = trainer.generate_images(num_imgs)
                    # Calculate grid size roughly square
                    nrow = int(num_imgs**0.5)
                    b64_img = trainer.images_to_base64(fake_images, nrow=nrow)
                    st.markdown(f'<img src="data:image/png;base64,{b64_img}" width="100%"/>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Generation failed: {e}")