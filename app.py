"""
Gradio Application for Indic-CLIP Demo

@description
This script launches an interactive Gradio interface to demonstrate the capabilities
of the trained Indic-CLIP model. It allows users to perform:
- Image-to-Text Retrieval: Find relevant text descriptions for an input image.
- Text-to-Image Retrieval: Find relevant images for an input text query (Hindi/Sanskrit).
- Zero-Shot Classification: Classify an image based on user-provided text labels without
  specific training on those labels.

The application loads a pre-trained Indic-CLIP model checkpoint and its associated tokenizer.
It uses a small, pre-defined gallery of text descriptions and sample images for the retrieval demos.

@dependencies
- gradio: For building the web interface.
- torch: Core PyTorch library.
- PIL (Pillow): For image processing.
- indic_clip library: Contains the core model, data handling, and inference logic.
  (Needs to be installed in the environment where this app runs).
- transformers, timm, sentencepiece, fastai: Dependencies of indic_clip.

@environment_variables (Optional - for Hugging Face Spaces)
- CHECKPOINT_FILENAME: Name of the checkpoint file (e.g., 'best_valid_loss.pth'). Default: 'best_valid_loss.pth'.
- CHECKPOINT_DIR: Path to the directory containing checkpoints, relative to app root. Default: 'models/checkpoints'.
- TOKENIZER_DIR: Path to the directory containing the tokenizer files, relative to app root. Default: 'models/tokenizer'.
- SAMPLE_DIR: Path to the directory containing sample images, relative to app root. Default: 'data/samples'.
- IMAGE_SIZE: Image size expected by the model. Default: 224.
- TOP_K: Number of retrieval results to display. Default: 5.
- MODEL_VISION_BACKBONE: Name of the vision backbone used during training. Default: 'resnet50'.
- MODEL_TEXT_BACKBONE: Name of the text backbone used during training. Default: 'ai4bharat/indic-bert'.
- MODEL_EMBED_DIM: Embedding dimension of the trained model. Default: 512.

@notes
- Ensure the specified checkpoint file, tokenizer files, and sample images exist in the expected paths
  within the Hugging Face Space repository or are downloaded/mounted correctly.
- Model configuration (embed_dim, backbones) MUST match the loaded checkpoint.
- Pre-encoding gallery features significantly improves demo speed. If loading fails, retrieval will be slower.
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import random
import logging
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional

# --- Project Imports ---
# Try importing assuming 'indic_clip' is installed
try:
    from indic_clip.inference import (
        load_indic_clip_model,
        extract_image_features,
        extract_text_features,
        compute_similarity
    )
    from indic_clip.core import (
        get_logger, setup_logging, CHECKPOINT_PATH as DEFAULT_CHECKPOINT_PATH,
        TOKENIZER_PATH as DEFAULT_TOKENIZER_PATH, PRETRAINED_TOKENIZER_NAME,
        DEFAULT_EMBED_DIM, DEFAULT_IMAGE_SIZE, PROJECT_ROOT as DEFAULT_PROJECT_ROOT
    )
    from indic_clip.data.tokenization import IndicBERTTokenizer
    from indic_clip.model.clip import IndicCLIP
    from indic_clip.evaluation.benchmarks import (
        DEFAULT_PROMPT_TEMPLATES_HI, DEFAULT_PROMPT_TEMPLATES_SA, DEFAULT_PROMPT_TEMPLATES_EN
    )
    PROJECT_MODULES_LOADED = True
except ImportError as e:
    PROJECT_MODULES_LOADED = False
    print(f"Error importing indic_clip modules: {e}")
    print("Falling back to dummy definitions or exiting if critical components missing.")
    # Define dummy logger if core is missing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("indic_clip_app_fallback")
    def get_logger(name): return logging.getLogger(name)
    def setup_logging(): pass
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_EMBED_DIM = 512
    PRETRAINED_TOKENIZER_NAME = "ai4bharat/indic-bert" # Fallback
    DEFAULT_CHECKPOINT_PATH = Path("./models/checkpoints") # Fallback
    DEFAULT_TOKENIZER_PATH = Path("./models/tokenizer") # Fallback
    DEFAULT_PROJECT_ROOT = Path(".") # Fallback
    # Dummies for functionality - these will likely cause errors later if used
    class IndicCLIP(torch.nn.Module): pass
    class IndicBERTTokenizer: pass
    def load_indic_clip_model(*args, **kwargs): raise RuntimeError("Model loading failed - indic_clip not found")
    def extract_image_features(*args, **kwargs): raise RuntimeError("Inference failed - indic_clip not found")
    def extract_text_features(*args, **kwargs): raise RuntimeError("Inference failed - indic_clip not found")
    def compute_similarity(*args, **kwargs): raise RuntimeError("Inference failed - indic_clip not found")
    DEFAULT_PROMPT_TEMPLATES_EN = ["a photo of a {}"] # Fallback

# --- Configuration ---
setup_logging()
logger = get_logger("indic_clip_app")

CHECKPOINT_FILENAME = os.getenv("CHECKPOINT_FILENAME", 'best_valid_loss.pth')
HF_SPACE_ROOT = Path(".") # Assume current dir is repo root in HF Spaces
CHECKPOINT_DIR = HF_SPACE_ROOT / os.getenv("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_PATH)
TOKENIZER_DIR = HF_SPACE_ROOT / os.getenv("TOKENIZER_DIR", DEFAULT_TOKENIZER_PATH)
SAMPLE_DIR = HF_SPACE_ROOT / os.getenv("SAMPLE_DIR", DEFAULT_PROJECT_ROOT / "data/samples") # Use project root default if not set

CHECKPOINT_FILE_PATH = CHECKPOINT_DIR / CHECKPOINT_FILENAME
TOKENIZER_DIR_PATH = TOKENIZER_DIR
SAMPLE_IMAGE_DIR = SAMPLE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", DEFAULT_IMAGE_SIZE))
TOP_K = int(os.getenv("TOP_K", 5))

MODEL_VISION_BACKBONE = os.getenv("MODEL_VISION_BACKBONE", "resnet50")
MODEL_TEXT_BACKBONE = os.getenv("MODEL_TEXT_BACKBONE", PRETRAINED_TOKENIZER_NAME)
MODEL_EMBED_DIM = int(os.getenv("MODEL_EMBED_DIM", DEFAULT_EMBED_DIM)) # Get default from core

logger.info(f"--- App Configuration ---")
logger.info(f"Device: {DEVICE}")
logger.info(f"Checkpoint Path: {CHECKPOINT_FILE_PATH}")
logger.info(f"Tokenizer Path: {TOKENIZER_DIR_PATH}")
logger.info(f"Sample Image Dir: {SAMPLE_IMAGE_DIR}")
logger.info(f"Image Size: {IMAGE_SIZE}")
logger.info(f"Vision Backbone: {MODEL_VISION_BACKBONE}")
logger.info(f"Text Backbone: {MODEL_TEXT_BACKBONE}")
logger.info(f"Embedding Dim: {MODEL_EMBED_DIM}")
logger.info(f"Top K Retrieval: {TOP_K}")
logger.info(f"Project Modules Loaded: {PROJECT_MODULES_LOADED}")
logger.info(f"-------------------------")


# --- Load Tokenizer ---
tokenizer = None
if PROJECT_MODULES_LOADED and TOKENIZER_DIR_PATH.exists():
    try:
        tokenizer = IndicBERTTokenizer.load_tokenizer(TOKENIZER_DIR_PATH)
        logger.info(f"Tokenizer loaded successfully from {TOKENIZER_DIR_PATH}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {TOKENIZER_DIR_PATH}: {e}", exc_info=True)
elif PROJECT_MODULES_LOADED:
    logger.error(f"Tokenizer directory not found at {TOKENIZER_DIR_PATH}. Cannot start app.")

# --- Load Model ---
model: IndicCLIP = None
if PROJECT_MODULES_LOADED and tokenizer is not None:
    MODEL_CONFIGURATION = {
        'embed_dim': MODEL_EMBED_DIM,
        'vision_model_name': MODEL_VISION_BACKBONE,
        'vision_pretrained': False, # Doesn't affect loading state_dict
        'text_model_name': MODEL_TEXT_BACKBONE,
        'text_pretrained': False,
        'tokenizer': tokenizer
    }
    if CHECKPOINT_FILE_PATH.is_file():
        try:
            model = load_indic_clip_model(
                checkpoint_path=CHECKPOINT_FILE_PATH,
                model_config=MODEL_CONFIGURATION,
                device=DEVICE
            )
            logger.info(f"Model loaded successfully from {CHECKPOINT_FILE_PATH} to device {DEVICE}.")
        except Exception as e:
            logger.error(f"Error loading model from {CHECKPOINT_FILE_PATH}: {e}", exc_info=True)
    else:
         logger.error(f"Checkpoint file not found: {CHECKPOINT_FILE_PATH}")
else:
    logger.error("Tokenizer not loaded or project modules missing, skipping model load.")

if model is None:
    logger.critical("Model could not be loaded. The application cannot function properly.")


# --- Sample Gallery Data ---
TEXT_GALLERY = [
    "एक लड़का फुटबॉल खेल रहा है।",
    "समुद्र तट पर सूर्यास्त।",
    "एक बिल्ली सोफे पर सो रही है।",
    "पारंपरिक साड़ी पहने एक महिला।",
    "एक मंदिर का प्रवेश द्वार।",
    "देवता गणेश की मूर्ति।",
    "एक व्यस्त भारतीय बाज़ार।",
    "लैपटॉप पर काम करता हुआ व्यक्ति।",
    "मेज़ पर रखी किताबों का ढेर।",
    "एक लाल रंग की स्पोर्ट्स कार।"
]

# Define image filenames expected in SAMPLE_IMAGE_DIR
IMAGE_GALLERY_FILENAMES = [
    'cat.jpg',
    'dog_park.jpg',
    'sunset_beach.jpg',
    'woman_saree.jpg',
    'temple.jpg',
    'ganesh.jpg',
    'market.jpg',
    'laptop.jpg',
    'books.jpg',
    'car.jpg'
]

# Verify image existence and create lists for display and processing
IMAGE_GALLERY_DISPLAY = [] # List of string paths for Gradio
valid_image_gallery_files = [] # List of Path objects for processing
SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
for filename in IMAGE_GALLERY_FILENAMES:
     full_img_path = SAMPLE_IMAGE_DIR / filename
     if full_img_path.is_file():
         IMAGE_GALLERY_DISPLAY.append(str(full_img_path)) # Use path relative to app root
         valid_image_gallery_files.append(full_img_path)
     else:
         logger.warning(f"Sample image not found: {full_img_path}")

logger.info(f"Found {len(valid_image_gallery_files)} valid sample images in {SAMPLE_IMAGE_DIR}")

# --- Pre-encode Gallery Features ---
text_gallery_features: Optional[torch.Tensor] = None
image_gallery_features: Optional[torch.Tensor] = None

if model is not None and tokenizer is not None:
    logger.info("Attempting to pre-encode gallery features...")
    try:
        text_gallery_features = extract_text_features(model, tokenizer, TEXT_GALLERY, device=DEVICE)
        logger.info(f"Encoded {len(TEXT_GALLERY)} text gallery items. Shape: {text_gallery_features.shape if text_gallery_features is not None else 'None'}")
    except Exception as e:
        logger.error(f"Failed to pre-encode text gallery: {e}", exc_info=True)

    if valid_image_gallery_files:
        try:
            image_gallery_features = extract_image_features(model, valid_image_gallery_files, img_size=IMAGE_SIZE, device=DEVICE)
            logger.info(f"Encoded {len(valid_image_gallery_files)} image gallery items. Shape: {image_gallery_features.shape if image_gallery_features is not None else 'None'}")
        except Exception as e:
            logger.error(f"Failed to pre-encode image gallery: {e}", exc_info=True)
else:
    logger.warning("Model or tokenizer not loaded, cannot pre-encode gallery features.")


# --- Gradio Interface Functions ---

def predict_text_from_image(image_input: Image.Image) -> str:
    """Gradio interface function for Image-to-Text retrieval."""
    if model is None or tokenizer is None or text_gallery_features is None:
        return "Error: Model, tokenizer, or text gallery features not loaded. Cannot perform retrieval."
    if image_input is None:
        return "Error: Please upload an image."

    logger.info("Performing Image-to-Text retrieval...")
    try:
        img_feat = extract_image_features(model, image_input, img_size=IMAGE_SIZE, device=DEVICE)
        if img_feat is None or img_feat.nelement() == 0:
             return "Error: Could not extract features from the image."

        similarity = compute_similarity(model, img_feat, text_gallery_features)
        scores, indices = torch.topk(similarity.squeeze(0), k=min(TOP_K, len(TEXT_GALLERY)), dim=-1)

        results = "\n".join([f"{scores[i].item():.4f}: {TEXT_GALLERY[indices[i].item()]}" for i in range(len(indices))])
        logger.info("Image-to-Text retrieval successful.")
        return results

    except Exception as e:
        logger.error(f"Error in predict_text_from_image: {e}", exc_info=True)
        return f"An error occurred during text retrieval: {e}"

def predict_image_from_text(text_input: str) -> List[Tuple[str, str]]:
    """Gradio interface function for Text-to-Image retrieval."""
    # Provide a placeholder image path for errors
    error_img_placeholder = "https://dummyimage.com/150x150/ff0000/ffffff.png&text=Error"

    if model is None or tokenizer is None or image_gallery_features is None or not IMAGE_GALLERY_DISPLAY:
        return [(error_img_placeholder, "Error: Model, tokenizer, or image gallery features not loaded.")]
    if not text_input or not text_input.strip():
        return [(error_img_placeholder, "Error: Please enter text query.")]

    logger.info(f"Performing Text-to-Image retrieval for query: '{text_input}'")
    try:
        txt_feat = extract_text_features(model, tokenizer, text_input, device=DEVICE)
        if txt_feat is None or txt_feat.nelement() == 0:
             return [(error_img_placeholder, "Error: Could not extract features from the text.")]

        similarity = compute_similarity(model, image_gallery_features, txt_feat) # T2I order: (Img, Txt) -> [N_img, N_txt=1]
        scores, indices = torch.topk(similarity.squeeze(-1), k=min(TOP_K, len(IMAGE_GALLERY_DISPLAY)), dim=0)

        results = []
        for i in range(len(indices)):
            img_index = indices[i].item()
            # Use the verified display path list
            display_path = IMAGE_GALLERY_DISPLAY[img_index]
            score = scores[i].item()
            caption = f"{score:.4f}: {Path(display_path).name}" # Use filename in caption
            results.append((display_path, caption))

        logger.info("Text-to-Image retrieval successful.")
        return results
    except Exception as e:
        logger.error(f"Error in predict_image_from_text: {e}", exc_info=True)
        return [(error_img_placeholder, f"An error occurred during image retrieval: {e}")]

def predict_zero_shot(image_input: Image.Image, candidate_labels_text: str) -> Dict[str, float]:
    """Gradio interface function for Zero-Shot Classification."""
    if model is None or tokenizer is None:
        return {"Error": 1.0, "Message": "Model or tokenizer not loaded."}
    if image_input is None:
        return {"Error": 1.0, "Message": "Please upload an image."}
    if not candidate_labels_text or not candidate_labels_text.strip():
        return {"Error": 1.0, "Message": "Please enter candidate labels (comma-separated)."}

    logger.info(f"Performing Zero-Shot classification for labels: '{candidate_labels_text}'")
    try:
        class_names = [label.strip() for label in candidate_labels_text.split(',') if label.strip()]
        if not class_names:
            return {"Error": 1.0, "Message": "Invalid label format. Enter comma-separated labels."}

        # Using English templates as default for broader applicability
        templates = DEFAULT_PROMPT_TEMPLATES_EN

        img_feat = extract_image_features(model, image_input, img_size=IMAGE_SIZE, device=DEVICE)
        if img_feat is None or img_feat.nelement() == 0:
             return {"Error": 1.0, "Message": "Could not extract features from image."}

        all_prompts = [tmpl.format(cn) for tmpl in templates for cn in class_names]
        text_embeddings = extract_text_features(model, tokenizer, all_prompts, device=DEVICE)
        if text_embeddings is None or text_embeddings.nelement() == 0:
             return {"Error": 1.0, "Message": "Could not extract features from text labels."}

        if len(templates) > 1:
            num_classes = len(class_names)
            text_embeddings = text_embeddings.view(len(templates), num_classes, -1).mean(dim=0)

        # Ensure final normalization
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Compute similarity
        similarity = compute_similarity(model, img_feat, text_embeddings).squeeze()

        # Apply softmax to get probabilities
        # Add temperature scaling if needed (usually included in compute_similarity via logit_scale)
        probs = F.softmax(similarity, dim=-1)

        results = {class_names[i]: probs[i].item() for i in range(len(class_names))}
        logger.info(f"Zero-Shot classification successful. Results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error in predict_zero_shot: {e}", exc_info=True)
        return {"Error": 1.0, "Message": f"An error occurred: {e}"}

# --- Gradio Interface Definition ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: black; background: black; }
input[type='range'] { accent-color: black; }
.dark input[type='range'] { accent-color: #dfdqdq; }
.container { max-width: 1100px; margin: auto; padding-top: 1.5rem; }
#gallery { min-height: 22rem; margin-bottom: 15px; margin-left: auto; margin-right: auto; }
#gallery>div>.h-full { min-height: 20rem; }
.details:hover { text-decoration: underline; }
.feedback { font-size: 0.8rem; margin-bottom: 5px; }
.feedback textarea { font-size: 0.8rem; }
.feedback button { margin: 0; }
.gradio-container { max-width: 1140px !important; }
footer {visibility: hidden}
"""

block = gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue"))

with block:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 1000px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem;">
            Indic-CLIP <span style="font-size: 1.5rem">🖼️<->📝</span>
        </h1>
        <p style="margin-bottom: 10px; font-size: 94%">
            Multimodal Vision-Language Model for Indic Languages (Hindi/Sanskrit)
         </p>
         <p>Provide an image or text to retrieve corresponding matches, or perform zero-shot classification.</p>
         <p><strong>Note:</strong> This demo uses a small, fixed gallery for retrieval. Model checkpoint: <code>{}</code></p>
       </div>
        """.format(CHECKPOINT_FILENAME)
    )
    with gr.Tabs():
        with gr.TabItem("🖼️ Image-to-Text Retrieval"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    input_image_i2t = gr.Image(type="pil", label="Input Image")
                    submit_i2t = gr.Button("Retrieve Text", variant="primary")
                with gr.Column():
                    output_text_i2t = gr.Textbox(lines=TOP_K, label=f"Top {TOP_K} Text Matches (Score: Text)", interactive=False)
            gr.Examples(
                examples=IMAGE_GALLERY_DISPLAY[:min(5, len(IMAGE_GALLERY_DISPLAY))], # Use verified paths
                inputs=input_image_i2t,
                label="Sample Images (Click to Load)"
            )

        with gr.TabItem("📝 Text-to-Image Retrieval"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    input_text_t2i = gr.Textbox(label="Input Text (e.g., Hindi, Sanskrit, English)", placeholder="उदाहरण: एक बिल्ली सोफे पर सो रही है।")
                    submit_t2i = gr.Button("Retrieve Images", variant="primary")
                with gr.Column():
                    output_gallery_t2i = gr.Gallery(label=f"Top {TOP_K} Image Matches (Score: Filename)", show_label=True).style(columns=TOP_K, height="auto", object_fit="contain")
            gr.Examples(
                examples=TEXT_GALLERY[:min(5, len(TEXT_GALLERY))],
                inputs=input_text_t2i,
                label="Sample Text Queries (Click to Load)"
            )

        with gr.TabItem("🏷️ Zero-Shot Classification"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    input_image_zs = gr.Image(type="pil", label="Input Image")
                with gr.Column():
                    candidate_labels_zs = gr.Textbox(label="Candidate Labels (Comma-separated)", placeholder="e.g., बिल्ली, कुत्ता, पक्षी, कार, मंदिर")
                    submit_zs = gr.Button("Classify Image", variant="primary")
                    output_labels_zs = gr.Label(num_top_classes=max(3, len(candidate_labels_zs.value.split(',')) if candidate_labels_zs.value else 3), label="Classification Probabilities")
            gr.Examples(
                examples=[
                    [img_path, "बिल्ली, कुत्ता, पक्षी"] for img_path in IMAGE_GALLERY_DISPLAY if "cat" in img_path
                ] + [
                    [img_path, "साड़ी, कुर्ता, पोशाक"] for img_path in IMAGE_GALLERY_DISPLAY if "saree" in img_path
                ] + [
                    [img_path, "मंदिर, मस्जिद, चर्च"] for img_path in IMAGE_GALLERY_DISPLAY if "temple" in img_path
                ] + [
                    [img_path, "कार, बस, मोटरबाइक"] for img_path in IMAGE_GALLERY_DISPLAY if "car" in img_path
                ]
                ,
                inputs=[input_image_zs, candidate_labels_zs],
                outputs=output_labels_zs,
                label="Sample Images and Labels (Click to Load)",
                cache_examples=False # Avoid caching issues with file paths
            )

    # Define button click actions
    submit_i2t.click(
        predict_text_from_image,
        inputs=[input_image_i2t],
        outputs=[output_text_i2t],
        api_name="image_to_text_retrieval"
    )
    submit_t2i.click(
        predict_image_from_text,
        inputs=[input_text_t2i],
        outputs=[output_gallery_t2i],
        api_name="text_to_image_retrieval"
    )
    submit_zs.click(
        predict_zero_shot,
        inputs=[input_image_zs, candidate_labels_zs],
        outputs=[output_labels_zs],
        api_name="zero_shot_classification"
    )

# --- Launch ---
if __name__ == "__main__":
    if not PROJECT_MODULES_LOADED:
         print("\nERROR: Indic-CLIP project modules could not be loaded.")
         print("Please ensure the library is installed correctly ('pip install -e .')")
         print("Cannot launch Gradio app.\n")
         # Display error in Gradio if possible
         try:
             with block:
                 gr.Markdown("<h2 style='color:red;'>ERROR: Indic-CLIP project code not found. Cannot launch application.</h2>")
             block.launch()
         except NameError: # If block wasn't defined due to earlier errors
             pass
    elif model is None:
        logger.error("Model is None. Cannot launch Gradio app.")
        # Display error in Gradio
        with block:
             gr.Markdown("<h2 style='color:red;'>ERROR: Model failed to load. Application cannot start. Please check logs and ensure checkpoint/tokenizer exist.</h2>")
        block.launch()
    else:
        logger.info("Launching Gradio interface...")
        block.launch() # share=True for public link