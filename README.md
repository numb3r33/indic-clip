# Indic-CLIP Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/[your-username]/[your-space-name]) <!-- Replace with your Space URL -->

This Space demonstrates the capabilities of **Indic-CLIP**, a multimodal vision-language model adapted for Indic languages (Hindi/Sanskrit) based on the CLIP architecture.

## Features

This demo allows you to interact with the model in three ways:

1.  **🖼️ Image-to-Text Retrieval:** Upload an image, and the model will retrieve the most relevant text descriptions from a predefined gallery.
2.  **📝 Text-to-Image Retrieval:** Enter a text query (e.g., in Hindi, Sanskrit, or English if the model supports it), and the model will retrieve the most similar images from a predefined gallery.
3.  **🏷️ Zero-Shot Classification:** Upload an image and provide a comma-separated list of candidate labels (e.g., "बिल्ली, कुत्ता, पक्षी"). The model will predict the probability of the image belonging to each label without having been explicitly trained on them.

## How to Use

1.  Select one of the tabs: "Image-to-Text Retrieval", "Text-to-Image Retrieval", or "Zero-Shot Classification".
2.  Follow the instructions within the tab:
    *   Upload an image using the image component.
    *   Enter text queries or labels in the text boxes.
3.  Click the corresponding button ("Retrieve Text", "Retrieve Images", "Classify Image") to get results.
4.  You can also click on the examples provided below each section to quickly try out the demo.

## Model Information

*   **Architecture:** CLIP adaptation using fast.ai.
*   **Vision Encoder:** (e.g., ResNet50, ViT-B/16) - Specify the backbone used.
*   **Text Encoder:** (e.g., ai4bharat/indic-bert) - Specify the backbone used.
*   **Training Data:** (e.g., Flickr8k-Hindi) - Specify the primary training dataset.
*   **Languages:** Hindi, Sanskrit (Specify based on training).

*Note: The retrieval galleries in this demo are small and fixed for demonstration purposes.*

## Find More

*   **GitHub Repository:** [Link to your main project repository] <!-- Add link -->
*   **Paper (if applicable):** [Link to your paper] <!-- Add link -->

Built with [Gradio](https://gradio.app/) and hosted on [Hugging Face Spaces](https://huggingface.co/spaces).