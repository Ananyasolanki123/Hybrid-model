from flask import Flask, request, render_template, jsonify
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load FashionCLIP model from Hugging Face
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Trend definitions
vibe_prompts = [
    "cottagecore aesthetic with florals",
    "futuristic streetwear with neon",
    "vintage 90s style with denim",
    "chic minimalist black and white outfit"
]

attr_prompts = [
    "pleated midi skirt in pastel color",
    "oversized blazer with shoulder pads",
    "leather biker jacket with studs",
    "denim overalls with pockets"
]

# Trend-based color palettes
trend_color_palettes = {
    "cottagecore aesthetic with florals": np.array([[201, 186, 145], [155, 196, 170], [237, 223, 204]]),
    "futuristic streetwear with neon": np.array([[57, 255, 20], [0, 255, 255], [255, 0, 255]]),
    "vintage 90s style with denim": np.array([[0, 0, 128], [128, 0, 128], [255, 255, 0]]),
    "chic minimalist black and white outfit": np.array([[0, 0, 0], [255, 255, 255]])
}

# Precompute embeddings
def get_text_embeddings(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        return model.get_text_features(**inputs)

vibe_text_emb = get_text_embeddings(vibe_prompts)
attr_text_emb = get_text_embeddings(attr_prompts)

# Color analysis functions
def extract_dominant_colors(image, k=5):
    image = image.resize((150, 150))
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10).fit(pixels)
    return kmeans.cluster_centers_

def color_coherence_score(dominant_colors, palette):
    distances = []
    for color in dominant_colors:
        min_dist = np.min(np.linalg.norm(palette - color, axis=1))
        distances.append(min_dist)
    avg_dist = np.mean(distances)
    return round(max(0, 100 - avg_dist), 2)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/trend-match', methods=['POST'])
def trend_match():
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Get image embedding
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)

        # Cosine similarity scores
        vibe_scores = torch.nn.functional.cosine_similarity(image_emb, vibe_text_emb)
        attr_scores = torch.nn.functional.cosine_similarity(image_emb, attr_text_emb)

        # Top matches
        top_vibe_idx = vibe_scores.argmax().item()
        top_attr_idx = attr_scores.argmax().item()
        top_trend = vibe_prompts[top_vibe_idx]
        top_attr = attr_prompts[top_attr_idx]

        # Final match score
        vibe_score = vibe_scores[top_vibe_idx].item()
        attr_score = attr_scores[top_attr_idx].item()
        match_score = round((0.6 * vibe_score + 0.4 * attr_score) * 100, 2)

        # Color coherence
        dominant_colors = extract_dominant_colors(image)
        palette = trend_color_palettes.get(top_trend, np.array([[128, 128, 128]]))  # fallback
        color_score = color_coherence_score(dominant_colors, palette)

        # Weighted final score
        final_score = round((0.5 * match_score + 0.5 * color_score), 2)

        return jsonify({
            "vibe_trend": top_trend,
            "vibe_score": round(vibe_score, 2),
            "attribute_trend": top_attr,
            "attribute_score": round(attr_score, 2),
            "trend_match_score": match_score,
            "color_coherence_score": color_score,
            "final_weighted_score": final_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
