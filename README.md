# 👗 Hybrid Fashion Trend Matching Model — Style Studio (Game Zone)

This  is  Flask-based **AI scoring microservice** powers real-time fashion trend analysis in **Style Studio’s Game Zone**. It compares user-submitted outfit images with **the trends list**, using a **hybrid of FashionBERT + CLIP**, and generates a **trend alignment score** for leaderboard ranking and winner selection.

---
Style Studio Repo-https://github.com/Ananyasolanki123/StyleStudio

## 🎯 Purpose

This API is used as the **core scoring backend** in the **Design Analysis system** for Style Studio. It determines how well a user’s outfit matches current fashion trends, with scores feeding directly into the Game Zone **ranking and competition winner logic**.

---

## 🧠 Scoring Methodology

1. 🔤 **FashionBERT** encodes trend text embeddings based on fashion-specific vocabulary.
2. 🧠 **CLIP** evaluates the semantic alignment between the outfit image and trending text prompts.
3. 🎨 **KMeans clustering** extracts dominant colors from the image and compares them with colors from trending looks.
4. 📊 A weighted score is generated, combining:
   {
  "color_coherence_score",
  "final_weighted_score",
  "trend_match_score",
  "vibe_score",
  "vibe_trend"
}

---

## 🧪 Example Response

json
{
  "color_coherence_score": 56.15,
  "final_weighted_score": 39.44,
  "trend_match_score": 22.74,
  "vibe_score": 0.23,
  "vibe_trend": "isPartial"
}

🛠️ API Endpoint
POST /trend-match
Upload a fashion outfit image and get a trend score.

🧩 Components
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Component	         |    Description         
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FashionBERT	Text   |   embedding for trend prompts (HuggingFace)
Fashion-CLIP	     |   Image-text alignment model (patrickjohncyh)
KMeans	           |   Color palette clustering


📦 Project Structure 
Hybrid Model/
├── main.py                  # Main Flask app and scoring logic
├── templates/form.html      # Optional web UI

🏆 Game Zone Integration

| Step | Role of Model                                      |
| ---- | -------------------------------------------------- |
| 1.   | Users submit outfit designs                        |
| 2.   | Model analyzes similarity with trending fashion    |
| 3.   | Color and semantic scores are calculated           |
| 4.   | Final scores are fed into Game Zone leaderboard    |
| 5.   | **Top scorers are declared winners** automatically |

⚙️ Setup Instructions
1.Run the API:
python main.py

2.Open in browser:
http://localhost:5000


🤖 Model Links
FashionBERT (Hugging Face)
FashionCLIP (Hugging Face)


