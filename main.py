import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from review_auditor import analyze_product_reviews

load_dotenv()

app = FastAPI(title="TrueFind Verified Product Search API")
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "TrueFind Verified Product Search API is running.",
        "documentation": "/docs",
        "endpoints": {
            "search": "/api/search (POST)"
        }
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "YOUR_SERPAPI_KEY")
genai_client = genai.Client(api_key=GEMINI_API_KEY)

MIN_RATING = 4.2
MIN_REVIEWS = 50
MEAN_BASELINE_RATING = 4.3


def extract_product_query_from_image(image_bytes: bytes, mime_type: str) -> str:
    prompt = (
        "Identify the exact product in this image for an e-commerce search. "
        "Return ONLY the specific brand, model name, color, and product type in 5-8 words."
    )
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text.strip()


def calculate_bayesian_score(rating: float, reviews: int) -> float:
    v = float(reviews)
    m = float(MIN_REVIEWS)
    R = float(rating)
    C = MEAN_BASELINE_RATING
    return (v / (v + m)) * R + (m / (v + m)) * C


def fetch_and_filter_products(query: str, country_code: str = "in"):
    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": country_code,
        "hl": "en",
        "api_key": SERPAPI_KEY,
    }

    response = requests.get("https://serpapi.com/search", params=params, timeout=15)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to fetch from e-commerce aggregator")

    data = response.json()
    raw_results = data.get("shopping_results", [])
    verified_products = []

    for item in raw_results:
        rating = item.get("rating")
        reviews = item.get("reviews")
        price = item.get("extracted_price")
        title = item.get("title")
        source = item.get("source")
        link = item.get("product_link") or item.get("link")
        thumbnail = item.get("thumbnail")

        if not all([rating, reviews, price, title, source, link]):
            continue

        if float(rating) < MIN_RATING or int(reviews) < MIN_REVIEWS:
            continue

        trust_score = round(calculate_bayesian_score(float(rating), int(reviews)), 2)

        verified_products.append({
            "title": title,
            "price": item.get("price"),
            "extracted_price": price,
            "rating": float(rating),
            "reviews": int(reviews),
            "trust_score": trust_score,
            "merchant": source,
            "link": link,
            "thumbnail": thumbnail,
            "delivery": item.get("delivery", "Standard shipping"),
        })

    verified_products.sort(key=lambda x: x["trust_score"], reverse=True)
    return verified_products


@app.post("/api/search")
async def search_products(
    query: str = Form(None),
    country: str = Form("in"),
    image: UploadFile = File(None),
):
    search_term = query

    if image is not None:
        image_bytes = await image.read()
        detected_product = extract_product_query_from_image(image_bytes, image.content_type or "image/jpeg")
        search_term = f"{detected_product} {query}" if query else detected_product

    if not search_term:
        raise HTTPException(status_code=400, detail="Provide either a text query or an image.")

    results = fetch_and_filter_products(search_term, country_code=country)

    return {
        "resolved_query": search_term,
        "total_verified_matches": len(results),
        "filters_applied": {
            "min_rating": MIN_RATING,
            "min_reviews": MIN_REVIEWS,
            "ranking_algorithm": "Bayesian Weighted Average",
        },
        "products": results,
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)