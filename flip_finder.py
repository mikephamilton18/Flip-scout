import os
import json
import time
import requests
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- CONFIGURATION ---
TARGET_AREAS = "Barry, Vale of Glamorgan, Penarth, Dinas Powys, Llantwit Major, Cardiff"
MIN_ESTIMATED_PROFIT_GBP = 35
SEEN_DB_FILE = "seen_items.json"

# Initialize Gemini Client (reads GEMINI_API_KEY from environment)
client = genai.Client()

# --- STRUCTURED OUTPUT SCHEMAS ---
class FlipOpportunity(BaseModel):
    item_title: str = Field(description="Clean title of the free item found")
    location: str = Field(description="Specific town/area (e.g., Barry, Penarth, Cardiff)")
    source_platform: str = Field(description="Platform where it is listed (e.g., Gumtree, Freecycle, Olio)")
    listing_url: str = Field(description="Direct URL to the listing or search page from grounding sources")
    condition_notes: str = Field(description="Condition details mentioned in the listing")
    estimated_market_value_gbp: int = Field(description="Average realistic UK secondhand resale value in GBP based on sold comps")
    quick_sale_price_gbp: int = Field(description="Conservative price for a fast 48-hour local sale in GBP")
    sold_comps_evidence: str = Field(description="Brief summary of real sold prices/comps found via Google Search")
    prep_and_upsell_tips: str = Field(description="Quick 15-min clean/photo tip to maximize resale price")
    outreach_message: str = Field(description="Ready-to-paste message to send the seller immediately")

class FlipReport(BaseModel):
    opportunities: List[FlipOpportunity]

# --- HELPER FUNCTIONS ---
def load_seen_items() -> set:
    if os.path.exists(SEEN_DB_FILE):
        with open(SEEN_DB_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_items(seen: set):
    with open(SEEN_DB_FILE, "w") as f:
        json.dump(list(seen), f, indent=2)

def send_telegram_alert(opp: FlipOpportunity):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    msg = (
        f"🔥 *NEW FREE FLIP FOUND ({opp.location})*\n\n"
        f"*Item:* {opp.item_title}\n"
        f"*Est. Value:* £{opp.estimated_market_value_gbp} (Quick Flip: £{opp.quick_sale_price_gbp})\n"
        f"*Platform:* {opp.source_platform}\n"
        f"*Comps:* {opp.sold_comps_evidence}\n"
        f"*Prep:* {opp.prep_and_upsell_tips}\n\n"
        f"🔗 [View Listing]({opp.listing_url})\n\n"
        f"📋 *Copy & Send:*\n`{opp.outreach_message}`"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})

# --- MAIN GEMINI SEARCH & VALUATION PIPELINE ---
def run_flip_finder():
    print(f"🔍 Searching local freebies & sold comps around {TARGET_AREAS}...")
    seen_items = load_seen_items()

    prompt = f"""
    You are an expert UK secondhand resale and arbitrage scout operating in South Wales.
    
    TASK 1: SEARCH LOCAL FREEBIES
    Use Google Search to find current, active FREE items listed for collection in or near: {TARGET_AREAS}.
    Focus on platforms like Gumtree Freebies, Freecycle, Trash Nothing, and Olio.
    Look specifically for high-resale-value categories:
    - Solid wood furniture (oak/pine chests of drawers, dining chairs, desks, coffee tables, mirrors)
    - Garden & outdoor equipment (patio sets, garden benches, lawnmowers, pressure washers)
    - Fitness & sports gear (exercise bikes, weight benches, dumbbells, bikes/scooters)
    - Working appliances, tools, or branded baby/kids gear (e.g., Stokke, wooden cots)
    
    STRICTLY IGNORE: rubble, soil, hardcore, paving slabs, single mattresses, broken/non-working electronics, free quotes/surveys, and low-value clutter (hangers, cardboard boxes, gravel).

    TASK 2: RESEARCH SOLD MARKET VALUE
    For every promising free item you find, run a Google Search for UK sold prices / completed listings (e.g., eBay UK sold prices, Gumtree secondhand prices) for that exact item or brand/type.
    Only include items that can realistically be cleaned up and resold locally for at least £{MIN_ESTIMATED_PROFIT_GBP}.
    
    Extract the exact source URLs from your Google Search grounding results.
    """

    # Enable Google Search Grounding
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            response_mime_type="application/json",
            response_schema=FlipReport,
            temperature=0.2,  # Low temperature for factual, grounded extraction
        ),
    )

    # Parse structured JSON output
    try:
        report_data = json.loads(response.text)
        report = FlipReport(**report_data)
    except Exception as e:
        print(f"⚠️ Failed to parse response: {e}")
        print(response.text)
        return

    # Also extract raw grounding URLs as a fallback verification check
    grounding_urls = []
    if response.candidates and response.candidates[0].grounding_metadata:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks or []
        grounding_urls = [c.web.uri for c in chunks if c.web and c.web.uri]

    new_finds = 0
    for opp in report.opportunities:
        unique_key = f"{opp.item_title.lower().strip()}_{opp.location.lower().strip()}"
        if unique_key in seen_items:
            continue

        seen_items.add(unique_key)
        new_finds += 1

        print("\n" + "=" * 60)
        print(f"💰 {opp.item_title} — {opp.location} ({opp.source_platform})")
        print(f"📈 Market Value: £{opp.estimated_market_value_gbp} | Quick Flip: £{opp.quick_sale_price_gbp}")
        print(f"📊 Sold Comps: {opp.sold_comps_evidence}")
        print(f"🧽 Prep Tip: {opp.prep_and_upsell_tips}")
        print(f"🔗 URL: {opp.listing_url}")
        print(f"💬 Outreach: {opp.outreach_message}")

        send_telegram_alert(opp)

    save_seen_items(seen_items)
    print(f"\n✅ Scan complete. {new_finds} new profitable opportunities found.")
    if grounding_urls:
        print(f"🔎 Grounded across {len(grounding_urls)} live web sources.")

if __name__ == "__main__":
    run_flip_finder()
