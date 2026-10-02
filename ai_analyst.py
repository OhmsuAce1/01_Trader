import json
import os
import datetime
from typing import Dict, Any, List, Optional
import yfinance as yf
import plotly.graph_objects as go
from google import genai
from google.genai import types

def fetch_stock_news(ticker: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Fetches latest news stories for a ticker using yfinance.
    Tries yf.Search first (standard in yfinance >= 0.2.40), then falls back to Ticker.news.
    """
    cleaned_news = []
    try:
        ticker = ticker.upper().strip()
        search_res = yf.Search(ticker, news_count=limit)
        raw_news = getattr(search_res, "news", []) or []
        
        if not raw_news:
            t = yf.Ticker(ticker)
            raw_news = getattr(t, "news", []) or []

        for item in raw_news[:limit]:
            title = item.get("title", "")
            publisher = item.get("publisher", "Unknown")
            link = item.get("link", "")
            pub_time = item.get("providerPublishTime")
            
            pub_str = ""
            if pub_time:
                try:
                    pub_str = datetime.datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pub_str = ""

            if title:
                cleaned_news.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "publish_time": pub_str,
                })
    except Exception:
        pass

    return cleaned_news

def analyze_stock_with_gemini(
    ticker: str,
    api_key: str,
    model_name: str = "gemini-flash-lite-latest",
    company_info: Optional[Dict[str, Any]] = None,
    news_list: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Calls Google Gemini API via the modern google-genai SDK to analyze
    News Sentiment, Key Catalysts & Drivers, and Risk/Reward parameters.
    """
    if not api_key or not api_key.strip():
        return {
            "error": "กรุณาระบุ GEMINI_API_KEY ในแถบด้านข้าง (Sidebar) ก่อนเริ่มการวิเคราะห์"
        }

    ticker = ticker.upper().strip()
    company_info = company_info or {}
    news_list = news_list or []

    # Format news into a readable text block
    news_text = ""
    if news_list:
        for i, n in enumerate(news_list, 1):
            time_info = f" ({n['publish_time']})" if n.get('publish_time') else ""
            news_text += f"{i}. [{n.get('publisher', 'News')}] {n.get('title', '')}{time_info}\n"
    else:
        news_text = "ไม่มีพาดหัวข่าวล่าสุดเฉพาะเจาะจง ให้วิเคราะห์จากบริบทตลาดและปัจจัยพื้นฐานทั่วไปของหุ้นตัวนี้"

    # Context about current price & company
    company_context = f"""
สัญลักษณ์หุ้น: {ticker}
ชื่อบริษัท: {company_info.get('name', ticker)}
กลุ่มธุรกิจ (Sector): {company_info.get('sector', 'N/A')}
ราคาปัจจุบัน: ${company_info.get('current_price', 'N/A')} (เปลี่ยนแปลงวันนี้: {company_info.get('change_pct', 0):+.2f}%)
กรอบราคาวันนี้: High ${company_info.get('day_high', 'N/A')} / Low ${company_info.get('day_low', 'N/A')}
กรอบ 52 สัปดาห์: ${company_info.get('fifty_two_week_low', 'N/A')} - ${company_info.get('fifty_two_week_high', 'N/A')}
Market Cap: ${(company_info.get('market_cap', 0) / 1e9):,.2f}B
"""

    prompt = f"""คุณคือนักวิเคราะห์หุ้นสหรัฐฯ มืออาชีพ (Senior Equity Analyst) เชี่ยวชาญการวิเคราะห์ Sentiment ข่าวสาร และพฤติกรรมราคาหุ้นบน Webull
โปรดวิเคราะห์ข้อมูลหุ้น {ticker} จากปัจจัยพื้นฐานและข่าวสารล่าสุดต่อไปนี้:

=== ข้อมูลบริษัทและราคาล่าสุด ===
{company_context}

=== พาดหัวข่าวล่าสุด ===
{news_text}

=== ภารกิจของคุณ ===
โปรดวิเคราะห์ข้อมูลและส่งออกผลลัพธ์เป็น JSON ตามโครงสร้างด้านล่างอย่างเคร่งครัด (ภาษาไทย กระชับ คมชัด ตรงประเด็น):
1. **News Sentiment Score**: ให้คะแนนความเชื่อมั่นรวมจากข่าวและบรรยากาศการลงทุน ตั้งแต่ -100 (Bearish รุนแรงมาก) ถึง +100 (Bullish รุนแรงมาก) โดย 0 คือเป็นกลาง (Neutral)
2. **Sentiment Label**: ระบุระดับ เช่น "Extreme Bullish", "Bullish", "Neutral", "Bearish", "Extreme Bearish"
3. **Sentiment Reasoning**: อธิบายเหตุผลที่ให้คะแนนนี้ 2-3 ประโยค
4. **Key Catalysts & Drivers**: สรุปสั้นๆ 3-5 ข้อว่าปัจจัยขับเคลื่อนราคาหลักในปัจจุบันคืออะไร (เช่น ผลประกอบการ, ราคากระแส Bitcoin/คริปโต, อัตราดอกเบี้ย Fed, ความต้องการชิป AI, คำสั่งซื้อ, พันธมิตรธุรกิจ ฯลฯ)
5. **Risk / Reward Summary**: วิเคราะห์ความเสี่ยงสำคัญ 2-3 ข้อ และจุดที่นักเทรดบน Webull ควรระมัดระวัง (เช่น ความผันผวนสูง, วันประกาศงบ, การ Dilution, การเทขายของบอท)
6. **Actionable Takeaway**: สรุปคำแนะนำเชิงกลยุทธ์สำหรับนักเทรด 1-2 ประโยค (เช่น ควรเน้นเข้าจังหวะย่อตัว Pullback, วาง Stop เผื่อความผันผวน, หรือรอความชัดเจน)

รูปแบบ JSON ที่ต้องส่งกลับ (ห้ามมีข้อความอื่นนอกเหนือจาก JSON):
{{
  "sentiment_score": 65,
  "sentiment_label": "Bullish",
  "sentiment_reasoning": "คำอธิบายเหตุผล...",
  "key_catalysts": [
    "ปัจจัยที่ 1: ...",
    "ปัจจัยที่ 2: ...",
    "ปัจจัยที่ 3: ..."
  ],
  "risks": [
    "ความเสี่ยงที่ 1: ...",
    "ความเสี่ยงที่ 2: ..."
  ],
  "actionable_takeaway": "คำแนะนำเชิงกลยุทธ์..."
}}
"""

    # Prioritize active models with highest available free-tier quota (flash-lite has separate quota from flash)
    valid_active_models = [
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-flash-latest",
        "gemini-3.8-flash"
    ]
    if model_name not in valid_active_models:
        model_name = "gemini-flash-lite-latest"

    models_to_try = [model_name] + [m for m in valid_active_models if m != model_name]

    client = genai.Client(api_key=api_key.strip())
    last_error = None

    for m in models_to_try:
        try:
            config = types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config=config
            )
            raw_text = getattr(response, "text", "") or ""
            if not raw_text:
                continue

            # Strip markdown codeblocks if returned
            text_to_parse = raw_text.strip()
            if text_to_parse.startswith("```json"):
                text_to_parse = text_to_parse[7:]
            elif text_to_parse.startswith("```"):
                text_to_parse = text_to_parse[3:]
            if text_to_parse.endswith("```"):
                text_to_parse = text_to_parse[:-3]

            result_data = json.loads(text_to_parse.strip())
            result_data["model_used"] = m
            result_data["news_analyzed_count"] = len(news_list)
            return result_data

        except Exception as e:
            last_error = e
            continue

    return {
        "error": f"เกิดข้อผิดพลาดในการเรียกใช้ Gemini API: {str(last_error)}"
    }

def create_sentiment_gauge(score: int, label: str) -> go.Figure:
    """
    Renders a Webull-style speed-gauge chart for News Sentiment (-100 to +100).
    """
    # Color logic
    if score >= 35:
        bar_color = "#10B981" # Green
    elif score <= -35:
        bar_color = "#EF4444" # Red
    else:
        bar_color = "#F59E0B" # Yellow/Amber

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment: {label}", 'font': {'size': 18, 'color': '#E2E8F0', 'family': 'sans-serif'}},
        number={'suffix': " pts", 'font': {'size': 32, 'color': bar_color, 'family': 'sans-serif'}},
        gauge={
            'axis': {
                'range': [-100, 100],
                'tickwidth': 1,
                'tickcolor': "#64748B",
                'tickmode': "array",
                'tickvals': [-100, -50, 0, 50, 100],
                'ticktext': ["-100 (Bear)", "-50", "0 (Neutral)", "+50", "+100 (Bull)"]
            },
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "#151B26",
            'borderwidth': 1,
            'bordercolor': "#243044",
            'steps': [
                {'range': [-100, -35], 'color': "rgba(239, 68, 68, 0.20)"},
                {'range': [-35, 35], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [35, 100], 'color': "rgba(16, 185, 129, 0.20)"}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0B0E14",
        font={'color': "#E2E8F0"},
        height=260,
        margin=dict(l=25, r=25, t=40, b=20)
    )

    return fig
