# ⚡ Webull US Stock Pro Trader & Anti-Stop Hunt Analyzer

เว็บแอปพลิเคชันสำหรับวิเคราะห์หุ้นสหรัฐฯ พร้อมระบบคำนวณแผนเทรดป้องกัน **Stop Hunt (Liquidity Sweep)** และฟังก์ชัน **AI Stock Analyst** ด้วย Google Gemini พัฒนาด้วย **Streamlit + Plotly + yfinance**

---

## 🎯 แอปนี้คืออะไร? (Overview)

แอปพลิเคชันเครื่องมือช่วยเทรดหุ้นสหรัฐฯ (NYSE/NASDAQ) สำหรับนักเทรด โดยเน้นฟังก์ชันสำคัญ:
1. **Interactive Technical Charts:** ดูกราฟแท่งเทียนแบบเรียลไทม์ พร้อมเส้น EMA (20, 50, 200), RSI, MACD และ Volume Analysis
2. **Anti-Stop Hunt Strategy & Webull Position Sizing:** วางแผนจุดเข้า, จุด Safe Stop Loss (คำนวณผ่าน 1.3x ATR เพื่อหลีกเลี่ยงการโดนกวาดสภาพคล่อง), Take Profit และคำนวณขนาดไม้ตามความเสี่ยงพอร์ต
3. **Daily Stock Screener:** สแกนหุ้นชั้นนำ 30 ตัว กรองตามเงื่อนไขโมเมนตัม (EMA20, RSI, Volume Spikes, MACD Bullish)
4. **🤖 AI Sentiment & News Analyst:** วิเคราะห์ข่าวล่าสุดและความเชื่อมั่นของตลาดด้วย Google Gemini API (คะแนน Sentiment, Key Catalysts และข้อควรระวัง)
5. **My Watchlist:** บันทึกและติดตามรายชื่อหุ้นโปรดส่วนตัวแบบอัตโนมัติ

---

## 🚀 วิธีติดตั้งและรันในเครื่อง (Local Setup)

### 1. โคลนโปรเจกต์และเข้าสู่โฟลเดอร์
```bash
git clone <URL_ของ_REPOSITORY_คุณ>
cd 01_Trader
```

### 2. สร้างและเปิดใช้งาน Virtual Environment (แนะนำ)
- **Windows (PowerShell/CMD):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install -r requirements.txt
```

### 4. (ทางเลือก) ตั้งค่า Gemini API Key
สร้างไฟล์ `.streamlit/secrets.toml` (จะไม่ถูกอัปโหลดขึ้น GitHub ตาม `.gitignore`):
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
*(หากไม่ใส่ สามารถกรอก API Key ผ่านแถบ Sidebar บนหน้าเว็บได้เช่นกัน)*

### 5. สั่งรันแอปพลิเคชัน
```bash
streamlit run app.py
```
> สำหรับ Windows สามารถดับเบิลคลิกไฟล์ `run.bat` เพื่อเปิดแอปได้ทันที

---

## ☁️ วิธี Deploy ขึ้น Streamlit Community Cloud

1. นำโค้ดขึ้น GitHub Repository ของคุณ (ดูวิธี Git ในหัวข้อถัดไป)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io/) แล้วคลิก **"New app"**
3. เลือก Repository, Branch และระบุไฟล์หลัก:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. คลิก **"Advanced settings"** -> ที่หัวข้อ **Secrets** ใส่:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   ```
5. คลิก **"Deploy!"** ระบบจะติดตั้ง Dependencies ตาม `requirements.txt` และเปิดใช้งานทันที

---

## 📦 การนำโค้ดขึ้น GitHub ครั้งแรก

```bash
git init
git add .
git commit -m "Initial commit: Webull US Stock Pro Trader"
git branch -M main
git remote add origin <URL_ของ_REPOSITORY_บน_GITHUB>
git push -u origin main
```

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
├── .streamlit/
│   ├── config.toml       # การตั้งค่า UI ธีม Dark Mode
│   └── secrets.toml      # (Local เท่านั้น) API Key ไม่ขึ้น Git
├── .gitignore            # กำหนดไฟล์ที่ไม่ให้อัปโหลดขึ้น Git
├── requirements.txt      # รายการ Dependencies สำหรับรันแอป
├── app.py                # ไฟล์หลักของ Streamlit App (Main entry point)
├── ai_analyst.py         # โมดูลวิเคราะห์ข่าวและ Sentiment ด้วย Google Gemini
├── backtester.py         # โมดูลทดสอบย้อนหลังกลยุทธ์ Anti-Stop Hunt
├── charts.py             # โมดูลแสดงผลกราฟ Plotly Interactive
├── data_loader.py        # โมดูลดึงข้อมูลราคาและตัวชี้วัดจาก yfinance
├── indicators.py         # สูตรคำนวณทางเทคนิคและ Anti-Stop Hunt
├── watchlist_manager.py  # โมดูลจัดการรายการ Watchlist ส่วนตัว
├── run.bat               # สคริปต์รันบน Windows แบบ 1-Click
└── README.md             # เอกสารแนะนำและคู่มือการใช้งาน
```
