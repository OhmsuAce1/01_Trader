from google import genai
import time

api_key = "AQ.Ab8RN6LcGzlDOfvldOuDt4L4nrUexigfEhP-4733EGTXGxnDRw"
client = genai.Client(api_key=api_key)

prompt = """วิเคราะห์หุ้น RIOT สั้นๆ เป็น JSON:
{"sentiment_score": 50, "sentiment_label": "Neutral", "sentiment_reasoning": "...", "key_catalysts": ["ข้อ 1"], "risks": ["ข้อ 1"], "actionable_takeaway": "..."}
"""

t0 = time.time()
print("Calling interactions.create...", flush=True)
resp = client.interactions.create(
    model="gemini-flash-latest",
    input=prompt,
    store=False
)
print("Finished in", time.time() - t0, "seconds", flush=True)
print("Output text:", resp.output_text, flush=True)
