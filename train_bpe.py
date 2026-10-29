import json  # لقراءة جدول تكرار الكلمات الذي حفظناه سابقاً
import time  # لقياس زمن التدريب
from bpe_trainer import BPETrainer  # نستورد الـ class التي كتبناها للتو

with open("data/word_freq.json", encoding="utf-8") as f:  # نفتح ملف تكرار الكلمات بترميز يدعم العربية
    word_freq = json.load(f)  # نقرأه كقاموس: كلمة ← عدد مرات ظهورها
start = time.time()  # نسجّل لحظة البدء
trainer = BPETrainer(vocab_size=16000, num_special=5)  # ننشئ المدرّب بحجم مفردات 16000 ونحجز 5 أماكن للرموز الخاصة
trainer.train(word_freq)  # نبدأ التدريب، وسيطبع تقدمه كل ألف دمج
print("time:", round(time.time() - start), "seconds")  # نطبع الزمن الكلي بالثواني
trainer.save("data/bpe.json")  # نحفظ الرموز الابتدائية وقائمة الدمج في ملف
print("first 20 merges:", trainer.merges[:20])  # أول 20 دمجاً، وهي الأكثر شيوعاً في العربية
print("last 10 merges:", trainer.merges[-10:])  # آخر 10 دمجات، وهي الأندر
print("some alphabet symbols:", trainer.alphabet[:40])  # عيّنة من الرموز الابتدائية لنراها
