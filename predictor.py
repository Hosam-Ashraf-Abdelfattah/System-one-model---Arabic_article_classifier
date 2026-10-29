import json  # لقراءة درجة الحرارة المحفوظة
import numpy as np  # للحسابات العددية
import torch  # مكتبة PyTorch
from tokenizer import BPETokenizer  # الـ tokenizer الذي بنيناه
from model import build_model  # دالة بناء النموذج
from metrics import softmax  # دالة softmax بـ numpy
import config as C  # الإعدادات

device = "cuda"  # نستعمل كرت الشاشة
tok = BPETokenizer("data/bpe.json")  # نحمّل الـ tokenizer
model = build_model(tok.vocab_size, tok.pad_id).to(device)  # نبني النموذج بنفس الإعدادات
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device)["model"])  # نحمّل أوزان أفضل نقطة استعادة
model.eval()  # نوقف الـ dropout
T = json.load(open("checkpoints/temperature.json"))["T"]  # درجة الحرارة التي تعلمناها في المعايرة

@torch.no_grad()  # لا نحتاج اشتقاقات في التنبؤ
def predict_probs(texts, batch_size=64):  # تستقبل قائمة نصوص وتعيد احتمالات معايَرة (عدد النصوص × 7)
    out = []  # قائمة نتائج الدفعات
    for s in range(0, len(texts), batch_size):  # نمر على النصوص دفعة دفعة
        chunk = texts[s: s + batch_size]  # نصوص هذه الدفعة
        ids_list = [tok.encode(t, max_len=C.MAX_LEN) for t in chunk]  # نرمّزها مع الاقتطاع
        longest = max(len(x) for x in ids_list)  # أطولها في الدفعة
        batch = torch.full((len(chunk), longest), tok.pad_id, dtype=torch.long)  # دفعة مملوءة بالحشو
        for i, ids in enumerate(ids_list):  # نملأ كل صف
            batch[i, : len(ids)] = torch.tensor(ids)  # والباقي يبقى حشواً
        logits = model(batch.to(device)).cpu().numpy()  # الـ logits كمصفوفة numpy
        out.append(softmax(logits / T))  # نقسم على T ونحوّلها إلى احتمالات معايَرة
    return np.concatenate(out)  # نجمع كل الدفعات في مصفوفة واحدة
