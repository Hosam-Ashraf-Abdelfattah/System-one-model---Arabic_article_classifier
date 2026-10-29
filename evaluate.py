import numpy as np  # لحفظ الـ logits
import torch  # مكتبة PyTorch
from tokenizer import BPETokenizer  # الـ tokenizer
from dataset import make_loader  # الـ DataLoader
from model import build_model  # بناء النموذج
from metrics import report  # دالة التقرير التي كتبناها للتو
import config as C  # الإعدادات

SPLIT = "validation"  # نقيّم على validation، ولا نلمس test إلا في النهاية
CKPT = "checkpoints/best.pt"  # نقطة الاستعادة التي نقيّمها
device = "cuda"  # كرت الشاشة
tok = BPETokenizer("data/bpe.json")  # نحمّل الـ tokenizer
model = build_model(tok.vocab_size, tok.pad_id).to(device)  # نبني النموذج بنفس الإعدادات
ckpt = torch.load(CKPT, map_location=device)  # نقرأ نقطة الاستعادة
model.load_state_dict(ckpt["model"])  # نحمّل الأوزان المحفوظة
model.eval()  # نوقف الـ dropout
print("loaded", CKPT, "| epoch", ckpt["epoch"], "| saved val loss", round(ckpt["val_loss"], 4), "| saved val acc", round(ckpt["val_acc"] * 100, 2))  # نتأكد أننا حمّلنا النموذج المقصود
loader = make_loader(SPLIT, 64, False)  # بيانات الجزء بدون خلط
all_logits, all_labels = [], []  # قوائم لجمع النتائج
with torch.no_grad():  # لا نحتاج اشتقاقات
    for ids, labels in loader:  # لكل دفعة
        all_logits.append(model(ids.to(device)).cpu())  # نحسب الـ logits وننقلها للمعالج
        all_labels.append(labels)  # ونحفظ الفئات الحقيقية
logits = torch.cat(all_logits).numpy()  # نجمعها في مصفوفة numpy واحدة
labels = torch.cat(all_labels).numpy()  # وكذلك الفئات
np.save(f"data/{SPLIT}_logits.npy", logits)  # نحفظ الـ logits لنعاير عليها لاحقاً دون إعادة تشغيل النموذج
np.save(f"data/{SPLIT}_true_labels.npy", labels)  # ونحفظ الفئات
report(logits, labels, C.CLASS_NAMES)  # نطبع التقرير الكامل
