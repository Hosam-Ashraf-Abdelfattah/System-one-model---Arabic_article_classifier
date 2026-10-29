import time  # لقياس زمن الترميز
import numpy as np  # لحفظ الأرقام في مصفوفات مضغوطة وسريعة القراءة
from datasets import load_from_disk  # لقراءة البيانات المقسّمة التي حفظناها
from tokenizer import BPETokenizer  # الـ tokenizer الذي بنيناه

MAX_LEN = 256  # الحد الأقصى للطول الذي قررناه
tok = BPETokenizer("data/bpe.json")  # نحمّل الـ tokenizer
data = load_from_disk("data/sanad_split")  # نقرأ الأجزاء الثلاثة: train وvalidation وtest
for split in ["train", "validation", "test"]:  # نعالج كل جزء على حدة
    articles = data[split]["Article"]  # قائمة نصوص المقالات
    labels = np.array(data[split]["label"], dtype=np.int64)  # الفئات كمصفوفة أرقام صحيحة
    ids = np.full((len(articles), MAX_LEN), tok.pad_id, dtype=np.uint16)  # مصفوفة (عدد المقالات × 256) مملوءة بالحشو، بنوع uint16 لتوفير الذاكرة
    lengths = np.zeros(len(articles), dtype=np.int32)  # أطوال المقالات الفعلية قبل الحشو، لنقيس نسبة الحشو
    t0 = time.time()  # نبدأ ساعة هذا الجزء
    for i, a in enumerate(articles):  # نمر على كل مقال
        enc = tok.encode(a, max_len=MAX_LEN)  # نرمّزه مع CLS وSEP واقتطاع عند 256
        ids[i, : len(enc)] = enc  # نكتبه في أول الصف، والباقي يبقى حشواً
        lengths[i] = len(enc)  # نسجّل طوله الفعلي
        if i % 20000 == 0:  # كل 20 ألف مقال
            print(split, i, round(time.time() - t0), "s")  # نطبع التقدم
    np.save(f"data/{split}_ids.npy", ids)  # نحفظ مصفوفة الأرقام على القرص
    np.save(f"data/{split}_labels.npy", labels)  # ونحفظ الفئات بنفس الترتيب
    pad_frac = 1 - lengths.mean() / MAX_LEN  # نسبة الخانات التي هي حشو من المجموع
    unk = int((ids == tok.unk_id).sum())  # عدد الرموز المجهولة في هذا الجزء
    print(split, ids.shape, "| mean length", round(float(lengths.mean()), 1), "| pad fraction", round(float(pad_frac), 3), "| UNK", unk, "|", round(time.time() - t0), "s")  # ملخص الجزء
