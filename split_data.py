import numpy as np  # مكتبة الحساب العددي، سنستخدمها لحساب متوسط أطوال المقالات
from collections import Counter  # أداة جاهزة لعدّ تكرار كل قيمة في قائمة
from datasets import load_from_disk, DatasetDict  # نستورد دالة قراءة ما حفظناه سابقاً، وحاوية لعدة أجزاء
full = load_from_disk("data/sanad_raw")["train"]  # نقرأ البيانات المحفوظة ونأخذ جزء train الوحيد الموجود
names = full.features["label"].names  # نقرأ أسماء الفئات السبع لنطبعها بدل الأرقام
tmp = full.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")  # نفصل 20% جانباً، مع ثبات نسب الفئات ورقم seed ثابت
holdout = tmp["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")  # نقسم الـ20% المفصولة نصفين: validation وtest
data = DatasetDict({"train": tmp["train"], "validation": holdout["train"], "test": holdout["test"]})  # نجمع الأجزاء الثلاثة في كائن واحد بأسماء واضحة
for split in data:  # نمر على كل جزء من الأجزاء الثلاثة
    counts = Counter(data[split]["label"])  # نعدّ كم مقالاً في كل فئة داخل هذا الجزء
    print(split, len(data[split]))  # نطبع اسم الجزء وعدد مقالاته
    for i, n in sorted(counts.items()):  # نمر على الفئات مرتبة برقمها
        print("   ", names[i], n)  # نطبع اسم الفئة وعدد مقالاتها، فنتأكد أن التوزيع متشابه بين الأجزاء
sample = data["train"].shuffle(seed=0).select(range(5000))  # نأخذ عينة عشوائية من 5000 مقال حتى لا نحسب على الكل
lengths = np.array([len(a.split()) for a in sample["Article"]])  # نحسب عدد الكلمات في كل مقال بتقسيمه عند المسافات
print("mean words:", lengths.mean())  # متوسط طول المقال بالكلمات
print("percentiles 50/90/99:", np.percentile(lengths, [50, 90, 99]))  # الوسيط، وطول يتجاوزه 10% من المقالات، وطول يتجاوزه 1% فقط
data.save_to_disk("data/sanad_split")  # نحفظ التقسيم النهائي على القرص لنستخدم نفس التقسيم في كل الخطوات القادمة
