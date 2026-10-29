import re  # مكتبة التعبيرات النمطية للبحث عن الأنماط في النص
import json  # لحفظ جدول تكرار الكلمات في ملف نصي منظم
from collections import Counter  # قاموس متخصص في العدّ، يزيد عدّاد الكلمة كلما رأيناها

class PreTokenizer:  # class مسؤولة عن شيء واحد: تقطيع النص إلى كلمات وعلامات ترقيم
    def __init__(self):  # دالة التجهيز، تعمل مرة واحدة عند إنشاء الكائن
        self.pattern = re.compile(r"\w+|[^\w\s]")  # نبني النمط مرة واحدة: كلمة كاملة، أو علامة ترقيم منفردة

    def __call__(self, text):  # تُنفَّذ عند كتابة pretok(text)، وتستقبل نصاً وتعيد قائمة قطع
        return self.pattern.findall(text)  # تعيد كل ما يطابق النمط بالترتيب، فتخرج الكلمات وعلامات الترقيم في قائمة

if __name__ == "__main__":  # هذا الجزء يعمل فقط عند تشغيل الملف مباشرة
    from datasets import load_from_disk  # لقراءة البيانات المحفوظة
    from normalizer import ArabicNormalizer  # نستورد الـ class التي كتبناها في الخطوة السابقة
    norm = ArabicNormalizer()  # ننشئ كائن التطبيع
    pretok = PreTokenizer()  # وننشئ كائن التقطيع
    print(pretok("ذكر المغرب، اليوم 2026!"))  # اختبار صغير، والمتوقع: ['ذكر', 'المغرب', '،', 'اليوم', '2026', '!']
    train = load_from_disk("data/sanad_split")["train"]  # نقرأ جزء التدريب فقط، ولا نلمس validation أو test لأنهما للامتحان
    freq = Counter()  # جدول العدّ الفارغ: كلمة ← عدد مرات ظهورها
    for i, article in enumerate(train["Article"]):  # نمر على كل مقال في التدريب، ومعه رقمه
        freq.update(pretok(norm(article)))  # نطبّع المقال ثم نقطّعه ثم نضيف كل كلمة إلى العدّاد
        if i % 20000 == 0:  # كل 20 ألف مقال
            print("processed", i)  # نطبع التقدم لنعرف أن البرنامج يعمل
    total = sum(freq.values())  # مجموع كل الكلمات في التدريب بتكرارها
    print("total words:", total)  # نطبعه، فنعرف حجم النص بالكلمات
    print("unique words:", len(freq))  # عدد الكلمات المختلفة، وهو حجم المفردات لو اعتمدنا طريقة كلمة بكلمة
    print("top 15:", freq.most_common(15))  # أكثر 15 قطعة تكراراً مع أعدادها
    once = sum(1 for c in freq.values() if c == 1)  # نعدّ الكلمات التي ظهرت مرة واحدة فقط في كل البيانات
    print("words seen once:", once)  # نطبعها، فنرى كم من المفردات نادر جداً
    running = 0  # عدّاد تراكمي لنسبة التغطية
    for rank, (w, c) in enumerate(freq.most_common(), start=1):  # نمر على الكلمات من الأكثر تكراراً، ومعها ترتيبها
        running += c  # نضيف تكرار هذه الكلمة إلى المجموع التراكمي
        if rank in (1000, 5000, 20000, 50000):  # عند هذه الأرقام تحديداً
            print("top", rank, "words cover", round(100 * running / total, 1), "% of text")  # نطبع نسبة النص التي تغطيها هذه الكلمات
    with open("data/word_freq.json", "w", encoding="utf-8") as f:  # نفتح ملفاً للكتابة بترميز يدعم العربية
        json.dump(freq, f, ensure_ascii=False)  # نحفظ الجدول كما هو، دون تحويل العربية إلى رموز escape
