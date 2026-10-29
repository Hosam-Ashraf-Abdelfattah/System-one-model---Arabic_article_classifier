import json  # لحفظ نتائج التدريب في ملف منظم
import heapq  # أداة الكومة (heap) للحصول على الأعلى تكراراً بسرعة دون مسح كل الأزواج
from collections import Counter, defaultdict  # Counter للعدّ، وdefaultdict لقاموس تُنشأ قيمته الفارغة تلقائياً

class BPETrainer:  # class مسؤولة عن شيء واحد: تعلّم قائمة الدمج من جدول تكرار الكلمات
    def __init__(self, vocab_size, num_special=5):  # دالة التجهيز
        self.vocab_size = vocab_size  # حجم المفردات النهائي المطلوب
        self.num_special = num_special  # عدد الرموز الخاصة التي سنحجز لها مكاناً
        self.merges = []  # قائمة الدمج المتعلَّمة بالترتيب، وهي "عقل" الـ tokenizer
        self.alphabet = []  # الرموز الابتدائية قبل أي دمج

    @staticmethod  # دالة لا تحتاج self لأنها لا تستعمل بيانات الكائن
    def merge_word(symbols, pair):  # تدمج زوجاً معيناً داخل كلمة واحدة
        a, b = pair  # نفكّ الزوج إلى رمزيه
        out = []  # القائمة الجديدة
        i = 0  # المؤشر الحالي
        while i < len(symbols):  # نمشي على الرموز
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:  # إن بدأ الزوج هنا
                out.append(a + b)  # نضع الرمز الجديد الملتصق
                i += 2  # نتجاوز الرمزين
            else:  # وإلا
                out.append(symbols[i])  # نبقي الرمز
                i += 1  # ننتقل للتالي
        return out  # نعيد الكلمة بعد الدمج

    def train(self, word_freq, log_every=1000):  # التدريب نفسه، يستقبل قاموس كلمة ← تكرارها
        items = list(word_freq.items())  # نحوّل القاموس إلى قائمة أزواج لنتعامل مع كل كلمة برقمها
        words = [list(w[:-1]) + [w[-1] + "</w>"] for w, _ in items]  # كل كلمة إلى قائمة حروف، وآخر حرف يلتصق به علامة النهاية
        freqs = [f for _, f in items]  # قائمة التكرارات بنفس ترتيب الكلمات
        self.alphabet = sorted({s for sym in words for s in sym})  # مجموعة كل الرموز الابتدائية المختلفة، مرتبة
        num_merges = self.vocab_size - self.num_special - len(self.alphabet)  # عدد الدمجات المسموحة حتى نبلغ الحجم المطلوب
        print("initial symbols:", len(self.alphabet), "| merges to learn:", num_merges)  # نطبعهما لنعرف الميزانية
        if num_merges <= 0:  # إن كانت الرموز الابتدائية وحدها تملأ الميزانية
            raise ValueError("vocab_size too small for the alphabet")  # نوقف البرنامج برسالة واضحة
        pair_counts = Counter()  # جدول: زوج ← مجموع تكراره في كل النص
        where = defaultdict(set)  # فهرس: زوج ← مجموعة أرقام الكلمات التي يظهر فيها
        for idx, sym in enumerate(words):  # نبني الجدولين مرة واحدة في البداية
            for pair in zip(sym, sym[1:]):  # كل زوج متجاور في الكلمة
                pair_counts[pair] += freqs[idx]  # نزيد عدّاد الزوج بتكرار الكلمة
                where[pair].add(idx)  # ونسجّل أن هذه الكلمة تحتوي الزوج
        heap = [(-c, p) for p, c in pair_counts.items()]  # كومة عناصرها (سالب العدد، الزوج)، والسالب لأن الكومة في بايثون تُخرج الأصغر أولاً
        heapq.heapify(heap)  # نرتّب القائمة لتصبح كومة صالحة
        for step in range(num_merges):  # حلقة التدريب الرئيسية
            best = None  # الزوج الفائز، لا نعرفه بعد
            while heap:  # نسحب من الكومة حتى نجد عنصراً صالحاً
                neg, pair = heapq.heappop(heap)  # نسحب أعلى عنصر في الكومة
                if pair_counts.get(pair, 0) == -neg:  # هل ما زال عدّه الحالي مطابقاً لما سُجّل؟ إن نعم فهو صالح
                    best = pair  # وجدنا الفائز
                    break  # نخرج من حلقة السحب
            if best is None:  # لو فرغت الكومة ولم نجد شيئاً
                break  # نتوقف مبكراً
            best_count = pair_counts[best]  # نحفظ عدّه قبل أن يتغير
            self.merges.append(best)  # نسجّل الدمج في القائمة
            changed = set()  # مجموعة الأزواج التي تغيّر عدّها في هذه الدورة
            for idx in list(where[best]):  # فقط الكلمات التي تحتوي الزوج، وهذا سر السرعة
                sym = words[idx]  # رموز الكلمة قبل الدمج
                f = freqs[idx]  # تكرارها
                for pair in zip(sym, sym[1:]):  # نسحب مساهمة أزواج هذه الكلمة القديمة من الجدول
                    pair_counts[pair] -= f  # ننقص العدّاد
                    changed.add(pair)  # نسجّل أن هذا الزوج تغيّر
                    if pair_counts[pair] == 0:  # لو صفر
                        del pair_counts[pair]  # نحذفه من الجدول
                new_sym = self.merge_word(sym, best)  # ندمج الزوج في هذه الكلمة
                words[idx] = new_sym  # نخزّن الشكل الجديد
                for pair in zip(new_sym, new_sym[1:]):  # نضيف مساهمة أزواج الشكل الجديد
                    pair_counts[pair] += f  # نزيد العدّاد
                    changed.add(pair)  # نسجّل التغيير
                    where[pair].add(idx)  # ونحدّث الفهرس
            where.pop(best, None)  # الزوج المدموج انتهى، نحذفه من الفهرس
            pair_counts.pop(best, None)  # ونتأكد أنه خرج من جدول العدّ
            for pair in changed:  # لكل زوج تغيّر عدّه
                c = pair_counts.get(pair, 0)  # عدّه الجديد
                if c > 0:  # إن بقي موجوداً
                    heapq.heappush(heap, (-c, pair))  # ندفع له سجلاً جديداً في الكومة، والسجلات القديمة تسقط تلقائياً عند فحصها
            if step % log_every == 0 or step == num_merges - 1:  # كل ألف دمج، وعند الدمج الأخير
                print("merge", step + 1, best, best_count)  # نطبع التقدم
        return self.merges  # نعيد قائمة الدمج

    def save(self, path):  # حفظ النتيجة في ملف
        with open(path, "w", encoding="utf-8") as f:  # نفتح الملف بترميز يدعم العربية
            json.dump({"alphabet": self.alphabet, "merges": [list(m) for m in self.merges]}, f, ensure_ascii=False)  # نحفظ الرموز الابتدائية وقائمة الدمج
