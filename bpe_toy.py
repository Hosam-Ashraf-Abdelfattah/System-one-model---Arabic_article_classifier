from collections import Counter  # أداة العدّ الجاهزة

def get_pair_counts(words):  # دالة تعدّ الأزواج المتجاورة في كل الكلمات، وتستقبل قاموس: كلمة (كقائمة رموز) ← تكرارها
    pairs = Counter()  # جدول العدّ الفارغ: زوج ← عدد تكراره
    for symbols, freq in words.items():  # نمر على كل كلمة مع عدد مرات ظهورها
        for a, b in zip(symbols, symbols[1:]):  # نكوّن أزواجاً متجاورة: الأول مع الثاني، الثاني مع الثالث، وهكذا
            pairs[(a, b)] += freq  # نزيد عدّاد الزوج بتكرار الكلمة كاملاً، فالكلمة الشائعة تُحسب أكثر
    return pairs  # نعيد جدول الأزواج

def merge_pair(pair, words):  # دالة تدمج زوجاً معيناً في كل الكلمات
    new_words = {}  # قاموس الكلمات بعد الدمج
    for symbols, freq in words.items():  # نمر على كل كلمة
        out = []  # القائمة الجديدة للكلمة بعد الدمج
        i = 0  # مؤشر الموضع الحالي داخل الكلمة
        while i < len(symbols):  # نمشي على رموز الكلمة واحداً واحداً
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:  # إن وجدنا الزوج المطلوب دمجه في هذا الموضع
                out.append(symbols[i] + symbols[i + 1])  # نضع الرمزين ملتصقين كرمز واحد جديد
                i += 2  # نقفز فوق الرمزين اللذين دمجناهما
            else:  # وإلا فهذا الموضع ليس بداية الزوج
                out.append(symbols[i])  # نبقي الرمز كما هو
                i += 1  # ننتقل للرمز التالي
        new_words[tuple(out)] = new_words.get(tuple(out), 0) + freq  # نخزّن الكلمة الجديدة بنفس تكرارها
    return new_words  # نعيد القاموس بعد الدمج

corpus = {"كتاب": 6, "الكتاب": 4, "كتب": 5, "مكتبة": 3}  # مجموعة كلمات صغيرة مع تكرار كل منها
words = {tuple(w[:-1]) + (w[-1] + "</w>",): f for w, f in corpus.items()}  # نحوّل كل كلمة إلى حروف مفصولة، ونلصق علامة النهاية بآخر حرف
print(words)  # نطبع البداية لنرى التمثيل الأولي
for step in range(8):  # ننفذ 8 دورات دمج
    pairs = get_pair_counts(words)  # نعدّ الأزواج في الوضع الحالي
    best = max(pairs, key=pairs.get)  # نختار الزوج الأعلى تكراراً
    print(step + 1, best, pairs[best])  # نطبع رقم الدورة والزوج الفائز وعدده
    words = merge_pair(best, words)  # ندمج هذا الزوج في كل الكلمات
    print("   ", list(words.keys()))  # نطبع الكلمات بعد الدمج لنرى كيف تتجمع الحروف
