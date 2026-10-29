import json  # لقراءة ملف الدمج الذي حفظه المدرّب
from normalizer import ArabicNormalizer  # أداة توحيد النص من المرحلة 1
from pretokenizer import PreTokenizer  # أداة التقطيع إلى كلمات من المرحلة 2
from bpe_trainer import BPETrainer  # نستعير منها دالة دمج زوج داخل كلمة

class BPETokenizer:  # class واحدة تحوّل النص إلى أرقام وتعيده، وتحمل كل المراحل السابقة
    SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # الرموز الخاصة، وترتيبها يحدد أرقامها: PAD=0 وUNK=1 وCLS=2 وSEP=3 وMASK=4

    def __init__(self, path):  # دالة التجهيز، تستقبل مسار ملف bpe.json
        with open(path, encoding="utf-8") as f:  # نفتح الملف بترميز يدعم العربية
            data = json.load(f)  # نقرأ الرموز الابتدائية وقائمة الدمج
        self.norm = ArabicNormalizer()  # كائن التطبيع
        self.pretok = PreTokenizer()  # كائن التقطيع
        self.merges = [tuple(m) for m in data["merges"]]  # نحوّل كل دمج من قائمة إلى tuple ليصلح مفتاحاً في قاموس
        self.rank = {pair: i for i, pair in enumerate(self.merges)}  # رتبة كل دمج: الأصغر هو الأقدم تعلّماً والأولى بالتطبيق
        candidates = self.SPECIALS + data["alphabet"] + [a + b for a, b in self.merges]  # كل الرموز بالترتيب: خاصة ثم ابتدائية ثم ناتجة عن دمج
        self.id_to_token = []  # قائمة الرموز، وموضع الرمز فيها هو رقمه
        seen = set()  # مجموعة لاكتشاف التكرار
        for t in candidates:  # نمر على كل الرموز
            if t not in seen:  # قد ينتج نفس النص من مسارين دمج مختلفين، فنحتفظ بأول ظهور فقط
                seen.add(t)  # نسجّله كمرئي
                self.id_to_token.append(t)  # ونضيفه للقائمة
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}  # القاموس المعاكس: رمز ← رقم
        self.cache = {}  # ذاكرة مؤقتة: كلمة ← رموزها، حتى لا نعيد الحساب
        self.pad_id = self.token_to_id["[PAD]"]  # رقم رمز الحشو
        self.unk_id = self.token_to_id["[UNK]"]  # رقم رمز المجهول
        self.cls_id = self.token_to_id["[CLS]"]  # رقم رمز بداية النص
        self.sep_id = self.token_to_id["[SEP]"]  # رقم رمز نهاية النص
        self.mask_id = self.token_to_id["[MASK]"]  # رقم رمز الإخفاء

    @property  # يجعل الدالة تُستدعى كأنها خاصية: tokenizer.vocab_size بدون أقواس
    def vocab_size(self):  # حجم المفردات الفعلي
        return len(self.id_to_token)  # عدد الرموز في القائمة

    def _bpe(self, word):  # تحوّل كلمة واحدة إلى قائمة رموز بتطبيق الدمجات
        if word in self.cache:  # إن سبق أن عالجنا هذه الكلمة
            return self.cache[word]  # نعيد النتيجة المخزّنة
        symbols = list(word[:-1]) + [word[-1] + "</w>"]  # حروف الكلمة، وآخر حرف تلتصق به علامة النهاية
        while len(symbols) > 1:  # نكرر ما دام فيها أكثر من رمز
            pairs = set(zip(symbols, symbols[1:]))  # الأزواج المتجاورة الموجودة الآن
            best = min(pairs, key=lambda p: self.rank.get(p, float("inf")))  # الزوج الأقدم تعلّماً، ومن ليس في القائمة رتبته لا نهائية
            if best not in self.rank:  # لو لم يبقَ أي زوج قابل للدمج
                break  # نتوقف
            symbols = BPETrainer.merge_word(symbols, best)  # ندمج هذا الزوج في الكلمة
        self.cache[word] = symbols  # نخزّن النتيجة
        return symbols  # ونعيدها

    def tokenize(self, text):  # تحوّل نصاً كاملاً إلى قائمة رموز نصية
        tokens = []  # القائمة الناتجة
        for word in self.pretok(self.norm(text)):  # نطبّع النص ثم نقطّعه إلى كلمات
            tokens.extend(self._bpe(word))  # نضيف رموز كل كلمة
        return tokens  # نعيدها

    def encode(self, text, max_len=None):  # تحوّل نصاً إلى أرقام، مع CLS وSEP واقتطاع اختياري
        ids = [self.token_to_id.get(t, self.unk_id) for t in self.tokenize(text)]  # كل رمز إلى رقمه، والمجهول إلى UNK
        if max_len is not None:  # إن طُلب حدّ أقصى للطول
            ids = ids[: max_len - 2]  # نقتطع من النهاية، ونترك مكانين لـ CLS وSEP
        return [self.cls_id] + ids + [self.sep_id]  # نضع CLS في الأول وSEP في الآخر

    def decode(self, ids):  # تحوّل أرقاماً إلى نص مقروء
        skip = {self.pad_id, self.cls_id, self.sep_id}  # الرموز التي لا نعرضها
        tokens = [self.id_to_token[i] for i in ids if i not in skip]  # نحوّل كل رقم إلى رمزه ونتجاهل الخاصة
        return "".join(tokens).replace("</w>", " ").strip()  # نلصق الرموز، وعلامة النهاية تصير مسافة
