import math  # للدوال الرياضية مثل log وsqrt
import torch  # مكتبة PyTorch الأساسية
import torch.nn as nn  # وحدات بناء الشبكات العصبية مثل Embedding وDropout

class Embeddings(nn.Module):  # class تحوّل أرقام الرموز إلى متجهات تحمل المعنى والموضع
    def __init__(self, vocab_size, d_model, max_len, pad_id, dropout=0.1):  # دالة التجهيز
        super().__init__()  # نستدعي تجهيز الأب nn.Module، وهو ضروري في كل class شبكة
        self.d_model = d_model  # نحفظ بُعد المتجه لاستعماله لاحقاً
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # جدول بحث: vocab_size صفاً وd_model عموداً، وصف الحشو لا يتلقى تعديلاً أثناء التدريب
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=d_model ** -0.5)  # نهيّئ القيم عشوائياً بانحراف معياري 1/sqrt(d)
        with torch.no_grad():  # نعطّل تتبع الاشتقاق لأننا نعدّل القيم يدوياً
            self.token_emb.weight[pad_id].zero_()  # نعيد صفّ الحشو إلى أصفار لأن التهيئة العشوائية غيّرته
        pe = torch.zeros(max_len, d_model)  # مصفوفة فارغة للترميز الموضعي: صف لكل موضع
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # الأرقام 0 إلى max_len-1 كعمود، شكله (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))  # يحسب 1/10000^(2i/d) لكل زوج أبعاد، شكله (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # الأبعاد الزوجية (0 و2 و4...) تأخذ sin
        pe[:, 1::2] = torch.cos(position * div_term)  # الأبعاد الفردية (1 و3 و5...) تأخذ cos
        self.register_buffer("pe", pe)  # نسجّلها كـ buffer: ليست معاملاً يتدرب، لكنها تُحفظ وتنتقل للـ GPU مع النموذج
        self.dropout = nn.Dropout(dropout)  # يصفّر 10% من القيم عشوائياً أثناء التدريب فقط، ليمنع الحفظ الأعمى

    def forward(self, ids):  # تُنفَّذ عند استدعاء الكائن، وتستقبل أرقاماً شكلها (batch, seq_len)
        x = self.token_emb(ids) * math.sqrt(self.d_model)  # نجلب متجه كل رمز ونكبّره ليوازن الترميز الموضعي، الشكل (batch, seq_len, d_model)
        x = x + self.pe[: ids.size(1)]  # نضيف ترميز المواضع بعدد طول التسلسل، ويُبثّ تلقائياً على كل مقالات الدفعة
        return self.dropout(x)  # نطبّق الـ dropout ونعيد النتيجة
