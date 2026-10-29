import torch.nn as nn  # وحدات الشبكات العصبية
from encoder import Encoder  # الجسم الذي بنيناه
import config as C  # إعداداتنا المركزية

class ChoiceHead(nn.Module):  # رأس اختيار من عدة خيارات: تجميع ثم طبقة خطية تعطي logits
    def __init__(self, d_model, n_choices, dropout=0.1):  # بُعد النموذج وعدد الخيارات
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        self.dropout = nn.Dropout(dropout)  # يصفّر بعض القيم عشوائياً أثناء التدريب
        self.fc = nn.Linear(d_model, n_choices)  # يحوّل المتجه المجمَّع إلى رقم خام لكل خيار

    def forward(self, h, mask):  # h شكلها (B, L, d_model)، وmask شكلها (B, L) وقيمتها True عند الرموز الحقيقية
        m = mask.unsqueeze(-1).to(h.dtype)  # نحوّل القناع إلى أعداد (1 أو 0) ونضيف بُعداً أخيراً، الشكل (B, L, 1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)  # نصفّر متجهات الحشو ونجمع ونقسم على عدد الرموز الحقيقية، والـ clamp تمنع القسمة على صفر
        return self.fc(self.dropout(pooled))  # الـ logits الخام، الشكل (B, n_choices)، ولا نطبق softmax هنا

class DecisionModel(nn.Module):  # النموذج الكامل: Encoder ثم رأس قرار
    def __init__(self, encoder, head):  # نستقبل الجسم والرأس جاهزين
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        self.encoder = encoder  # الجسم الذي يفهم النص
        self.head = head  # الرأس الذي يحوّل الفهم إلى قرار

    def forward(self, ids):  # ids شكلها (B, L)
        h, mask = self.encoder(ids)  # متجهات الرموز مع قناع الحشو
        return self.head(h, mask)  # نمرّرها إلى الرأس ونعيد الـ logits

def build_model(vocab_size, pad_id):  # دالة تبني النموذج الكامل بإعدادات config، فنستعملها في التدريب والتقييم دون تكرار
    enc = Encoder(vocab_size, C.D_MODEL, C.N_HEADS, C.N_LAYERS, C.D_FF, C.MAX_LEN, pad_id, C.DROPOUT)  # الجسم
    head = ChoiceHead(C.D_MODEL, C.N_CLASSES, C.DROPOUT)  # رأس الاختيار
    return DecisionModel(enc, head)  # النموذج الكامل
