import math  # لدالة الجذر التربيعي
import torch  # مكتبة PyTorch
import torch.nn as nn  # وحدات الشبكات العصبية

class SelfAttentionHead(nn.Module):  # رأس attention واحد: يحوّل متجهات الرموز إلى متجهات تعرف سياقها
    def __init__(self, d_model, d_head):  # d_model بُعد المدخل، وd_head بُعد مخرج هذا الرأس
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        self.d_head = d_head  # نحفظه لاستعماله في القسمة على الجذر
        self.W_q = nn.Linear(d_model, d_head, bias=False)  # يصنع Query لكل رمز: ضرب في مصفوفة متعلَّمة بلا bias
        self.W_k = nn.Linear(d_model, d_head, bias=False)  # يصنع Key لكل رمز
        self.W_v = nn.Linear(d_model, d_head, bias=False)  # يصنع Value لكل رمز

    def forward(self, x, pad_mask=None):  # x شكلها (B, L, d_model)، وpad_mask شكلها (B, L) وقيمتها True عند الرموز الحقيقية
        q = self.W_q(x)  # الـ Queries، الشكل (B, L, d_head)
        k = self.W_k(x)  # الـ Keys، الشكل (B, L, d_head)
        v = self.W_v(x)  # الـ Values، الشكل (B, L, d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # نضرب كل Query في كل Key ونقسم على الجذر، الشكل (B, L, L)
        if pad_mask is not None:  # إن وُجد قناع حشو
            scores = scores.masked_fill(~pad_mask[:, None, :], float("-inf"))  # نضع -∞ عند مفاتيح الحشو، والتوسيع [:, None, :] يجعل القناع صالحاً لكل الاستعلامات
        weights = torch.softmax(scores, dim=-1)  # softmax على آخر بُعد: كل صف يصير أوزاناً مجموعها 1
        out = weights @ v  # متوسط مرجّح للـ Values، الشكل (B, L, d_head)
        return out, weights  # نعيد المخرج والأوزان معاً لنستطيع فحصها


class MultiHeadAttention(nn.Module):  # attention بعدة رؤوس تعمل بالتوازي، ثم تُدمج مخارجها
    def __init__(self, d_model, n_heads, dropout=0.1):  # d_model بُعد النموذج، وn_heads عدد الرؤوس
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # نتأكد أن البُعد يقبل القسمة على عدد الرؤوس، وإلا نوقف البرنامج برسالة
        self.n_heads = n_heads  # نحفظ عدد الرؤوس
        self.d_head = d_model // n_heads  # بُعد كل رأس، وهو 64 في إعدادنا
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # مصفوفات Query لكل الرؤوس موضوعة جنباً إلى جنب
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # مصفوفات Key لكل الرؤوس
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # مصفوفات Value لكل الرؤوس
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # مصفوفة المزج النهائية بعد لصق مخارج الرؤوس
        self.attn_dropout = nn.Dropout(dropout)  # يصفّر بعض الأوزان عشوائياً أثناء التدريب ليمنع الاعتماد على رمز واحد

    def split_heads(self, t):  # تحوّل (B, L, d_model) إلى (B, n_heads, L, d_head)
        B, L, _ = t.shape  # نقرأ حجم الدفعة والطول، ونتجاهل البُعد الأخير
        return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # نقسم البُعد الأخير إلى رؤوس ثم نقدّم بُعد الرؤوس قبل الطول

    def forward(self, x, pad_mask=None):  # x شكلها (B, L, d_model)، وpad_mask شكلها (B, L)
        q = self.split_heads(self.W_q(x))  # Queries لكل الرؤوس، الشكل (B, h, L, d_head)
        k = self.split_heads(self.W_k(x))  # Keys لكل الرؤوس
        v = self.split_heads(self.W_v(x))  # Values لكل الرؤوس
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # درجات كل رأس، الشكل (B, h, L, L)
        if pad_mask is not None:  # إن وُجد قناع حشو
            scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))  # نخفي مفاتيح الحشو، والتوسيع بمحورين يجعله صالحاً لكل الرؤوس وكل الاستعلامات
        weights = torch.softmax(scores, dim=-1)  # أوزان الانتباه لكل رأس، كل صف مجموعه 1
        out = self.attn_dropout(weights) @ v  # نطبّق الـ dropout على الأوزان ثم نأخذ المتوسط المرجّح، الشكل (B, h, L, d_head)
        B, _, L, _ = out.shape  # نقرأ حجم الدفعة والطول
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # نعيد الرؤوس إلى جنب بعضها (لصق)، والـ contiguous ضرورية قبل view، الشكل (B, L, d_model)
        return self.W_o(out), weights  # نمزج الرؤوس بـ W_o ونعيد المخرج مع الأوزان للفحص
