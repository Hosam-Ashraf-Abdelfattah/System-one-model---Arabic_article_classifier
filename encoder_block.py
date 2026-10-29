import torch.nn as nn  # وحدات الشبكات العصبية
from attention import MultiHeadAttention  # طبقة الـ attention التي بنيناها
from feedforward import FeedForward  # الشبكة التغذوية التي بنيناها

class EncoderBlock(nn.Module):  # كتلة Encoder واحدة بترتيب Pre-LN: تطبيع ثم طبقة ثم إضافة المدخل
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):  # الأبعاد ونسبة الـ dropout
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        self.ln1 = nn.LayerNorm(d_model)  # تطبيع قبل الـ attention، بمعاملين γ وβ لكل بُعد
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)  # طبقة الرؤوس المتعددة
        self.ln2 = nn.LayerNorm(d_model)  # تطبيع قبل الشبكة التغذوية
        self.ff = FeedForward(d_model, d_ff, dropout)  # الشبكة التغذوية، وفيها dropout خاص بها
        self.dropout = nn.Dropout(dropout)  # dropout لمخرج الـ attention قبل إضافته للمدخل

    def forward(self, x, pad_mask=None):  # x شكلها (B, L, d_model)، وpad_mask شكلها (B, L)
        attn_out, weights = self.attn(self.ln1(x), pad_mask)  # نطبّع x أولاً ثم نمرّره على الـ attention
        x = x + self.dropout(attn_out)  # الاتصال المتبقي الأول: نضيف مخرج الـ attention إلى x الأصلي غير المطبَّع
        x = x + self.ff(self.ln2(x))  # الاتصال المتبقي الثاني: نطبّع ثم نمرّر على الشبكة التغذوية ثم نضيف إلى x
        return x, weights  # نعيد المخرج مع أوزان الانتباه لنستطيع فحصها
