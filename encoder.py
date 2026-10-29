import torch.nn as nn  # وحدات الشبكات العصبية
from embeddings import Embeddings  # طبقة الـ embedding من الخطوة 6
from encoder_block import EncoderBlock  # الكتلة من الخطوة 10

class Encoder(nn.Module):  # الجسم الكامل للنموذج: embedding ثم عدة كتل ثم تطبيع نهائي
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, pad_id, dropout=0.1):  # كل إعدادات النموذج
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        self.pad_id = pad_id  # نحفظ رقم الحشو لنبني منه القناع
        self.embeddings = Embeddings(vocab_size, d_model, max_len, pad_id, dropout)  # طبقة الـ embedding والموضع
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])  # قائمة الكتل، وModuleList تجعل PyTorch ترى معاملاتها
        self.final_ln = nn.LayerNorm(d_model)  # التطبيع النهائي الذي يتطلبه ترتيب Pre-LN

    def forward(self, ids):  # ids شكلها (B, L) وهي أرقام الرموز
        pad_mask = ids != self.pad_id  # قناع: True عند الرموز الحقيقية وFalse عند الحشو
        x = self.embeddings(ids)  # نحوّل الأرقام إلى متجهات مع الموضع، الشكل (B, L, d_model)
        for block in self.blocks:  # نمر على الكتل بالترتيب
            x, _ = block(x, pad_mask)  # كل كتلة تأخذ مخرج سابقتها، ونتجاهل أوزان الانتباه هنا
        return self.final_ln(x), pad_mask  # نطبّع ونعيد المتجهات مع القناع لتستعملهما رؤوس القرار
