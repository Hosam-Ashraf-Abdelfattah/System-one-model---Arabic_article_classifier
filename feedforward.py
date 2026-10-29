import torch.nn as nn  # وحدات الشبكات العصبية

class FeedForward(nn.Module):  # شبكة تغذوية تُطبَّق على كل رمز على حدة، وتعطي النموذج قدرة حساب غير خطية
    def __init__(self, d_model, d_ff, dropout=0.1):  # d_model بُعد النموذج، وd_ff بُعد الطبقة الوسطى الأوسع
        super().__init__()  # نستدعي تجهيز الأب nn.Module
        self.fc1 = nn.Linear(d_model, d_ff)  # الطبقة الأولى: توسّع من 256 إلى 1024، ومعها bias
        self.act = nn.GELU()  # دالة التفعيل الناعمة التي تكسر الخطية
        self.fc2 = nn.Linear(d_ff, d_model)  # الطبقة الثانية: تضغط من 1024 إلى 256
        self.dropout = nn.Dropout(dropout)  # يصفّر بعض القيم عشوائياً أثناء التدريب فقط

    def forward(self, x):  # x شكلها (B, L, d_model)
        x = self.fc1(x)  # التوسيع، الشكل (B, L, d_ff)
        x = self.act(x)  # نطبّق GELU على كل رقم
        x = self.fc2(x)  # الضغط، الشكل (B, L, d_model)
        return self.dropout(x)  # نطبّق الـ dropout ونعيد النتيجة
