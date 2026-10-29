MAX_LEN = 256  # الحد الأقصى لطول المقال بالرموز
D_MODEL = 256  # بُعد المتجهات داخل النموذج
N_HEADS = 4  # عدد رؤوس الـ attention
N_LAYERS = 4  # عدد كتل الـ Encoder
D_FF = 1024  # بُعد الطبقة الوسطى في الشبكة التغذوية
DROPOUT = 0.1  # نسبة الـ dropout في كل مكان
N_CLASSES = 7  # عدد فئات SANAD
CLASS_NAMES = ["Tech", "Finance", "Politics", "Religion", "Medical", "Culture", "Sports"]  # أسماء الفئات بترتيب أرقامها
