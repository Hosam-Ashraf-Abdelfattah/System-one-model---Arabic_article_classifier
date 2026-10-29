import numpy as np  # لقراءة المصفوفات المحفوظة
import torch  # مكتبة PyTorch
from torch.utils.data import Dataset, DataLoader  # الحاويات القياسية لتغذية النموذج بالبيانات

class TokenDataset(Dataset):  # class تخبر PyTorch كيف تقرأ مثالاً واحداً من بياناتنا
    def __init__(self, split):  # split هو اسم الجزء: train أو validation أو test
        self.ids = np.load(f"data/{split}_ids.npy")  # مصفوفة الأرقام (عدد المقالات × 256)
        self.labels = np.load(f"data/{split}_labels.npy")  # الفئات بنفس الترتيب

    def __len__(self):  # PyTorch تسأل: كم مثالاً عندك؟
        return len(self.labels)  # عدد المقالات

    def __getitem__(self, i):  # PyTorch تقول: أعطني المثال رقم i
        ids = torch.from_numpy(self.ids[i].astype(np.int64))  # نحوّل الصف إلى int64 لأن طبقة الـ embedding تتطلبه
        return ids, int(self.labels[i])  # نعيد الأرقام والفئة

def make_loader(split, batch_size, shuffle):  # دالة تبني DataLoader جاهزاً
    ds = TokenDataset(split)  # ننشئ الـ Dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)  # الـ shuffle للتدريب فقط، وdrop_last يحذف الدفعة الأخيرة الناقصة في التدريب فقط
