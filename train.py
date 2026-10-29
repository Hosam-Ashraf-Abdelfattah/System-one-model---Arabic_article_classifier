import math  # لدالة cos في جدولة معدل التعلم
import os  # لإنشاء مجلد نقاط الاستعادة
import time  # لقياس زمن التدريب
import torch  # مكتبة PyTorch
import torch.nn as nn  # لدالة الخسارة
from tokenizer import BPETokenizer  # الـ tokenizer
from dataset import make_loader  # الـ DataLoader الذي بنيناه
from model import build_model  # دالة بناء النموذج
import config as C  # الإعدادات المركزية

QUICK = False  # True = اختبار سريع لثلاثمئة خطوة للتأكد أن الحلقة سليمة، وغيّرها إلى False للتدريب الكامل
EPOCHS = 4  # عدد مرات المرور على كل البيانات في التدريب الكامل
BATCH = 32  # حجم الدفعة
PEAK_LR, FLOOR_LR = 3e-4, 3e-5  # أعلى معدل تعلم وأدناه
WARMUP = 50 if QUICK else 300  # عدد خطوات التسخين
CLIP = 1.0  # الحد الأقصى لطول الاشتقاقات
LOG_EVERY = 50 if QUICK else 200  # كل كم خطوة نطبع
device = "cuda"  # نستعمل كرت الشاشة
torch.manual_seed(42)  # نثبّت العشوائية لتتكرر النتائج

def lr_at(step, warmup, total, peak, floor):  # تحسب معدل التعلم عند خطوة معينة
    if step < warmup:  # أثناء التسخين
        return peak * (step + 1) / warmup  # صعود خطي من الصفر تقريباً إلى القمة
    progress = (step - warmup) / max(1, total - warmup)  # نسبة ما قطعناه من مرحلة التناقص، من 0 إلى 1
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))  # تناقص جيبي ناعم من القمة إلى القاع

@torch.no_grad()  # المقيِّم لا يحتاج اشتقاقات
def evaluate(model, loader, loss_fn, max_batches=None):  # يحسب الخسارة والدقة على جزء من البيانات
    model.eval()  # نوقف الـ dropout
    total_loss, correct, n = 0.0, 0, 0  # مجموع الخسارة وعدد الصحيح وعدد الأمثلة
    for b, (ids, labels) in enumerate(loader):  # نمر على الدفعات
        if max_batches is not None and b >= max_batches:  # لو حددنا حداً أقصى للدفعات
            break  # نتوقف
        ids, labels = ids.to(device), labels.to(device)  # ننقلها إلى الكرت
        logits = model(ids)  # التنبؤ
        total_loss += loss_fn(logits, labels).item() * len(labels)  # الخسارة متوسط الدفعة، فنضربها في حجم الدفعة لنجمع المجموع
        correct += (logits.argmax(-1) == labels).sum().item()  # نعدّ الإجابات الصحيحة
        n += len(labels)  # نعدّ الأمثلة
    model.train()  # نعيد النموذج لوضع التدريب
    return total_loss / n, correct / n  # متوسط الخسارة والدقة

tok = BPETokenizer("data/bpe.json")  # نحمّل الـ tokenizer
model = build_model(tok.vocab_size, tok.pad_id).to(device)  # نبني النموذج وننقله للكرت
print("params:", sum(p.numel() for p in model.parameters()))  # عدد المعاملات، المتوقع 7253255
train_loader = make_loader("train", BATCH, True)  # بيانات التدريب مع الخلط
val_loader = make_loader("validation", 64, False)  # بيانات التحقق بدون خلط
total = 300 if QUICK else EPOCHS * len(train_loader)  # إجمالي الخطوات، ويلزم لجدولة معدل التعلم
loss_fn = nn.CrossEntropyLoss()  # دالة الخسارة
decay = [p for p in model.parameters() if p.dim() >= 2]  # المعاملات التي نطبّق عليها weight decay: المصفوفات والـ embedding
no_decay = [p for p in model.parameters() if p.dim() < 2]  # المعاملات التي نستثنيها: الـ bias وLayerNorm
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=PEAK_LR)  # المُحسِّن بمجموعتين
print("baseline to beat: loss 1.802 | accuracy 27.75% | total steps:", total)  # الأرقام التي حسبناها لنموذج يعرف تكرار الفئات فقط
os.makedirs("checkpoints", exist_ok=True)  # ننشئ مجلد نقاط الاستعادة إن لم يوجد
best_val, step, done, start = float("inf"), 0, False, time.time()  # أفضل خسارة تحقق وعدّاد الخطوات وعلامة الانتهاء وساعة البدء
model.train()  # وضع التدريب
for epoch in range(EPOCHS):  # حلقة الـ epochs
    running, count = 0.0, 0  # مجموع الخسارة منذ آخر طباعة وعدد الخطوات
    for ids, labels in train_loader:  # حلقة الدفعات
        lr = lr_at(step, WARMUP, total, PEAK_LR, FLOOR_LR)  # معدل التعلم لهذه الخطوة
        for g in opt.param_groups:  # نمر على مجموعتي المعاملات
            g["lr"] = lr  # نضبط معدل التعلم الجديد
        ids, labels = ids.to(device), labels.to(device)  # ننقل الدفعة إلى الكرت
        opt.zero_grad(set_to_none=True)  # نمسح اشتقاقات الخطوة السابقة
        loss = loss_fn(model(ids), labels)  # التمرير الأمامي وحساب الخسارة
        loss.backward()  # التمرير العكسي: حساب الاشتقاقات
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # نقصّ الاشتقاقات ونحصل على طولها قبل القص
        opt.step()  # تحديث المعاملات
        step += 1  # نعدّ الخطوة
        running += loss.item()  # نجمع الخسارة
        count += 1  # نعدّ الخطوات منذ آخر طباعة
        if step % LOG_EVERY == 0:  # كل LOG_EVERY خطوة
            print(f"step {step}/{total} | loss {running / count:.4f} | lr {lr:.2e} | grad norm {gnorm:.2f} | {(time.time() - start) / 60:.1f} min")  # نطبع التقدم
            running, count = 0.0, 0  # نصفّر العدّادات
        if step >= total:  # لو بلغنا الإجمالي
            done = True  # نعلّم الانتهاء
            break  # ونخرج من حلقة الدفعات
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, max_batches=50 if QUICK else None)  # نقيّم على validation
    print(f"== epoch {epoch + 1} | val loss {val_loss:.4f} | val acc {val_acc * 100:.2f}% | {(time.time() - start) / 60:.1f} min")  # ملخص الـ epoch
    if val_loss < best_val:  # لو هذا أفضل نموذج حتى الآن
        best_val = val_loss  # نحدّث أفضل خسارة
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc}, "checkpoints/best.pt")  # نحفظ أوزانه
        print("   saved checkpoints/best.pt")  # نخبر المستخدم
    if done:  # لو انتهينا
        break  # نخرج من حلقة الـ epochs
