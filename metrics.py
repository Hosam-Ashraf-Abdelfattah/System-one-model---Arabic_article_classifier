import numpy as np  # للحسابات الإحصائية

def softmax(logits):  # تحوّل logits إلى احتمالات، صفاً بصف
    z = logits - logits.max(axis=1, keepdims=True)  # نطرح أكبر قيمة في كل صف للاستقرار العددي، ولا يغير النتيجة
    e = np.exp(z)  # الأُسّ
    return e / e.sum(axis=1, keepdims=True)  # نقسم على المجموع فيصير مجموع كل صف 1

def confusion_matrix(labels, preds, n_classes):  # مصفوفة الالتباس
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)  # مصفوفة أصفار: الصفوف للحقيقي والأعمدة للمتنبَّأ
    np.add.at(cm, (labels, preds), 1)  # نزيد خانة (الحقيقي، المتنبَّأ) بواحد لكل مقال
    return cm  # نعيدها

def per_class_report(cm):  # precision وrecall وF1 لكل فئة من مصفوفة الالتباس
    tp = np.diag(cm).astype(float)  # الصحيح لكل فئة: القطر
    precision = tp / np.maximum(cm.sum(axis=0), 1)  # الصحيح مقسوماً على كل ما قيل إنه هذه الفئة (مجموع العمود)
    recall = tp / np.maximum(cm.sum(axis=1), 1)  # الصحيح مقسوماً على كل مقالات هذه الفئة الحقيقية (مجموع الصف)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)  # المتوسط التوافقي، مع حماية من القسمة على صفر
    return precision, recall, f1  # نعيدها

def calibration_table(probs, labels, n_bins=10):  # جدول المعايرة وقيمة ECE
    conf = probs.max(axis=1)  # ثقة النموذج: أعلى احتمال أعطاه
    pred = probs.argmax(axis=1)  # فئته المختارة
    correct = (pred == labels).astype(float)  # 1 إن أصاب و0 إن أخطأ
    edges = np.linspace(0.0, 1.0, n_bins + 1)  # حدود الشرائح: 0 و0.1 و... و1
    rows, ece = [], 0.0  # صفوف الجدول وقيمة ECE المتراكمة
    for lo, hi in zip(edges[:-1], edges[1:]):  # لكل شريحة
        m = (conf > lo) & (conf <= hi)  # الإجابات التي وقعت ثقتها في هذه الشريحة
        if m.sum() == 0:  # لو كانت فارغة
            continue  # نتخطاها
        nb, cb, ab = int(m.sum()), conf[m].mean(), correct[m].mean()  # عددها ومتوسط ثقتها ودقتها الفعلية
        rows.append((lo, hi, nb, cb, ab))  # نسجّل الصف
        ece += (nb / len(labels)) * abs(ab - cb)  # نضيف الفجوة مرجّحة بحجم الشريحة
    return rows, ece  # الجدول وECE

def report(logits, labels, class_names):  # تطبع تقريراً كاملاً من logits وفئات حقيقية
    probs = softmax(logits)  # الاحتمالات
    preds = probs.argmax(axis=1)  # الفئات المتنبَّأ بها
    n = len(labels)  # عدد الأمثلة
    acc = (preds == labels).mean()  # الدقة
    nll = -np.log(np.maximum(probs[np.arange(n), labels], 1e-12)).mean()  # خسارة Cross-Entropy: سالب لوغاريتم احتمال الفئة الصحيحة
    cm = confusion_matrix(labels, preds, len(class_names))  # مصفوفة الالتباس
    p, r, f1 = per_class_report(cm)  # المقاييس لكل فئة
    print(f"accuracy {acc * 100:.2f}% | loss {nll:.4f} | macro-F1 {f1.mean() * 100:.2f}%")  # الملخص
    print("\nconfusion matrix (rows = true, columns = predicted):")  # عنوان المصفوفة
    print(" " * 10 + "".join(f"{c[:7]:>8}" for c in class_names))  # أسماء الأعمدة
    for i, name in enumerate(class_names):  # لكل صف
        print(f"{name:<10}" + "".join(f"{v:>8}" for v in cm[i]))  # اسم الفئة ثم أرقام صفها
    print("\nper class        precision  recall      F1  support")  # عنوان الجدول
    for i, name in enumerate(class_names):  # لكل فئة
        print(f"{name:<10} {p[i] * 100:12.2f} {r[i] * 100:8.2f} {f1[i] * 100:7.2f} {cm[i].sum():8d}")  # مقاييسها
    rows, ece = calibration_table(probs, labels)  # جدول المعايرة
    print("\nconfidence bin     count   mean confidence   accuracy")  # عنوان الجدول
    for lo, hi, nb, cb, ab in rows:  # لكل شريحة
        print(f"({lo:.1f}, {hi:.1f}] {nb:10d} {cb * 100:14.2f}% {ab * 100:10.2f}%")  # صفها
    conf = probs.max(axis=1)  # ثقة كل إجابة
    print(f"\nECE {ece * 100:.2f}% | mean confidence {conf.mean() * 100:.2f}% vs accuracy {acc * 100:.2f}%")  # ملخص المعايرة
    wrong = preds != labels  # الإجابات الخاطئة
    if wrong.any():  # لو وُجدت أخطاء
        print(f"errors: {int(wrong.sum())} | mean confidence on errors: {conf[wrong].mean() * 100:.1f}% | errors made with confidence > 90%: {(conf[wrong] > 0.9).mean() * 100:.1f}%")  # ثقة النموذج في أخطائه
