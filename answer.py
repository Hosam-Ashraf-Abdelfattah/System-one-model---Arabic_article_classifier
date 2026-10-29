import json  # لطباعة الجواب بصيغة منظمة
import numpy as np  # للحسابات العددية
from predictor import predict_probs  # دالة التنبؤ المعايَر التي جرّبتها أنت فعلاً في فحص الاختصار
import config as C  # أسماء الفئات

ABSTAIN_BELOW = 0.70  # لو كانت الثقة المعايَرة أقل من هذا نمتنع عن الجواب بدل أن نخمّن

def answer(text, question="ما موضوع هذا النص؟", abstain_below=ABSTAIN_BELOW):  # تستقبل نصاً وتعيد جواباً مكتوب النوع مع ثقته
    p = predict_probs([text])[0]  # احتمالات الفئات السبع المعايَرة لهذا النص
    order = np.argsort(-p)  # ترتيب الفئات من الأرجح إلى الأقل
    best = int(order[0])  # أرجح فئة
    return {  # الجواب
        "question": question,  # السؤال للعرض فقط: النموذج يجيب عن سؤال واحد ثابت هو الموضوع
        "type": "choice",  # نوع الجواب: اختيار من خيارات
        "answer": C.CLASS_NAMES[best] if p[best] >= abstain_below else None,  # الفئة، أو None إن كانت الثقة دون الحد
        "confidence": round(float(p[best]), 3),  # الثقة المعايَرة في الجواب
        "top3": [(C.CLASS_NAMES[i], round(float(p[i]), 3)) for i in order[:3]],  # أعلى ثلاث فئات واحتمالاتها
    }

if __name__ == "__main__":  # هذا الجزء يعمل عند تشغيل الملف مباشرة
    examples = [  # أمثلة قصيرة من تأليفي لتجربة سريعة، وقد لا تشبه أسلوب SANAD تماماً
        "فاز الفريق بهدفين مقابل هدف في المباراة النهائية أمام جماهيره",
        "أعلنت الشركة ارتفاع أرباحها السنوية وتوزيع أسهم مجانية على المساهمين",
        "أكد الوزير في تصريح للصحافة أن الحكومة ستعرض مشروع القانون على البرلمان",
        "ينصح الأطباء بالإكثار من شرب الماء لتجنب الجفاف وحصى الكلى",
        "كلام لا معنى له عن أشياء كثيرة مختلفة",
    ]
    for t in examples:  # لكل مثال
        print(json.dumps(answer(t), ensure_ascii=False))  # نطبع الجواب
    print("\nاكتب نصاً عربياً (سطر فارغ للخروج):")  # تعليمات
    while True:  # حلقة تفاعلية
        t = input("> ").strip()  # نقرأ نصاً من لوحة المفاتيح
        if not t:  # سطر فارغ
            break  # نخرج
        print(json.dumps(answer(t), ensure_ascii=False, indent=1))  # نطبع الجواب
