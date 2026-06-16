import os
import glob
import cv2
import easyocr
import numpy as np

# 1. تهيئة المنظومة الذكية
print("⏳ جاري استدعاء عيون الذكاء الاصطناعي (EasyOCR)...")
reader = easyocr.Reader(['en'])

# 2. إعدادات المجلدات
RAW_DIR = "raw_img"
CLEAN_DIR = "clean_img"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# 3. جلب الصور
images = glob.glob(os.path.join(RAW_DIR, "*"))

if not images:
    print("❌ مجلد raw_img فارغ! حط صور وتعال عيني.")
    exit()

print(f" Let's Go! جاري معالجة {len(images)} صور في Level 1...")

for img_path in images:
    img_name = os.path.basename(img_path)
    print(f"🧼 جاري تطهير: {img_name}...")
    
    # قراءة الصورة الأصلية
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ الصورة {img_name} معطوبة، تخطيناها.")
        continue
        
    # صناعة القناع الخام (كل شي أسود حالياً)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    
    # تشغيل الـ OCR
    results = reader.readtext(img)
    
    # رسم مربعات النصوص على القناع
    for (bbox, text, prob) in results:
        # تحويل الإحداثيات إلى أرقام صحيحة وضمان عدم الخروج عن أبعاد الصورة
        p1 = (max(0, int(bbox[0][0])), max(0, int(bbox[0][1])))
        p2 = (min(img.shape[1], int(bbox[2][0])), min(img.shape[0], int(bbox[2][1])))
        
        # رسم مربع أبيض ممتلئ مكان النص
        cv2.rectangle(mask, p1, p2, 255, -1)
    
    # 🌟 [سر Level 1] تمديد القناع (Dilation)
    # هنا نصنع مصفوفة (Kernel) بحجم 9x9 بكسل تمثل مقدار النفخ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # عملية النفخ السحرية تضخم المربعات البيضاء حتى تاكل الحواف الميتة
    smart_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # عملية التبييض الاحترافية باستخدام خوارزمية تيسير (Navier-Stokes) بدل القديمة
    cleaned_img = cv2.inpaint(img, smart_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    # حفظ النتيجة
    output_path = os.path.join(CLEAN_DIR, img_name)
    cv2.imwrite(output_path, cleaned_img)
    print(f"✅ تم تنظيف وحفظ: {img_name}")

print("🎉 كفو! قفلنا Level 1 بنجاح ساحق!")
