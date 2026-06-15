import os
import glob
import cv2
import easyocr
import numpy as np

# 1. تهيئة قارئ النصوص الذكي (لغة إنجليزية فقط)
print("⏳ جاري تشغيل عيون الذكاء الاصطناعي EasyOCR...")
reader = easyocr.Reader(['en'])

# 2. تحديد المجلدات
RAW_DIR = "raw_img"
CLEAN_DIR = "clean_img"

# إنشاء المجلدات إذا ما كانت موجودة
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# 3. جلب جميع الصور من مجلد raw_img
images = glob.glob(os.path.join(RAW_DIR, "*"))

if not images:
    print("❌ ماكو أي صورة بمجلد raw_img! المنظومة في وضع الاستعداد.")
    exit()

print(f" Let's Go! لقينا {len(images)} صور جاهزة للتطهير...")

for img_path in images:
    img_name = os.path.basename(img_path)
    print(f"🧼 جاري تبييض الصورة: {img_name}...")
    
    # قراءة الصورة الأصلية باستخدام OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ خطأ بقراءة الصورة {img_name}، راح نتخطاها.")
        continue
        
    # صناعة قناع أسود (Mask) بنفس أبعاد الصورة الأصلية
    mask = np.zeros(img.shape[:2], dtype="uint8")
    
    # تشغيل الـ OCR لقراءة النصوص وتحديد إحداثياتها
    results = reader.readtext(img)
    
    # رسم مربعات بيضاء على القناع الأسود في مكان النصوص المكتشفة
    for (bbox, text, prob) in results:
        #bbox يحتوي على إحداثيات الزوايا الأربعة للمربع [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        p1 = tuple(map(int, bbox[0])) # الزاوية العليا اليسرى
        p2 = tuple(map(int, bbox[2])) # الزاوية السفلى اليمنى
        
        # رسم مربع أبيض ممتلئ (thickness=-1) فوق النص في صورة القناع
        cv2.rectangle(mask, p1, p2, 255, -1)
    
    # عملية التبييض السحرية (Inpainting)
    # يأخذ الصورة الأصلية والقناع، ويمسح الأبيض ويصب ألوان الخلفية (بقطر معالجة 3 بكسل)
    cleaned_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # حفظ الصورة النظيفة في مجلد clean_img بنفس الاسم
    output_path = os.path.join(CLEAN_DIR, img_name)
    cv2.imwrite(output_path, cleaned_img)
    print(f"✅ تم تنظيف وحفظ: {img_name}")

print("🎉 كفو! تم تبييض جميع الصور بنجاح ساحق!")
