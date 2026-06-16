import os
import glob
import cv2
import easyocr
import numpy as np
import fal_client

# 🌟 تأمين المفتاح بداخل السيرفر
# لازم تضيف المفتاح بداخل إعدادات الـ Repository في جيت هوب باسم FAL_KEY
os.environ["FAL_KEY"] = os.getenv("FAL_KEY", "c31845c1-898a-4397-8d71-548906dfc09b:3d15c3778f28749eb3b171f3f46fd91e")

print("⏳ جاري تشغيل عيون الـ OCR الذكية...")
reader = easyocr.Reader(['en'])

RAW_DIR = "raw_img"
CLEAN_DIR = "clean_img"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

images = glob.glob(os.path.join(RAW_DIR, "*"))

if not images:
    print("❌ ماكو صور بمجلد raw_img!")
    exit()

print(f"🔥 صعدنا لـ Level 6 (الجيل التوليدي)! جاري تبييض {len(images)} صور...")

for img_path in images:
    img_name = os.path.basename(img_path)
    print(f"🧼 جاري التطهير التوليدي لـ: {img_name}...")
    
    img = cv2.imread(img_path)
    if img is None: continue
        
    # صناعة القناع
    mask = np.zeros(img.shape[:2], dtype="uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    res_norm = reader.readtext(img)
    res_inv = reader.readtext(cv2.bitwise_not(gray))
    
    for (bbox, text, prob) in (res_norm + res_inv):
        # تمديد ذكي بـ 5 بكسل لضمان عزل تام للحواف
        p1 = (max(0, int(bbox[0][0]) - 5), max(0, int(bbox[0][1]) - 5))
        p2 = (min(img.shape[1], int(bbox[2][0]) + 5), min(img.shape[0], int(bbox[2][1]) + 5))
        cv2.rectangle(mask, p1, p2, 255, -1)
        
    # حفظ القناع مؤقتاً بصيغة png لأن السيرفر يحتاج يرفعه
    temp_img_path = f"temp_{img_name}"
    temp_mask_path = f"mask_{img_name}"
    
    cv2.imwrite(temp_img_path, img)
    cv2.imwrite(temp_mask_path, mask)
    
    try:
        print("🚀 جاري إرسال الكتلة العملاقة لـ وحش الذكاء الاصطناعي...")
        
        # رفع الملفات بشكل مؤقت وآمن للحصول على روابط سريعة
        image_url = fal_client.upload_file(temp_img_path)
        mask_url = fal_client.upload_file(temp_mask_path)
        
        # استدعاء نموذج التبييض السياقي الفاخر (فهم الخلفية وإعادة الرسم)
        result = fal_client.subscribe(
            "fal-ai/lama",
            input={
                "image_url": image_url,
                "mask_url": mask_url
            },
            with_logs=False,
        )
        
        # جلب رابط الصورة النقية الناتجة وتحميلها
        output_url = result["image"]["url"]
        cleaned_img_bytes = fal_client.download_file(output_url)
        
        # حفظ النتيجة النهائية النظيفة
        output_path = os.path.join(CLEAN_DIR, img_name)
        with open(output_path, "wb") as f:
            f.write(cleaned_img_bytes)
            
        print(f"✅ تم التبييض التوليدي الخرافي وحفظ: {img_name}")
        
    except Exception as e:
        print(f"⚠️ صار مشكلة أثناء التوليد للأسف: {e}")
        # إذا صار أي خطأ بالنت، نرجع للنظام الاحتياطي السريع حتى ما يموت السيرفر
        backup_fill = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), backup_fill)
        
    finally:
        # تنظيف الملفات المؤقتة
        if os.path.exists(temp_img_path): os.remove(temp_img_path)
        if os.path.exists(temp_mask_path): os.remove(temp_mask_path)

print("🎉 قفلنا المود التوليدي لـ Level 6! عوف الغباء وروح شوف الإبداع!")
