import os
import glob
import cv2
import easyocr
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MangaScourX import PatchMatchInpainter

print("⏳ جاري تشغيل عيون الـ OCR الذكية...")
reader = easyocr.Reader(['en'])

# تهيئة المحرك المحلي مالتنا بالإعدادات الإنتاجية القصوى
inpainter = PatchMatchInpainter(
    patch_size=7,
    pyramid_levels=5,
    iterations=6,
    knn=3,
    use_rotation=True,   # تفعيل دوران الأنسجة
    use_scale=True,      # تفعيل تحجيم الأنسجة
    use_coherence=True,  # الحفاظ على الخطوط الهيكلية للمانجا
    verbose=True
)

RAW_DIR = "raw_img"
CLEAN_DIR = "clean_img"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

images = glob.glob(os.path.join(RAW_DIR, "*"))

if not images:
    print("❌ ماكو صور بمجلد raw_img!")
    exit()

print(f"🚀 المحرك المحلي الخارق جاهز! جاري تطهير {len(images)} صور بواسطة MangaScourX...")

for img_path in images:
    img_name = os.path.basename(img_path)
    print(f"\n✨ جاري المعالجة الرياضية لـ: {img_name}...")
    
    img = cv2.imread(img_path)
    if img is None: continue
        
    # 1. صناعة قناع الحظر والعزل
    mask = np.zeros(img.shape[:2], dtype="uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    res_norm = reader.readtext(img)
    res_inv = reader.readtext(cv2.bitwise_not(gray))
    
    for (bbox, text, prob) in (res_norm + res_inv):
        # تمديد ذكي بـ 5 بكسل لضمان عزل تام وتغطية حواف النص
        p1 = (max(0, int(bbox[0][0]) - 5), max(0, int(bbox[0][1]) - 5))
        p2 = (min(img.shape[1], int(bbox[2][0]) + 5), min(img.shape[0], int(bbox[2][1]) + 5))
        cv2.rectangle(mask, p1, p2, 255, -1)
        
    try:
        print("🎯 جاري استدعاء الـ 5D PatchMatch Engine لإعادة بناء النسيج...")
        
        # تشغيل التبييض والتنظيف الخارق المبني على الخوارزميات الرياضية الذكية
        # المحرك يتوقع ماسك True للـ خراب (النص) و False للـ سليم
        known_mask = mask == 0
        cleaned_img = inpainter.run(img, known_mask)
        
        # حفظ النتيجة النهائية الفخمة
        output_path = os.path.join(CLEAN_DIR, img_name)
        cv2.imwrite(output_path, cleaned_img)
        print(f"✅ تم التطهير المحلي وحفظ: {img_name}")
        
    except Exception as e:
        print(f"⚠️ خطأ غير متوقع أثناء المعالجة: {e}")
        # نظام الطوارئ السريع والتقليدي إذا انضربت أي مصفوفة بالـ Memory
        backup_fill = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), backup_fill)

print("\n🎉 قفلنا السيرفر بالكامل! عوف غباء الـ API وروح شوف الإبداع المحلي الخارق!")
