import os
import glob
import cv2
import easyocr
import numpy as np
from typing import List, Tuple, Optional

# ================== إعدادات النظام ==================
RAW_DIR = "raw_img"
CLEAN_DIR = "clean_img"
LANGUAGES = ['en']          # يمكن إضافة 'ar' للعربية
CONFIDENCE_THRESHOLD = 0.5  # تجاهل النصوص الأقل ثقة
BUBBLE_AREA_THRESHOLD = 500 # أقل مساحة لفقاعة الكلام (بكسل)
INPAINT_RADIUS_FREE_TEXT = 2   # نصف قطر الترميم للنصوص الحرة (صغير جداً)
INPAINT_RADIUS_BUBBLE = 5      # نصف قطر الترميم داخل الفقاعة (أكبر قليلاً)
# ===================================================

print("⏳ جاري تحميل نموذج EasyOCR...")
reader = easyocr.Reader(LANGUAGES, gpu=False)  # gpu=True إذا كان متوفراً

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

def detect_speech_bubbles(gray: np.ndarray) -> List[np.ndarray]:
    """
    كشف فقاعات الكلام البيضاء باستخدام العتبة والـ Contours.
    تعيد قائمة من الأقنعة (Masks) لكل فقاعة.
    """
    # عتبة عالية لالتقاط المناطق البيضاء فقط
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # عمليات مورفولوجية لإغلاق الثقوب الصغيرة داخل الفقاعات
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_masks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > BUBBLE_AREA_THRESHOLD:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            bubble_masks.append(mask)
    return bubble_masks

def is_inside_bubble(text_mask: np.ndarray, bubble_masks: List[np.ndarray]) -> bool:
    """
    تحديد ما إذا كان النص (الموضح بقناع) يقع داخل أي فقاعة.
    معيار التداخل: أكثر من 30% من مساحة النص داخل الفقاعة.
    """
    if not bubble_masks:
        return False
    text_pixels = cv2.countNonZero(text_mask)
    if text_pixels == 0:
        return False
    for bubble_mask in bubble_masks:
        intersection = cv2.bitwise_and(text_mask, bubble_mask)
        inter_pixels = cv2.countNonZero(intersection)
        if inter_pixels / text_pixels > 0.3:
            return True
    return False

def refine_text_mask(gray: np.ndarray, bbox_points: np.ndarray) -> np.ndarray:
    """
    إنشاء قناع دقيق للنص على مستوى الحروف باستخدام Adaptive Threshold وعمليات مورفولوجية.
    """
    # قناع خام للمنطقة المستطيلة للنص
    raw_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(raw_mask, [bbox_points], 255)
    
    # استخراج منطقة النص فقط
    text_region = cv2.bitwise_and(gray, raw_mask)
    
    # عتبة محلية (Adaptive) لعزل الحروف عن الخلفية
    thresh = cv2.adaptiveThreshold(text_region, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # تنظيف القناع: إغلاق الثقوب الصغيرة + تآكل طفيف لعزل الحروف بدقة
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    # إعادة تطبيق القناع الأصلي للحفاظ على حدود المنطقة
    refined_mask = cv2.bitwise_and(thresh, raw_mask)
    return refined_mask

# ================== المعالجة الرئيسية ==================
images = glob.glob(os.path.join(RAW_DIR, "*"))
if not images:
    print("❌ لا توجد صور في مجلد raw_img!")
    exit()

print(f"✨ تم العثور على {len(images)} صورة. بدء عملية التبييض الذكية...")

for img_path in images:
    img_name = os.path.basename(img_path)
    print(f"\n🧼 معالجة: {img_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ خطأ في قراءة {img_name}، تخطي.")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = img.copy()
    
    # 1. كشف فقاعات الكلام
    bubble_masks = detect_speech_bubbles(gray)
    print(f"   - تم كشف {len(bubble_masks)} فقاعة كلام.")
    
    # 2. كشف النصوص باستخدام EasyOCR
    try:
        text_regions = reader.readtext(gray)
    except Exception as e:
        print(f"⚠️ فشل كشف النصوص: {e}")
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), result)
        continue
    
    if not text_regions:
        print("   - لا توجد نصوص في هذه الصورة.")
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), result)
        continue
    
    # 3. معالجة كل منطقة نصية
    for (bbox, text, confidence) in text_regions:
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        # تحويل bbox إلى مصفوفة نقاط (رباعية الأضلاع)
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        
        # قناع مؤقت لهذه المنطقة النصية (مستطيل تقريبي)
        temp_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(temp_mask, [pts], 255)
        
        # تحديد ما إذا كان النص داخل فقاعة
        inside = is_inside_bubble(temp_mask, bubble_masks)
        
        if inside:
            # النظام الأول: تبييض كامل داخل الفقاعة (ملء المنطقة باللون الأبيض)
            cv2.fillPoly(result, [pts], (255, 255, 255))
            print(f"   - نص '{text[:20]}...' داخل فقاعة → تبييض كامل.")
        else:
            # النظام الثاني: قناع دقيق وترميم دقيق
            refined_mask = refine_text_mask(gray, pts)
            # ترميم بنصف قطر صغير جداً للحفاظ على تفاصيل الخلفية
            result = cv2.inpaint(result, refined_mask, INPAINT_RADIUS_FREE_TEXT, cv2.INPAINT_TELEA)
            print(f"   - نص حر '{text[:20]}...' → ترميم دقيق (قطر {INPAINT_RADIUS_FREE_TEXT} px).")
    
    # حفظ النتيجة النهائية
    output_path = os.path.join(CLEAN_DIR, img_name)
    cv2.imwrite(output_path, result)
    print(f"✅ تم حفظ الصورة النظيفة: {img_name}")

print("\n🎉 اكتملت المعالجة بنجاح! جميع الصور أصبحت خالية من النصوص مع الحفاظ على تفاصيل الخلفية.")
