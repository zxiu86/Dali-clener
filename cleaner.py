import os
import cv2
# 1. الاستدعاء الشامل والموحد لـ "مود الوحش" (v1.0.5 ✅)
import mangascourx.msx_all as msx

def main():
    input_dir = "raw_img"
    output_dir = "clean_img"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 تشغيل السيرفر الملوكي الموحد (msx_all v1.0.5) قيد الطحن الكاسح...")
    
    # 2. استدعاء الفئة الشاملة مباشرة من الاختصار الملوكي مالتك
    pipeline = msx.MangaCleanPipeline(
        inpainting_method="patchmatch",
        patch_size=7,
        denoise_level=3,
        whiten_background=True
    )
    
    if not os.path.exists(input_dir):
        print(f"⚠️ المجلد '{input_dir}' غير موجود! ضيف الصور يا خوي.")
        return

    # جلب ملفات الصور المدعومة بالفولدر
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not files:
        print("📥 مجلد raw_img فارغ، لا توجد صفحات لتنظيفها.")
        return
        
    print(f"📸 تم العثور على {len(files)} صورة، جاري التطهير الشامل بالـ PatchMatch...")

    # 3. المعالجة الذكية بدون أي خطأ Indentation أو مسافات زايدة
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        try:
            print(f"🧼 جاري تبييض وتنظيف: {file_name} ...")
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة ملف الصورة: {file_name}")
                continue
                
            # تشغيل المحرك الموحد
            output = pipeline.run(image)
            
            # استخراج النتيجة النهائية الصافية مباشرة حسب الـ API الجديد
            cleaned_page = output["final_page"]
            
            cv2.imwrite(output_path, cleaned_page)
            print(f"✅ تم التنظيف الاحترافي والحفظ في: {output_path}")
            
        except Exception as e:
            print(f"💥 حدث خطأ أثناء معالجة الصورة {file_name}: {str(e)}")

    print("\n🎉 تم بنجاح! السيرفر طحن الصور كلها والاختصار الشامل اشتغل طيران!")

if __name__ == "__main__":
    main()
