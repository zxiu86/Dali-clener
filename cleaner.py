import os
import cv2
# 1. استدعاء الحزمة الرسمية الموحدة (V1.0.4 المستقرة ✅)
import mangascourx as msx
from mangascourx import TextRemovePipeline

def main():
    # مجلد المدخلات والمخرجات حسب طلبك
    input_dir = "raw_img"
    output_dir = "clean_img"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 وحش التبييض والتحرير الآلي (MangaScourX v1.0.4) قيد التشغيل الملوكي...")
    
    # 2. بناء خط الأنابيب حسب الدليل الفريش اللي نسخته بيدك
    pipeline = TextRemovePipeline(
        inpainter_type="patchmatch",  # تشغيل خوارزمية الـ PatchMatch الأسطورية
        enable_bubbles=True,          # تفعيل كشف فقاعات الكلام والكنتور
        verbose=True                  # طباعة تفاصيل المعالجة والمصفوفات
    )
    
    if not os.path.exists(input_dir):
        print(f"⚠️ المجلد '{input_dir}' غير موجود! ضيف الصور بداخله أولاً.")
        return

    # جلب جميع الصور المدعومة بالفولدر
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not files:
        print("📥 مجلد raw_img فارغ، السيرفر ينتظر صوراً لتطهيرها.")
        return
        
    print(f"📸 تم العثور على {len(files)} صورة، جاري طحن النصوص وتبييض الخلفيات...")

    # 3. الدوران على الصور ومعالجتها
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"🧼 جاري تبييض وتنظيف: {file_name} ...")
        
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة ملف الصورة: {file_name}")
                continue
                
            # 4. معالجة الصورة واستخراج النتيجة النهائية الصافية مباشرة
            result = pipeline.run(image)
            
            # حفظ النتيجة النظيفة في مجلد clean_img عشان الـ Actions يسحبها تلقائياً
            cv2.imwrite(output_path, result)
            print(f"✅ تم التنظيف الاحترافي والحفظ في: {output_path}")
            
        except Exception as e:
            print(f"💥 حدث خطأ غير متوقع أثناء معالجة الصورة {file_name}: {str(e)}")

    print("\n🎉 تم الانتهاء من تطهير جميع الصور بنجاح! السيرفر جاهز للـ Commit والـ Push.")

if __name__ == "__main__":
    main()
        
        print(f"🧼 جاري تبييض: {file_name} ...")
        
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة: {file_name}")
                continue
                
            result = pipeline.run(image)
            cleaned_page = result["final_page"]
            cv2.imwrite(output_path, cleaned_page)
            print(f"✅ تم الحفظ: {output_path}")
            
        except Exception as e:
            print(f"💥 خطأ في {file_name}: {str(e)}")

    print("🎉 تم الانتهاء!")

if __name__ == "__main__":
    main()
