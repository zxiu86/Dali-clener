import os
import cv2
# الاستدعاء القطعي الصحيح من الحزمة الموحدة بعد كشف الأوراق
import mangascourx as msx
from mangascourx import MangaCleanPipeline

def main():
    input_dir = "raw_img"
    output_dir = "clean_img"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 وحش التبييض والتحرير الآلي الشامل (MangaCleanPipeline v1.0.4) بدأ بالعمل...")
    
    # تهيئة الفئة الشاملة الصحيحة بدون معاملات متضاربة لتشتغل بالـ PatchMatch التلقائي
    pipeline = MangaCleanPipeline()
    
    if not os.path.exists(input_dir):
        print(f"⚠️ المجلد '{input_dir}' غير موجود!")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not files:
        print("📥 مجلد raw_img فارغ، السيرفر ينتظر صوراً لتطهيرها.")
        return
        
    print(f"📸 تم العثور على {len(files)} صورة، جاري الطحن الاحترافي...")

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        try:
            print(f"🧼 جاري تبييض وتنظيف: {file_name} ...")
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة ملف الصورة: {file_name}")
                continue
                
            # تشغيل معجزة التبييض وسحب المصفوفات المستقرة
            result = pipeline.run(image)
            
            # استخراج الصفحة النهائية النظيفة (حسب هندسة الفئة الشاملة مالت المبرمج)
            if isinstance(result, dict) and "final_page" in result:
                cleaned_page = result["final_page"]
            else:
                cleaned_page = result # fallback إذا كانت ترجع الصورة مباشرة
            
            cv2.imwrite(output_path, cleaned_page)
            print(f"✅ تم التنظيف بنجاح والحفظ في: {output_path}")
            
        except Exception as e:
            print(f"💥 حدث خطأ أثناء معالجة الصورة {file_name}: {str(e)}")

    print("\n🎉 انتهى التنظيف بالكامل! السيرفر جاهز للـ Commit الأخضر الأسطوري!")

if __name__ == "__main__":
    main()
