import os
import cv2
import mangascourx as msx
from mangascourx import TextRemovePipeline

def main():
    input_dir = "raw_img"
    output_dir = "clean_img"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 وحش التبييض والتحرير الآلي (MangaScourX v1.0.4) قيد التشغيل الملوكي...")
    import inspect
    print("🔍 الأسماء المقبولة بداخل الـ Pipeline هي:", inspect.signature(TextRemovePipeline.__init__))

    pipeline = TextRemovePipeline(
        inpainter_type="patchmatch",
        enable_bubbles=True,
        verbose=True
    )
    
    if not os.path.exists(input_dir):
        print(f"⚠️ المجلد '{input_dir}' غير موجود!")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not files:
        print("📥 مجلد raw_img فارغ، السيرفر ينتظر صوراً لتطهيرها.")
        return
        
    print(f"📸 تم العثور على {len(files)} صورة، جاري طحن النصوص...")

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        # السطر 64 المصحح والمحاذي بالملي داخل حلقة الـ for ودالة الـ try
        try:
            print(f"🧼 جاري تبييض وتنظيف: {file_name} ...")
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة ملف الصورة: {file_name}")
                continue
                
            result = pipeline.run(image)
            cv2.imwrite(output_path, result)
            print(f"✅ تم التنظيف الاحترافي والحفظ في: {output_path}")
            
        except Exception as e:
            print(f"💥 حدث خطأ أثناء معالجة الصورة {file_name}: {str(e)}")

    print("\n🎉 تم الانتهاء من تطهير جميع الصور بنجاح!")

if __name__ == "__main__":
    main()
