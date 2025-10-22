# -*- coding:utf-8 -*-
# create: @time: 9/22/23 16:11

from PIL import Image, ImageEnhance, ImageFilter
from .image_processing_nougat import NougatImageProcessor

class NougatLaTexProcessor(NougatImageProcessor):
    def __init__(self, img_height=768, img_width=1024, enhance_contrast=True, sharpen=True, **kwargs):
        super().__init__(**kwargs)
        self.imgH = img_height
        self.imgW = img_width
        self.enhance_contrast = enhance_contrast
        self.sharpen = sharpen

    def __call__(self, images, **kwargs):
        if isinstance(images, list):
            return [self.preprocess(self._rescale(img), **kwargs) for img in images]
        else:
            return self.preprocess(self._rescale(images), **kwargs)

    def _rescale(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_width, original_height = image.size
        #print(f"Original size: {original_width}x{original_height}")

        canvasW, canvasH = self.imgW, self.imgH
        aspect_img = original_width / original_height
        aspect_canvas = canvasW / canvasH

        # Resize to match longest side
        if aspect_img > aspect_canvas:
            scale_factor = canvasW / original_width
        else:
            scale_factor = canvasH / original_height

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create canvas and paste centered image
        new_image = Image.new("RGB", (canvasW, canvasH), (255, 255, 255))
        offset_x = (canvasW - new_width) // 2
        offset_y = (canvasH - new_height) // 2
        new_image.paste(image, (offset_x, offset_y))

        # Optional contrast enhancement
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(new_image)
            new_image = enhancer.enhance(1.5)

        # ✅ Step 3: Optional sharpening for better edge clarity
        if self.sharpen:
            new_image = new_image.filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=1))

        #print(f"Processed canvas size: {new_image.width}x{new_image.height}")
        return new_image