from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     revision="fp16",
#     torch_dtype=torch.float16,
# )
# pipe = StableDiffusionInpaintPipeline.from_ckpt("../ckpts/sd-v1-5-inpainting.ckpt")
pipe = StableDiffusionInpaintPipeline.from_single_file("./ckpts/sd-v1-5-inpainting.ckpt")
# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

image = Image.open("./data/scenery/scenery_1,058.jpg")
mask_image = Image.open("./data/tmps/scenery_1,058_mask.jpg")

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
# image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image = pipe(prompt="Expand the image", image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")
