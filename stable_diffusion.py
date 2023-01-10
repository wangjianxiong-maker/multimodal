import gradio as gr
from PIL import Image
import torch

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)

# device="cuda"
model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

pipe_text2img = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe_img2img = StableDiffusionImg2ImgPipeline(**pipe_text2img.components)


# pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(model_id).to(device)    # work
# pipe_inpaint = StableDiffusionInpaintPipeline(**pipe_text2img.components)   # not work
# def infer_text2img(prompt, guide, steps, width, height):
#     output = pipe_text2img(prompt, width=width, height=height, guidance_scale=guide, num_inference_steps=steps,)
#     image = output.images[0]
#     return image

def infer_text2img(prompt, guide, steps, width, height, image_in, strength):
    if image_in is not None:
        init_image = image_in.convert("RGB").resize((width, height))
        output = pipe_img2img(prompt, init_image=init_image, strength=strength, width=width, height=height, guidance_scale=guide, num_inference_steps=steps)
    else:
        output = pipe_text2img(prompt, width=width, height=height, guidance_scale=guide, num_inference_steps=steps,)
    image = output.images[0]
    return image

def infer_inpaint(prompt, guide, steps, width, height, image_in):
    init_image = image_in["image"].convert("RGB").resize((width, height))
    mask = image_in["mask"].convert("RGB").resize((width, height))

    output = pipe_inpaint(prompt, \
                        init_image=init_image, mask_image=mask, \
                        width=width, height=height, \
                        guidance_scale=7.5, num_inference_steps=20)
    image = output.images[0]
    return image

with gr.Blocks() as demo:
    examples = [
                ["飞流直下三千尺, 疑是银河落九天, 瀑布, 插画"],
                ["东临碣石, 以观沧海, 波涛汹涌, 插画"],
                ["孤帆远影碧空尽，惟见长江天际流,油画"],
                ["女孩背影, 日落, 唯美插画"],
                ]
    with gr.Row():
        with gr.Column(scale=1, ):
            image_out = gr.Image(label = '输出(output)')
        with gr.Column(scale=1, ):
            image_in = gr.Image(source='upload', elem_id="image_upload", type="pil", label="参考图（非必须）(ref)")
            prompt = gr.Textbox(label = '提示词(prompt)')
            submit_btn = gr.Button("生成图像(Generate)")
            with gr.Row(scale=0.5 ):
                guide = gr.Slider(2, 15, value = 7, step = 0.1, label = '文本引导强度(guidance scale)')
                steps = gr.Slider(10, 30, value = 20, step = 1, label = '迭代次数(inference steps)')
                width = gr.Slider(384, 640, value = 512, step = 64, label = '宽度(width)')
                height = gr.Slider(384, 640, value = 512, step = 64, label = '高度(height)')
                strength = gr.Slider(0, 1.0, value = 0.8, step = 0.02, label = '参考图改变程度(strength)')
                ex = gr.Examples(examples, fn=infer_text2img, inputs=[prompt, guide, steps, width, height], outputs=image_out)

        # with gr.Column(scale=1, ):
        #     image_in = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload")
        #     inpaint_prompt = gr.Textbox(label = '提示词(prompt)')
        #     inpaint_btn = gr.Button("图像编辑(Inpaint)")
            # img2img_prompt = gr.Textbox(label = '提示词(prompt)')
            # img2img_btn = gr.Button("图像编辑(Inpaint)")
        submit_btn.click(fn = infer_text2img, inputs = [prompt, guide, steps, width, height, image_in, strength], outputs = image_out)
        # inpaint_btn.click(fn = infer_inpaint, inputs = [inpaint_prompt, width, height, image_in], outputs = image_out)
        # img2img_btn.click(fn = infer_img2img, inputs = [img2img_prompt, width, height, image_in], outputs = image_out)
demo.queue(concurrency_count=1, max_size=8).launch()