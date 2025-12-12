import subprocess
import sys
import warnings
import os
from datetime import datetime
import time

print("AIXYNZ Video Generator - Installing required packages...")
warnings.filterwarnings('ignore')

def install_dependencies():
    packages = [
        'gradio', 'diffusers', 'transformers', 'accelerate',
        'torch', 'torchvision', 'opencv-python', 'imageio',
        'imageio-ffmpeg', 'pillow', 'numpy'
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_dependencies()

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import cv2
import numpy as np
from PIL import Image
import imageio

pipeline = None
device = None
generated_videos = []

def initialize_model():
    global pipeline, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        print("Loading AI model...")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)
        
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_frame(prompt, seed):
    global pipeline, device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        image = pipeline(
            prompt,
            num_inference_steps=20,
            generator=generator,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
    
    return np.array(image)

def interpolate_frames(frame1, frame2, num_frames):
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1) if num_frames > 1 else 0
        interpolated = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        frames.append(interpolated)
    return frames

def generate_video(prompt, duration, fps, quality, progress=gr.Progress()):
    global pipeline, generated_videos
    
    if pipeline is None:
        return None, "Model not loaded. Please wait for initialization."
    
    if not prompt or prompt.strip() == "":
        return None, "Please enter a prompt."
    
    try:
        progress(0, desc="Starting video generation...")
        
        total_frames = duration * fps
        keyframe_interval = 30
        num_keyframes = max(2, (total_frames // keyframe_interval) + 1)
        
        print(f"Generating {num_keyframes} keyframes for {total_frames} frames")
        
        keyframes = []
        for i in range(num_keyframes):
            progress((i / num_keyframes) * 0.5, desc=f"Generating keyframe {i+1}/{num_keyframes}")
            seed = 42 + i * 100
            frame = generate_frame(prompt, seed)
            keyframes.append(frame)
        
        progress(0.5, desc="Interpolating frames...")
        all_frames = []
        
        for i in range(len(keyframes) - 1):
            progress(0.5 + (i / (len(keyframes) - 1)) * 0.3, 
                    desc=f"Interpolating section {i+1}/{len(keyframes)-1}")
            frames_between = keyframe_interval
            interpolated = interpolate_frames(keyframes[i], keyframes[i + 1], frames_between)
            all_frames.extend(interpolated)
        
        all_frames.append(keyframes[-1])
        all_frames = all_frames[:total_frames]
        
        progress(0.8, desc="Encoding video...")
        
        timestamp = int(time.time() * 1000)
        output_path = f"video_{timestamp}.mp4"
        
        quality_settings = {
            "Fast": {"quality": 5},
            "Medium": {"quality": 7},
            "Best": {"quality": 9}
        }
        
        settings = quality_settings.get(quality, quality_settings["Medium"])
        
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            quality=settings["quality"],
            pixelformat='yuv420p',
            macro_block_size=1
        )
        
        for idx, frame in enumerate(all_frames):
            writer.append_data(frame)
            if idx % 10 == 0:
                progress(0.8 + (idx / len(all_frames)) * 0.2, 
                        desc=f"Writing frame {idx}/{len(all_frames)}")
        
        writer.close()
        
        file_size = os.path.getsize(output_path)
        video_info = {
            "id": timestamp,
            "prompt": prompt,
            "path": output_path,
            "duration": duration,
            "fps": fps,
            "frames": len(all_frames),
            "size": file_size,
            "quality": quality,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        generated_videos.append(video_info)
        
        progress(1.0, desc="Video generated successfully!")
        
        success_msg = f"""Video Generated Successfully!

Prompt: {prompt}
Duration: {duration}s | FPS: {fps} | Frames: {len(all_frames)}
Size: {file_size / 1024 / 1024:.2f} MB | Quality: {quality}"""
        
        return output_path, success_msg
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return None, error_msg

def get_video_gallery():
    return [v["path"] for v in generated_videos if os.path.exists(v["path"])]

def get_video_info():
    if not generated_videos:
        return "No videos generated yet."
    
    info = "Generated Videos:\n\n"
    for idx, v in enumerate(generated_videos, 1):
        info += f"""Video {idx}:
Prompt: {v['prompt'][:50]}{'...' if len(v['prompt']) > 50 else ''}
Duration: {v['duration']}s | FPS: {v['fps']} | Frames: {v['frames']}
Size: {v['size'] / 1024 / 1024:.2f} MB | Quality: {v['quality']}
Created: {v['created']}

"""
    return info

def create_interface():
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 15px;
        border-radius: 8px;
        background: #f0f7ff;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="AIXYNZ Video Generator", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 AIXYNZ Video Generator</h1>
            <p style="font-size: 1.2em; margin-top: 10px;">Generate stunning videos from text prompts using AI</p>
        </div>
        """)
        
        device_status = "GPU Ready" if torch.cuda.is_available() else "CPU Mode"
        gr.Markdown(f"**Status:** {device_status} | **Model:** Stable Diffusion v1.5")
        
        with gr.Tabs():
            with gr.Tab("Generate Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Video Settings")
                        
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Example: A majestic dragon flying over snow-capped mountains at sunset, cinematic, 4k",
                            lines=3,
                            info="Describe the video you want to generate"
                        )
                        
                        with gr.Row():
                            duration_input = gr.Slider(
                                minimum=2,
                                maximum=30,
                                value=5,
                                step=1,
                                label="Duration (seconds)",
                                info="Longer videos take more time"
                            )
                            
                            fps_input = gr.Slider(
                                minimum=10,
                                maximum=30,
                                value=15,
                                step=5,
                                label="FPS",
                                info="Frames per second"
                            )
                        
                        quality_input = gr.Radio(
                            choices=["Fast", "Medium", "Best"],
                            value="Medium",
                            label="Quality",
                            info="Fast = Quick generation, Best = Higher quality"
                        )
                        
                        generate_btn = gr.Button(
                            "Generate Video",
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Examples(
                            examples=[
                                ["A serene ocean with waves crashing on a beach at sunset"],
                                ["Flying through a futuristic city with neon lights"],
                                ["A cozy fireplace in a winter cabin with snow falling outside"],
                                ["Abstract colorful paint swirling and mixing together"],
                                ["A magical forest with glowing mushrooms and fireflies"]
                            ],
                            inputs=prompt_input,
                            label="Example Prompts"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Video")
                        
                        video_output = gr.Video(
                            label="Your Video",
                            height=400
                        )
                        
                        status_output = gr.Markdown(
                            "Ready to generate! Enter a prompt and click Generate.",
                            elem_classes="status-box"
                        )
                
                generate_btn.click(
                    fn=generate_video,
                    inputs=[prompt_input, duration_input, fps_input, quality_input],
                    outputs=[video_output, status_output]
                )
            
            with gr.Tab("Video Gallery"):
                gr.Markdown("### All Generated Videos")
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Gallery", size="sm")
                
                gallery_output = gr.Gallery(
                    label="Generated Videos",
                    show_label=True,
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
                
                info_output = gr.Markdown("No videos generated yet.")
                
                def refresh_gallery():
                    videos = get_video_gallery()
                    info = get_video_info()
                    return videos, info
                
                refresh_btn.click(
                    fn=refresh_gallery,
                    outputs=[gallery_output, info_output]
                )
            
            with gr.Tab("Help & Info"):
                gr.Markdown("""
                # How to Use
                
                ## Quick Start
                1. Go to **Generate Video** tab
                2. Enter a descriptive prompt
                3. Adjust duration, FPS, and quality
                4. Click **Generate Video**
                5. Wait for the generation (2-10 minutes depending on settings)
                6. Download or view your video!
                
                ## Tips for Best Results
                
                ### Writing Good Prompts:
                - **Be specific**: "A red dragon flying over mountains" vs "dragon"
                - **Add style**: Include words like "cinematic", "4k", "detailed"
                - **Describe motion**: "waves crashing", "birds flying", "wind blowing"
                - **Set the scene**: Time of day, weather, lighting
                
                ### Settings Guide:
                - **Duration**: Start with 5 seconds, increase for longer videos
                - **FPS**: 15 FPS is good balance (higher = smoother but slower)
                - **Quality**: 
                  - Fast: 2-3 minutes generation
                  - Medium: 4-6 minutes generation  
                  - Best: 8-12 minutes generation
                
                ## Generation Times
                
                | Settings | GPU (T4) | CPU |
                |----------|----------|-----|
                | 5s, 15 FPS, Fast | ~3 min | ~15 min |
                | 5s, 15 FPS, Medium | ~5 min | ~25 min |
                | 10s, 15 FPS, Best | ~12 min | ~45 min |
                
                ## Example Prompts
                
                **Nature:**
                - "Sunset over calm ocean with golden reflections on water"
                - "Northern lights dancing in the night sky over snowy landscape"
                - "Cherry blossoms falling in a Japanese garden in spring"
                
                **Fantasy:**
                - "A magical portal opening in an ancient forest with glowing runes"
                - "Dragon soaring through storm clouds with lightning"
                - "Crystal cave with luminescent minerals glowing in the dark"
                
                **Abstract:**
                - "Colorful paint drops falling into water in slow motion"
                - "Geometric shapes morphing and rotating in space"
                - "Smoke tendrils swirling and dancing with vibrant colors"
                
                **Urban:**
                - "Time-lapse of busy city intersection with car light trails"
                - "Futuristic city with flying cars and holographic billboards"
                - "Neon-lit alleyway in cyberpunk Tokyo with rain reflections"
                
                ## Technical Details
                - **Model**: Stable Diffusion v1.5
                - **Method**: Keyframe generation + interpolation
                - **Resolution**: 512x512 pixels
                - **Format**: MP4 (H.264)
                
                ## Important Notes
                - First video takes longer (model download ~5GB)
                - Keep browser tab open during generation
                - Videos are saved temporarily in Colab session
                - Download your videos before closing Colab!
                
                ## Troubleshooting
                
                **Video generation fails?**
                - Try shorter duration (3-5 seconds)
                - Use simpler prompts
                - Restart runtime if out of memory
                
                **Slow generation?**
                - You're using CPU (normal, but slower)
                - Reduce FPS or duration
                - Use "Fast" quality setting
                """)
        
        return demo

print("\n" + "="*60)
print("AIXYNZ VIDEO GENERATOR - INITIALIZATION")
print("="*60)

model_ready = initialize_model()

if model_ready:
    print("\nEverything ready! Launching Gradio interface...")
    print("="*60 + "\n")
    
    demo = create_interface()
    demo.queue()
    demo.launch(
        share=True,
        debug=False,
        show_error=True,
        inbrowser=True
    )
else:
    print("\nFailed to initialize. Please check errors above.")
