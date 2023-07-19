import cv2
import tensorflow as tf
import numpy as np
import imageio
import subprocess
import os   
from utils.image_utils import get_image
from train import Trainer

class StyleTransfer:
    def __init__(self, style_path):
        self.trainer = Trainer(
            style_path=style_path, 
            content_file_path="/home/max/Downloads/train2017", 
            epochs=2, 
            batch_size=8,
            content_weight=1e0,
            style_weight=4e1,
            tv_weight=2e2,
            learning_rate=1e-3,
            log_period=100,
            save_period=1000,
            content_layers=["conv4_2"],
            style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
            content_layer_weights=[1],
            style_layer_weights=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

    
    def _preprocess(self, frame):
        img = tf.convert_to_tensor(frame, dtype=tf.float32)
        img = tf.expand_dims(img, 0)
        return img
    
    def practice(self):
        self.trainer.run()

    def _postprocess(self, img_tensor):
        img_tensor = tf.clip_by_value(img_tensor, 0, 255)
        img = tf.squeeze(img_tensor).numpy().astype(np.uint8)
        return img

    def apply_to_image(self, image_path, output_path):
        img = get_image(image_path)
        img_tensor = self._preprocess(img)
        res = self.trainer.transform.model(img_tensor)
        res = self._postprocess(res)
        imageio.imwrite(output_path, res)

    def apply_to_video(self, input_video_path, output_video_path):
        video = cv2.VideoCapture(input_video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self._preprocess(frame)
            output_tensor = self.trainer.transform.model(input_tensor)
            output_frame = self._postprocess(output_tensor)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            out.write(output_frame)

        video.release()
        out.release()

    def apply_to_video_with_sound(self, video_path: str, output_path: str):
        temp_output = "temp_output.mp4"
        self.apply_to_video(video_path, temp_output)

        audio_file = "audio.aac"
        # Extract audio
        subprocess.call(["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", audio_file])

        # Merge audio and video
        subprocess.call(["ffmpeg", "-i", temp_output, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_path])

        # Delete temporary files
        os.remove(temp_output)
        os.remove(audio_file)
    
    
### Usage Example
if __name__ == "__main__":
    painting = 'starry-night'
    style_transfer = StyleTransfer(f"dataset/styles/{painting}.jpg")
    # style_transfer.practice()
    # style_transfer.apply_to_image("dataset/inferences/profile.jpeg", f"{painting}_profile.jpg")
    style_transfer.apply_to_video_with_sound("dataset/inferences/beach.mp4", f"{painting}_beach.mp4")
