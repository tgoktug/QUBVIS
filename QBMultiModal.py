from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import random
import torch
import clip
import numpy as np
from PIL import Image
import yt_dlp
import json
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import MultiHeadAttention, BatchNormalization, Dropout, Dense
from tensorflow import keras
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from VidCapModel import load_video_captioning_model  # VidCapModel.py dosyasındaki fonksiyonu import ediyoruz
import torch
import torch.nn as nn
import torch.nn.functional as F
from QBSumModel2 import load_video_summary_model
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from pytube import YouTube
from segment_selector import get_relevant_segments

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
captioning_model = load_video_captioning_model("./bml-selfattn-gpt_torch_2.pth")
whisper_model = whisper.load_model("base")



# Loading model
model2 = load_video_summary_model(weights_path='./queryselfattn_weights.weights.h5')

# **📌 Flask App**
app = Flask(__name__)   

# **📌 For saving files to static folder**
app.config['VIDEO_FOLDER'] = os.path.join(os.getcwd(), 'static', 'videos')  # Videolar burada kaydedilecek

# **📌Video Processing Functions**
# **📌 Video İşleme Fonksiyonları**
def download_youtube_video(youtube_id, output_path):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    #os.makedirs(output_path)
    video_output = os.path.join(output_path, f"{youtube_id}.mp4")
    #os.makedirs(output_path)
    print(video_output)
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst',
        'outtmpl': video_output,
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError:
        print(f"[ERROR] Video unavailable: {youtube_id}")
        return None
    return video_output if os.path.exists(video_output) else None

def extract_keyframes(video_path, keyframe_folder):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if not os.path.exists(keyframe_folder):
        os.makedirs(keyframe_folder)
    keyframe_count = total_frames // int(fps)
    for second in range(keyframe_count):
        video.set(cv2.CAP_PROP_POS_FRAMES, second * fps)
        ret, frame = video.read()
        if ret and random.random() < 0.5:
            keyframe_filename = os.path.join(keyframe_folder, f"keyframe_{second}.jpg")
            cv2.imwrite(keyframe_filename, frame)
    video.release()

def get_clip_features(keyframe_folder):
    keyframes = os.listdir(keyframe_folder)
    clip_features = []
    for keyframe in keyframes:
        keyframe_path = os.path.join(keyframe_folder, keyframe)
        image = preprocess(Image.open(keyframe_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = clip_model.encode_image(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            clip_features.append(feature.cpu().numpy().flatten())
    return clip_features

def get_query_clip_features(query):
    query = query.lower()
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_features = clip_model.encode_text(text)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        return query_features.cpu().numpy().flatten()

def pad_clip_features(clip_features, target_size=512):
    current_size = len(clip_features)
    if current_size < target_size:
        padding_size = target_size - current_size
        clip_features.extend([[0.0] * 512] * padding_size)
    return clip_features

def create_summary_video(video_path, predicted_summary, output_path, fps):
    video_clip = VideoFileClip(video_path)
    clips_to_include = []
    for second in range(len(predicted_summary)):
        if predicted_summary[second] == 1:
            start_time = second
            end_time = start_time + 1
            clip = video_clip.subclip(start_time, end_time)
            clips_to_include.append(clip)
    final_clip = concatenate_videoclips(clips_to_include)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video_clip.close()
    final_clip.close()

# **📌 Top-k sampling**
def top_k_sampling(predictions, k=1):
    predictions = predictions[:, -1, :]
    top_k_values, top_k_indices = torch.topk(predictions, k=k, dim=-1)
    top_k_probs = torch.nn.functional.softmax(top_k_values, dim=-1)
    sampled_index = torch.multinomial(top_k_probs, num_samples=1)
    return top_k_indices.gather(-1, sampled_index).squeeze()

# **📌 Caption generation**
def generate_caption(x_test_video, tokenizer, model, max_caption_length=32, top_k=1, device="cuda"):
    model.eval()
    x_test_video = torch.tensor(x_test_video, dtype=torch.float32).unsqueeze(0).to(device)  # (batch_size, num_frames, feature_size)
    start_token = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)  # (1, 1)
    
    generated_caption = []
    caption_input = start_token
    

    for _ in range(max_caption_length - 1):
        with torch.no_grad():
            predictions = model(x_test_video, caption_input)  
        next_token = top_k_sampling(predictions, k=top_k)
        if next_token == tokenizer.eos_token_id or next_token == tokenizer.pad_token_id:
            break
        generated_caption.append(next_token) 
        caption_input = torch.cat([caption_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=-1) 

    return tokenizer.decode(generated_caption, skip_special_tokens=True)  
# **📌 API Endpoints**

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video_api():
    try:
        print("Processing started...")

        # API'den gelen veriyi al
        data = request.get_json()
        video_url = data.get('video_url')
        query = data.get('query')

        if not video_url or not query:
            print("Error: Video URL or query is missing.")
            return jsonify({"error": "Video URL ve query parametreleri gereklidir."}), 400

        print(f"Video URL: {video_url}, Query: {query}")

        video_id = video_url.split('v=')[-1]
        query_safe = query.replace(" ", "_").lower()
        video_folder = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_{query_safe}_summary")
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        summary_video_path = os.path.join(app.config['VIDEO_FOLDER'], f"{query_safe}_summary.mp4")
        print(f"Summary video will be saved to: {summary_video_path}")

        video_path = download_youtube_video(video_id, video_folder)
        print(video_path)
        if not video_path:
            print("Error: Video could not be downloaded.")
            return jsonify({'error': 'Video indirilemedi.'}), 500

        print(f"Video downloaded to: {video_path}")

        keyframe_folder = os.path.join(video_folder, 'keyframes')
        extract_keyframes(video_path, keyframe_folder)
        print(f"Keyframes extracted to: {keyframe_folder}")

        clip_features = get_clip_features(keyframe_folder)
        print(f"Extracted {len(clip_features)} CLIP features.")

        clip_features = pad_clip_features(clip_features)
        query_features = get_query_clip_features(query)
        print(f"Query features extracted.")

        clip_features = np.array(clip_features).reshape(1, 512, 512)
        query_features = np.array(query_features).reshape(1, 512)

        predicted_mask = model2.predict([clip_features, query_features], verbose=0)
        print(f"Predicted mask shape: {predicted_mask.shape}")

        # Eğer predicted_mask'teki tüm değerler 0 ise, özet video oluşturulamaz
        if np.all(predicted_mask == 0):
            print("No relevant frames found for summarization.")
            return jsonify({'message': 'Alakalı bölüm bulunamadı. Video özetlenemedi.'}), 400
        print(predicted_mask)
        predicted_mask = (predicted_mask.squeeze() > 0.5).astype(np.int32)
        print(f"Predicted mask after thresholding: {predicted_mask}")

        video_data = {
            'video_id': video_id,
            'query': query,
            'key_frame_sayisi': len(clip_features[0]),
            'clip_features': clip_features.tolist(),
            'predicted_summary': predicted_mask.tolist(),
            'model_predictions': {
                'frames': predicted_mask.tolist(),
                'description': "Bu özet video, video içeriğini özetlemek için model tahminlerini kullanır."
            }
        }

        json_filename = f"{video_id}_{query_safe}_summary.json"
        json_path = os.path.join(video_folder, json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(video_data, json_file)

        create_summary_video(video_path, predicted_mask, summary_video_path, fps=30)
        print(f"Summary video created: {summary_video_path}")

        # Özet video maskesine göre CLIP özelliklerini seç
        summary_clip_features = []
        for idx, mask in enumerate(predicted_mask):
            if mask == 1:  # Maskteki 1'ler özet videoya dahil olan frame'leri gösteriyor
                summary_clip_features.append(clip_features[0][idx])  # Maskteki 1 olan her frame'in CLIP özelliğini al

        print(f"Selected {len(summary_clip_features)} CLIP features based on the mask.")

        # Eğer özet video 15 frame'den fazla ise, 15'er frame'lik dilimler oluşturulacak
        captions = []
        for i in range(0, len(summary_clip_features), 15):
            clip_chunk = summary_clip_features[i:i+15]

            # Eğer 15 frame'den az kaldıysa padding yapılacak
            if len(clip_chunk) < 15:
                clip_chunk = pad_clip_features(clip_chunk, target_size=15)  # Padding yapılacak

            # clip_chunk'un şekli (15, 512) olmalı
            print(f"clip_chunk shape: {len(clip_chunk)} frames")

            # Eğer boyut doğruysa, (15, 512) formatında modele gönderilmeli
            clip_chunk = np.array(clip_chunk).reshape(15, 512)  # [batch_size, num_frames, feature_size]

            # Caption üretme
            print(f"Generating caption for chunk {i//15 + 1} with {clip_chunk.shape[1]} frames.")
            caption = generate_caption(clip_chunk, tokenizer, captioning_model, max_caption_length=32, top_k=5, device=device)
            captions.append(caption)
            print(caption)

        print(f"Generated {len(captions)} captions.")

        result_json = {
            'message': f'Video {video_id} başarıyla indirildi, özet çıkarıldı ve caption üretildi.',
            'video_path': video_path,
            'summary_video': f'/static/videos/{query_safe}_summary.mp4',
            'captions': captions
        }

        # JSON dosyasını kaydedelim
        result_json_filename = f"{video_id}_{query_safe}_result.json"
        result_json_path = os.path.join(video_folder, result_json_filename)
        with open(result_json_path, 'w') as result_json_file:
            json.dump(result_json, result_json_file)

        return jsonify(result_json)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe2', methods=['POST'])
def transcribe_audio_backup():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        query = data.get('query')
        query_safe = query.replace(" ", "_").lower()
        video_folder = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_{query_safe}_summary")
        video_output = os.path.join(video_folder, f"{video_id}.mp4")
        if not video_id:
            return jsonify({"error": "video_id eksik"}), 400

        video_path = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}.mp4")
        if not os.path.exists(video_output):
            return jsonify({"error": "Video dosyası bulunamadı."}), 404

        result = whisper_model.transcribe(video_output, verbose=False)

        # Zaman damgalı transcriptleri JSON olarak kaydet
        transcript_output_path = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_transcript.json")
        with open(transcript_output_path, 'w') as f:
            json.dump(result['segments'], f, indent=2)

        return jsonify({
            "video_id": video_id,
            "segments": result['segments'],
            "message": "Transkript başarıyla çıkarıldı."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def transcribe_audio(video_id,query):
    try:
        if not video_id or not query:
            return jsonify({"error": "video_id ve query gereklidir."}), 400

        query_safe = query.replace(" ", "_").lower()
        video_folder = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_{query_safe}_summary")
        video_output = os.path.join(video_folder, f"{video_id}.mp4")

        if not os.path.exists(video_output):
            return jsonify({"error": "Video dosyası bulunamadı."}), 404

        # Transkript çıkar
        result = whisper_model.transcribe(video_output, verbose=False)
        segments = result['segments']

        # 🔍 Model ile segment seç
        relevant_segments = get_relevant_segments(query, segments, threshold=0.5)

        # JSON çıktısı
        output_path = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_relevant_segments.json")
        with open(output_path, 'w') as f:
            json.dump(relevant_segments, f, indent=2)

        return jsonify({
            "video_id": video_id,
            "query": query,
            "relevant_segments": relevant_segments,
            "message": "Transkript çıkarıldı ve özet segmentler seçildi."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/static/videos/<path:filename>', methods=['GET'])
def serve_video(filename):
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

@app.route('/multimodal', methods=['POST'])
def multimodal_summary():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        query = data.get('query')
        is_checkbox_checked = data.get('checkbox')

        if not video_id or not query:
            return jsonify({"error": "video_id ve query gereklidir."}), 400

        if not is_checkbox_checked:
            return process_video_api()

        query_safe = query.replace(" ", "_").lower()
        
        video_folder = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_{query_safe}_summary")
        video_path = download_youtube_video(video_id, video_folder)
        summary_json_path = os.path.join(video_folder, f"{video_id}_{query_safe}_summary.json")
        if not os.path.exists(summary_json_path):
            # Eğer summary dosyası yoksa, özetleme işlemini burada yap
            keyframe_folder = os.path.join(video_folder, 'keyframes')
            video_path = os.path.join(video_folder, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                video_path = download_youtube_video(video_id, video_folder)
                if not video_path:
                    return jsonify({"error": "Video indirilemedi."}), 500

            extract_keyframes(video_path, keyframe_folder)
            clip_features = get_clip_features(keyframe_folder)
            clip_features = pad_clip_features(clip_features)
            query_features = get_query_clip_features(query)
            clip_features_np = np.array(clip_features).reshape(1, 512, 512)
            query_features_np = np.array(query_features).reshape(1, 512)
            predicted_mask = model2.predict([clip_features_np, query_features_np], verbose=0)
            predicted_mask = (predicted_mask.squeeze() > 0.5).astype(np.int32)
            video_data = {
                'video_id': video_id,
                'query': query,
                'key_frame_sayisi': len(clip_features),
                'clip_features': clip_features_np.tolist(),
                'predicted_summary': predicted_mask.tolist(),
                'model_predictions': {
                    'frames': predicted_mask.tolist(),
                    'description': "Generated by fallback inside /multimodal endpoint."
                }
            }
        with open(summary_json_path, 'r') as f:
            summary_data = json.load(f)
            frame_mask = summary_data['predicted_summary']
        print(frame_mask)
        transcribe_audio(video_id,query)
        segments_path = os.path.join(app.config['VIDEO_FOLDER'], f"{video_id}_relevant_segments.json")
        if not os.path.exists(segments_path):
            return jsonify({"error": "Transkript segment dosyası bulunamadı."}), 404

        with open(segments_path, 'r') as f:
            segments = json.load(f)

        audio_seconds = set()
        for seg in segments:
            start = int(seg['start'])
            end = int(seg['end'])
            audio_seconds.update(range(start, end + 1))

        combined_mask = [1 if i in audio_seconds or frame_mask[i] == 1 else 0 for i in range(len(frame_mask))]

        combined_json_path = os.path.join(video_folder, f"{video_id}_{query_safe}_multimodal_summary.json")
        with open(combined_json_path, 'w') as f:
            json.dump({
                "video_id": video_id,
                "query": query,
                "multimodal_summary_mask": combined_mask
            }, f, indent=2)

        # 🔽 Eklenen işlemler: özet video oluştur, caption üret
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        summary_video_path = os.path.join(app.config['VIDEO_FOLDER'], f"{query_safe}_summary_multimodal.mp4")
        create_summary_video(video_path, combined_mask, summary_video_path, fps=30)

        # CLIP özniteliklerini kullanarak caption üret
        summary_clip_features = []
        clip_features_array = np.array(summary_data['clip_features'])[0]
        for idx, m in enumerate(combined_mask):
            if m == 1:
                summary_clip_features.append(clip_features_array[idx])

        captions = []
        for i in range(0, len(summary_clip_features), 15):
            chunk = summary_clip_features[i:i+15]
            if len(chunk) < 15:
                chunk = pad_clip_features(chunk, 15)
            chunk = np.array(chunk).reshape(15, 512)
            caption = generate_caption(chunk, tokenizer, captioning_model, max_caption_length=32, top_k=5, device=device)
            captions.append(caption)

        return jsonify({
            "video_id": video_id,
            "query": query,
            "combined_summary_mask": combined_mask,
            "summary_video": f"/static/videos/{query_safe}_summary_multimodal.mp4",
            "captions": captions,
            "message": "Multimodal summary and captions generated successfully."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
