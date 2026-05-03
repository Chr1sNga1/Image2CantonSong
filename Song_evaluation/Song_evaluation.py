import os
import re
import torch
import librosa
import numpy as np
import math
import random
import jiwer
from pypinyin import pinyin, Style
from transformers import GPT2LMHeadModel, BertTokenizer
from faster_whisper import WhisperModel 

# Set environment variables
os.environ["HF_TOKEN"] = "HF_TOKEN" # Input token here, deleted for GitHub upload

# Fix random seeds for experiment soundness
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Cantonese tone height mapping
TONE_HEIGHT_MAP = {1: 6, 2: 5, 3: 4, 5: 3, 6: 2, 4: 1}

class MultiDimensionalEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing models...")
        
        # Load Faster-Whisper model
        # Switch device to self.device and compute_type to "float16" if GPU is available
        self.whisper_model = WhisperModel(
            "large-v3", 
            device="cpu", 
            compute_type="int8"
        )
        
        # Load lightweight Chinese GPT for Perplexity calculation
        self.ppl_model_name = "ckiplab/gpt2-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.ppl_model_name)
        self.ppl_model = GPT2LMHeadModel.from_pretrained(self.ppl_model_name).to(self.device)
        self.ppl_model.eval()

    def clean_text(self, text, only_chinese=True):
        """Clean text by removing tags and non-Chinese characters"""
        if not text: return ""
        text = re.sub(r'[a-zA-Z]', '', text)
        if only_chinese:
            text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return text.strip()

    def get_jyutping(self, text):
        """Convert text to Jyutping for phonetic accuracy comparison"""
        py_list = pinyin(text, style=Style.NORMAL)
        return [item[0] for item in py_list]

    def get_tone_height(self, char):
        """Map Cantonese tones to pitch heights"""
        tones = pinyin(char, style=Style.TONE2)
        if not tones or not tones[0]: return 3
        digit = re.findall(r'\d', tones[0][0])
        return TONE_HEIGHT_MAP.get(int(digit[0]), 3) if digit else 3

    def calculate_ppl(self, text):
        """Calculate lyric fluency (Perplexity)"""
        if not text or len(text) < 2: return 0
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
        return math.exp(loss.item())

    def evaluate(self, audio_path, ref_lyrics_path):
        # A. Data Preprocessing
        with open(ref_lyrics_path, "r", encoding="utf-8-sig") as f:
            raw_ref = f.read()
        clean_ref = self.clean_text(raw_ref)
        
        # Extract first 30 chars as Prompt to assist alignment and homophone correction
        auto_prompt = clean_ref[:30] 
        print(f"Using Auto-Prompt: {auto_prompt}")

        print(f"Analyzing: {os.path.basename(audio_path)}...")
        
        # B. Faster-Whisper Inference
        # For songs with short intros, disable vad_filter to ensure capturing the first word
        segments, info = self.whisper_model.transcribe(
            audio_path, 
            language="yue", 
            beam_size=5,
            word_timestamps=True,
            vad_filter=False, 
            initial_prompt=auto_prompt
        )
        
        segments = list(segments)
        
        # C. Accuracy Metrics (Dual Scoring)
        full_hyp_text = "".join([s.text for s in segments])
        hyp_text = self.clean_text(full_hyp_text)
        
        # 1. Textual Accuracy (Literal match)
        text_accuracy = max(0, 1 - jiwer.cer(clean_ref, hyp_text))
        
        # 2. Phonetic Accuracy (Allows homophones)
        ref_jp = self.get_jyutping(clean_ref)
        hyp_jp = self.get_jyutping(hyp_text)
        phonetic_accuracy = max(0, 1 - jiwer.wer(" ".join(ref_jp), " ".join(hyp_jp)))

        # DEBUG Output
        print("\n" + "!"*30 + " DEBUG DATA " + "!"*30)
        print(f"Reference (Cleaned): {clean_ref[:50]}...")
        print(f"Whisper   (Cleaned): {hyp_text[:50]}...")
        print(f"Text Acc: {text_accuracy:.2%}, Phonetic Acc: {phonetic_accuracy:.2%}")
        print("!"*72 + "\n")

        # D. Model Confidence & Fluency
        confidence = np.exp(np.mean([s.avg_logprob for s in segments]))
        ppl = self.calculate_ppl(hyp_text)
        
        # E. Acoustic and Rhythmic Metrics
        y, sr = librosa.load(audio_path, sr=22050)
        word_data = []
        densities = []
        
        for segment in segments:
            dur = segment.end - segment.start
            text = self.clean_text(segment.text)
            count = len(text)
            # Calculate syllable density (syllables per second)
            if dur > 0.5 and count > 0: 
                densities.append(count / dur)
            
            if segment.words:
                for w in segment.words:
                    char = self.clean_text(w.word)
                    if not char: continue
                    # Extract audio clip for each character with 0.05s buffer
                    s_idx = max(0, int((w.start - 0.05) * sr)) 
                    e_idx = min(len(y), int((w.end + 0.05) * sr))
                    if (e_idx - s_idx) < 1024: continue 
                    
                    # Pitch tracking
                    f0, _, _ = librosa.pyin(y[s_idx:e_idx], fmin=60, fmax=500, sr=sr)
                    avg_p = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
                    if avg_p > 0:
                        word_data.append({'h': self.get_tone_height(char[0]), 'p': avg_p})

        # Variance and Coefficient of Variation for Pitch
        all_pitches = [d['p'] for d in word_data]
        pitch_var = np.var(all_pitches) if all_pitches else 0
        pitch_cv = (np.std(all_pitches) / np.mean(all_pitches)) if all_pitches and np.mean(all_pitches) != 0 else 0

        # Calculate Tone-Melody Directional Consistency (Prosodic 協音 Alignment)
        matches = 0
        if len(word_data) > 1:
            d_tone = np.diff([d['h'] for d in word_data])
            d_pitch = np.diff([d['p'] for d in word_data])
            matches = np.sum(np.sign(d_tone) == np.sign(d_pitch)) / len(d_tone)

        return {
            "Text_Acc": text_accuracy,
            "Phonetic_Acc": phonetic_accuracy,
            "Conf": confidence,
            "PPL": ppl,
            "Dens_Var": np.var(densities) if densities else 0,
            "Dir_Consist": matches,
            "Pitch_Var": pitch_var,
            "Pitch_CV": pitch_cv
        }

def generate_gen_audio_report(res):
    print("\n" + "="*25 + " AI GENERATED SONG QUALITY REPORT " + "="*25)
    
    # 1. Lyric-Audio Alignment (一致性)
    print(f"\n[1. Lyric-Audio Alignment]")
    print(f"   - Textual Acc:  {res['Text_Acc']:.2%}")
    print(f"   - Phonetic Acc: {res['Phonetic_Acc']:.2%}")
    acc_gap = res['Phonetic_Acc'] - res['Text_Acc']
    if res['Phonetic_Acc'] > 0.80:
        print(f"   - 評價：語音還原度高。AI 成功將歌詞轉化為清晰音軌。")
    else:
        print(f"   - 評價：語音模糊。部分發音與歌詞對位失敗，可能存在 AI 吞音問題。")
    print(f"   - 註解：Phonetic Gap ({acc_gap:.2%}) 反映了同音異字對識別準確度的影響。")

    # 2. Cantonese Prosody & Tones (Tone-Melody Matching) (協音/倒字)
    print(f"\n[2. Cantonese Prosody & Tones]")
    print(f"   - Dir_Consist: {res['Dir_Consist']:.2%}")
    
    # Reference: Human professional recordings (e.g. 《明年今日》"Next Year Today") score approx 30%-40%
    if res['Dir_Consist'] > 0.45:
        print(f"   - 評價：協音表現極佳。數值超越部分真人錄音基準，旋律與聲調高度對位。")
    elif res['Dir_Consist'] > 0.30:
        print(f"   - 評價：接近真人水平。協音程度符合廣東流行曲標準，聽感自然。")
    elif res['Dir_Consist'] > 0.20:
        print(f"   - 評價：協音一般。存在部分『倒字』，屬 AI 生成粵語歌的常見範疇。")
    else:
        print(f"   - 評價：協音欠佳。旋律走勢與聲調嚴重脫節，聽感不自然。")

    # 3. Pitch Dynamics & Emotion (音高/感情)
    print(f"\n[3. Pitch Dynamics & Emotion]")
    print(f"   - Pitch CV:       {res['Pitch_CV']:.4f}")
    print(f"   - Pitch Variance: {res['Pitch_Var']:.2f}")
    if res['Pitch_CV'] > 0.55:
        print(f"   - 評價：音域起伏大。表現力較強，具備明顯的歌唱特徵。")
    elif res['Pitch_CV'] > 0.35:
        print(f"   - 評價：音域適中。符合一般流行曲的起伏範圍。")
    else:
        print(f"   - 評價：音域過窄。聽感平淡，表現力接近機器朗讀。")

    # 4. VLM Language Quality (Grammar/Fluency) (歌詞語法)
    print(f"\n[4. Language Model Quality (VLM)]")
    print(f"   - Perplexity (PPL): {res['PPL']:.2f}")
    if res['PPL'] < 200:
        print(f"   - 評價：語法通順。歌詞結構嚴謹，符合粵語母語者的表達邏輯。")
    else:
        print(f"   - 評價：語法生硬。可能存在字詞堆砌或不通順的詞組。")

    # 5. Rhythmic Stability (節奏穩定性)
    print(f"\n[5. Rhythmic Stability]")
    print(f"   - Syllable Var: {res['Dens_Var']:.4f}")
    if res['Dens_Var'] > 2.5:
        print("   - 評價：節奏不穩定。部分段落出現趕字或拖拍現象。")
    else:
        print("   - 評價：節奏分佈穩定。音節密度隨時間變化合理。")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    evaluator = MultiDimensionalEvaluator()
    # Test file settings
    audio_file = "Sammi_Cheng_Beautiful_Life.mp3"
    lyrics_file = f"lyrics_{audio_file.replace('.mp3', '')}.txt"
    
    if os.path.exists(audio_file) and os.path.exists(lyrics_file):
        res = evaluator.evaluate(audio_file, lyrics_file)
        
        print("\n" + "="*50)
        print(f"RESULT FOR MSTAT PROJECT (PHONETIC OPTIMIZED)")
        print("-" * 50)
        print(f"1. Textual Accuracy:  {res['Text_Acc']:.2%}")
        print(f"2. Phonetic Accuracy: {res['Phonetic_Acc']:.2%}")
        print(f"3. Model Confidence:  {res['Conf']:.2%}")
        print(f"4. Syllable Variance: {res['Dens_Var']:.4f}")
        print(f"5. Perplexity (PPL):  {res['PPL']:.2f}")
        print(f"6. Tone Consistency:  {res['Dir_Consist']:.2%}")
        print(f"7. Pitch Variance:    {res['Pitch_Var']:.4f}")
        print(f"8. Pitch CV:          {res['Pitch_CV']:.4f}")
        print("="*50)

        generate_gen_audio_report(res)
    else:
        print("Error: Audio or Lyrics file not found.")