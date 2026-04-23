quit
esc
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
gpu-interactive 
exit
conda create -n my_env python=3.11
~/anaconda3/bin/conda init
conda create -n my_env python=3.11
conda init
~/anaconda3/bin/conda init
conda create -n my_env python=3.11
conda init
~/anaconda3/bin/conda init
conda create -n my_env python=3.11
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
conda create -n canton_GPU  python=3.11
conda activate canton_GPU
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
exit
hostname -I 
cd
ls
conda activate canton_GPU
unzip canto_demo_v19_mmllm_auto_yue.zip
ls
cd canto_demo_v19_mmllm_auto_yue/
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; import huggingface_hub; print('ok')"
git --version
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME
ls
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
hostname -I 
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
exit
gpu-interactive 
exit
ls
rm canto_demo_v19_mmllm_auto_yue
rm canto_demo_v19_mmllm_auto_yue.zip 
ls
exit
gpu-interactive 
exit 
exit
ls
unzip canto_demo_v20_no_fallback.zip 
cd canto_demo_v20_no_fallback/
stream run app.py
pip install -r requirements.txt
conda activate canton_GPU
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
hostname -I
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
exit
gpu-interactive 
exit
ls
rm canto_demo_v19_mmllm_auto_yue/
rm canto_demo_v20_no_fallback.zip 
ls
unzip canto_demo_v20_1_debug.zip 
cd canto_demo_v20_1_debug/
conda activate canton_GPU
pip install -r requirements.txt 
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
nano /userhome/cs5/u3665806/canto_demo_v20_1_debug/modules/mm_direct_gen.py
nano /userhome/cs5/u3665806/canto_demo_v20_1_debug/app.py
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
exit
gpu-interactive 
exit
ls
rm canto_demo_v20_1_debug.zip 
ls
rmdir canto_demo_v19_mmllm_auto_yue/
rm -r canto_demo_v19_mmllm_auto_yue/
ls
rm -r canto_demo_v20_no_fallback/
ls
rmdir canto_demo_v20_no_fallback/
rm -r canto_demo_v20_no_fallback/
ls
cd canto_demo_v20_1_debug/
nano /userhome/cs5/u3665806/canto_demo_v20_1_debug/modules/mm_direct_gen.py
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
exit
gpu-interactive 
exit
streamlit run app.py
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
cd canto_demo_v20_1_debug/
streamlit run app.py
exit
gpu-interactive 
exit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
cd canto_demo_v20_1_debug/
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
streamlit run app.py --server.port 8501
exit
gpu-interactive 
exit
cd canto_demo_v20_1_debug/
ls
nano /userhome/cs5/u3665806/canto_demo_v20_1_debug/modules/mm_direct_gen.py
conda activate canton_GPU
nvdia smi
nvdia-smi
exit
gpu-interactive 
pip install flash-attn --no-build-isolation
conda activate canton_GPU
pip install flash-attn --no-build-isolation
nvdia-smi
exit
nvdia-smi
nvidia-smi
exit
conda activate canton_GPU
streamlit run app.py --server.port 8501
cd canto_demo_v20_1_debug/
streamlit run app.py --server.port 8501
exit
conda activate canton_GPU
gpu-interactive 
exit
gpu-interactive 
ps -u $USER
pkill -u $USER
conda activate canton_GPU
exit
nvdia-smi
nvidia-smi
conda activate canton_GPU
nvidia-smi
gpu-interactive 
nvidia-smi
exit
ps -u $USER
gpu-interactive 
conda activate canton_GPU
unzip levo2_demo_v1.zip -d levo2_demo_v1
ls
unzip levo2_demo_v1 -d levo2_demo_v1
ls
zip canto_demo_v20_1_debug/
python - <<'PY'
import zipfile
p = "levo2_demo_v1.zip"
print("is_zip:", zipfile.is_zipfile(p))
PY

rm -f levo2_demo_v1.zip 
ls
exit
ls
rmdir New\ Folder/
ls
unzip levo2_demo_v1.zip 
conda activate canton_GPU
unzip levo2_demo_v1.zip 
unzip levo2_demo_v1.zip -d levo2_demo_v1
gpu-interactive 
exit
nvidia-smi
conda activate canton_GPU
ls
unzip levo2_demo_v1.zip
ls
unzip canto_demo_v21_yue_aligned.zip 
ls
rm canto_demo_v21_yue_aligned.zip 
rm levo2_demo_v1
rm levo2_demo_v1.zip 
ls
cd levo2_demo_v1/
ls
streamlit run app.py --server.port 8501
cd ~
ls
cd canto_demo_v21_yue_aligned/
ls
pip install -r  requirements.txt
streamlit run app.py --server.port 8501
python - <<'PY'
from pathlib import Path

p = Path("/userhome/cs5/u3665806/canto_demo_v21_yue_aligned/.cache_yue_runtime/YuE/inference/xcodec_mini_infer/post_process_audio.py")
txt = p.read_text(encoding="utf-8", errors="replace")

txt = txt.replace("import numpy as np.functional as F", "import torchaudio.functional as F")

if "import numpy as np" not in txt:
    if "import soundfile as sf" in txt:
        txt = txt.replace("import soundfile as sf", "import soundfile as sf\nimport numpy as np")
    else:
        txt = "import numpy as np\n" + txt

p.write_text(txt, encoding="utf-8")
print("patched:", p)
PY

python -m py_compile /userhome/cs5/u3665806/canto_demo_v21_yue_aligned/.cache_yue_runtime/YuE/inference/xcodec_mini_infer/post_process_audio.py
python - <<'PY'
from pathlib import Path

p = Path("/userhome/cs5/u3665806/canto_demo_v21_yue_aligned/.cache_yue_runtime/YuE/inference/xcodec_mini_infer/post_process_audio.py")
txt = p.read_text(encoding="utf-8", errors="replace")

txt = txt.replace("import numpy as np.functional as F", "import torchaudio.functional as F")

if "import numpy as np" not in txt:
    if "import soundfile as sf" in txt:
        txt = txt.replace("import soundfile as sf", "import soundfile as sf\nimport numpy as np")
    else:
        txt = "import numpy as np\n" + txt

p.write_text(txt, encoding="utf-8")
print("patched:", p)
PY

eixt
exit
ls
conda activate canton_GPU
gpu-interactive 
exit
conda activate yue_clean
unzip canto_project_clean_yue_demo.zip 
rm canto_project_clean_yue_demo.zip 
cd canto_project_clean_yue_demo/
ls
pip install -r  requirements.txt
exit
conda create -n yue_clean python=3.8 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_clean
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
pip install -r <(curl -sSL https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/requirements.txt)
ls
unzip canto_project_clean_yue_demo.zip 
gpu-interactive 
exit
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n yue_official python=3.8 -y
conda activate yue_official
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
pip install -r <(curl -sSL https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/requirements.txt)
pip install flash-attn --no-build-isolation
cd ~
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE/inference
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
cd ~
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE/inference
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
mkdir -p ~/yue_test_inputs
cat > ~/yue_test_inputs/genre.txt <<'EOF'
Cantopop female uplifting pop airy vocal bright vocal Cantonese
EOF

cat > ~/yue_test_inputs/lyrics.txt <<'EOF'
[verse]
霓虹落在長街轉角
夜風吹起舊日承諾
你的背影仍留心幕
一步一步使我失落

[chorus]
今晚星光照不穿寂寞
我仍站在原地等結果
若然緣分終於都飄泊
仍願為你高聲唱這首歌
EOF

cd ~/YuE/inference
python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_cot   --max_new_tokens 600   --repetition_penalty 1.05
cd ~/YuE/inference
python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_cot   --max_new_tokens 600   --repetition_penalty 1.05
ps
exit
gpu-interactive 
exit
nvidia-smi
gpu-interactive 
nvidia-smi
cexit
exit
hostname -I
exit
nvidia-smi
conda activate canto_GPU
nvidia-smi
gpu-interactive 
exit
nvidia-smi
exit
gpu-interactive 
gpu-interactive 
hostname -I
nvidia-smi
exit
nvidia-smi
hostname -I
source ~/anaconda3/etc/profile.d/conda.sh && \ conda activate yue_official && \ cd ~/YuE/inference && \ python infer.py \ --cuda_idx 0 \ --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot \ --stage2_model m-a-p/YuE-s2-1B-general \ --genre_txt ~/yue_test_inputs/genre.txt \ --lyrics_txt ~/yue_test_inputs/lyrics.txt \ --run_n_segments 2 \ --stage2_batch_size 1 \ --output_dir ~/yue_output_16g \ --max_new_tokens 6000 \ --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_16g_v2   --max_new_tokens 6000   --repetition_penalty 1.05
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_16g_fix   --max_new_tokens 3000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-icl   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_16g_icl_chorus   --max_new_tokens 1200   --repetition_penalty 1.1   --use_audio_prompt   --audio_prompt_path ~/ref_canto_chorus_30s.mp3   --prompt_start_time 0   --prompt_end_time 30
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 4   --stage2_batch_size 1   --output_dir ~/yue_output_16g_fix   --max_new_tokens 3000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 4   --stage2_batch_size 1   --output_dir ~/yue_output_16g_seg   --max_new_tokens 3000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 3   --stage2_batch_size 1   --output_dir ~/yue_output_16g_seg3   --max_new_tokens 3000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 5   --stage2_batch_size 1   --output_dir ~/yue_output_16g_seg5   --max_new_tokens 3000   --repetition_penalty 1.1
exit
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_16g   --max_new_tokens 6000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_16g   --max_new_tokens 4000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 2   --stage2_batch_size 1   --output_dir ~/yue_output_16g   --max_new_tokens 3600   --repetition_penalty 1.1
gpu-interactive 
exit
conda activate 
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 5   --stage2_batch_size 1   --output_dir ~/yue_output_16g_end   --max_new_tokens 6000   --repetition_penalty 1.1
exit
gpu-interactive 
exit
nvidia-smi
exit
hostname -I
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 5   --stage2_batch_size 2   --output_dir ~/yue_output_16g_end_fast3   --max_new_tokens 3000   --repetition_penalty 1.1
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_official && cd ~/YuE/inference && python infer.py   --cuda_idx 0   --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot   --stage2_model m-a-p/YuE-s2-1B-general   --genre_txt ~/yue_test_inputs/genre.txt   --lyrics_txt ~/yue_test_inputs/lyrics.txt   --run_n_segments 5   --stage2_batch_size 4   --output_dir ~/yue_output_16g_end_fast3   --max_new_tokens 3000   --repetition_penalty 1.1
conda activate yue_project
cd~
cd ~
cd canto_project_official_yue_bridge_demo/
ls
pip install -r requirements.txt 
streamlit run app.py -serverhost 8501
streamlit run app.py
source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda remove -n yue_project --all -y && conda create -n yue_project python=3.10 -y && conda activate yue_project && conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && conda install -y mkl mkl-service intel-openmp -c defaults && pip install streamlit==1.40.1 transformers==4.49.0 accelerate==1.2.1 huggingface_hub>=0.24 soundfile>=0.12 sentencepiece>=0.2.0 einops>=0.8.0 matplotlib>=3.7 pillow>=10.0
streamlit run app.py
source ~/anaconda3/etc/profile.d/conda.sh && conda activate yue_project && python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project
pip uninstall -y torch torchvision torchaudio
conda remove -y pytorch torchvision torchaudio pytorch-cuda mkl mkl-service intel-openmp || true
conda clean -a -ypython -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y mkl=2024.0 mkl-service intel-openmp=2024.0 -c defaults
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
conda install -y ittapi -c conda-forge
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project
conda install -y "mkl=2024.0.*" "intel-openmp=2024.0.*" "mkl-service<3"
conda install -y "pytorch=2.4.*" "torchvision=0.19.*" "torchaudio=2.4.*" "pytorch-cuda=12.1" -c pytorch -c nvidia
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
conda install -y ittapi -c conda-forge
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
exit
gpu
gpu-interactive 
exit
conda env list
conda remove -n canto_GPU
conda remove -n canton_GPU
conda remove -n canton_GPU --all
conda env list
conda remove -n yue_clean --all
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project
export LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so${LD_PRELOAD:+:$LD_PRELOAD}"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project
export LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so:$CONDA_PREFIX/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project
export LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so:$CONDA_PREFIX/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda create -n yue_project_clean python=3.10 -y
conda activate yue_project_clean
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
conda 
conda remove -n yue_project_clean -all
conda remove -n yue_project_clean --all
exit
nvidia-smi
exit
nvidia-smi
ps
nvidia-smi
nvidia-smie
exit
hostname -I
conda env list
conda remove -n yue_project --all
conda env list
conda remove -n yue_project --all
conda env list
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda create -n yue_project_clean python=3.10 -y
conda activate yue_project_clean
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -m pip install streamlit==1.40.1 transformers==4.49.0 accelerate==1.2.1 huggingface_hub soundfile sentencepiece einops matplotlib pillow
cd ~/canto_project_official_yue_bridge_demo
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project_clean
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
streamlit run app.py --server.port 8501
ps -u $USER
pkill -u $USER
k
ps -u $USER
streamlit run app.py --server.port 8501
cd ~
unzip canto_project_official_yue_bridge_demo_v2.zip 
cd canto_project_official_yue_bridge_demo_v2/
ls
pip install -r requirements.txt 
streamlit run app.py --server.port 8501
exit
gpu-interactive 
exit
