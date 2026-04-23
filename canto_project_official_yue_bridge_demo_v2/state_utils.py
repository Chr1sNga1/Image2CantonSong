import streamlit as st
from schemas import LyricsPromptBundle, SongResult

def init_state():
    defaults = {
        "uploaded_image_bytes": None,
        "uploaded_image_name": None,
        "uploader_version": 0,
        "lyrics_prompt_raw": LyricsPromptBundle(),
        "lyrics_prompt_confirmed": LyricsPromptBundle(),
        "song_result": SongResult(),
        "step_1_done": False,
        "step_2_done": False,
        "step_3_done": False,
        "step_4_done": False,
        "last_error": "",
        "last_debug_log": "",
        "last_metadata_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def hard_reset():
    next_version = st.session_state.get("uploader_version", 0) + 1
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["uploader_version"] = next_version
