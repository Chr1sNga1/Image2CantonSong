import pandas as pd
import re
import os

# 1. Setup Input Path
input_file = 'cantopop_corpus_final_598.csv'
filename_base, file_extension = os.path.splitext(input_file)

# Load raw dataset
df = pd.read_csv(input_file)

def v8_pro_cleaner_strict(text):
    # If null or empty, mark as None for later removal
    if pd.isna(text) or not str(text).strip():
        return None
    
    # Strict HTML fragment check: DROP song if <i> or /i> tags remain
    if re.search(r'/i>|<i>|<[iI]', text):
        return None

    # Pre-check: If lyrics lack [ ] tags, YuE model cannot process them; DROP
    if not re.search(r'\[.*?\]', text):
        return None

    # --- 1. Basic Noise and HTML Cleanup ---
    text = re.sub(r'^\d*.*Contributors', '', text)
    text = re.sub(r'Embed$', '', text).strip()
    text = re.sub(r'\[.*?歌詞\]', '', text)
    
    # Enhanced cleanup for HTML residues (e.g., /i> issues)
    text = re.sub(r'<.*?>|/[iI]>|<[iI]', '', text)
    
    # Handle newlines inside tags
    text = re.sub(r'\[[^\]]*?\n[^\]]*?\]', lambda m: m.group(0).replace('\n', ' '), text)
    
    # --- 2. Complete Removal of Monologue Sections (including content) ---
    # This removes everything from a monologue tag (e.g., [張國榮獨白]) to the next bracket
    monologue_pattern = r'\n?\[[^\]]*?(讀白|獨白|口白|說話|Spoken|Monologue)[^\]]*?\].*?(?=\n\[|$)'
    text = re.sub(monologue_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # --- 3. Standard Tag Normalization (YuE Format) ---
    # Using [^\]]*? to capture tags with various prefixes/suffixes
    # 1. Handle Pre-Chorus first (must be placed before Chorus)
    text = re.sub(r'\[[^\]]*?(Pre-Chorus|Pre|導歌)[^\]]*?\]', '[Pre-Chorus]', text, flags=re.IGNORECASE)
    
    # 2. Handle Chorus, excluding cases already marked as "Pre-Chorus"
    # Uses negative lookbehind (?<!Pre-) to avoid overwriting Pre-Chorus
    text = re.sub(r'\[[^\]]*?(?<!Pre-)(Chorus|副歌|副|Refrain|Hook)[^\]]*?\]', '[Chorus]', text, flags=re.IGNORECASE)

    # Consolidate Bridge, Instrumental, and Solo keywords
    bridge_pattern = r'\[[^\]]*?(Bridge|橋段|過渡|間奏)[^\]]*?\]'
    text = re.sub(bridge_pattern, '[Bridge]', text, flags=re.IGNORECASE)

    text = re.sub(r'\[[^\]]*?(Verse|主歌|主|Main)[^\]]*?\]', '[Verse]', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\[[^\]]*?(Intro|前奏|引子|序|引|頭)[^\]]*?\]', '[Intro]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[^\]]*?(Outro|結尾|尾聲|終|完|尾奏|Ending)[^\]]*?\]', '[Outro]', text, flags=re.IGNORECASE)
    
    # --- [Optimization] Position-aware handling for 「獨奏」/「Solo」 tags ---
    # Temporarily mark as [TEMP_SOLO] to detect position
    solo_pattern = r'\[[^\]]*?(獨奏|樂器|Solo|Instrumental)[^\]]*?\]'
    text = re.sub(solo_pattern, '[TEMP_SOLO]', text, flags=re.IGNORECASE)
    
    lines = text.split('\n')
    # Find indices of all non-empty lines
    content_indices = [i for i, line in enumerate(lines) if line.strip()]
    
    if content_indices:
        first_idx = content_indices[0]
        last_idx = content_indices[-1]
        
        # If the first content line is a Solo -> treat as Intro
        if '[TEMP_SOLO]' in lines[first_idx]:
            lines[first_idx] = lines[first_idx].replace('[TEMP_SOLO]', '[Intro]')
        
        # If the last content line is a Solo -> treat as Outro
        if '[TEMP_SOLO]' in lines[last_idx]:
            lines[last_idx] = lines[last_idx].replace('[TEMP_SOLO]', '[Outro]')
            
    # Convert remaining middle Solo tags to Bridge
    text = '\n'.join(lines).replace('[TEMP_SOLO]', '[Bridge]')
    
    # --- 4. Intelligent Parentheses () Handling ---
    def handle_parentheses(match):
        content = match.group(1).strip()
        # Remove meta-info like (Male/Female/Chorus/Language)
        if re.match(r'^(男|女|合|合唱版|x\d+|\d+|粵語|國語)$', content):
            return ""
        return f"({content})"

    text = re.sub(r'\((.*?)\)', handle_parentheses, text)

    # --- 5. Final Cleanup and Structural Organization ---
    # Wipe out any residual tags that do not conform to YuE standards
    text = re.sub(r'\[(?!Verse|Chorus|Bridge|Pre-Chorus|Intro|Outro).*?\]', '', text)
    
    # Remove redundant empty lines caused by monologue deletion
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Final check: If no valid structural tags remain after cleanup, DROP
    if not re.search(r'\[(Verse|Chorus|Bridge|Pre-Chorus|Intro|Outro)\]', text):
        return None
    
    return text.strip()

# Execute batch processing
df['Lyrics_YuE'] = df['Lyrics_Raw'].apply(v8_pro_cleaner_strict)

# Execute DROP operation: Remove all rows where Lyrics_YuE is None
original_count = len(df)
df = df.dropna(subset=['Lyrics_YuE'])

# Deduplicate by Title, keeping the first artist instance
df = df.drop_duplicates(subset=['Title'], keep='first')

final_count = len(df)

# --- Dynamic Output Filename Generation ---
new_filename_base = re.sub(r'\d+', str(final_count), filename_base)
output_file = f"{new_filename_base}_yue{file_extension}"

# Save results
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Lyrics processing complete!")
print(f"------------------------------------")
print(f"Original songs: {original_count}")
print(f"Dropped songs:  {original_count - final_count}")
print(f"Final songs:    {final_count}")
print(f"------------------------------------")
print(f"Output file renamed to: {output_file}")