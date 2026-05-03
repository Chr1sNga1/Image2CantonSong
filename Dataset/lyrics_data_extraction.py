import lyricsgenius
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import random

# ================= CONFIGURATION =================
GENIUS_ACCESS_TOKEN = "GENIUS_ACCESS_TOKEN" # Input token here, deleted for GitHub upload

HIGH_QUALITY_LYRICISTS = [
    "鄭國江", "黃霑", "James Wong", "潘偉源", "潘源良", "Calvin Poon", "盧國沾",
    "林夕", "Lin Xi", "Albert Leung", "黃偉文", "Wyman Wong", "Wyman", 
    "周耀輝", "Chow Yiu Fai", "林若寧", "Riley Lam", "小克", "Siu Hak",
    "陳詠謙", "Chan Wing Him", "林寶", "Gary Lam", "梁栢堅", "林振強", "Richard Lam",
    "周博賢", "Adrian Chow", "陳少琪", "Keith Chan", "向雪懷", "Jolland Chan", 
    "李峻一", "Joe Lei", "小美", "Siu May", "劉卓輝", "Lau Cheuk Fai"
]

TARGET_ARTISTS = [
    "張國榮 (Leslie Cheung)", "梅艷芳 (Anita Mui)", "Beyond", "張學友 (Jacky Cheung)", 
    "王菲 (Faye Wong)", "陳百強 (Danny Chan)", "黎明 (Leon Lai)", "陳奕迅 (Eason Chan)", 
    "容祖兒 (Joey Yung)", "古巨基 (Leo Ku)", "楊千嬅 (Miriam Yeung)", "謝安琪 (Kay Tse)", 
    "張敬軒 (Hins Cheung)", "何韻詩 (HOCC)", "李克勤 (Hacken Lee)", "許志安 (Andy Hui)", 
    "謝霆鋒 (Nicholas Tse)", "衛蘭 (Janice Vidal)", "鄭秀文 (Sammi Cheng)", "彭羚 (Cass Phang)", 
    "林峯 (Raymond Lam)", "陳柏宇 (Jason Chan)", "周殷廷 (Yan Ting)", "陳慧琳 (Kelly Chen)", 
    "吳雨霏 (Kary Ng)", "C AllStar", "泳兒 (Vincy Chan)", "王菀之 (Ivana Wong)", "方皓玟 (Charmaine Fong)", 
    "Twins (HKG)", "梁詠琪 (Gigi Leung)", "Dear Jane", "林家謙 (Terence Lam)", "MC 張天賦 (Cheung Tinfu)",
    "郭富城 (Aaron Kwok)", "側田 (Justin Lo)", "薛凱琪 (Fiona Sit)", "梁漢文 (Edmond Leung)", 
    "李幸倪 (Gin Lee)", "許廷鏗 (Alfred Hui)", "麥浚龍 (Juno Mak)", "RubberBand",
    "林憶蓮 (Sandy Lam)", "葉倩文 (Sally Yeh)", "鄧麗欣 (Stephy Tang)"
]

MAX_SONGS_PER_ARTIST = 40 
# =================================================

class CantopopLyricsScraper:
    def __init__(self, token):
        self.genius = lyricsgenius.Genius(token, sleep_time=1.5, retries=10, timeout=60)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
        self.genius._session.headers.update(self.headers)

    def verify_cantonese(self, song_url):
        """
        Verify if the song is truly Cantonese using a dual-filter approach.
        This mitigates noise from Mandarin versions listed in 'Interpolations'[cite: 1].
        """
        if not song_url: return False
        try:
            # Random jitter to avoid rate-limiting during large-scale scraping[cite: 1]
            time.sleep(random.uniform(2.0, 3.5))
            response = requests.get(f"{song_url}/credits", headers=self.headers, timeout=20)
            if response.status_code == 200:
                soup_text = BeautifulSoup(response.text, 'html.parser').get_text().lower()
                
                # 1. Whitelist: Check for explicit Cantonese markers[cite: 1]
                has_canto_tag = any(word in soup_text for word in ['chinese (cantonese)', 'cantopop', '廣東歌', '粵語'])
                
                # 2. Blacklist: Strictly exclude Mandarin variants to prevent data contamination[cite: 1]
                is_mandarin = any(word in soup_text for word in ['mandarin version', '國語版', '普通話', 'mandopop'])
                
                return has_canto_tag and not is_mandarin
            return False
        except Exception:
            return False

    def execute_extraction(self):
        extracted_data = []
        for query_name in TARGET_ARTISTS:
            print(f"\n[Processing] Target: {query_name}")
            
            try:
                time.sleep(random.uniform(3, 6))
                
                search_term = query_name
                artist = self.genius.search_artist(search_term, max_songs=MAX_SONGS_PER_ARTIST, sort="popularity")
                
                if not artist:
                    print(f"  [Error] No results for {query_name}")
                    continue

                for song in artist.songs:
                    try:
                        meta = song.to_dict()
                        raw_lyr = meta.get('lyrics', '')
                        
                        # Lyricist validation (Checks metadata and raw text for backup)
                        writers = [w.get('name', '') for w in meta.get('writer_artists', [])]
                        matched_lyricist = next((l for l in HIGH_QUALITY_LYRICISTS if any(l in w for w in writers) or l in raw_lyr), None)

                        if matched_lyricist:
                            if self.verify_cantonese(song.url):
                                clean_text = re.sub(r'^\d*.*Contributors', '', raw_lyr)
                                clean_text = re.sub(r'Embed$', '', clean_text).strip()
                                
                                extracted_data.append({
                                    "Artist": artist.name,
                                    "Title": song.title,
                                    "Lyricist": matched_lyricist,
                                    "Lyrics_Raw": clean_text,
                                    "Released on": meta.get('release_date_for_display', 'Unknown'),
                                    "Genius_ID": meta.get('id', 'Unknown')
                                })
                                print(f"  [Success] {song.title}")
                    except: continue
                
                if extracted_data:
                    pd.DataFrame(extracted_data).to_csv("lyrics_final_checkpoint.csv", index=False, encoding='utf-8-sig')
                    
            except Exception as e:
                print(f"  [Critical] Interruption during {query_name}: {e}")
                time.sleep(40) # Safety cooldown
                continue

        return extracted_data

if __name__ == "__main__":
    scraper = CantopopLyricsScraper(GENIUS_ACCESS_TOKEN)
    final_results = scraper.execute_extraction()
    
    if final_results:
        df = pd.DataFrame(final_results).drop_duplicates(subset=['Title', 'Artist'])
        df.to_csv(f"cantopop_corpus_final_{len(df)}.csv", index=False, encoding='utf-8-sig')
        print(f"\n[Finished] Total unique songs collected: {len(df)}")