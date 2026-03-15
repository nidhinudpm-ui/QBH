
import os
import sys
import pickle

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from match import match_query

# Target the latest query found in logs
query_path = os.path.join(PROJECT_DIR, "uploads", "test_good.wav")
target_song = "Nenjakame.wav"

print(f"--- DIAGNOSTIC TEST ---")
print(f"Query: {os.path.basename(query_path)}")
print(f"Target: {target_song}")

results = match_query(
    query_path, 
    target_song=target_song, 
    disable_prefilter=False, 
    return_results=True
)

print("\n--- RESULTS ---")
for i, res in enumerate(results[:5]):
    print(f"{i+1}. {res['song_name']} - Score: {res['melody_score']:.4f}")
