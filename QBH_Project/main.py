import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert
import extract_features
import record
import match

def display_menu():
    print("\n" + "*"*45)
    print("*   QUERY BY HUMMING (QBH) — Melody Match  *")
    print("*"*45)
    print("1. Convert songs to WAV (convert.py)")
    print("2. Build melody database (extract_features.py)")
    print("3. Record humming (record.py)")
    print("4. Match humming to database (match.py)")
    print("5. Full pipeline (Record → Match)")
    print("6. Exit")
    print("*"*45)

def main():
    while True:
        display_menu()
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            print("\n--- Converting songs to WAV ---")
            convert.convert_all_songs()
        elif choice == '2':
            print("\n--- Building melody database ---")
            extract_features.build_melody_database()
        elif choice == '3':
            print("\n--- Recording humming ---")
            record.record_audio()
        elif choice == '4':
            print("\n--- Matching ---")
            match.match_query()
        elif choice == '5':
            print("\n--- Full pipeline ---")
            if record.record_audio():
                match.match_query()
        elif choice == '6':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
