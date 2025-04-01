import sys
import termios
import tty

INPUT_FILE = "core/data/unlabeled_bullets.txt"
OUTPUT_FILE = "core/data/depth_data.txt"

def get_single_char() -> str:
    """
    Reads a single character from stdin without requiring Enter. 
    Unix-only approach. 
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def label_bullets():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        bullets = [line.strip() for line in f if line.strip()]

    labeled = []

    print("🔍 Label each bullet as:")
    print("[1] Deep / Detailed")
    print("[0] Shallow / Generic")
    print("[s] Skip bullet")
    print("[q] Quit\n")

    for i, bullet in enumerate(bullets, start=1):
        print(f"\n{i}/{len(bullets)} → {bullet}")
        print("Label (1/0/s/q)? ", end="", flush=True)
        
        # Single-character read (no Enter required)
        label = get_single_char()
        print(label)  # Echo the typed character so user sees it

        if label == "1" or label == "0":
            labeled.append(f"{bullet} | {label}")
        elif label == "s":
            continue
        elif label == "q":
            break

    # Append labeled bullets to output
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for entry in labeled:
            out.write(entry + "\n")

    print(f"\n✅ Saved {len(labeled)} labeled bullets to {OUTPUT_FILE}")

if __name__ == "__main__":
    label_bullets()
