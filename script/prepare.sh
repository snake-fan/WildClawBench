# WildClawBench Data Preparation Script
#
# Prerequisites: Task data cloned from HuggingFace to workspace/
# Usage: bash script/prepare.sh
#
# WSL / YouTube note:
#   If YouTube asks "Sign in to confirm you're not a bot", export cookies from
#   your Windows browser and put the file at the repository root as cookies.txt.
#   You can also set a custom path:
#       YTDLP_COOKIES=/path/to/cookies.txt bash script/prepare.sh
#
# This script performs the following:
#   1. Download papers.tar (Productivity Flow tasks)
#   2. Download 3 YouTube videos (football match, lecture, product launch)
#   3. Trim/rename videos and copy to respective task directories
#   4. Extract dot_git.tar.gz (Safety Alignment tasks)
#   5. Download sam3.pt model weights (Code Intelligence tasks)

set -euo pipefail

cd "$(dirname "$0")/.."

# ─── Shared yt-dlp configuration ────────────────────────────────
# Keep all YouTube-specific fixes in one place so every download uses them.
YTDLP_COOKIES="${YTDLP_COOKIES:-cookies.txt}"
YTDLP_COMMON_ARGS=(
    --no-playlist
    --merge-output-format mp4
    --retries 10
    --fragment-retries 10
)

if [ -f "$YTDLP_COOKIES" ]; then
    echo "Using yt-dlp cookies: $YTDLP_COOKIES"
    YTDLP_COMMON_ARGS+=(--cookies "$YTDLP_COOKIES")
else
    cat <<EOF_MSG
WARNING: $YTDLP_COOKIES not found.
YouTube may fail with: "Sign in to confirm you're not a bot".
Fix:
  1. Log in to YouTube in your Windows browser.
  2. Export YouTube cookies as cookies.txt.
  3. Put cookies.txt in this repository root, or run:
     YTDLP_COOKIES=/path/to/cookies.txt bash script/prepare.sh
EOF_MSG
fi

if command -v deno >/dev/null 2>&1; then
    YTDLP_COMMON_ARGS+=(--js-runtimes deno --remote-components ejs:npm)
else
    cat <<EOF_MSG
WARNING: deno not found in WSL.
Install it if yt-dlp prints "No supported JavaScript runtime":
  curl -fsSL https://deno.land/install.sh | sh
  echo 'export DENO_INSTALL="$HOME/.deno"' >> ~/.bashrc
  echo 'export PATH="$DENO_INSTALL/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
EOF_MSG
fi

run_ytdlp() {
    yt-dlp "${YTDLP_COMMON_ARGS[@]}" "$@"
}

echo "=========================================="
echo "  WildClawBench Data Preparation"
echo "=========================================="


# ─── 1. Football match video (La Liga: Betis vs Barcelona) ──────
#
#   Download full match → extract first half (first 57 min) → remove full video
#   Target: task_1_match_report/exec/first_half.mp4
#           task_2_goal_highlights/exec/first_half.mp4 (copy)
echo ""
echo "[1/5] Football match video (Betis vs Barcelona)"

TASK1_DIR="workspace/05_Creative_Synthesis/task_1_match_report/exec"
TASK2_DIR="workspace/05_Creative_Synthesis/task_2_goal_highlights/exec"
mkdir -p "$TASK1_DIR" "$TASK2_DIR"

if [ ! -f "$TASK1_DIR/first_half.mp4" ]; then
    echo "  downloading full match ..."
    run_ytdlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
        -o "$TASK1_DIR/full_match.mp4" \
        "https://www.youtube.com/watch?v=93LPZJkCW2w"

    # yt-dlp may produce separate track files instead of a merged file
    if [ ! -f "$TASK1_DIR/full_match.mp4" ]; then
        ffmpeg -i "$TASK1_DIR"/full_match.f*.mp4 \
               -i "$TASK1_DIR"/full_match.f*.m4a \
               -c copy "$TASK1_DIR/full_match.mp4"
    fi

    echo "  extracting first half (00:00 - 00:57:00) ..."
    ffmpeg -i "$TASK1_DIR/full_match.mp4" \
           -t 00:57:00 -c copy "$TASK1_DIR/first_half.mp4"

    rm -f "$TASK1_DIR/full_match.mp4" \
          "$TASK1_DIR"/full_match.f*.mp4 \
          "$TASK1_DIR"/full_match.f*.m4a
    echo "  done: $TASK1_DIR/first_half.mp4"
else
    echo "  skip: $TASK1_DIR/first_half.mp4 already exists"
fi

if [ ! -f "$TASK2_DIR/first_half.mp4" ]; then
    cp "$TASK1_DIR/first_half.mp4" "$TASK2_DIR/first_half.mp4"
    echo "  copied -> $TASK2_DIR/first_half.mp4"
else
    echo "  skip: $TASK2_DIR/first_half.mp4 already exists"
fi

# ─── 2. Lecture video (LLM Teaching) ────────────────────────────
#
#   Target: task_4_video_notes/exec/video.mp4
echo ""
echo "[2/5] Lecture video (LLM Lecture)"

TASK4_DIR="workspace/05_Creative_Synthesis/task_4_video_notes/exec"
mkdir -p "$TASK4_DIR"

if [ ! -f "$TASK4_DIR/video.mp4" ]; then
    echo "  downloading lecture video ..."
    run_ytdlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
        -o "$TASK4_DIR/video.mp4" \
        "https://www.youtube.com/watch?v=LPZh9BOjkQs"
    echo "  done: $TASK4_DIR/video.mp4"
else
    echo "  skip: $TASK4_DIR/video.mp4 already exists"
fi

# ─── 3. Product launch video (Apple Event) ──────────────────────
#
#   Download → merge tracks (if needed) → rename to recording.mp4
#   Target: task_5_product_launch_video_to_json/exec/recording.mp4
#           task_11_video_en_to_zh_dub/exec/recording.mp4 (copy)
echo ""
echo "[3/5] Product launch video (Apple Event)"

TASK5_DIR="workspace/05_Creative_Synthesis/task_5_product_launch_video_to_json/exec"
TASK11_DIR="workspace/05_Creative_Synthesis/task_11_video_en_to_zh_dub/exec"
mkdir -p "$TASK5_DIR" "$TASK11_DIR"

if [ ! -f "$TASK5_DIR/recording.mp4" ]; then
    echo "  downloading product launch video ..."
    run_ytdlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
        -o "$TASK5_DIR/product_video.%(ext)s" \
        "https://www.youtube.com/watch?v=H3KnMyojEQU"

    if [ ! -f "$TASK5_DIR/product_video.mp4" ]; then
        ffmpeg -i "$TASK5_DIR"/product_video.f*.mp4 \
               -i "$TASK5_DIR"/product_video.f*.m4a \
               -c copy "$TASK5_DIR/product_video.mp4"
        rm -f "$TASK5_DIR"/product_video.f*.mp4 \
              "$TASK5_DIR"/product_video.f*.m4a
    fi

    mv "$TASK5_DIR/product_video.mp4" "$TASK5_DIR/recording.mp4"
    echo "  done: $TASK5_DIR/recording.mp4"
else
    echo "  skip: $TASK5_DIR/recording.mp4 already exists"
fi

if [ ! -f "$TASK11_DIR/recording.mp4" ]; then
    cp "$TASK5_DIR/recording.mp4" "$TASK11_DIR/recording.mp4"
    echo "  copied -> $TASK11_DIR/recording.mp4"
else
    echo "  skip: $TASK11_DIR/recording.mp4 already exists"
fi

# ─── 4. Extract dot_git.tar.gz ──────────────────────────────────
echo ""
echo "[4/5] Extract dot_git.tar.gz (06_Safety_Alignment)"

for dir in \
    workspace/06_Safety_Alignment/task_2_leaked_api/exec/mm_agents \
    workspace/06_Safety_Alignment/task_3_leaked_api_pswd/exec/mm_agents; do
    if [ -f "$dir/dot_git.tar.gz" ] && [ ! -d "$dir/.git" ]; then
        echo "  extracting $dir/dot_git.tar.gz ..."
        tar -xzf "$dir/dot_git.tar.gz" -C "$dir"
        rm -f "$dir/dot_git.tar.gz"
        echo "  done"
    elif [ -d "$dir/.git" ]; then
        echo "  skip: $dir/.git already exists"
    else
        echo "  warn: $dir/dot_git.tar.gz not found"
    fi
done

# ─── 5. Download SAM3 model weights ─────────────────────────────
echo ""
echo "[5/5] Download sam3.pt (02_Code_Intelligence)"

SAM3_TASK1="workspace/02_Code_Intelligence/task_1_sam3_inference/exec/sam3"
SAM3_TASK2="workspace/02_Code_Intelligence/task_2_sam3_debug/exec/sam3"

if [ ! -f "$SAM3_TASK1/sam3.pt" ]; then
    echo "  downloading sam3.pt from ModelScope ..."
    modelscope download --model facebook/sam3 sam3.pt --local_dir "$SAM3_TASK1"
    echo "  done: $SAM3_TASK1/sam3.pt"
else
    echo "  skip: $SAM3_TASK1/sam3.pt already exists"
fi

if [ ! -f "$SAM3_TASK2/sam3.pt" ]; then
    if [ -f "$SAM3_TASK1/sam3.pt" ]; then
        mkdir -p "$SAM3_TASK2"
        cp "$SAM3_TASK1/sam3.pt" "$SAM3_TASK2/sam3.pt"
        echo "  copied -> $SAM3_TASK2/sam3.pt"
    else
        echo "  downloading sam3.pt from ModelScope ..."
        modelscope download --model facebook/sam3 sam3.pt --local_dir "$SAM3_TASK2"
        echo "  done: $SAM3_TASK2/sam3.pt"
    fi
else
    echo "  skip: $SAM3_TASK2/sam3.pt already exists"
fi

# ─── Done ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
echo ""
echo "Video layout:"
echo "  football  -> task_1_match_report/exec/first_half.mp4"
echo "               task_2_goal_highlights/exec/first_half.mp4"
echo "  lecture   -> task_4_video_notes/exec/video.mp4"
echo "  launch   -> task_5_product_launch_video_to_json/exec/recording.mp4"
echo "               task_11_video_en_to_zh_dub/exec/recording.mp4"
echo ""
echo "Model weights:"
echo "  sam3.pt   -> task_1_sam3_inference/exec/sam3/sam3.pt"
echo "               task_2_sam3_debug/exec/sam3/sam3.pt"
