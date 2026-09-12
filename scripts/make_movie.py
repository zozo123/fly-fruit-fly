"""Compose the saved simulator frames; requires Pillow and system ffmpeg."""
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def main():
    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch)
        clips = []
        for name in ['baseline', 'ppo-8192']:
            folder = temp / name
            folder.mkdir()
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(ROOT / 'media' / f'{name}.mp4'),
                            str(folder / '%03d.png')], check=True)
            clips.append([Image.open(p).convert('RGB').resize((480, 360))
                          for p in sorted(folder.glob('*.png'))])
        frames = []
        for i in range(max(map(len, clips)) + 75):
            canvas = Image.new('RGB', (960, 520), '#0c1420')
            draw = ImageDraw.Draw(canvas)
            def text(x, y, value, size=18, color='#e8edf6'):
                draw.text((x, y), value, font=ImageFont.truetype(FONT, size), fill=color)
            text(20, 12, 'FLY FRUIT FLY  /  First measured attempt', 26)
            text(20, 50, 'Both fail. Stable flight has not been learned.', 20, '#ffba8a')
            for j, clip in enumerate(clips):
                x = j * 480
                canvas.paste(clip[min(i, len(clip)-1)], (x, 110))
                text(x+20, 85, ['Untrained wingbeat', 'PPO / 8,192 training steps'][j], 18)
                if i >= len(clip)-1:
                    draw.rectangle((x+10, 426, x+470, 462), fill='#0c1420')
                    text(x+20, 433, 'EPISODE ENDED / frame held', 18, '#ffba8a')
            text(20, 479, 'Seed 10000 | 10x slow motion | orange: simulated fly; ghost: target', 18)
            frames.append(canvas)
            canvas.save(temp / f'frame-{i:04d}.png')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '50', '-i',
                        str(temp/'frame-%04d.png'), '-c:v', 'libx264', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(ROOT/'media/comparison.mp4')], check=True)
        # Same frames/timing at half resolution and 25 fps for GitHub's README.
        thumbs = [frame.resize((720, 390)) for frame in frames[::2]]
        thumbs[0].save(ROOT/'media/comparison.gif', save_all=True,
                       append_images=thumbs[1:], duration=40, loop=0, optimize=True)
        frames[15].save(ROOT/'media/poster.png')

if __name__ == '__main__':
    main()
