import os
from subprocess import check_output

def get_fps(video_path):
    FFMPEG_ARGS = '2>&1 | sed -n "s/.*, \(.*\) fp.*/\\1/p"'
    args = f'-i {video_path} {FFMPEG_ARGS}'
    cmd = f"ffmpeg {args}"
    return float(check_output(cmd, shell=True).decode("utf-8"))

def call_ffmpeg(
        video_path: str,
        destination: str
) -> bool:
    fps = get_fps(video_path)
    FFMPEG_ARGS = '-r {} -qscale:v 2 -hide_banner'.format(fps)

    os.makedirs(destination, exist_ok=True)

    args = f'-i {video_path} {FFMPEG_ARGS} {os.path.join(destination, "%04d.jpg")}'
    cmd = f"ffmpeg {args}"

    if os.system(cmd) != 0:
        return False
    return True

dst = '/mnt/data/frames/altyn_original_fps_igor'
src = '/mnt/data/video/opt/labeler/static/video/altyn/'
shift = len(src)
max_len = []
for path, subdirs, files in os.walk(src):
    for name in files:
        file_src = os.path.join(path, name)
        #file_dst = os.path.join(dst, file_src[shift:-4]) #possible subfolder inside a main folder
        file_dst = os.path.join(dst, name[:-4]) #out of subfolders inside a main folder
        call_ffmpeg(file_src, file_dst)
