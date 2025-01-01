import cv2
import os
from PIL import Image
import imagehash


DATA_DIR = '/root/autodl-tmp/movie_clip/'
MOVIE_DIR = ''
def _extract_unique_frames(
    video_path,
    save_to,
    hash_size=8,
    cutoff=32,
    frame_count=0,
    end_frame=None
):
    if not os.path.exists(save_to):
        os.makedirs(save_to, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    last_hash = None

    while True:
        ret = cap.grab()
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        ret, frame = cap.retrieve()
        
        if not ret or frame_count > end_frame:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.average_hash(image, hash_size)
        if not last_hash or (curr_hash - last_hash) >= cutoff:
            image.save(save_to + '/frame_{:06d}.jpg'.format(frame_count))
            last_hash = curr_hash
    
    cap.release()

from typing import Annotated
def _extract_unique_frames(
    video_path,
    save_to,
    hash_size=8,
    cutoff=32,
    number_workers: Annotated[str, '多进程处理视频的进程数, 建议值:cpu数%4']=2
):
    import multiprocessing
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    _frame_per_worker = frame_count // number_workers
    _ps = []
    for i in range(number_workers):
        p = multiprocessing.Process(
            target=_extract_unique_frames,
            args=(
                video_path,
                save_to,
                hash_size,
                cutoff,
                i * _frame_per_worker,
                (i + 1) * _frame_per_worker-1
            )
        )
        _ps.append(p)
        p.start()
    for p in _ps:
        p.join()

def extract_unique_frames():
    _movies = os.listdir(MOVIE_DIR)
    for item in tqdm(_movies):
        if not item.endswith('.mkv'):
            continue
        print(f'Processing {item}...')
        _extract_unique_frames(
            'doubai223/'+ item,
            'data/' + item,
            number_workers=3
        )

from tqdm import tqdm
if __name__ == '__main__':
    extract_unique_frames()