#!/usr/bin/env python3
import argparse
import json
import os
import sys
import queue
import threading
import time
import cv2
import torch
import asyncio
from datetime import date
from pathlib import Path
from loguru import logger
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import uvicorn
import redis

# Globals
output_frame = None
lock = threading.Lock()
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / 'templates'
CONFIG_PATH = None
CONFIG_DATA = None

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def load_config(path):
    with open(path) as f:
        return json.load(f)


class FlowTracker:
    """Tracks directional flow (in/out) of people in a video stream."""

    def __init__(self, url, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        if not hasattr(self, 'conf_threshold'):
            self.conf_threshold = 0.4
        self.src_url = url
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
        logger.info(f"Loading YOLO model '{self.model_path}' on {self.device}")
        self.model = YOLO(self.model_path)
        if self.device.startswith('cuda'):
            self.model.model.to(self.device)
        self.tracker = DeepSort(max_age=5)
        self.frame_queue = queue.Queue(maxsize=10)
        self.tracks = {}
        self.redis = redis.Redis.from_url(getattr(self, 'redis_url', 'redis://localhost:6379/0'))
        self.in_count = int(self.redis.get('in_count') or 0)
        self.out_count = int(self.redis.get('out_count') or 0)
        stored_date = self.redis.get('count_date')
        self.prev_date = date.fromisoformat(stored_date.decode()) if stored_date else date.today()
        self.redis.mset({
            'in_count': self.in_count,
            'out_count': self.out_count,
            'count_date': self.prev_date.isoformat(),
        })
        self.running = True

    def capture_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.src_url)
            if not cap.isOpened():
                logger.error(f"Cannot open stream: {self.src_url}")
                time.sleep(self.retry_interval)
                continue
            logger.info(f"Stream opened: {self.src_url}")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        f"Lost stream, retry in {self.retry_interval}s")
                    break
                if self.frame_queue.full():
                    _ = self.frame_queue.get()
                self.frame_queue.put(frame)
            cap.release()
            time.sleep(self.retry_interval)

    def process_loop(self):
        global output_frame
        idx = 0
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            idx += 1
            if date.today() != self.prev_date:
                self.in_count = 0
                self.out_count = 0
                self.tracks.clear()
                self.prev_date = date.today()
                self.redis.mset({
                    'in_count': self.in_count,
                    'out_count': self.out_count,
                    'count_date': self.prev_date.isoformat(),
                })
                logger.info("Daily counts reset")
            if self.skip_frames and (idx % self.skip_frames):
                continue
            res = self.model.predict(frame, device=self.device, verbose=False)[0]
            h, w = frame.shape[:2]
            x_line = int(w * self.line_ratio)
            cv2.line(frame, (x_line, 0), (x_line, h), (255, 0, 0), 2)
            dets = []
            for *xyxy, conf, cls in res.boxes.data.tolist():
                if int(cls) != 0 or conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                dets.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
            tracks = self.tracker.update_tracks(dets, frame=frame)
            now = time.time()
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tid = tr.track_id
                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                cx = (x1 + x2) // 2
                zone = 'left' if cx < x_line else 'right'
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        'zone': zone,
                        'cx': cx,
                        'time': now,
                        'last_counted': None,
                    }
                prev = self.tracks[tid]
                if zone != prev['zone'] and abs(cx-prev['cx']) > self.v_thresh and now - prev['time'] > self.debounce:
                    direction = (
                        'Entering' if prev['zone'] == 'left' and zone == 'right' else
                        'Exiting' if prev['zone'] == 'right' and zone == 'left' else
                        None
                    )
                    if direction:
                        if prev['last_counted'] is None:
                            if direction == 'Entering':
                                self.in_count += 1
                            else:
                                self.out_count += 1
                            self.redis.mset({
                                'in_count': self.in_count,
                                'out_count': self.out_count,
                            })
                            prev['last_counted'] = direction
                            logger.info(
                                f"{direction} ID{tid}: In={self.in_count}, Out={self.out_count}")
                        elif prev['last_counted'] != direction:
                            if prev['last_counted'] == 'Entering':
                                self.in_count -= 1
                            else:
                                self.out_count -= 1
                            prev['last_counted'] = None
                            logger.info(f"Reversed flow for ID{tid}")
                            self.redis.mset({
                                'in_count': self.in_count,
                                'out_count': self.out_count,
                            })
                        prev['time'] = now
                prev['zone'], prev['cx'] = zone, cx
                color = (0, 255, 0) if zone == 'right' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                conf = tr.get_det_conf()
                label = f"ID{tid}"
                if conf is not None:
                    label += f" {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Entering: {self.in_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exiting: {self.out_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with lock:
                output_frame = frame.copy()
            time.sleep(1/self.fps)


app = FastAPI()
counter: FlowTracker = None


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        'dashboard.html',
        {
            'request': request,
            'MAX_CAPACITY': counter.max_capacity,
            'WARN_THRESHOLD': counter.warn_threshold,
        }
    )


@app.get("/video_feed")
async def video_feed():
    async def gen():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    await asyncio.sleep(0.01)
                    continue
                _, buf = cv2.imencode('.jpg', output_frame)
                frame = buf.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            await asyncio.sleep(1/counter.fps)
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket("/ws/stats")
async def ws_stats(ws: WebSocket):
    await ws.accept()
    while True:
        in_count = counter.in_count
        out_count = counter.out_count
        max_cap = counter.max_capacity
        warn_lim = max_cap * counter.warn_threshold / 100
        status = (
            'green' if in_count < warn_lim else
            'yellow' if in_count < max_cap else
            'red'
        )
        await ws.send_json({
            'in_count': in_count,
            'out_count': out_count,
            'status': status,
        })
        await asyncio.sleep(1)


@app.get('/settings')
def settings_page(request: Request):
    return templates.TemplateResponse('settings.html', {
        'request': request,
        'cfg': CONFIG_DATA,
    })


@app.post('/settings')
async def update_settings(request: Request):
    form = await request.form()
    CONFIG_DATA['max_capacity'] = int(form['max_capacity'])
    CONFIG_DATA['warn_threshold'] = int(form['warn_threshold'])
    if 'conf_threshold' in form:
        CONFIG_DATA['conf_threshold'] = float(form['conf_threshold'])
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIG_DATA, f, indent=2)
    counter.max_capacity = CONFIG_DATA['max_capacity']
    counter.warn_threshold = CONFIG_DATA['warn_threshold']
    counter.conf_threshold = CONFIG_DATA.get('conf_threshold', counter.conf_threshold)
    return templates.TemplateResponse('settings.html', {
        'request': request,
        'cfg': CONFIG_DATA,
        'saved': True,
    })


def main():
    global counter, CONFIG_PATH, CONFIG_DATA
    p = argparse.ArgumentParser()
    p.add_argument('stream_url', nargs='?')
    p.add_argument('-c', '--config', default='config.json')
    p.add_argument('-w', '--workers', type=int, default=None)
    args = p.parse_args()

    here = BASE_DIR
    CONFIG_PATH = args.config if os.path.isabs(args.config) else os.path.join(here, args.config)
    CONFIG_DATA = load_config(CONFIG_PATH)
    url = args.stream_url or CONFIG_DATA['stream_url']

    cores = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else CONFIG_DATA['default_workers']
    w = max((cores - 1 if workers == -1 else (1 if workers == 0 else workers)), 1)
    cv2.setNumThreads(w)
    torch.set_num_threads(w)

    logger.info(f"Threads={w}, cores={cores}")
    counter = FlowTracker(url, CONFIG_DATA)
    threading.Thread(target=counter.capture_loop, daemon=True).start()
    threading.Thread(target=counter.process_loop, daemon=True).start()

    logger.info(
        f"Server @ http://0.0.0.0:{CONFIG_DATA['port']}  Stream={url}")
    uvicorn.run(app, host='0.0.0.0', port=CONFIG_DATA['port'], log_config=None)


if __name__ == '__main__':
    main()
