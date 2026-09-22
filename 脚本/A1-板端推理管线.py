#!/usr/bin/env python3
"""
Phase A1 真机: Mac编排 → 板端 SNPE CPU/GPU 推理
KITTI → YOLOv5s INT8 → InceptionV3 INT8 → scene → decision
"""

import json, os, subprocess, sys, time, tempfile
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

# ── Config ─────────────────────────────────────────────
BOARD_MODELS = "/data/local/tmp/models"
BOARD_WORK = "/data/local/tmp/phase_a1"
SNPE_RUN = "/data/local/tmp/snpe-net-run"
SNPE_LIBS = "/data/local/tmp/qnn_libs"
SNPE_ENV = f"export LD_LIBRARY_PATH={SNPE_LIBS}:$LD_LIBRARY_PATH"

YOLO_DLC = f"{BOARD_MODELS}/yolov5s.dlc"
INCEPTION_DLC = f"{BOARD_MODELS}/inception_v3.dlc"
YOLO_SIZE = (640, 640)
INCEPTION_SIZE = (299, 299)
MEAN_RGB = (128, 128, 128)
CONF_THRES = 0.25
IOU_THRES = 0.45

LABELS_PATH = os.path.expanduser("~/Documents/端测/项目文档/参考资料/imagenet_slim_labels.txt")

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

OBSTACLE_CLASSES = {
    "car","truck","bus","motorcycle","bicycle","boat","airplane","train",
    "traffic light","stop sign","fire hydrant","parking meter","bench",
    "potted plant","suitcase","backpack","handbag","umbrella","sports ball",
    "dog","cat","horse","sheep","cow","bird",
    "chair","couch","dining table","bed","toilet","tv",
    "refrigerator","microwave","oven","sink",
    "vase","teddy bear","scissors"
}

# ── Helpers ─────────────────────────────────────────────
def adb(cmd, timeout=30):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def adb_push(local, remote):
    subprocess.run(["adb", "push", str(local), remote], capture_output=True, check=True)

def adb_pull(remote, local):
    subprocess.run(["adb", "pull", remote, str(local)], capture_output=True, check=True)

def board_run_snpe(dlc, input_list, output_dir, timeout=90):
    """Run snpe-net-run on board with GPU acceleration, returns True on success."""
    cmd = f"{SNPE_ENV} && {SNPE_RUN} --container {dlc} --input_list {input_list} --output_dir {output_dir}"
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stdout + r.stderr

# ── YOLO preprocessing ─────────────────────────────────
def preprocess_yolo(img_path, raw_path):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, YOLO_SIZE)
    mean = np.empty((YOLO_SIZE[0], YOLO_SIZE[1], 3), dtype=np.float32)
    for c in range(3): mean[:,:,c] = MEAN_RGB[c]
    raw = (img.astype(np.float32) - mean) / 128.0
    raw.tofile(str(raw_path))
    return img

# ── YOLO NMS ───────────────────────────────────────────
def filter_boxes(org_box, conf_thres=0.25, iou_thres=0.45):
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf]
    if len(box) == 0: return np.array([])
    cls_scores = box[..., 5:]
    best_cls = np.argmax(cls_scores, axis=1)
    box[:, 4] = np.max(cls_scores, axis=1)
    box[:, 5] = best_cls
    box = box[:, :6]
    output = []
    for cid in np.unique(best_cls).astype(int):
        m = box[:, 5] == cid; y = np.copy(box[m])
        y[:, 0] = box[m][:,0] - box[m][:,2]/2; y[:, 1] = box[m][:,1] - box[m][:,3]/2
        y[:, 2] = box[m][:,0] + box[m][:,2]/2; y[:, 3] = box[m][:,1] + box[m][:,3]/2
        x1,y1_,x2,y2_ = y[:,0], y[:,1], y[:,2], y[:,3]
        scores = y[:,4]; areas = (y2_-y1_+1)*(x2-x1+1)
        order = scores.argsort()[::-1]; keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1=np.maximum(x1[i], x1[order[1:]]); yy1=np.maximum(y1_[i], y1_[order[1:]])
            xx2=np.minimum(x2[i], x2[order[1:]]); yy2=np.minimum(y2_[i], y2_[order[1:]])
            w=np.maximum(0, xx2-xx1+1); h=np.maximum(0, yy2-yy1+1)
            ious = w*h/(areas[i]+areas[order[1:]]-w*h)
            order = order[np.where(ious<=iou_thres)[0]+1]
        output.append(y[keep])
    return np.concatenate(output,axis=0) if output else np.array([])

def parse_yolo(raw_path):
    data = np.fromfile(str(raw_path), dtype=np.float32).reshape(1,25200,85)
    boxes = filter_boxes(data, CONF_THRES, IOU_THRES)
    dets = []
    if boxes.size > 0:
        for b in boxes:
            x1,y1,x2,y2 = b[:4].astype(float)
            label = CLASSES[int(b[5])] if int(b[5]) < len(CLASSES) else f"cls_{int(b[5])}"
            dets.append({"label":label,"confidence":round(float(b[4]),3),
                         "bbox":[round(x1,1),round(y1,1),round(x2,1),round(y2,1)]})
    return dets

# ── InceptionV3 ────────────────────────────────────────
def preprocess_inception_crop(img, bbox, raw_path):
    x1,y1,x2,y2 = [max(0, int(round(v))) for v in bbox]
    x2=min(YOLO_SIZE[0],x2); y2=min(YOLO_SIZE[1],y2)
    if x2<=x1 or y2<=y1: return None
    crop = cv2.resize(img[y1:y2, x1:x2], INCEPTION_SIZE)
    mean = np.empty((INCEPTION_SIZE[0], INCEPTION_SIZE[1], 3), dtype=np.float32)
    for c in range(3): mean[:,:,c] = MEAN_RGB[c]
    raw = (crop.astype(np.float32) - mean) / 128.0
    raw.tofile(str(raw_path))
    return raw_path

# ── Scene → Decision ───────────────────────────────────
def detections_to_scene(fid, detections):
    obstacles = []
    for d in detections:
        lbl = d.get("inception_label") or d["label"]
        if lbl not in OBSTACLE_CLASSES and d["label"] not in OBSTACLE_CLASSES: continue
        x1,y1,x2,y2 = d["bbox"]
        bw,bh = max(x2-x1,1), max(y2-y1,1)
        cx=(x1+x2)/2/YOLO_SIZE[0]; cy=(y1+y2)/2/YOLO_SIZE[1]
        nw=bw/YOLO_SIZE[0]; nh=bh/YOLO_SIZE[1]
        lane = "left" if cx<1/3 else "right" if cx>2/3 else "center"
        dist = round(min(80.0, 0.43/max(nh, 0.001)), 2)
        obstacles.append({
            "type":lbl,"yolo_label":d["label"],"yolo_conf":d["confidence"],
            "inception_label":d.get("inception_label"),
            "inception_conf":d.get("inception_confidence",0),
            "x":round(cx,3),"y":round(cy,3),"w":round(nw,3),"h":round(nh,3),
            "distance_est":dist,"lane_hint":lane,"confidence":d["confidence"]})
    lanes = {"left":0.0,"center":0.0,"right":0.0}
    for o in obstacles:
        lanes[o["lane_hint"]] += o["w"]*(0.65 if o["distance_est"]<2.0 else 0.35)
    fs = {n:round(max(0.0, min(1.0, 1.0-min(v,0.95))),2) for n,v in lanes.items()}
    return {"frame_id":fid,"obstacles":obstacles,"free_space":fs,
            "ego_state":{"speed_mps":1.0,"speed_level":"slow","heading":"forward"}}

def decide(scene):
    free=scene["free_space"]; ego=scene["ego_state"]
    obs=scene.get("obstacles",[])
    nearest = min((o["distance_est"] for o in obs), default=999.0)
    co = [o for o in obs if o["lane_hint"]=="center"]
    nc = min((o["distance_est"] for o in co), default=999.0)
    cb = free["center"]<0.25 or nc<1.8
    lb = free["left"]-free["right"]>0.12
    rb = free["right"]-free["left"]>0.12
    if nearest<0.8: return {"action":"STOP","confidence":0.98,"reason":f"emergency:{nearest:.2f}m","source":"rule_engine"}
    if cb and free["left"]<0.25 and free["right"]<0.25:
        if ego["speed_level"] in {"medium","fast"} or nc<1.2:
            return {"action":"STOP","confidence":0.93,"reason":"center blocked, sides unsafe","source":"rule_engine"}
        return {"action":"WAIT","confidence":0.81,"reason":"center blocked, sides narrow","source":"rule_engine"}
    if nc<2.0 and cb:
        if lb and free["left"]>0.40:
            c=round(min(0.55+min(free["left"],.9)*.25+min(max(nc,0),3)*.06,.95),2)
            return {"action":"BYPASS_LEFT","confidence":c,"reason":"left safer","source":"rule_engine"}
        if rb and free["right"]>0.40:
            c=round(min(0.55+min(free["right"],.9)*.25+min(max(nc,0),3)*.06,.95),2)
            return {"action":"BYPASS_RIGHT","confidence":c,"reason":"right safer","source":"rule_engine"}
        return {"action":"SLOW_DOWN","confidence":0.72,"reason":"center obstacle, slowing","source":"rule_engine"}
    if free["center"]>=0.45 and nearest>2.0:
        return {"action":"KEEP_CENTER","confidence":0.88,"reason":"center clear","source":"rule_engine"}
    if free["left"]>0.55 and lb:
        return {"action":"BYPASS_LEFT","confidence":0.70,"reason":"left wider","source":"rule_engine"}
    if free["right"]>0.55 and rb:
        return {"action":"BYPASS_RIGHT","confidence":0.70,"reason":"right wider","source":"rule_engine"}
    return {"action":"SLOW_DOWN","confidence":0.60,"reason":"ambiguous fallback","source":"rule_engine"}

# ── Main ─────────────────────────────────────────────────
def main():
    imagenet_labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            imagenet_labels = [l.strip() for l in f if l.strip()]
    print(f"Labels loaded: {len(imagenet_labels)}")

    KITTI = os.path.expanduser("~/Documents/端测/测试数据/KITTI-城市道路131帧")
    OUT = os.path.expanduser("~/Documents/端测/输出/城市道路131帧-FP32-GPU")
    os.makedirs(OUT, exist_ok=True)

    images = sorted(Path(KITTI).glob("*.png"))
    print(f"Phase A1 Board: {len(images)} frames → YOLO INT8 → Incept INT8 → decision\n")

    # Verify
    if "OK" not in adb("echo OK"):
        print("ERROR: adb not responding", file=sys.stderr); return 1
    # Clean board workdir
    adb(f"rm -rf {BOARD_WORK} && mkdir -p {BOARD_WORK}")

    all_results = []; t0_total = time.time()

    for idx, img_path in enumerate(images):
        tag = img_path.stem
        print(f"[{idx+1}/{len(images)}] Frame {tag} …", end="", flush=True)
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmp:
            # 1. YOLO preprocess
            raw_local = Path(tmp) / "yolo.raw"
            yolo_img = preprocess_yolo(img_path, raw_local)
            board_raw = f"{BOARD_WORK}/f{tag}.raw"
            adb_push(raw_local, board_raw)

            # 2. YOLO inference
            list_local = Path(tmp) / "list.txt"
            list_local.write_text(board_raw + "\n")
            board_list = f"{BOARD_WORK}/f{tag}_list.txt"
            adb_push(list_local, board_list)

            yolo_out = f"{BOARD_WORK}/yolo_{tag}"
            ok, log = board_run_snpe(YOLO_DLC, board_list, yolo_out)
            t_yolo = round((time.time()-t0)*1000)

            # Pull & parse
            adb_pull(f"{yolo_out}/Result_0/output0.raw", Path(tmp)/"yolo_out.raw")
            dets = parse_yolo(Path(tmp)/"yolo_out.raw") if Path(tmp,"yolo_out.raw").exists() else []
            ob_dets = [d for d in dets if d["label"] in OBSTACLE_CLASSES]
            print(f" YOLO:{len(dets)}/{len(ob_dets)}@{t_yolo}ms", flush=True)

            # 3. InceptionV3 per bbox
            for i, d in enumerate(dets):
                crop_local = Path(tmp) / f"crop_{i}.raw"
                r = preprocess_inception_crop(yolo_img, d["bbox"], crop_local)
                if r is None:
                    d.update(inception_label=None, inception_confidence=0.0); continue

                board_crop = f"{BOARD_WORK}/f{tag}_c{i}.raw"
                adb_push(crop_local, board_crop)
                cl_local = Path(tmp) / f"cl_{i}.txt"
                cl_local.write_text(board_crop + "\n")
                board_cl = f"{BOARD_WORK}/f{tag}_cl_{i}.txt"
                adb_push(cl_local, board_cl)

                incept_out = f"{BOARD_WORK}/in_{tag}_{i}"
                board_run_snpe(INCEPTION_DLC, board_cl, incept_out)

                adb_pull(f"{incept_out}/Result_0/InceptionV3/Predictions/Reshape_1:0.raw",
                         Path(tmp)/f"in_{i}.raw")
                ip = Path(tmp)/f"in_{i}.raw"
                if ip.exists():
                    probs = np.fromfile(str(ip), dtype=np.float32)
                    if len(probs) > 0:
                        ti = int(np.argmax(probs))
                        d["inception_confidence"] = round(float(probs[ti]), 4)
                        d["inception_label"] = imagenet_labels[ti] if ti < len(imagenet_labels) else f"c{ti}"
                    else:
                        d.update(inception_label=None, inception_confidence=0.0)
                else:
                    d.update(inception_label=None, inception_confidence=0.0)

            for d in ob_dets:
                il = d.get("inception_label","-")
                ic = d.get("inception_confidence",0)
                print(f"  {d['label']}→{il}({ic:.3f})", flush=True)

            # 4. Scene + decision
            scene = detections_to_scene(idx, dets)
            dec = decide(scene)
            t_tot = round((time.time()-t0)*1000)
            print(f"  → {dec['action']} conf={dec['confidence']} [{t_tot}ms]\n", flush=True)

            all_results.append({
                "frame_id":idx,"filename":img_path.name,
                "detections":dets,"scene_context":scene,"decision":dec,
                "timing_ms":{"total":t_tot,"yolo":t_yolo}})

            # Cleanup board
            adb(f"rm -rf {BOARD_WORK}/f{tag}* {BOARD_WORK}/yolo_{tag} {BOARD_WORK}/in_{tag}_*", timeout=5)

    # ── Summary ───────────────────────────────────
    total_t = time.time()-t0_total
    print(f"{'='*60}\nDone: {len(all_results)} frames in {total_t:.1f}s")

    summary = Path(OUT) / "phase_a1_board_summary.json"
    with open(summary, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary}")

    acts = Counter(r["decision"]["action"] for r in all_results)
    print("Decisions:"); [print(f"  {a}: {c}") for a,c in acts.most_common()]
    incept_n = sum(1 for r in all_results for d in r["detections"] if d.get("inception_label"))
    print(f"InceptionV3 enriched: {incept_n} detections")

    yt = [r["timing_ms"]["yolo"] for r in all_results]
    tt = [r["timing_ms"]["total"] for r in all_results]
    if yt: print(f"YOLO avg: {sum(yt)//len(yt)}ms, Total avg: {sum(tt)//len(tt)}ms")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
