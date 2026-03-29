def start_camera(source,url="",event_id=None,threshold=10):
    global camera,camera_active,camera_source,esp32_url
    global capture_thread,session_data,current_event_id,current_threshold,prev_gray
    camera_source=source; esp32_url=url
    current_event_id=event_id; current_threshold=threshold; prev_gray=None

    if source=="webcam":
        camera=cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        if not camera.isOpened(): return False
        for _ in range(20): camera.read()

    load_model()
    sid=str(uuid.uuid4())[:8]
    camera_active=True

    # Reset frame state for fresh session
    global latest_frame, pending_raw
    with frame_lock: latest_frame = None
    with pending_lock: pending_raw = None

    session_data.update({"start_time":datetime.now(),"total_logs":0,
                         "alert_events":0,"max_crowd":0,"session_id":sid})
    if event_id:
        try:
            conn=get_db()
            conn.execute("INSERT INTO mon_sessions VALUES (?,?,?,NULL,0,0,0,?)",
                (sid,event_id,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),threshold))
            conn.execute("UPDATE events SET status='active' WHERE id=?",(event_id,))
            conn.commit(); conn.close()
        except: pass

    # Thread 1: frame grabber (fast, non-blocking)
    capture_thread=threading.Thread(target=capture_worker,daemon=True)
    capture_thread.start()
    # Thread 2: heavy detection (slow, runs in background)
    det_thread=threading.Thread(target=detection_worker,daemon=True)
    det_thread.start()
    return True