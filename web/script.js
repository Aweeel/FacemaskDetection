// ── Theme toggle ──────────────────────────────────────────────
const toggle = document.getElementById("themeToggle");
toggle.addEventListener("click", () => {
    const html = document.documentElement;
    const current = html.getAttribute("data-theme");
    html.setAttribute("data-theme", current === "dark" ? "light" : "dark");
});

// ── View switching ────────────────────────────────────────────
const singleViewBtn = document.getElementById("singleViewBtn");
const gridViewBtn   = document.getElementById("gridViewBtn");
const singleView    = document.getElementById("singleView");
const gridView      = document.getElementById("gridView");
const cameraSelect  = document.getElementById("cameraSelect");
const camLabel      = document.getElementById("camLabel");

let currentMode      = 'single';
let activeCameraIndex = 0;
let videoStreams      = [null, null, null, null];

singleViewBtn.addEventListener("click", () => {
    currentMode = 'single';
    singleView.classList.add("active");
    gridView.classList.remove("active");
    singleViewBtn.classList.add("active");
    gridViewBtn.classList.remove("active");
    updateSingleViewLabel();
    updateMainVideoStream();
});

gridViewBtn.addEventListener("click", () => {
    currentMode = 'grid';
    singleView.classList.remove("active");
    gridView.classList.add("active");
    singleViewBtn.classList.remove("active");
    gridViewBtn.classList.add("active");
    updatePlaceholders();
});

cameraSelect.addEventListener("change", (e) => {
    activeCameraIndex = parseInt(e.target.value);
    console.log("[cameraSelect] Changed to index", activeCameraIndex, "hasStream=", !!videoStreams[activeCameraIndex]);
    updateSingleViewLabel();
    if (currentMode === 'single') {
        console.log("[cameraSelect] In single mode, calling updateMainVideoStream");
        updateMainVideoStream();
        updatePlaceholders();
    }
});

function updateSingleViewLabel() {
    const hasSelectedStream = !!videoStreams[activeCameraIndex];
    camLabel.textContent = `CAM ${String(activeCameraIndex + 1).padStart(2, '0')} | ${hasSelectedStream ? 'LIVE FEED' : 'NO CAMERA DETECTED'}`;
}

function updateMainVideoStream() {
    const mainVideo = document.getElementById("mainVideo");
    const stream = videoStreams[activeCameraIndex] || null;
    console.log("[updateMainVideoStream] Setting mainVideo.srcObject to stream at index", activeCameraIndex, "stream=", !!stream);
    mainVideo.srcObject = stream;
    updateSingleViewLabel();
    updatePlaceholders();
}

// Click grid camera → switch to single view (only if camera exists)
document.querySelectorAll("#gridView .video-wrapper").forEach((wrapper, index) => {
    wrapper.addEventListener("click", () => {
        // Only switch to single view if this camera slot has a stream
        if (!videoStreams[index]) {
            console.log(`[gridClick] Cam ${index + 1} is empty, not switching`);
            return; // Don't switch to single view for empty cameras
        }
        cameraSelect.value = index;
        activeCameraIndex = index;
        updateSingleViewLabel();
        singleViewBtn.click();
    });
});

// ── Camera start / stop ───────────────────────────────────────
document.getElementById("startBtn").addEventListener("click", async () => {
    try {
        // First request permission — this populates device IDs in enumerateDevices
        const defaultStream = await navigator.mediaDevices.getUserMedia({ video: true });
        defaultStream.getTracks().forEach(t => t.stop()); // release immediately

        const devices      = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        if (videoDevices.length === 0) {
            showError("No cameras found.");
            return;
        }

        for (let i = 0; i < Math.min(videoDevices.length, 4); i++) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: videoDevices[i].deviceId } }
                });
                videoStreams[i] = stream;
                const vid = document.getElementById(`video${i + 1}`);
                if (vid) vid.srcObject = stream;
            } catch (camErr) {
                console.warn(`Camera ${i + 1} failed to initialize:`, camErr.message);
                // Continue with next camera instead of failing completely
            }
        }
        
        // Check if at least one camera started successfully
        const activeStreams = videoStreams.filter(s => s !== null);
        if (activeStreams.length === 0) {
            throw new Error("Could not start any camera. Check if cameras are in use, permissions are granted, or devices are connected.");
        }
        
        updateMainVideoStream();

        document.getElementById("startBtn").disabled  = true;
        document.getElementById("stopBtn").disabled   = false;
        document.getElementById("detectBtn").disabled = false;

        showError(null);
        updatePlaceholders();
    } catch (err) {
        showError("Error accessing cameras: " + (err.message || err.name || "Permission denied. Allow camera access in your browser."));
    }
});

document.getElementById("stopBtn").addEventListener("click", () => {
    stopDetection();
    videoStreams.forEach(s => s && s.getTracks().forEach(t => t.stop()));
    videoStreams = [null, null, null, null];
    document.querySelectorAll("video").forEach(v => v.srcObject = null);
    clearCanvas();

    document.getElementById("startBtn").disabled  = false;
    document.getElementById("stopBtn").disabled   = true;
    document.getElementById("detectBtn").disabled = true;

    updatePlaceholders();
});

// ── Detection loop ────────────────────────────────────────────
let detecting       = false;
let detectInterval  = null;
const DETECT_MS     = 500; // send a frame every 500ms

const detectBtn     = document.getElementById("detectBtn");
const captureCanvas = document.createElement("canvas");

detectBtn.addEventListener("click", () => {
    if (!detecting) {
        startDetection();
    } else {
        stopDetection();
        clearCanvas();
    }
});

function startDetection() {
    detecting = true;
    detectBtn.textContent   = "Stop Detect";
    detectBtn.style.background = "var(--red)";
    detectInterval = setInterval(detectFrame, DETECT_MS);
}

function stopDetection() {
    detecting = false;
    detectBtn.textContent  = "Detect";
    detectBtn.style.background = "";
    clearInterval(detectInterval);
    detectInterval = null;
}

async function detectFrame() {
    // In single view use mainVideo, in grid view use all active streams
    if (currentMode === 'single') {
        const video = document.getElementById("mainVideo");
        const canvas = document.getElementById("overlayCanvas");
        await detectOnVideo(video, canvas);
    } else {
        for (let i = 0; i < 4; i++) {
            if (!videoStreams[i]) continue;
            const video  = document.getElementById(`video${i + 1}`);
            const canvas = document.getElementById(`overlayCanvas${i + 1}`);
            await detectOnVideo(video, canvas);
        }
    }
}

// Update placeholders visibility for empty camera slots
function updatePlaceholders() {
    console.log("[updatePlaceholders] called");
    // main view placeholder
    const mainVideo = document.getElementById("mainVideo");
    const mainWrapper = document.getElementById("mainVideoWrapper");
    const mainHasStream = !!(mainVideo && mainVideo.srcObject);
    console.log("[updatePlaceholders] main: hasStream=", mainHasStream);
    if (mainWrapper) mainWrapper.classList.toggle("empty-state", !mainHasStream);

    // grid placeholders
    for (let i = 0; i < 4; i++) {
        const wrapper = document.querySelector(`#gridView .video-wrapper[data-cam="${i + 1}"]`);
        const vid  = document.getElementById(`video${i + 1}`);
        const hasStream = !!(videoStreams[i] && vid && vid.srcObject);
        console.log(`[updatePlaceholders] grid[${i}]: stream=${!!videoStreams[i]}, vid.srcObject=${!!(vid && vid.srcObject)}, hasStream=${hasStream}`);
        if (wrapper) wrapper.classList.toggle("empty-state", !hasStream);
    }
}

async function detectOnVideo(video, canvas) {
    if (!video || !video.srcObject || video.readyState < 2) return;

    captureCanvas.width  = video.videoWidth  || 640;
    captureCanvas.height = video.videoHeight || 480;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const imageBase64 = captureCanvas.toDataURL("image/jpeg", 0.8);

    try {
        // Determine which camera this is (by matching the video element)
        let cameraIndex = 0;
        if (currentMode === 'grid') {
            for (let i = 0; i < 4; i++) {
                if (document.getElementById(`video${i + 1}`) === video) {
                    cameraIndex = i;
                    break;
                }
            }
        } else {
            cameraIndex = activeCameraIndex;
        }

        const res  = await fetch("/predict", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ image: imageBase64, camera: cameraIndex })
        });
        const data = await res.json();
        drawOverlay(data, video, canvas);
        
        // Refresh logs after detection
        if (data.results && data.results.length > 0) {
            refreshLogs();
        }
    } catch (err) {
        showError("Backend unreachable. Is app.py running?");
        stopDetection();
    }
}

function drawOverlay(data, video, overlay) {
    overlay.width  = video.offsetWidth;
    overlay.height = video.offsetHeight;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (data.status === "no_face" || !data.results) return;

    const scaleX = overlay.width  / (video.videoWidth  || 640);
    const scaleY = overlay.height / (video.videoHeight || 480);

    data.results.forEach(r => {
        const [bx, by, bw, bh] = r.box;
        const color = r.label === "with_mask" ? "#00ff88" : "#ff4d4d";

        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.strokeRect(bx * scaleX, by * scaleY, bw * scaleX, bh * scaleY);

        const label = `${r.label} ${r.confidence}%`;
        ctx.font      = "bold 14px Consolas, monospace";
        const tw      = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(bx * scaleX, (by * scaleY) - 22, tw + 10, 20);
        ctx.fillStyle = "#000";
        ctx.fillText(label, bx * scaleX + 5, (by * scaleY) - 6);
    });
}

function clearCanvas() {
    ["overlayCanvas", "overlayCanvas1", "overlayCanvas2", "overlayCanvas3", "overlayCanvas4"].forEach(id => {
        const overlay = document.getElementById(id);
        if (overlay) {
            overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
        }
    });
}

function showError(msg) {
    const el = document.getElementById("errorMessage");
    if (!msg) { el.style.display = "none"; return; }
    el.textContent  = msg;
    el.style.display = "block";
}

// ── Logging Panel ──────────────────────────────────────
async function refreshLogs() {
    try {
        const res = await fetch("/logs");
        const data = await res.json();
        displayLogs(data.logs);
    } catch (err) {
        console.error("Failed to fetch logs:", err);
    }
}

function displayLogs(logs) {
    const panel = document.getElementById("logPanel");
    
    if (!logs || logs.length === 0) {
        panel.innerHTML = '<p class="log-empty">No detections yet</p>';
        return;
    }

    let html = "";
    logs.forEach(log => {
        const timestamp = log.timestamp;
        const cameraNum = log.camera + 1;
        const maskClass = log.label === "with_mask" ? "with-mask" : "no-mask";
        const maskText = log.label === "with_mask" ? "✓ MASK" : "✗ NO MASK";
        
        html += `
            <div class="log-entry ${maskClass}">
                <div class="log-time">${timestamp} | CAM ${cameraNum}</div>
                <div class="log-text">${maskText} (${log.confidence}%)</div>
            </div>
        `;
    });
    
    panel.innerHTML = html;
}

// Clear logs button
document.getElementById("clearLogsBtn").addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear all logs?")) return;
    try {
        await fetch("/logs/clear", { method: "POST" });
        displayLogs([]);
        console.log("Logs cleared");
    } catch (err) {
        showError("Failed to clear logs");
    }
});

// Load logs on page load
window.addEventListener("load", () => {
    refreshLogs();
});