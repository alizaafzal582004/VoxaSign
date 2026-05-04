import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

// Canvas elements
const video = document.getElementById("webcam");
const liveCanvas = document.getElementById("output_canvas");
const liveCtx = liveCanvas.getContext("2d");
const imageCanvas = document.getElementById("image_canvas");

// State
let handLandmarker;
let model;
let currentPrediction = "";
let lastVideoTime = -1;
let activeMode = null; // 'live' | 'upload'
let cameraStream = null;

// Expose prediction to parent UI
window.currentPrediction = "";
window.clearCurrentPrediction = () => {
    currentPrediction = "";
    window.currentPrediction = "";
    if (window.updatePrediction) window.updatePrediction("", 0);
};

const labelMap = [
    "A","B","Blank","C","D","E","F","G","H","I",
    "J","K","L","M","N","O","P","Q","R","S",
    "T","U","V","W","X","Y","Z"
];

// ── INIT ──
async function initialize() {
    console.log("🚀 Initializing VoxaSign...");
    try {
        console.log("Loading MediaPipe HandLandmarker...");
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        console.log("✅ MediaPipe HandLandmarker loaded");
        
        console.log("Loading TensorFlow model from ./web_model/model.json...");
        model = await tf.loadLayersModel('./web_model/model.json');
        console.log("✅ TensorFlow model loaded");
        console.log("✅ AI Engine Ready - You can now use the app!");
    } catch (error) {
        console.error("❌ Initialization failed:", error);
        alert("Failed to load AI models. Please check the console for details.\n\nMake sure you're running this from a web server (not file://)");
    }
}

// ── CAMERA ──
async function startCamera() {
    if (cameraStream) return; // already running
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = cameraStream;
        video.onloadedmetadata = () => {
            video.play();
            predictWebcam();
        };
    } catch (err) {
        console.error("Camera access denied:", err);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        video.srcObject = null;
    }
}

// ── MODE CHANGE LISTENER ──
window.addEventListener('modeChange', async (e) => {
    activeMode = e.detail;

    if (activeMode === 'live') {
        await startCamera();
    } else {
        stopCamera();
        currentPrediction = "";
        window.currentPrediction = "";
        if (window.updatePrediction) window.updatePrediction("", 0);
    }
});

// ── DRAW UI on canvas ──
function drawLandmarkBox(ctx, landmarks, w, h) {
    const x = landmarks.map(l => l.x * w);
    const y = landmarks.map(l => l.y * h);
    const minX = Math.min(...x), maxX = Math.max(...x);
    const minY = Math.min(...y), maxY = Math.max(...y);
    const pad = 20;

    // Glow box
    ctx.shadowColor = "#00c6ff";
    ctx.shadowBlur = 12;
    ctx.strokeStyle = "#00c6ff";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    // Use rect instead of roundRect for better browser compatibility
    ctx.rect(minX - pad, minY - pad, (maxX - minX) + pad * 2, (maxY - minY) + pad * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Landmark dots
    ctx.fillStyle = "rgba(0,198,255,0.8)";
    landmarks.forEach(l => {
        ctx.beginPath();
        ctx.arc(l.x * w, l.y * h, 3, 0, Math.PI * 2);
        ctx.fill();
    });
}

// ── PROCESS LANDMARKS ──
function processLandmarks(landmarks) {
    const wrist = landmarks[0];
    let coords = [];
    for (let i = 0; i < landmarks.length; i++) {
        coords.push(landmarks[i].x - wrist.x);
        coords.push(landmarks[i].y - wrist.y);
        coords.push(landmarks[i].z - wrist.z);
    }
    const maxVal = Math.max(...coords.map(Math.abs)) || 1;
    return coords.map(c => c / maxVal);
}

// ── INFERENCE ──
async function runInference(landmarks) {
    if (!model) {
        console.error("Model not loaded yet");
        return;
    }
    
    const inputData = processLandmarks(landmarks);
    const inputTensor = tf.tensor2d(inputData, [1, 63]);
    const prediction = model.predict(inputTensor);
    const scores = await prediction.data();
    const maxIdx = scores.indexOf(Math.max(...scores));
    const confidence = scores[maxIdx];
    const label = labelMap[maxIdx];

    console.log(`Prediction: ${label} (${(confidence * 100).toFixed(1)}%)`);

    if (confidence > 0.75 && label !== "Blank") {
        currentPrediction = label;
        window.currentPrediction = label;
        if (window.updatePrediction) window.updatePrediction(label, confidence);
    } else {
        currentPrediction = "";
        window.currentPrediction = "";
        if (window.updatePrediction) window.updatePrediction("", 0);
    }

    inputTensor.dispose();
    prediction.dispose();
}

// ── LIVE WEBCAM LOOP ──
async function predictWebcam() {
    if (activeMode !== 'live') return;

    if (video.currentTime !== lastVideoTime) {
        liveCanvas.width = video.videoWidth;
        liveCanvas.height = video.videoHeight;
        lastVideoTime = video.currentTime;

        const result = handLandmarker.detectForVideo(video, performance.now());
        liveCtx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);

        if (result.landmarks && result.landmarks.length > 0) {
            drawLandmarkBox(liveCtx, result.landmarks[0], liveCanvas.width, liveCanvas.height);
            await runInference(result.landmarks[0]);
        } else {
            currentPrediction = "";
            window.currentPrediction = "";
            if (window.updatePrediction) window.updatePrediction("", 0);
        }
    }

    if (activeMode === 'live') {
        window.requestAnimationFrame(predictWebcam);
    }
}

// ── IMAGE UPLOAD INFERENCE ──
document.getElementById("imageUpload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file || !handLandmarker || !model) {
        console.log("Upload skipped - not ready:", { file: !!file, handLandmarker: !!handLandmarker, model: !!model });
        return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        try {
            // The display canvas is handled by studio.html
            // We run inference on the original image
            await handLandmarker.setOptions({ runningMode: "IMAGE" });
            const result = await handLandmarker.detect(img);

            if (result.landmarks && result.landmarks.length > 0) {
                console.log("✓ Hand detected in uploaded image");
                // Draw box on image_canvas after it's been drawn by studio.html
                setTimeout(() => {
                    const ctx2 = imageCanvas.getContext('2d');
                    drawLandmarkBox(ctx2, result.landmarks[0], imageCanvas.width, imageCanvas.height);
                }, 100);
                await runInference(result.landmarks[0]);
            } else {
                console.log("⚠ No hand detected in uploaded image");
                if (window.updatePrediction) window.updatePrediction("", 0);
            }

            // Switch back to VIDEO mode for live camera
            await handLandmarker.setOptions({ runningMode: "VIDEO" });
        } catch (error) {
            console.error("Image inference error:", error);
            if (window.updatePrediction) window.updatePrediction("", 0);
        }
    };
});

initialize();


