import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

// UI Elements
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const alphabetBox = document.getElementById("predictedAlphabet");
const wordBox = document.getElementById("cumulativeWord");
const sentenceBox = document.getElementById("generatedSentence");
const imageUpload = document.getElementById("imageUpload");
const clearCanvasBtn = document.getElementById("clearCanvasBtn"); // NEW

// Variables
let handLandmarker;
let model;
let currentPrediction = "";
let lastVideoTime = -1;
let isUsingUpload = false; // Flag to track if we are showing an image or video

const labelMap = ["A", "B", "Blank", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];

async function initialize() {
    try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" },
            runningMode: "VIDEO", 
            numHands: 1
        });
        model = await tf.loadLayersModel('./web_model/model.json');
        setupCamera();
    } catch (error) {
        console.error("Initialization failed:", error);
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    video.onloadedmetadata = () => { video.play(); predictWebcam(); };
}

function drawUI(landmarks) {
    const x = landmarks.map(l => l.x * canvas.width);
    const y = landmarks.map(l => l.y * canvas.height);
    const minX = Math.min(...x); const maxX = Math.max(...x);
    const minY = Math.min(...y); const maxY = Math.max(...y);
    ctx.strokeStyle = "#e74c3c"; ctx.lineWidth = 4;
    ctx.strokeRect(minX - 20, minY - 20, (maxX - minX) + 40, (maxY - minY) + 40);
}

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

async function runInference(landmarks) {
    const inputData = processLandmarks(landmarks);
    const inputTensor = tf.tensor2d(inputData, [1, 63]);
    const prediction = model.predict(inputTensor);
    const scores = await prediction.data();
    let maxIdx = scores.indexOf(Math.max(...scores));
    const detectedLabel = labelMap[maxIdx];

    if (scores[maxIdx] > 0.75 && detectedLabel !== "Blank") {
        alphabetBox.value = detectedLabel;
        currentPrediction = detectedLabel;
    } else {
        alphabetBox.value = "---";
        currentPrediction = "";
    }
    inputTensor.dispose();
    prediction.dispose();
}

// WEBCAM LOOP
async function predictWebcam() {
    // Only run webcam loop if we aren't displaying an uploaded image
    if (!isUsingUpload && video.currentTime !== lastVideoTime) {
        canvas.width = video.videoWidth; 
        canvas.height = video.videoHeight;
        lastVideoTime = video.currentTime;
        const result = handLandmarker.detectForVideo(video, performance.now());
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (result.landmarks && result.landmarks.length > 0) {
            drawUI(result.landmarks[0]);
            await runInference(result.landmarks[0]);
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

// IMAGE UPLOAD HANDLER
imageUpload.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    isUsingUpload = true; // Stop webcam processing
    clearCanvasBtn.style.display = "flex"; // Show the delete icon

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        await handLandmarker.setOptions({ runningMode: "IMAGE" });
        const result = await handLandmarker.detect(img);
        if (result.landmarks && result.landmarks.length > 0) {
            drawUI(result.landmarks[0]);
            await runInference(result.landmarks[0]);
            // NOTICE: I removed the wordBox.value update here. 
            // The user must now click "Add Letter" manually.
        }
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    };
});

// NEW: CLEAR IMAGE ICON LOGIC
clearCanvasBtn.addEventListener("click", () => {
    isUsingUpload = false; // Restart webcam loop
    clearCanvasBtn.style.display = "none";
    imageUpload.value = ""; // Reset file input
    alphabetBox.value = "";
    currentPrediction = "";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// BUTTONS
document.getElementById("submitBtn").addEventListener("click", () => {
    if (currentPrediction) wordBox.value += currentPrediction;
});

document.getElementById("clearBtn").addEventListener("click", () => {
    wordBox.value = ""; sentenceBox.value = ""; alphabetBox.value = ""; currentPrediction = "";
});

document.getElementById("generateBtn").addEventListener("click", () => {
    const word = wordBox.value;
    if (word) {
        sentenceBox.value = `The user signed: ${word}`;
        const speech = new SpeechSynthesisUtterance(word);
        window.speechSynthesis.speak(speech);
    }
});

initialize();