# app.py
import os
import io
import json
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import cv2
import numpy as np
from pymongo import MongoClient
import gridfs
from scipy.spatial import distance as dist

# ------------------ Load Environment ------------------
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ------------------ MongoDB Setup ------------------
client = MongoClient(MONGO_URL)
db = client["beautygan_db"]
fs = gridfs.GridFS(db)
logs_collection = db["analysis_logs"]

# ------------------ Optional Groq Client ------------------
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    GROQ_AVAILABLE = groq_client is not None
except Exception:
    groq_client = None
    GROQ_AVAILABLE = False

# ------------------ MediaPipe Setup ------------------
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ------------------ Landmark Indices ------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_FULL = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ------------------ Utilities ------------------
def prepare_image(img_pil: Image.Image):
    img_pil = ImageOps.exif_transpose(img_pil)
    img_pil = img_pil.convert("RGB")
    image = np.array(img_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    elif max(h, w) < 480:
        scale = 480 / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return image

def detect_face_region(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if not results.detections:
        return image, None
    det = results.detections[0]
    box = det.location_data.relative_bounding_box
    h, w, _ = image.shape
    padding = 0.2
    x1 = max(0, int((box.xmin - padding*box.width)*w))
    y1 = max(0, int((box.ymin - padding*box.height)*h))
    x2 = min(w, int((box.xmin + (1+padding)*box.width)*w))
    y2 = min(h, int((box.ymin + (1+padding)*box.height)*h))
    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)

def get_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    h, w, _ = image.shape
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = np.array([(lm.x*w, lm.y*h) for lm in face_landmarks.landmark])
    return landmarks

# ------------------ Facial Analysis Functions ------------------
# Include all your previous functions here: 
# analyze_face_shape, calculate_angle, analyze_eye_shape, calculate_eyelid_curvature, 
# analyze_lip_shape, calculate_polygon_area, analyze_eyebrow_shape, analyze_skin_condition, get_region_points, etc.
# For brevity, I am skipping pasting all functions here. You can copy them from your existing code.
def analyze_face_shape(landmarks):
    """Comprehensive face shape analysis using multiple measurements"""
    try:
        # Key measurement points
        forehead_width = dist.euclidean(landmarks[21], landmarks[251])
        cheekbone_width = dist.euclidean(landmarks[234], landmarks[454])
        jawline_width = dist.euclidean(landmarks[172], landmarks[397])
        face_length = dist.euclidean(landmarks[10], landmarks[152])
        
        # Jaw angles
        left_jaw_angle = calculate_angle(landmarks[172], landmarks[152], landmarks[234])
        right_jaw_angle = calculate_angle(landmarks[397], landmarks[152], landmarks[454])
        avg_jaw_angle = (left_jaw_angle + right_jaw_angle) / 2
        
        # Chin prominence
        chin_point = landmarks[152]
        jaw_left = landmarks[172]
        jaw_right = landmarks[397]
        chin_prominence = dist.euclidean(chin_point, [(jaw_left[0] + jaw_right[0])/2, jaw_left[1]])
        
        # Ratios
        width_ratio = cheekbone_width / face_length
        jaw_to_cheek = jawline_width / cheekbone_width
        forehead_to_cheek = forehead_width / cheekbone_width
        
        # Classification logic
        if width_ratio > 0.75:
            if jaw_to_cheek > 0.90 and avg_jaw_angle > 120:
                return "square", 0.85
            elif forehead_to_cheek < 0.85 and jaw_to_cheek < 0.85:
                return "diamond", 0.82
            else:
                return "round", 0.80
        elif width_ratio < 0.60:
            if jaw_to_cheek < 0.70:
                return "heart", 0.83
            elif avg_jaw_angle < 100:
                return "triangular", 0.78
            else:
                return "oblong", 0.81
        else:
            if forehead_to_cheek > 1.05:
                return "inverted_triangle", 0.79
            elif jaw_to_cheek > 0.95:
                return "rectangular", 0.82
            elif forehead_to_cheek < 0.92 and jaw_to_cheek < 0.92:
                return "diamond", 0.84
            elif abs(forehead_width - cheekbone_width) < 5 and abs(cheekbone_width - jawline_width) < 5:
                return "oval", 0.88
            else:
                return "oval", 0.75
                
    except Exception as e:
        return "unknown", 0.0


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


# ------------------- Eye Shape Analysis -------------------

def analyze_eye_shape(landmarks):
    """Advanced eye shape analysis"""
    try:
        left_eye_pts = landmarks[LEFT_EYE]
        right_eye_pts = landmarks[RIGHT_EYE]
        
        def analyze_single_eye(eye_pts):
            # Eye aspect ratio
            vertical_1 = dist.euclidean(eye_pts[1], eye_pts[5])
            vertical_2 = dist.euclidean(eye_pts[2], eye_pts[4])
            horizontal = dist.euclidean(eye_pts[0], eye_pts[3])
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            
            # Eye corners angle
            inner = eye_pts[0]
            outer = eye_pts[3]
            top = eye_pts[1]
            bottom = eye_pts[5]
            
            # Upturned vs downturned
            corner_angle = np.arctan2(outer[1] - inner[1], outer[0] - inner[0])
            corner_angle_deg = np.degrees(corner_angle)
            
            # Eyelid exposure
            eyelid_curve = calculate_eyelid_curvature(eye_pts)
            
            return ear, corner_angle_deg, eyelid_curve
        
        left_ear, left_angle, left_curve = analyze_single_eye(left_eye_pts)
        right_ear, right_angle, right_curve = analyze_single_eye(right_eye_pts)
        
        avg_ear = (left_ear + right_ear) / 2
        avg_angle = (abs(left_angle) + abs(right_angle)) / 2
        avg_curve = (left_curve + right_curve) / 2
        
        # Classification
        if avg_ear < 0.18:
            if avg_curve < 0.15:
                eye_type = "monolid"
                confidence = 0.82
            else:
                eye_type = "hooded"
                confidence = 0.85
        elif avg_ear < 0.22:
            if avg_angle < 2:
                eye_type = "almond"
                confidence = 0.88
            elif avg_angle > 5:
                eye_type = "upturned"
                confidence = 0.83
            else:
                eye_type = "downturned"
                confidence = 0.80
        elif avg_ear < 0.28:
            if avg_curve > 0.25:
                eye_type = "round"
                confidence = 0.86
            else:
                eye_type = "almond"
                confidence = 0.82
        else:
            eye_type = "protruding"
            confidence = 0.79
            
        # Check for deep-set
        if avg_curve < 0.12 and avg_ear < 0.20:
            eye_type = "deep_set"
            confidence = 0.81
            
        return eye_type, confidence
        
    except Exception as e:
        return "unknown", 0.0


def calculate_eyelid_curvature(eye_pts):
    """Calculate eyelid curvature"""
    try:
        upper_pts = eye_pts[1:4]
        x = [p[0] for p in upper_pts]
        y = [p[1] for p in upper_pts]
        
        if len(x) >= 3:
            coeffs = np.polyfit(x, y, 2)
            curvature = abs(coeffs[0])
            return curvature
    except:
        return 0.0
    return 0.0


# ------------------- Lip Shape Analysis -------------------

def analyze_lip_shape(landmarks):
    """Detailed lip shape analysis"""
    try:
        upper_lip = landmarks[UPPER_LIP]
        lower_lip = landmarks[LOWER_LIP]
        full_lip = landmarks[LIPS_FULL]
        
        # Lip width and height
        lip_width = dist.euclidean(landmarks[61], landmarks[291])
        
        upper_height = max([p[1] for p in upper_lip]) - min([p[1] for p in upper_lip])
        lower_height = max([p[1] for p in lower_lip]) - min([p[1] for p in lower_lip])
        
        # Cupid's bow prominence
        cupids_bow_center = landmarks[0]
        cupids_bow_left = landmarks[37]
        cupids_bow_right = landmarks[267]
        
        bow_depth = abs(cupids_bow_center[1] - (cupids_bow_left[1] + cupids_bow_right[1]) / 2)
        bow_prominence = bow_depth / upper_height if upper_height > 0 else 0
        
        # Fullness ratio
        lip_area = calculate_polygon_area(full_lip)
        fullness_ratio = lip_area / (lip_width ** 2) if lip_width > 0 else 0
        
        # Upper to lower ratio
        ul_ratio = upper_height / lower_height if lower_height > 0 else 1.0
        
        # Classification
        if fullness_ratio < 0.025:
            lip_type = "thin"
            confidence = 0.84
        elif fullness_ratio < 0.040:
            if ul_ratio > 1.3:
                lip_type = "top_heavy"
                confidence = 0.81
            elif ul_ratio < 0.7:
                lip_type = "bottom_heavy"
                confidence = 0.82
            elif bow_prominence > 0.15:
                lip_type = "heart_shaped"
                confidence = 0.80
            else:
                lip_type = "round"
                confidence = 0.83
        else:
            if bow_prominence > 0.20:
                lip_type = "full_heart"
                confidence = 0.85
            else:
                lip_type = "full"
                confidence = 0.87
                
        return lip_type, confidence
        
    except Exception as e:
        return "unknown", 0.0


def calculate_polygon_area(points):
    """Calculate area of polygon"""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))


# ------------------- Eyebrow Shape Analysis -------------------

def analyze_eyebrow_shape(landmarks):
    """Comprehensive eyebrow shape analysis"""
    try:
        left_brow = landmarks[LEFT_EYEBROW]
        right_brow = landmarks[RIGHT_EYEBROW]
        
        def analyze_single_brow(brow_pts):
            # Fit spline to eyebrow
            x = [p[0] for p in brow_pts]
            y = [p[1] for p in brow_pts]
            
            # Calculate arch height
            arch_height = min(y) - max(y)
            brow_width = max(x) - min(x)
            arch_ratio = abs(arch_height) / brow_width if brow_width > 0 else 0
            
            # Find peak position
            min_y_idx = np.argmin(y)
            peak_position = (x[min_y_idx] - min(x)) / brow_width if brow_width > 0 else 0.5
            
            # Tail angle
            tail_angle = calculate_angle(brow_pts[0], brow_pts[-1], brow_pts[-2])
            
            # Thickness (approximate)
            thickness = arch_height / len(brow_pts) if len(brow_pts) > 0 else 0
            
            return arch_ratio, peak_position, tail_angle, thickness
        
        left_arch, left_peak, left_tail, left_thick = analyze_single_brow(left_brow)
        right_arch, right_peak, right_tail, right_thick = analyze_single_brow(right_brow)
        
        avg_arch = (left_arch + right_arch) / 2
        avg_peak = (left_peak + right_peak) / 2
        
        # Classification
        if avg_arch < 0.03:
            brow_type = "straight"
            confidence = 0.85
        elif avg_arch < 0.08:
            if avg_peak < 0.4:
                brow_type = "soft_angled"
                confidence = 0.82
            elif avg_peak > 0.6:
                brow_type = "low_arch"
                confidence = 0.80
            else:
                brow_type = "soft_arch"
                confidence = 0.84
        elif avg_arch < 0.15:
            if avg_peak < 0.35:
                brow_type = "high_arch"
                confidence = 0.86
            elif 0.4 < avg_peak < 0.6:
                brow_type = "rounded"
                confidence = 0.83
            else:
                brow_type = "s_shaped"
                confidence = 0.79
        else:
            brow_type = "sharp_arch"
            confidence = 0.81
            
        return brow_type, confidence
        
    except Exception as e:
        return "unknown", 0.0


# ------------------- Skin Condition Analysis -------------------

def analyze_skin_condition(image, landmarks):
    """Analyze skin condition from facial regions"""
    conditions = []
    try:
        h, w = image.shape[:2]
        
        # Define skin sampling regions (cheeks, forehead)
        left_cheek_region = get_region_points(landmarks, [234, 227, 137, 177, 215])
        right_cheek_region = get_region_points(landmarks, [454, 356, 454, 366, 435])
        forehead_region = get_region_points(landmarks, [10, 338, 297, 251, 21, 54])
        
        regions = [left_cheek_region, right_cheek_region, forehead_region]
        
        # Sample colors and textures
        region_colors = []
        region_textures = []
        
        for region in regions:
            if len(region) > 0:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(region, dtype=np.int32)], 255)
                
                # Color analysis
                mean_color = cv2.mean(image, mask=mask)[:3]
                region_colors.append(mean_color)
                
                # Texture analysis (variance)
                roi = cv2.bitwise_and(image, image, mask=mask)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                texture_var = np.var(gray_roi[mask > 0])
                region_textures.append(texture_var)
        
        if region_colors:
            avg_color = np.mean(region_colors, axis=0)
            avg_texture = np.mean(region_textures)
            
            b, g, r = avg_color
            
            # Redness detection
            if r > g + 20 and r > b + 20:
                conditions.append("redness")
            
            # Uneven skin tone (high texture variance)
            if avg_texture > 500:
                conditions.append("uneven_tone")
            
            # Dark spots / hyperpigmentation
            color_std = np.std([np.mean(c) for c in region_colors])
            if color_std > 15:
                conditions.append("hyperpigmentation")
            
            # Oiliness (higher blue channel in certain lighting)
            if b > (r + g) / 2 + 10:
                conditions.append("oily")
            
            # Dryness indicators (low variance, flat appearance)
            if avg_texture < 200:
                conditions.append("dry")
        
        if not conditions:
            conditions = ["clear"]
            
    except Exception as e:
        conditions = ["not_assessed"]
    
    return conditions


def get_region_points(landmarks, indices):
    """Get landmark points for a specific region"""
    return [landmarks[i].astype(int) for i in indices if i < len(landmarks)]



def sample_region_color(image, points):
    """Sample average color from region"""
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
        mean_color = cv2.mean(image, mask=mask)[:3]
        return np.array(mean_color)
    except:
        return np.array([0, 0, 0])

# ------------------ Groq AI Feedback ------------------
def get_ai_wellness_feedback(analysis_results):
    if not GROQ_AVAILABLE:
        return {
            "wellness_score": "N/A",
            "feedback": "AI feedback unavailable. Set GROQ_API_KEY in env.",
            "reasoning": ""
        }
    try:
        prompt = f"""You are a facial wellness and beauty analysis expert. 
Based on the following facial analysis data, provide a wellness score and brief 2-3 sentence feedback.
Analysis Data: {json.dumps(analysis_results, indent=2)}
Respond ONLY in JSON format with keys: wellness_score, feedback, reasoning.
"""
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system","content": "You are a professional facial wellness analyst."},
                {"role": "user","content": prompt}
            ],
            temperature=0.7
        )
        content = (resp.choices[0].message.content if resp.choices else "").strip()
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            ai_response = json.loads(json_str)
        except:
            ai_response = {"wellness_score":"7.5","feedback":content[:300],"reasoning":"Fallback"}
        return ai_response
    except:
        return {"wellness_score":"Error","feedback":"Could not generate AI feedback","reasoning":"Check GROQ_API_KEY and model"}
def analyze_face_complete(img: Image.Image):
    """Complete face analysis pipeline"""
    try:
        # Prepare image
        image = prepare_image(img)
        
        # Try with face detection crop first
        cropped, box = detect_face_region(image)
        
        # Get landmarks
        landmarks = None
        used_image = None
        
        if cropped is not None and cropped.size > 0:
            landmarks = get_landmarks(cropped)
            if landmarks is not None:
                used_image = cropped
        
        if landmarks is None:
            landmarks = get_landmarks(image)
            used_image = image
        
        if landmarks is None:
            return {
                "error": "No face detected. Please upload a clear, frontal face image.",
                "face_detected": False
            }
        
        # Perform all analyses
        face_shape, face_conf = analyze_face_shape(landmarks)
        eye_shape, eye_conf = analyze_eye_shape(landmarks)
        lip_shape, lip_conf = analyze_lip_shape(landmarks)
        eyebrow_shape, brow_conf = analyze_eyebrow_shape(landmarks)
        skin_conditions = analyze_skin_condition(used_image, landmarks)
    
        
        # Compile results
        results = {
            "face_detected": True,
            "face_shape": {
                "type": face_shape,
                "confidence": round(face_conf, 2)
            },
            "eye_shape": {
                "type": eye_shape,
                "confidence": round(eye_conf, 2)
            },
            "lip_shape": {
                "type": lip_shape,
                "confidence": round(lip_conf, 2)
            },
            "eyebrow_shape": {
                "type": eyebrow_shape,
                "type_label": eyebrow_shape
            },
            "skin_condition": skin_conditions,
            "image_info": {
                "original_size": image.shape[:2],
                "processed_size": used_image.shape[:2],
                "landmarks_count": len(landmarks)
            }
        }
        
        # Get AI wellness feedback
        ai_feedback = get_ai_wellness_feedback(results)
        results["ai_wellness_analysis"] = ai_feedback
        
        return results
        
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "face_detected": False
        }

# ------------------ Image & Log Helpers ------------------
def save_image_to_gridfs(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    file_id = fs.put(buf, content_type="image/png")
    return str(file_id)

def log_analysis(data: dict):
    logs_collection.insert_one(data)

# ------------------ FastAPI App ------------------
app = FastAPI(title="Advanced Facial Analysis API")

@app.post("/analyze/facial_features")
async def analyze_facial_features(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        img_id = save_image_to_gridfs(img)
        results = analyze_face_complete(img)
        results["source_image_id"] = img_id
        log_analysis(results)
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()})

@app.post("/analyze/ai_wellness")
async def analyze_ai_wellness(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        img_id = save_image_to_gridfs(img)
        results = analyze_face_complete(img)
        results["source_image_id"] = img_id
        ai_feedback = results.get("ai_wellness_analysis", {"wellness_score":"N/A"})
        log_analysis({"ai_feedback": ai_feedback, "source_image_id": img_id})
        return JSONResponse(ai_feedback)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()})

@app.get("/")
def root():
    return {"message":"Welcome to Advanced Facial Analysis API"}
