from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import io
import uuid
import cv2
import numpy as np
<<<<<<< Updated upstream

=======
import pymongo
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.yolo_v11_detection import run_yolo_v11_detection
from functools import lru_cache
from ultralytics import YOLO
>>>>>>> Stashed changes
from app.core.config import settings
from app.services.vectorizer import get_vectorizer
from app.services.datastore import get_datastore
from app.services.detection import run_detection

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


# ===========================
# Helper function for parallel processing
# ===========================

def extract_embedding_only(
    crop: Image.Image,
    bbox_list: List[int],
    label: str,
    conf: float,
    vectorizer,
) -> Dict[str, Any]:
    """
    Extract embedding from a single crop (no database save).
    This function is designed to be run in parallel threads.
    Returns: dict with vector, bbox_list, label, conf, crop_id
    """
    try:
        # Extract embedding
        vector = vectorizer.get_image_embedding(crop)
        
        # Generate unique ID
        crop_id = str(uuid.uuid4())
        
        return {
            "crop_id": crop_id,
            "vector": vector,
            "metadata": {
                "bbox": bbox_list,
                "label": label,
                "confidence": conf,
            },
            "bbox_list": bbox_list,
        }
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


# ===========================
# Models
# ===========================

class IndexShelfResponse(BaseModel):
    image_id: str
    num_detections: int
    num_indexed: int


class ShelfInfo(BaseModel):
    shelf_id: str
    num_objects: int


class MatchResult(BaseModel):
    # include shelf_id so we can search across all shelves
    shelf_id: str
    bbox: List[int]
    score: float


class SearchResponse(BaseModel):
    matches: List[MatchResult]


# ===========================
# Shelf indexing & listing
# ===========================

@router.post("/index-shelf", response_model=IndexShelfResponse)
async def index_shelf(
    file: UploadFile = File(...),
    prompt: str = "Bottle",
    # box_thresh: float = 0.3, <-- CHANGED: Removed, as run_detection no longer uses it
):
    """
    Upload a full shelf image, detect objects with GroundingDINO,
    and index them in the vector DB as a new shelf (identified by image_id).
    """
    # <-- CHANGED: Removed box_thresh validation
    # if not (0.0 <= box_thresh <= 1.0):
    #     raise HTTPException(status_code=400, detail="box_thresh must be between 0.0 and 1.0")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # <-- CHANGED: box_thresh is no longer passed
    detections = run_detection(image, prompt=prompt) 

    image_id = str(uuid.uuid4())
    vectorizer = get_vectorizer()
    datastore = get_datastore()

    # Prepare all crops and metadata for parallel processing
    crop_tasks = []
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        label = detections.data["class_name"][i]
        conf = float(detections.confidence[i])

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        # Skip very tiny boxes
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        crop = image.crop((x1, y1, x2, y2))
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        vector = vectorizer.get_image_embedding(crop)

        crop_id = str(uuid.uuid4())
        bbox_list = [x1, y1, x2, y2]

        datastore.save_object(
            image_id=image_id,
            crop_id=crop_id,
            vector=vector,
            metadata={
                "bbox": bbox_list,
                "label": label,
                "confidence": conf,
            },
        )

        all_bboxes.append(bbox_list)
        num_indexed += 1
=======

        crop_tasks.append((crop, bbox_list, label, conf))

=======

        crop_tasks.append((crop, bbox_list, label, conf))

>>>>>>> Stashed changes
=======

        crop_tasks.append((crop, bbox_list, label, conf))

>>>>>>> Stashed changes
=======

        crop_tasks.append((crop, bbox_list, label, conf))

>>>>>>> Stashed changes
=======

        crop_tasks.append((crop, bbox_list, label, conf))

>>>>>>> Stashed changes
=======

        crop_tasks.append((crop, bbox_list, label, conf))

>>>>>>> Stashed changes
=======

        crop_tasks.append((crop, bbox_list, label, conf))

>>>>>>> Stashed changes
    # Step 1: Extract embeddings in parallel (no database operations)
    all_bboxes: List[List[int]] = []
    objects_to_save: List[Dict[str, Any]] = []
    
    # Use ThreadPoolExecutor for parallel embedding extraction
    max_workers = min(len(crop_tasks), 8)  # Limit to 8 threads
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all embedding extraction tasks
        futures = [
            executor.submit(
                extract_embedding_only,
                crop,
                bbox_list,
                label,
                conf,
                vectorizer,
            )
            for crop, bbox_list, label, conf in crop_tasks
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                objects_to_save.append(result)
                all_bboxes.append(result["bbox_list"])
    
    # Step 2: Batch save all objects to database at once (much faster!)
    if objects_to_save:
        datastore.batch_save_objects(image_id, objects_to_save)
    
    num_indexed = len(objects_to_save)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    # Save this shelf image to disk under its image_id
    settings.SHELF_DIR.mkdir(parents=True, exist_ok=True)
    shelf_path = settings.SHELF_DIR / f"{image_id}.png"
    image.save(shelf_path)

    return IndexShelfResponse(
        image_id=image_id,
        num_detections=len(detections),
        num_indexed=num_indexed,
    )


@router.get("/shelves", response_model=List[ShelfInfo])
async def list_shelves():
    """List all shelves that have indexed objects."""
    datastore = get_datastore()
    raw = datastore.list_shelves()
    return [ShelfInfo(**s) for s in raw]


# ===========================
# Search (JSON, global-capable)
# ===========================

@router.post("/search", response_model=SearchResponse)
async def search_similar(
    file: UploadFile = File(...),
    shelf_id: str | None = None,
    # max_results: int = 10, <-- CHANGED: Removed
    match_threshold: float = 0.19,
    search_all: bool = False,
):
    """
    Upload a query image and search for visually similar objects.
    Returns ALL matches found above the match_threshold.

    - If search_all == False: search only within the given shelf_id.
    - If search_all == True: search across the whole database (all shelves),
      ignoring shelf_id.
    ...
    """
    if match_threshold <= 0:
        raise HTTPException(status_code=400, detail="match_threshold must be > 0 (distance).")

    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    # When not doing global search, enforce a valid shelf_id
    if not search_all:
        if not shelf_id or shelf_id.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="shelf_id is required when search_all is False",
            )
        if not datastore.has_shelf(shelf_id):
            raise HTTPException(status_code=404, detail=f"Shelf '{shelf_id}' not found")

    query_vector = vectorizer.get_image_embedding(query_image)
    
    # <-- CHANGED: Ask for a large fixed number of results to filter.
    # Adjust '1000' if your datastore has more items.
    results = datastore.query_similar(query_vector, n_results=1000)

    matches: List[MatchResult] = []

    for res in results:
        data = res["data"]
        parent_id = data.get("parent_image_id")
        if parent_id is None:
            continue

        # If not global search, restrict to a single shelf
        if not search_all and parent_id != shelf_id:
            continue

        bbox = data["bbox"]
        score = float(res["score"])

        # For this distance metric: smaller = more similar
        if score < match_threshold:
            matches.append(
                MatchResult(
                    shelf_id=parent_id,
                    bbox=bbox,
                    score=score,
                )
            )
        # <-- CHANGED: Removed the break condition
        #   if len(matches) >= max_results:
        #       break

    return SearchResponse(matches=matches)


# ===========================
# Search (Visual, GLOBAL)
# ===========================

@router.post("/search-visual")
async def search_visual(
    file: UploadFile = File(...),
    # max_results: int = 10, <-- CHANGED: Removed
    match_threshold: float = 0.19,
    only_matches: bool = True,
):
    """
    GLOBAL visual search.
    ...
    - Return that shelf image with bounding boxes drawn for ALL matches
      on that shelf.
    ...
    """
    if match_threshold <= 0:
        raise HTTPException(status_code=400, detail="match_threshold must be > 0 (distance).")

    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    query_vector = vectorizer.get_image_embedding(query_image)
    
    # <-- CHANGED: Ask for a large fixed number of results.
    # Adjust '5000' if your datastore has more items.
    results = datastore.query_similar(query_vector, n_results=5000)

    if not results:
        raise HTTPException(status_code=404, detail="No indexed objects found in database")

    # Pick the single best match overall to decide which shelf to visualize
    best = min(results, key=lambda r: float(r["score"]))
    target_shelf_id = best["data"].get("parent_image_id")
    if not target_shelf_id:
        raise HTTPException(status_code=500, detail="Indexed object has no parent_image_id")

    # Load that shelf image from disk
    shelf_path = settings.SHELF_DIR / f"{target_shelf_id}.png"
    if not shelf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Shelf image for '{target_shelf_id}' not found on disk",
        )

    shelf_image = Image.open(shelf_path).convert("RGB")
    draw_image = cv2.cvtColor(np.array(shelf_image), cv2.COLOR_RGB2BGR)

    # num_drawn = 0 <-- CHANGED: Removed

    # Now draw only detections that belong to the chosen shelf
    for res in results:
        data = res["data"]
        if data.get("parent_image_id") != target_shelf_id:
            continue

        bbox = data["bbox"]
        score = float(res["score"])
        x1, y1, x2, y2 = map(int, bbox)

        if score < match_threshold:
            color = (0, 255, 0)  # GREEN
        else:
<<<<<<< Updated upstream
            if only_matches:
                # Skip non-matches when only_matches is True
=======
            try:
                mongo_db = mongo_client.get_default_database()
            except Exception:
                mongo_db = mongo_client["cstore-ai"]

        collection = mongo_db["single_query_images"]
        return collection

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

@lru_cache(maxsize=1)
def get_yolo_v11_model() -> YOLO:
    # USE YOUR ACTUAL FILE PATH HERE
    model_path = "models/yolo/yolo_v11_best.pt"
    # or an absolute path if necessary, but keep it the same as Colab’s weights
    return YOLO(model_path)

@router.post("/index-shelf-yolo", response_model=IndexShelfResponse)
async def index_shelf_yolo(
    file: UploadFile = File(...),
    box_thresh: float = 0.2,
):
    """
    Upload a full shelf image, detect objects with YOLO v11,
    and index them in the vector DB as a new shelf (identified by image_id).
    """
    if not (0.0 <= box_thresh <= 1.0):
        raise HTTPException(status_code=400, detail="box_thresh must be between 0.0 and 1.0")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save uploaded file to a temp path (so YOLO uses 'source=path' like in Colab)
    try:
        suffix = Path(file.filename).suffix or ".jpg"
    except Exception:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Also load as PIL for cropping and saving shelf image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = get_yolo_v11_model()

    # --- YOLO PREDICT (Colab-style) ---
    results = model.predict(
        source=tmp_path,        # <--- path, same as Colab pattern
        imgsz=640,
        conf=box_thresh,        # equivalent of confidence threshold
        save=False,             # we handle saving ourselves
        show_labels=False,
        show_conf=False,
        verbose=False,
    )

    if not results:
        raise HTTPException(status_code=500, detail="YOLO v11 returned no results")

    result = results[0]
    boxes = result.boxes

    # DEBUG: you can temporarily print or log this to verify
    print("YOLO result:", result)
    print("Number of boxes:", 0 if boxes is None else len(boxes))

    image_id = str(uuid.uuid4())

    # If no detections, still save shelf image but index zero
    if boxes is None or len(boxes) == 0:
        shelves_dir = settings.SHELF_DIR
        shelves_dir.mkdir(parents=True, exist_ok=True)
        out_path = shelves_dir / f"{image_id}.png"
        image.save(out_path)

        return IndexShelfResponse(
            image_id=image_id,
            num_detections=0,
            num_indexed=0,
        )

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    names = result.names  # {class_id: class_name}

    # Prepare all crops and metadata for parallel processing
    crop_tasks = []
    for i in range(len(boxes)):
        box = boxes[i]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id))

        bbox_list = [int(x1), int(y1), int(x2), int(y2)]
        crop = image.crop((bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]))
        
        crop_tasks.append((crop, bbox_list, label, conf))

    # Step 1: Extract embeddings in parallel (no database operations)
    objects_to_save: List[Dict[str, Any]] = []
    
    # Use ThreadPoolExecutor for parallel embedding extraction
    max_workers = min(len(crop_tasks), 8)  # Limit to 8 threads
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all embedding extraction tasks
        futures = [
            executor.submit(
                extract_embedding_only,
                crop,
                bbox_list,
                label,
                conf,
                vectorizer,
            )
            for crop, bbox_list, label, conf in crop_tasks
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                objects_to_save.append(result)
    
    # Step 2: Batch save all objects to database at once (much faster!)
    if objects_to_save:
        datastore.batch_save_objects(image_id, objects_to_save)
    
    num_indexed = len(objects_to_save)

    shelves_dir = settings.SHELF_DIR
    shelves_dir.mkdir(parents=True, exist_ok=True)
    out_path = shelves_dir / f"{image_id}.png"
    image.save(out_path)

    return IndexShelfResponse(
        image_id=image_id,
        num_detections=len(boxes),
        num_indexed=num_indexed,
    )


@router.post("/ModelTraining")
async def upload_single_query_image(
    modelName: str = Form(...),
    skuDescription: str = Form(...),
    skuId: str = Form(...),
    tenantId: str = Form(...),

    # NEW optional fields (only these three are optional)
    clientId: str | None = Form(None),
    categoryId: str | None = Form(None),
    brandId: str | None = Form(None),

    image: UploadFile = File(...),
):
    """Upload a single query image with SKU + tenant info and store in MongoDB (base64)."""
    if not settings.MONGO_URI:
        raise HTTPException(status_code=500, detail="MongoDB URI is not configured")

    collection = get_single_query_images_collection()

    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    MAX_SIZE = 5 * 1024 * 1024
    file_bytes = await image.read()
    if len(file_bytes) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image file too large")

    image_b64 = base64.b64encode(file_bytes).decode("utf-8")

    doc_id = str(uuid.uuid4())
    document = {
        "_id": doc_id,
        "modelName": modelName,
        "skuDescription": skuDescription,
        "skuId": skuId,
        "tenantId": tenantId,
        "clientId": clientId,
        "categoryId": categoryId,
        "brandId": brandId,
        "filename": image.filename,
        "content_type": image.content_type,
        "image_base64": image_b64,
        "created_at": datetime.utcnow(),
    }

    try:
        result = collection.insert_one(document)
    except OperationFailure as e:
        try:
            errmsg = e.details.get("errmsg", "")
        except Exception:
            errmsg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"MongoDB write error: {errmsg}",
        )

    return {
        "id": str(result.inserted_id),
        "status": "stored",
    }





@router.get("/AvailableModel", response_model=List[SingleQueryImageOut])
async def list_query_images(
    modelName: str,
    skuId: str | None = None,
    tenantId: str | None = None,
    skuDescription: str | None = None,
    clientId: str | None = None,
    categoryId: str | None = None,
    brandId: str | None = None,
):
    """
    Return stored query images filtered by:
    - required: modelName
    - optional: skuId, tenantId, skuDescription, clientId, categoryId, brandId
    """
    collection = get_single_query_images_collection()

    # Build MongoDB query dynamically
    query: Dict[str, str] = {"modelName": modelName}

    if skuId:
        query["skuId"] = skuId
    if tenantId:
        query["tenantId"] = tenantId
    if skuDescription:
        query["skuDescription"] = skuDescription
    if clientId:
        query["clientId"] = clientId
    if categoryId:
        query["categoryId"] = categoryId
    if brandId:
        query["brandId"] = brandId

    try:
        docs = list(collection.find(query))
    except OperationFailure as e:
        errmsg = getattr(e, "details", {}).get("errmsg", str(e))
        raise HTTPException(status_code=500, detail=f"MongoDB read error: {errmsg}")

    results: List[SingleQueryImageOut] = []

    for doc in docs:
        results.append(
            SingleQueryImageOut(
                id=str(doc.get("_id")),
                modelName=doc.get("modelName", ""),
                skuDescription=doc.get("skuDescription"),
                skuId=doc.get("skuId"),
                tenantId=doc.get("tenantId"),
                clientId=doc.get("clientId"),
                categoryId=doc.get("categoryId"),
                brandId=doc.get("brandId"),
                image_base64=doc.get("image_base64", "")
            )
        )

    return results




 

#return models coreesponding images


@router.get("/search-visual-model")
async def search_visual_by_model(
    modelName: str,
    max_results: int = 10,
    match_threshold: float = 0.19,
    only_matches: bool = True,
    as_zip: bool = False,
):
    """
    Visual search across ALL shelves using all stored images for the given modelName.

    Works similarly to /search-visual-multi, but:
    - No image upload in the request.
    - Loads all images from MongoDB where modelName matches.
    """
    # 1. Get all query images for this modelName from MongoDB
    collection = get_single_query_images_collection()
    try:
        docs = list(collection.find({"modelName": modelName}))
    except OperationFailure as e:
        try:
            errmsg = e.details.get("errmsg", "")
        except Exception:
            errmsg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"MongoDB read error: {errmsg}",
        )

    if not docs:
        raise HTTPException(
            status_code=404,
            detail=f"No stored images found for modelName='{modelName}'",
        )

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    matches: List[MatchResult] = []

    # 2. For each stored image, run visual search and accumulate matches
    for doc in docs:
        image_b64 = doc.get("image_base64")
        if not image_b64:
            continue
        try:
            image_bytes = base64.b64decode(image_b64)
            query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            # Skip invalid or unreadable images
            continue

        query_vector = vectorizer.get_image_embedding(query_image)

        # Ask for extra results (similar to /search-visual-multi)
        results = datastore.query_similar(query_vector, n_results=max_results * 3)

        for res in results:
            data = res["data"]
            parent_id = data.get("parent_image_id")
            if parent_id is None:
>>>>>>> Stashed changes
                continue
            color = (0, 0, 255)  # RED

        cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            draw_image,
            f"{score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # <-- CHANGED: Removed break condition
        # num_drawn += 1
        # if num_drawn >= max_results:
        #     break

    # Encode to PNG for HTTP response
    _, buffer = cv2.imencode(".png", draw_image)
    bytes_io = io.BytesIO(buffer.tobytes())

    # Expose which shelf was used
    response = StreamingResponse(bytes_io, media_type="image/png")
    response.headers["X-Shelf-Id"] = target_shelf_id
    return response
