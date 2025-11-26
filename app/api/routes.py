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

from app.core.config import settings
from app.services.vectorizer import get_vectorizer
from app.services.datastore import get_datastore
from app.services.detection import run_detection

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


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
    box_thresh: float = 0.3,
):
    """
    Upload a full shelf image, detect objects with GroundingDINO,
    and index them in the vector DB as a new shelf (identified by image_id).
    """
    if not (0.0 <= box_thresh <= 1.0):
        raise HTTPException(status_code=400, detail="box_thresh must be between 0.0 and 1.0")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detections = run_detection(image, prompt=prompt, box_thresh=box_thresh)

    image_id = str(uuid.uuid4())
    vectorizer = get_vectorizer()
    datastore = get_datastore()

    all_bboxes: List[List[int]] = []
    num_indexed = 0

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
    max_results: int = 10,
    match_threshold: float = 0.19,
    search_all: bool = False,
):
    """
    Upload a query image and search for visually similar objects.

    - If search_all == False: search only within the given shelf_id.
    - If search_all == True: search across the whole database (all shelves),
      ignoring shelf_id.

    Returns JSON with:
    - shelf_id: which shelf (parent_image_id)
    - bbox: bounding box on that shelf image
    - score: distance (smaller = more similar)
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
    # Ask for extra results, then filter by threshold (and optional shelf)
    results = datastore.query_similar(query_vector, n_results=max_results * 3)

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
            if len(matches) >= max_results:
                break

    return SearchResponse(matches=matches)


# ===========================
# Search (Visual, GLOBAL)
# ===========================

@router.post("/search-visual")
async def search_visual(
    file: UploadFile = File(...),
    max_results: int = 10,
    match_threshold: float = 0.19,
    only_matches: bool = True,
):
    """
    GLOBAL visual search.

    - Search across ALL indexed objects in the vector DB.
    - Find the best-matching shelf (parent_image_id with lowest distance).
    - Return that shelf image with bounding boxes drawn:

      GREEN = match (score < match_threshold)
      RED   = non-match (score >= match_threshold), only if only_matches == False

    The response is a PNG image. The chosen shelf_id is exposed via header 'X-Shelf-Id'.
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
    # Ask for a larger pool of results so we have enough hits on the top shelf
    results = datastore.query_similar(query_vector, n_results=max_results * 10)

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

    num_drawn = 0

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
            if only_matches:
                # Skip non-matches when only_matches is True
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

        num_drawn += 1
        if num_drawn >= max_results:
            break

    # Encode to PNG for HTTP response
    _, buffer = cv2.imencode(".png", draw_image)
    bytes_io = io.BytesIO(buffer.tobytes())

    # Expose which shelf was used
    response = StreamingResponse(bytes_io, media_type="image/png")
    response.headers["X-Shelf-Id"] = target_shelf_id
    return response
