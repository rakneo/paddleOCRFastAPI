from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from service.ocr_service import recognize
from uuid import uuid4


BASE_PATH = "/Users/rakshith/PycharmProjects/pocrpoc"

app = FastAPI(
    title="ML OCR Viewer",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/run")
async def proccess_image_ocr(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    image = Image.open(BytesIO(file_bytes)).convert('RGB')
    results, output_image = recognize(image)
    im_show = Image.fromarray(output_image)
    uid = uuid4()
    im_show.save(f"{BASE_PATH}/results/{file.filename}_result_{uid}.jpg")
    return {"file_reference_uri": f"{BASE_PATH}/{file.filename}_result_{uid}.jpg", "results": results}

