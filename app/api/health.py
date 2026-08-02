from fastapi import APIRouter

router = APIRouter(prefix="health")


@router.get("/")
def root():
    return {"message": "ok"}
