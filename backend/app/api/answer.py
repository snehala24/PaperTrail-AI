from fastapi import APIRouter, Depends
from backend.app.schemas import AnswerRequest, AnswerResponse
from backend.app.claims.service import AnswerService, get_answer_service

router = APIRouter(prefix="/answer", tags=["Answer"])


@router.post("", response_model=AnswerResponse)
async def generate_answer(
    req: AnswerRequest,
    service: AnswerService = Depends(get_answer_service),
) -> AnswerResponse:
    """
    Generate an evidence-grounded answer to a scientific research question.
    Executes hybrid retrieval, Cross-Encoder reranking, untrusted-evidence LLM synthesis,
    3-layer citation verification, and strict zero-tolerance abstention.
    """
    return await service.answer(req)
