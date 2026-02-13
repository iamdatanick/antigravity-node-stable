"""Workers AI inference for PHUC platform."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import httpx
import os


class AIModel(Enum):
    """Available Workers AI models."""
    # Chat/Completion
    LLAMA_3_1_8B = "@cf/meta/llama-3.1-8b-instruct"
    LLAMA_3_1_70B = "@cf/meta/llama-3.1-70b-instruct"
    
    # Embeddings
    BGE_BASE = "@cf/baai/bge-base-en-v1.5"
    BGE_LARGE = "@cf/baai/bge-large-en-v1.5"
    
    # Image
    SDXL = "@cf/stabilityai/stable-diffusion-xl-base-1.0"


@dataclass
class AIConfig:
    """Workers AI configuration."""
    account_id: str = field(default_factory=lambda: os.getenv("CF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.getenv("CF_API_TOKEN", ""))
    gateway_id: str = "Phuc-ai"  # AI Gateway for routing


@dataclass
class ChatMessage:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class EmbeddingResult:
    """Embedding result."""
    text: str
    vector: list[float]
    dimensions: int


class WorkersAI:
    """Workers AI inference client."""
    
    BASE_URL = "https://api.cloudflare.com/client/v4"
    
    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
    
    @property
    def base_url(self) -> str:
        return f"{self.BASE_URL}/accounts/{self.config.account_id}/ai/run"
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=120.0)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def chat(
        self,
        messages: list[ChatMessage],
        model: AIModel = AIModel.LLAMA_3_1_8B,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Generate chat completion."""
        client = await self._get_client()
        
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        response = await client.post(
            f"{self.base_url}/{model.value}",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json().get("result", {})
        return result.get("response", "")
    
    async def generate(
        self,
        prompt: str,
        system: str = None,
        model: AIModel = AIModel.LLAMA_3_1_8B,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Simple text generation."""
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        
        return await self.chat(messages, model, max_tokens, temperature)
    
    async def embed(
        self,
        texts: list[str],
        model: AIModel = AIModel.BGE_BASE
    ) -> list[EmbeddingResult]:
        """Generate embeddings."""
        client = await self._get_client()
        
        payload = {"text": texts}
        
        response = await client.post(
            f"{self.base_url}/{model.value}",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json().get("result", {})
        vectors = result.get("data", [])
        
        embeddings = []
        for text, vec_data in zip(texts, vectors):
            embeddings.append(EmbeddingResult(
                text=text,
                vector=vec_data,
                dimensions=len(vec_data)
            ))
        
        return embeddings
    
    async def embed_single(self, text: str, model: AIModel = AIModel.BGE_BASE) -> list[float]:
        """Embed single text, return vector."""
        results = await self.embed([text], model)
        return results[0].vector if results else []
    
    # PHUC-specific methods
    async def generate_campaign(
        self,
        therapeutic_area: str,
        objectives: list[str],
        context: str = ""
    ) -> dict:
        """Generate campaign strategy."""
        system = """You are a pharmaceutical marketing strategist. Generate campaigns 
        that are compliant with FDA regulations and industry best practices.
        Return response as valid JSON with keys: name, objective, channels, messaging, kpis."""
        
        prompt = f"""Create a marketing campaign for {therapeutic_area}.

Objectives: {', '.join(objectives)}

Context:
{context}

Generate a detailed campaign strategy in JSON format."""
        
        response = await self.generate(
            prompt=prompt,
            system=system,
            model=AIModel.LLAMA_3_1_70B,
            temperature=0.7
        )
        
        # Parse JSON from response
        import json
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"raw_response": response}
    
    async def answer_pharma_question(
        self,
        question: str,
        context: list[str]
    ) -> str:
        """Answer pharma question with RAG context."""
        system = """You are a pharmaceutical marketing intelligence assistant.
        Use the provided context to answer questions accurately.
        Always cite your sources and note any limitations or uncertainties."""
        
        context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        
        prompt = f"""Context:
{context_str}

Question: {question}

Provide a detailed, accurate answer based on the context above."""
        
        return await self.generate(
            prompt=prompt,
            system=system,
            model=AIModel.LLAMA_3_1_8B,
            temperature=0.3  # Lower temp for factual Q&A
        )
    
    async def extract_entities(self, text: str) -> dict:
        """Extract pharmaceutical entities from text."""
        system = """Extract pharmaceutical entities from the text.
        Return JSON with keys: drugs (NDC codes or names), providers (NPI or names),
        conditions, companies, therapeutic_areas."""
        
        response = await self.generate(
            prompt=f"Extract entities from:\n{text}",
            system=system,
            model=AIModel.LLAMA_3_1_8B,
            temperature=0.1
        )
        
        import json
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"raw_response": response}
