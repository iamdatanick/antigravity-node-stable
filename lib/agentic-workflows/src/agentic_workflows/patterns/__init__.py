"""
Claude Cookbook Patterns
========================

Integrates patterns from https://github.com/anthropics/claude-cookbooks

Patterns:
- classification: Text and data classification
- rag: Retrieval Augmented Generation
- summarization: Text summarization
- tool_use: Tool integration (calculator, SQL, customer service)
- multimodal: Vision and image generation
- advanced: Sub-agents, PDF upload, JSON mode, moderation, caching

Third-Party Integrations:
- Pinecone (vector database)
- Voyage AI (embeddings)
- Wikipedia
- Web pages
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
import json

# SDK imports
try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class ClassificationResult(BaseModel):
    """Classification result"""
    label: str
    confidence: float
    reasoning: str = ""


@dataclass
class Classifier:
    """
    Text/Data Classification using Claude.

    Based on claude-cookbooks/capabilities/classification
    """
    categories: List[str]
    model: str = "claude-haiku-3-20240307"  # Use Haiku for fast classification
    multi_label: bool = False

    async def classify(self, text: str) -> Union[ClassificationResult, List[ClassificationResult]]:
        """Classify text into categories"""
        if not ANTHROPIC_AVAILABLE:
            return ClassificationResult(label=self.categories[0], confidence=0.5)

        client = AsyncAnthropic()

        prompt = f"""Classify the following text into {"one or more of" if self.multi_label else "one of"} these categories: {', '.join(self.categories)}

Text: {text}

Return JSON: {{"label": "category", "confidence": 0.0-1.0, "reasoning": "explanation"}}"""

        response = await client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            result = json.loads(response.content[0].text)
            return ClassificationResult(**result)
        except:
            return ClassificationResult(
                label=self.categories[0],
                confidence=0.5,
                reasoning=response.content[0].text
            )

    async def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts"""
        tasks = [self.classify(text) for text in texts]
        return await asyncio.gather(*tasks)


class SentimentAnalyzer(Classifier):
    """Sentiment analysis classifier"""

    def __init__(self):
        super().__init__(
            categories=["positive", "negative", "neutral", "mixed"],
            model="claude-haiku-3-20240307"
        )


class IntentClassifier(Classifier):
    """Intent classification for customer service"""

    def __init__(self, intents: List[str] = None):
        super().__init__(
            categories=intents or [
                "question", "complaint", "feedback",
                "request", "greeting", "other"
            ]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RAG PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievedDocument(BaseModel):
    """Retrieved document"""
    content: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """RAG response"""
    answer: str
    sources: List[str]
    confidence: float


@dataclass
class VectorStore:
    """
    Base vector store interface.

    Implementations for:
    - Pinecone
    - Cloudflare Vectorize
    - In-memory (for testing)
    """
    name: str = "base"

    async def upsert(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert documents to store"""
        raise NotImplementedError

    async def query(self, vector: List[float], top_k: int = 5) -> List[RetrievedDocument]:
        """Query similar documents"""
        raise NotImplementedError


@dataclass
class PineconeStore(VectorStore):
    """
    Pinecone vector store integration.

    Based on claude-cookbooks/third_party/Pinecone
    """
    name: str = "pinecone"
    api_key: str = ""
    environment: str = ""
    index_name: str = ""

    async def upsert(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert to Pinecone"""
        # Would use pinecone-client
        return len(documents)

    async def query(self, vector: List[float], top_k: int = 5) -> List[RetrievedDocument]:
        """Query Pinecone"""
        # Would use pinecone-client
        return []


@dataclass
class CloudflareVectorize(VectorStore):
    """
    Cloudflare Vectorize integration.

    Uses our deployed MCP endpoint.
    """
    name: str = "cloudflare-vectorize"
    endpoint: str = "https://agentic-workflows-mcp.nick-9a6.workers.dev"

    async def upsert(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert via MCP"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/tools",
                json={"tool": "vectorize_upsert", "args": {"vectors": documents}}
            )
            return len(documents)

    async def query(self, vector: List[float], top_k: int = 5) -> List[RetrievedDocument]:
        """Query via MCP"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/tools",
                json={"tool": "vectorize_query", "args": {"vector": vector, "top_k": top_k}}
            )
            data = response.json()
            return [RetrievedDocument(**doc) for doc in data.get("matches", [])]


@dataclass
class Embedder:
    """
    Generate embeddings for RAG.

    Supports:
    - Voyage AI
    - Cloudflare Workers AI
    - OpenAI
    """
    provider: str = "cloudflare"  # cloudflare, voyage, openai
    model: str = "@cf/baai/bge-base-en-v1.5"
    endpoint: str = "https://agentic-workflows-mcp.nick-9a6.workers.dev"

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.provider == "cloudflare":
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoint}/tools",
                    json={"tool": "ai_embed", "args": {"text": text}}
                )
                return response.json().get("embedding", [])
        return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)


@dataclass
class RAGPipeline:
    """
    Retrieval Augmented Generation Pipeline.

    Based on claude-cookbooks/capabilities/rag
    """
    vector_store: VectorStore
    embedder: Embedder
    model: str = "claude-sonnet-4-20250514"
    top_k: int = 5

    async def index(self, documents: List[Dict[str, str]]) -> int:
        """Index documents for retrieval"""
        # Generate embeddings
        texts = [doc.get("content", "") for doc in documents]
        embeddings = await self.embedder.embed_batch(texts)

        # Prepare for upsert
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vectors.append({
                "id": doc.get("id", str(hash(doc.get("content", "")))),
                "values": embedding,
                "metadata": doc
            })

        # Upsert to store
        return await self.vector_store.upsert(vectors)

    async def query(self, question: str) -> RAGResponse:
        """Query with RAG"""
        if not ANTHROPIC_AVAILABLE:
            return RAGResponse(answer="[RAG unavailable]", sources=[], confidence=0.0)

        # 1. Embed query
        query_embedding = await self.embedder.embed(question)

        # 2. Retrieve relevant documents
        docs = await self.vector_store.query(query_embedding, self.top_k)

        # 3. Build context
        context = "\n\n".join([
            f"[Source: {doc.source}]\n{doc.content}"
            for doc in docs
        ])

        # 4. Generate answer
        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Answer based on these sources:

{context}

Question: {question}

Provide your answer and cite sources."""
            }]
        )

        return RAGResponse(
            answer=response.content[0].text,
            sources=[doc.source for doc in docs],
            confidence=0.8 if docs else 0.3
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARIZATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class SummaryStyle(str, Enum):
    """Summary styles"""
    BRIEF = "brief"
    DETAILED = "detailed"
    BULLET = "bullet"
    EXECUTIVE = "executive"


@dataclass
class Summarizer:
    """
    Text Summarization using Claude.

    Based on claude-cookbooks/capabilities/summarization
    """
    model: str = "claude-sonnet-4-20250514"
    style: SummaryStyle = SummaryStyle.BRIEF
    max_length: int = 500

    async def summarize(self, text: str, style: SummaryStyle = None) -> str:
        """Summarize text"""
        style = style or self.style

        if not ANTHROPIC_AVAILABLE:
            return f"[Summary of {len(text)} chars]"

        style_instructions = {
            SummaryStyle.BRIEF: "Provide a brief 2-3 sentence summary.",
            SummaryStyle.DETAILED: "Provide a comprehensive summary covering all key points.",
            SummaryStyle.BULLET: "Provide a bullet-point summary of key points.",
            SummaryStyle.EXECUTIVE: "Provide an executive summary suitable for decision makers.",
        }

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_length,
            messages=[{
                "role": "user",
                "content": f"""{style_instructions[style]}

Text to summarize:
{text}"""
            }]
        )
        return response.content[0].text

    async def summarize_chain(self, documents: List[str]) -> str:
        """Summarize multiple documents using map-reduce"""
        # Map: Summarize each document
        summaries = await asyncio.gather(*[
            self.summarize(doc, SummaryStyle.BRIEF)
            for doc in documents
        ])

        # Reduce: Combine summaries
        combined = "\n\n".join([f"Document {i+1}: {s}" for i, s in enumerate(summaries)])
        return await self.summarize(combined, SummaryStyle.DETAILED)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL USE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

class ToolCallResult(BaseModel):
    """Tool call result"""
    tool: str
    args: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None


@dataclass
class Calculator:
    """
    Calculator Tool Integration.

    Based on claude-cookbooks/tool_use/calculator
    """

    async def calculate(self, expression: str) -> float:
        """Evaluate mathematical expression"""
        import ast
        import operator

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }

        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = eval_expr(node.left)
                right = eval_expr(node.right)
                return operators[type(node.op)](left, right)
            else:
                raise ValueError(f"Unsupported operation")

        try:
            tree = ast.parse(expression, mode='eval')
            return eval_expr(tree.body)
        except:
            return 0.0


@dataclass
class SQLTool:
    """
    SQL Query Tool.

    Based on claude-cookbooks/misc/sql_queries
    """
    connection_string: str = ""

    async def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query"""
        # Would use asyncpg, aiomysql, or similar
        return []

    async def generate_sql(self, question: str, schema: str) -> str:
        """Generate SQL from natural language"""
        if not ANTHROPIC_AVAILABLE:
            return "SELECT * FROM table"

        client = AsyncAnthropic()
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Given this database schema:
{schema}

Generate a SQL query to answer: {question}

Return only the SQL query, no explanation."""
            }]
        )
        return response.content[0].text


@dataclass
class CustomerServiceAgent:
    """
    Customer Service Agent with Tools.

    Based on claude-cookbooks/tool_use/customer_service_agent
    """
    model: str = "claude-sonnet-4-20250514"
    tools: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        # Default tools
        self.tools.update({
            "lookup_order": self._lookup_order,
            "check_inventory": self._check_inventory,
            "create_ticket": self._create_ticket,
            "refund_order": self._refund_order,
        })

    async def _lookup_order(self, order_id: str) -> Dict[str, Any]:
        """Look up order details"""
        return {"order_id": order_id, "status": "shipped", "items": []}

    async def _check_inventory(self, product_id: str) -> Dict[str, Any]:
        """Check product inventory"""
        return {"product_id": product_id, "in_stock": True, "quantity": 100}

    async def _create_ticket(self, issue: str, priority: str) -> Dict[str, Any]:
        """Create support ticket"""
        return {"ticket_id": "TKT-001", "status": "open"}

    async def _refund_order(self, order_id: str, reason: str) -> Dict[str, Any]:
        """Process refund"""
        return {"refund_id": "REF-001", "status": "processing"}

    async def handle(self, message: str) -> str:
        """Handle customer message"""
        if not ANTHROPIC_AVAILABLE:
            return "How can I help you today?"

        # Build tool definitions
        tool_defs = [
            {
                "name": name,
                "description": func.__doc__ or name,
                "input_schema": {"type": "object", "properties": {}}
            }
            for name, func in self.tools.items()
        ]

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            tools=tool_defs,
            messages=[{"role": "user", "content": message}]
        )

        # Process response and execute tools if needed
        result_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                result_text += block.text
            elif hasattr(block, 'type') and block.type == 'tool_use':
                # Execute tool
                tool_name = block.name
                if tool_name in self.tools:
                    tool_result = await self.tools[tool_name](**block.input)
                    result_text += f"\n[Tool {tool_name}: {tool_result}]"

        return result_text


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIMODAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisionAnalyzer:
    """
    Vision Analysis using Claude.

    Based on claude-cookbooks/multimodal
    """
    model: str = "claude-sonnet-4-20250514"

    async def analyze_image(self, image_path: str, prompt: str = "Describe this image") -> str:
        """Analyze an image"""
        if not ANTHROPIC_AVAILABLE:
            return "[Image analysis unavailable]"

        import base64
        from pathlib import Path

        # Read image
        image_data = Path(image_path).read_bytes()
        base64_image = base64.standard_b64encode(image_data).decode("utf-8")

        # Determine media type
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(suffix, "image/jpeg")

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return response.content[0].text

    async def extract_chart_data(self, image_path: str) -> Dict[str, Any]:
        """Extract data from charts/graphs"""
        result = await self.analyze_image(
            image_path,
            "Extract all data from this chart/graph. Return as JSON with labels, values, and any relevant metadata."
        )
        try:
            return json.loads(result)
        except:
            return {"raw": result}

    async def transcribe_form(self, image_path: str) -> Dict[str, Any]:
        """Extract form fields and values"""
        result = await self.analyze_image(
            image_path,
            "Extract all form fields and their values from this image. Return as JSON object."
        )
        try:
            return json.loads(result)
        except:
            return {"raw": result}


@dataclass
class ImageGenerator:
    """
    Image Generation using Claude + Stable Diffusion.

    Based on claude-cookbooks/misc/illustrated_responses
    """
    sd_endpoint: str = ""

    async def generate(self, prompt: str) -> bytes:
        """Generate image from prompt"""
        import httpx

        # First, enhance prompt with Claude
        if ANTHROPIC_AVAILABLE:
            client = AsyncAnthropic()
            enhancement = await client.messages.create(
                model="claude-haiku-3-20240307",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"Enhance this image prompt for Stable Diffusion: {prompt}"
                }]
            )
            enhanced_prompt = enhancement.content[0].text
        else:
            enhanced_prompt = prompt

        # Generate with Stable Diffusion
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                self.sd_endpoint,
                json={"prompt": enhanced_prompt}
            )
            return response.content


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JSONMode:
    """
    Enable consistent JSON output from Claude.

    Based on claude-cookbooks/misc/json_mode
    """
    model: str = "claude-sonnet-4-20250514"
    schema: Dict[str, Any] = field(default_factory=dict)

    async def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response"""
        if not ANTHROPIC_AVAILABLE:
            return {}

        schema_str = json.dumps(self.schema, indent=2) if self.schema else "any valid JSON"

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""{prompt}

Return your response as valid JSON matching this schema:
{schema_str}

Respond with ONLY the JSON, no explanation."""
            }]
        )

        try:
            return json.loads(response.content[0].text)
        except:
            return {"raw": response.content[0].text}


@dataclass
class ModerationFilter:
    """
    Content Moderation Filter using Claude.

    Based on claude-cookbooks/misc/moderation_filter
    """
    categories: List[str] = field(default_factory=lambda: [
        "violence", "hate_speech", "sexual_content",
        "harassment", "self_harm", "illegal_activity"
    ])
    threshold: float = 0.7

    async def check(self, content: str) -> Dict[str, Any]:
        """Check content for policy violations"""
        if not ANTHROPIC_AVAILABLE:
            return {"flagged": False, "categories": {}}

        client = AsyncAnthropic()
        response = await client.messages.create(
            model="claude-haiku-3-20240307",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Analyze this content for policy violations.
Categories to check: {', '.join(self.categories)}

Content: {content}

Return JSON: {{"flagged": bool, "categories": {{"category": score}}, "reason": "explanation"}}"""
            }]
        )

        try:
            result = json.loads(response.content[0].text)
            return result
        except:
            return {"flagged": False, "categories": {}}


@dataclass
class PromptCache:
    """
    Prompt Caching for efficient API usage.

    Based on claude-cookbooks/misc/prompt_caching
    """
    cache: Dict[str, str] = field(default_factory=dict)
    ttl_seconds: int = 3600

    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        return self.cache.get(key)

    def set(self, key: str, value: str):
        """Cache response"""
        self.cache[key] = value

    async def query(self, prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
        """Query with caching"""
        cache_key = f"{model}:{hash(prompt)}"

        # Check cache
        cached = self.get(cache_key)
        if cached:
            return cached

        if not ANTHROPIC_AVAILABLE:
            return "[Cache miss, API unavailable]"

        # Call API
        client = AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text

        # Cache result
        self.set(cache_key, result)

        return result


@dataclass
class PDFProcessor:
    """
    PDF Processing and Analysis.

    Based on claude-cookbooks/misc/pdf_upload
    """
    model: str = "claude-sonnet-4-20250514"

    async def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text
        except Exception as e:
            return f"Error: {e}"

    async def summarize_pdf(self, pdf_path: str) -> str:
        """Extract and summarize PDF"""
        text = await self.extract_text(pdf_path)

        if not ANTHROPIC_AVAILABLE:
            return f"[PDF: {len(text)} chars extracted]"

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"Summarize this document:\n\n{text[:50000]}"  # Limit for context
            }]
        )
        return response.content[0].text

    async def answer_question(self, pdf_path: str, question: str) -> str:
        """Answer question about PDF content"""
        text = await self.extract_text(pdf_path)

        if not ANTHROPIC_AVAILABLE:
            return "[PDF Q&A unavailable]"

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Based on this document:

{text[:50000]}

Answer this question: {question}"""
            }]
        )
        return response.content[0].text


@dataclass
class AutomatedEvaluator:
    """
    Automated Prompt Evaluation using Claude.

    Based on claude-cookbooks/misc/building_evals
    """
    model: str = "claude-sonnet-4-20250514"
    criteria: List[str] = field(default_factory=lambda: [
        "accuracy", "relevance", "completeness", "clarity"
    ])

    async def evaluate(self, prompt: str, response: str, reference: str = None) -> Dict[str, Any]:
        """Evaluate a prompt/response pair"""
        if not ANTHROPIC_AVAILABLE:
            return {"score": 0.5, "criteria": {}}

        eval_prompt = f"""Evaluate this AI response on these criteria: {', '.join(self.criteria)}

Prompt: {prompt}
Response: {response}
{"Reference answer: " + reference if reference else ""}

Return JSON: {{"overall_score": 0.0-1.0, "criteria": {{"criterion": score}}, "feedback": "explanation"}}"""

        client = AsyncAnthropic()
        result = await client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": eval_prompt}]
        )

        try:
            return json.loads(result.content[0].text)
        except:
            return {"score": 0.5, "raw": result.content[0].text}


# ═══════════════════════════════════════════════════════════════════════════════
# WEB INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WikipediaSearch:
    """
    Wikipedia Integration.

    Based on claude-cookbooks/third_party/Wikipedia
    """

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search Wikipedia"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json"
                }
            )
            data = response.json()
            return data.get("query", {}).get("search", [])

    async def get_page(self, title: str) -> str:
        """Get Wikipedia page content"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json"
                }
            )
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
            return ""


@dataclass
class WebPageReader:
    """
    Web Page Reading.

    Based on claude-cookbooks/misc/read_web_pages
    """
    model: str = "claude-haiku-3-20240307"  # Use Haiku for web reading

    async def fetch(self, url: str) -> str:
        """Fetch web page content"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.text

    async def extract_content(self, url: str) -> str:
        """Extract main content from web page"""
        html = await self.fetch(url)

        # Simple extraction - in production use BeautifulSoup or similar
        import re
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def summarize_url(self, url: str) -> str:
        """Fetch and summarize web page"""
        content = await self.extract_content(url)

        if not ANTHROPIC_AVAILABLE:
            return f"[Web page: {len(content)} chars]"

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"Summarize this web page content:\n\n{content[:20000]}"
            }]
        )
        return response.content[0].text


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Classification
    "ClassificationResult",
    "Classifier",
    "SentimentAnalyzer",
    "IntentClassifier",
    # RAG
    "RetrievedDocument",
    "RAGResponse",
    "VectorStore",
    "PineconeStore",
    "CloudflareVectorize",
    "Embedder",
    "RAGPipeline",
    # Summarization
    "SummaryStyle",
    "Summarizer",
    # Tool Use
    "ToolCallResult",
    "Calculator",
    "SQLTool",
    "CustomerServiceAgent",
    # Multimodal
    "VisionAnalyzer",
    "ImageGenerator",
    # Advanced
    "JSONMode",
    "ModerationFilter",
    "PromptCache",
    "PDFProcessor",
    "AutomatedEvaluator",
    # Web
    "WikipediaSearch",
    "WebPageReader",
]
