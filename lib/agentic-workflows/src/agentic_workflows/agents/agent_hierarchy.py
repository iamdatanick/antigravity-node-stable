"""
PHUC Agent Hierarchy - Full 100+ Agent System
==============================================

Organizes all agents into hierarchies with:
- Parent/child relationships
- MCP skill assignments
- Handoff routing
- Escalation paths

Hierarchy:
├── EXECUTIVE (Meta agents - orchestrate everything)
│   ├── orchestrator
│   ├── planner
│   └── task_router
├── ENGINEERING (Code + Ops)
│   ├── code/ (15 agents)
│   └── ops/ (12 agents)
├── DATA (Data + Research)
│   ├── data/ (12 agents)
│   └── research/ (10 agents)
├── BUSINESS (Business + Communication)
│   ├── business/ (10 agents)
│   └── communication/ (10 agents)
├── CREATIVE (Creative + Specialized)
│   ├── creative/ (10 agents)
│   └── specialized/ (6+ agents)
└── PIPELINE (Architect → Builder → Tester → Shipper)
    └── phuc/ (4 agents + 13 sub-agents)

Location: C:\\Users\\NickV\\agentic-workflows\\agentic-workflows\\src\\agentic_workflows\\agents\\agent_hierarchy.py
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
from pathlib import Path

# SDK imports
from agentic_workflows.agents.base import BaseAgent, AgentConfig, AgentResult
from agentic_workflows.protocols.mcp_client import MCPClient
from agentic_workflows.handoffs import HandoffManager, CheckpointManager
from agentic_workflows.observability import MetricsCollector

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MCP_ENDPOINT = "https://agentic-workflows-mcp.nick-9a6.workers.dev"
CAMARA_ENDPOINT = "https://mcp.camaramcp.com/sse"
CLAUDE_AGENTS_DIR = Path.home() / ".claude" / "agents"


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchyLevel(Enum):
    """Agent hierarchy levels"""
    EXECUTIVE = 0      # Top-level orchestrators
    DIRECTOR = 1       # Category leads
    MANAGER = 2        # Team leads
    SPECIALIST = 3     # Individual contributors
    PIPELINE = 4       # Pipeline agents (separate track)


class AgentDomain(Enum):
    """Agent domains"""
    EXECUTIVE = "executive"
    ENGINEERING = "engineering"
    DATA = "data"
    BUSINESS = "business"
    CREATIVE = "creative"
    PIPELINE = "pipeline"


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Which MCP skills each agent category gets
CATEGORY_SKILLS: Dict[str, List[str]] = {
    # Meta/Executive - all skills
    "meta": ["ai", "vectorize", "d1", "r2", "workers", "attribution", "campaign", 
             "reporting", "injection-defense", "scope-validator", "pii-detector"],
    
    # Code agents - AI + security + workers
    "code": ["ai", "workers", "injection-defense", "scope-validator", "pii-detector"],
    
    # Data agents - AI + D1 + R2 + vectorize + reporting
    "data": ["ai", "d1", "r2", "vectorize", "reporting"],
    
    # Ops agents - workers + D1 + R2 + reporting
    "ops": ["workers", "d1", "r2", "reporting", "injection-defense"],
    
    # Business agents - AI + analytics
    "business": ["ai", "attribution", "campaign", "reporting"],
    
    # Communication agents - AI only
    "communication": ["ai"],
    
    # Creative agents - AI + image
    "creative": ["ai"],
    
    # Research agents - AI + vectorize
    "research": ["ai", "vectorize"],
    
    # Specialized - varies
    "specialized": ["ai", "injection-defense", "pii-detector"],
}

# Specific agent skill overrides
AGENT_SKILLS: Dict[str, List[str]] = {
    # Security specialists get security skills
    "security_auditor": ["injection-defense", "scope-validator", "pii-detector", "ai"],
    "secops": ["injection-defense", "scope-validator", "pii-detector", "workers"],
    
    # Database specialists get D1
    "sql_expert": ["d1", "ai"],
    "dba": ["d1", "r2", "ai"],
    "data_modeler": ["d1", "ai"],
    
    # DevOps gets workers
    "devops": ["workers", "d1", "r2"],
    "sre": ["workers", "d1", "r2", "reporting"],
    "k8s_admin": ["workers"],
    
    # Analytics specialists
    "data_analyst": ["d1", "r2", "vectorize", "reporting", "ai"],
    "financial_analyst": ["d1", "reporting", "ai"],
    "market_researcher": ["ai", "vectorize", "reporting"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# HANDOFF MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Who can hand off to whom
HANDOFF_ROUTES: Dict[str, List[str]] = {
    # Executive layer routes to directors
    "orchestrator": ["planner", "task_router", "code_lead", "data_lead", "business_lead"],
    "planner": ["task_router", "decomposer"],
    "task_router": ["code_reviewer", "data_analyst", "strategist", "researcher"],
    
    # Code team handoffs
    "code_reviewer": ["debugger", "refactorer", "security_auditor", "test_writer"],
    "debugger": ["code_reviewer", "performance_tuner"],
    "security_auditor": ["code_reviewer", "secops"],
    "test_writer": ["code_reviewer"],
    "api_designer": ["schema_designer", "documentation_writer"],
    
    # Data team handoffs
    "data_analyst": ["sql_expert", "statistician", "visualization_expert"],
    "sql_expert": ["query_optimizer", "dba"],
    "etl_designer": ["data_validator", "data_cleaner"],
    "data_modeler": ["schema_migrator", "dba"],
    
    # Ops team handoffs
    "sre": ["devops", "monitoring", "incident_responder"],
    "devops": ["k8s_admin", "terraform_expert", "docker_expert"],
    "k8s_admin": ["networking", "monitoring"],
    "secops": ["security_auditor", "incident_responder"],
    
    # Business team handoffs
    "business_analyst": ["product_manager", "strategist"],
    "product_manager": ["business_analyst", "okr_planner"],
    "strategist": ["market_researcher", "competitive_analyst"],
    "financial_analyst": ["pricing_expert", "roi_calculator"],
    
    # Communication team handoffs
    "email_drafter": ["pr_writer", "memo_writer"],
    "proposal_writer": ["presentation_creator", "report_writer"],
    "pr_writer": ["social_media", "blog_writer"],
    
    # Creative team handoffs
    "brainstormer": ["naming_expert", "concept_developer"],
    "copywriter": ["tagline_creator", "brand_voice"],
    "creative_director": ["brainstormer", "concept_developer"],
    
    # Research team handoffs
    "researcher": ["fact_checker", "literature_reviewer"],
    "fact_checker": ["citation_finder"],
    "trend_analyst": ["market_researcher", "competitive_analyst"],
    
    # Pipeline handoffs (separate track)
    "architect": ["builder"],
    "builder": ["tester"],
    "tester": ["shipper", "builder"],  # Can loop back
    "shipper": [],  # Terminal
}

# Escalation paths (when agent can't handle, escalate up)
ESCALATION_PATHS: Dict[str, str] = {
    # Code team escalates to code_reviewer
    "debugger": "code_reviewer",
    "refactorer": "code_reviewer",
    "test_writer": "code_reviewer",
    "security_auditor": "code_reviewer",
    "code_reviewer": "orchestrator",
    
    # Data team escalates to data_analyst
    "sql_expert": "data_analyst",
    "statistician": "data_analyst",
    "data_cleaner": "data_analyst",
    "data_analyst": "orchestrator",
    
    # Ops team escalates to sre
    "devops": "sre",
    "k8s_admin": "sre",
    "monitoring": "sre",
    "sre": "orchestrator",
    
    # Business escalates to strategist
    "business_analyst": "strategist",
    "product_manager": "strategist",
    "strategist": "orchestrator",
    
    # Everyone ultimately escalates to orchestrator
    "orchestrator": None,  # Top of hierarchy
}


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentNode:
    """Agent in the hierarchy with skills and handoffs"""
    name: str
    category: str
    level: HierarchyLevel
    domain: AgentDomain
    description: str
    skills: List[str]
    handoff_to: List[str]
    escalate_to: Optional[str]
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    system_prompt: str = ""
    file_path: Optional[Path] = None
    
    def get_mcp_tools(self) -> List[str]:
        """Get all MCP tools available to this agent"""
        tools = []
        skill_tools = {
            "ai": ["ai_generate", "ai_embed", "ai_classify", "ai_summarize", "ai_image"],
            "d1": ["d1_query", "d1_execute", "d1_batch", "d1_backup"],
            "r2": ["r2_get", "r2_put", "r2_list", "r2_delete", "r2_presign"],
            "workers": ["workers_deploy", "workers_list", "workers_logs", "workers_secrets"],
            "vectorize": ["vectorize_insert", "vectorize_query", "vectorize_delete"],
            "attribution": ["attribution_track", "attribution_query", "attribution_report"],
            "campaign": ["campaign_create", "campaign_update", "campaign_analyze"],
            "reporting": ["report_generate", "report_schedule", "report_export"],
            "injection-defense": ["scan_prompt", "check_threat", "sanitize_input"],
            "scope-validator": ["validate_scope", "check_permissions", "enforce_policy"],
            "pii-detector": ["detect_pii", "mask_pii", "audit_pii", "redact_document"],
        }
        for skill in self.skills:
            if skill in skill_tools:
                tools.extend(skill_tools[skill])
        return tools


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchyBuilder:
    """Builds the complete agent hierarchy"""
    
    @staticmethod
    def build() -> Dict[str, AgentNode]:
        """Build complete hierarchy of all agents"""
        nodes: Dict[str, AgentNode] = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # EXECUTIVE LEVEL (Meta agents)
        # ═══════════════════════════════════════════════════════════════════
        
        nodes["orchestrator"] = AgentNode(
            name="orchestrator",
            category="meta",
            level=HierarchyLevel.EXECUTIVE,
            domain=AgentDomain.EXECUTIVE,
            description="Top-level orchestrator - routes tasks to appropriate agents",
            skills=CATEGORY_SKILLS["meta"],
            handoff_to=HANDOFF_ROUTES.get("orchestrator", []),
            escalate_to=None,
            children=["planner", "task_router", "reviewer", "synthesizer"],
        )
        
        nodes["planner"] = AgentNode(
            name="planner",
            category="meta",
            level=HierarchyLevel.DIRECTOR,
            domain=AgentDomain.EXECUTIVE,
            description="Task planning and decomposition",
            skills=["ai"],
            handoff_to=HANDOFF_ROUTES.get("planner", []),
            escalate_to="orchestrator",
            parent="orchestrator",
        )
        
        nodes["task_router"] = AgentNode(
            name="task_router",
            category="meta",
            level=HierarchyLevel.DIRECTOR,
            domain=AgentDomain.EXECUTIVE,
            description="Routes tasks to the right specialist",
            skills=["ai"],
            handoff_to=HANDOFF_ROUTES.get("task_router", []),
            escalate_to="orchestrator",
            parent="orchestrator",
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # CODE TEAM (15 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        code_agents = [
            ("code_reviewer", "Review code for bugs, style, best practices", HierarchyLevel.MANAGER),
            ("debugger", "Debug and fix code issues", HierarchyLevel.SPECIALIST),
            ("security_auditor", "Security vulnerability scanning", HierarchyLevel.SPECIALIST),
            ("test_writer", "Generate unit and integration tests", HierarchyLevel.SPECIALIST),
            ("refactorer", "Improve code structure", HierarchyLevel.SPECIALIST),
            ("performance_tuner", "Optimize for speed/memory", HierarchyLevel.SPECIALIST),
            ("api_designer", "Design RESTful/GraphQL APIs", HierarchyLevel.MANAGER),
            ("schema_designer", "Database schema design", HierarchyLevel.SPECIALIST),
            ("documentation_writer", "Technical documentation", HierarchyLevel.SPECIALIST),
            ("code_explainer", "Explain complex code", HierarchyLevel.SPECIALIST),
            ("migration_planner", "Plan code migrations", HierarchyLevel.SPECIALIST),
            ("dependency_analyzer", "Analyze dependencies", HierarchyLevel.SPECIALIST),
            ("error_handler", "Design error handling", HierarchyLevel.SPECIALIST),
            ("logging_expert", "Add proper logging", HierarchyLevel.SPECIALIST),
            ("type_annotator", "Add type hints", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in code_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["code"])
            nodes[name] = AgentNode(
                name=name,
                category="code",
                level=level,
                domain=AgentDomain.ENGINEERING,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "code_reviewer"),
                parent="code_reviewer" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # DATA TEAM (12 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        data_agents = [
            ("data_analyst", "Analyze datasets", HierarchyLevel.MANAGER),
            ("sql_expert", "SQL queries and optimization", HierarchyLevel.SPECIALIST),
            ("etl_designer", "ETL pipeline design", HierarchyLevel.SPECIALIST),
            ("statistician", "Statistical analysis", HierarchyLevel.SPECIALIST),
            ("data_modeler", "Data modeling", HierarchyLevel.SPECIALIST),
            ("visualization_expert", "Charts and dashboards", HierarchyLevel.SPECIALIST),
            ("data_cleaner", "Data cleaning/preprocessing", HierarchyLevel.SPECIALIST),
            ("feature_engineer", "ML feature engineering", HierarchyLevel.SPECIALIST),
            ("data_validator", "Data quality checks", HierarchyLevel.SPECIALIST),
            ("schema_migrator", "Database migrations", HierarchyLevel.SPECIALIST),
            ("query_optimizer", "Query performance", HierarchyLevel.SPECIALIST),
            ("data_documenter", "Data dictionary/docs", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in data_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["data"])
            nodes[name] = AgentNode(
                name=name,
                category="data",
                level=level,
                domain=AgentDomain.DATA,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "data_analyst"),
                parent="data_analyst" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # OPS TEAM (12 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        ops_agents = [
            ("sre", "Site reliability engineering", HierarchyLevel.MANAGER),
            ("devops", "CI/CD pipelines", HierarchyLevel.SPECIALIST),
            ("k8s_admin", "Kubernetes management", HierarchyLevel.SPECIALIST),
            ("monitoring", "Observability setup", HierarchyLevel.SPECIALIST),
            ("dba", "Database administration", HierarchyLevel.SPECIALIST),
            ("secops", "Security operations", HierarchyLevel.SPECIALIST),
            ("terraform_expert", "Infrastructure as code", HierarchyLevel.SPECIALIST),
            ("docker_expert", "Container optimization", HierarchyLevel.SPECIALIST),
            ("networking", "Network configuration", HierarchyLevel.SPECIALIST),
            ("incident_responder", "Incident management", HierarchyLevel.SPECIALIST),
            ("capacity_planner", "Resource planning", HierarchyLevel.SPECIALIST),
            ("cost_optimizer", "Cloud cost optimization", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in ops_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["ops"])
            nodes[name] = AgentNode(
                name=name,
                category="ops",
                level=level,
                domain=AgentDomain.ENGINEERING,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "sre"),
                parent="sre" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # BUSINESS TEAM (10 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        business_agents = [
            ("strategist", "Business strategy", HierarchyLevel.MANAGER),
            ("business_analyst", "Requirements analysis", HierarchyLevel.SPECIALIST),
            ("product_manager", "Product strategy", HierarchyLevel.SPECIALIST),
            ("financial_analyst", "Financial modeling", HierarchyLevel.SPECIALIST),
            ("market_researcher", "Market analysis", HierarchyLevel.SPECIALIST),
            ("competitive_analyst", "Competitor analysis", HierarchyLevel.SPECIALIST),
            ("pricing_expert", "Pricing strategy", HierarchyLevel.SPECIALIST),
            ("okr_planner", "OKR/goal setting", HierarchyLevel.SPECIALIST),
            ("stakeholder_manager", "Stakeholder comms", HierarchyLevel.SPECIALIST),
            ("roi_calculator", "ROI analysis", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in business_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["business"])
            nodes[name] = AgentNode(
                name=name,
                category="business",
                level=level,
                domain=AgentDomain.BUSINESS,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "strategist"),
                parent="strategist" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # COMMUNICATION TEAM (10 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        comm_agents = [
            ("email_drafter", "Professional emails", HierarchyLevel.MANAGER),
            ("pr_writer", "Press releases", HierarchyLevel.SPECIALIST),
            ("presentation_creator", "Slide decks", HierarchyLevel.SPECIALIST),
            ("proposal_writer", "Business proposals", HierarchyLevel.SPECIALIST),
            ("report_writer", "Business reports", HierarchyLevel.SPECIALIST),
            ("newsletter_writer", "Newsletters", HierarchyLevel.SPECIALIST),
            ("social_media", "Social content", HierarchyLevel.SPECIALIST),
            ("blog_writer", "Blog posts", HierarchyLevel.SPECIALIST),
            ("speech_writer", "Speeches/talks", HierarchyLevel.SPECIALIST),
            ("memo_writer", "Internal memos", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in comm_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["communication"])
            nodes[name] = AgentNode(
                name=name,
                category="communication",
                level=level,
                domain=AgentDomain.BUSINESS,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "email_drafter"),
                parent="email_drafter" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # CREATIVE TEAM (10 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        creative_agents = [
            ("creative_director", "Creative strategy", HierarchyLevel.MANAGER),
            ("brainstormer", "Idea generation", HierarchyLevel.SPECIALIST),
            ("naming_expert", "Product/feature naming", HierarchyLevel.SPECIALIST),
            ("storyteller", "Narrative creation", HierarchyLevel.SPECIALIST),
            ("copywriter", "Marketing copy", HierarchyLevel.SPECIALIST),
            ("tagline_creator", "Slogans/taglines", HierarchyLevel.SPECIALIST),
            ("concept_developer", "Concept development", HierarchyLevel.SPECIALIST),
            ("brand_voice", "Brand messaging", HierarchyLevel.SPECIALIST),
            ("pitch_creator", "Pitch development", HierarchyLevel.SPECIALIST),
            ("analogy_maker", "Explain via analogies", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in creative_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["creative"])
            nodes[name] = AgentNode(
                name=name,
                category="creative",
                level=level,
                domain=AgentDomain.CREATIVE,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "creative_director"),
                parent="creative_director" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # RESEARCH TEAM (10 agents)
        # ═══════════════════════════════════════════════════════════════════
        
        research_agents = [
            ("researcher", "Deep research", HierarchyLevel.MANAGER),
            ("fact_checker", "Verify facts", HierarchyLevel.SPECIALIST),
            ("trend_analyst", "Trend analysis", HierarchyLevel.SPECIALIST),
            ("literature_reviewer", "Academic review", HierarchyLevel.SPECIALIST),
            ("citation_finder", "Find sources", HierarchyLevel.SPECIALIST),
            ("comparison_analyst", "Compare options", HierarchyLevel.SPECIALIST),
            ("survey_designer", "Survey creation", HierarchyLevel.SPECIALIST),
            ("interview_analyst", "Interview analysis", HierarchyLevel.SPECIALIST),
            ("benchmark_analyst", "Benchmarking", HierarchyLevel.SPECIALIST),
            ("hypothesis_tester", "Test hypotheses", HierarchyLevel.SPECIALIST),
        ]
        
        for name, desc, level in research_agents:
            skills = AGENT_SKILLS.get(name, CATEGORY_SKILLS["research"])
            nodes[name] = AgentNode(
                name=name,
                category="research",
                level=level,
                domain=AgentDomain.DATA,
                description=desc,
                skills=skills,
                handoff_to=HANDOFF_ROUTES.get(name, []),
                escalate_to=ESCALATION_PATHS.get(name, "researcher"),
                parent="researcher" if level == HierarchyLevel.SPECIALIST else "orchestrator",
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # PIPELINE AGENTS (separate track)
        # ═══════════════════════════════════════════════════════════════════
        
        nodes["architect"] = AgentNode(
            name="architect",
            category="pipeline",
            level=HierarchyLevel.PIPELINE,
            domain=AgentDomain.PIPELINE,
            description="Design systems, create architecture",
            skills=["ai", "vectorize"],
            handoff_to=["builder"],
            escalate_to=None,
            children=["designer", "researcher", "planner"],
        )
        
        nodes["builder"] = AgentNode(
            name="builder",
            category="pipeline",
            level=HierarchyLevel.PIPELINE,
            domain=AgentDomain.PIPELINE,
            description="Write production code",
            skills=["ai", "workers", "d1", "r2"],
            handoff_to=["tester"],
            escalate_to="architect",
            children=["coder", "documenter", "refactorer"],
        )
        
        nodes["tester"] = AgentNode(
            name="tester",
            category="pipeline",
            level=HierarchyLevel.PIPELINE,
            domain=AgentDomain.PIPELINE,
            description="Test and validate",
            skills=["injection-defense", "scope-validator", "pii-detector"],
            handoff_to=["shipper", "builder"],
            escalate_to="builder",
            children=["unit_tester", "integration_tester", "security_auditor", "pii_checker"],
        )
        
        nodes["shipper"] = AgentNode(
            name="shipper",
            category="pipeline",
            level=HierarchyLevel.PIPELINE,
            domain=AgentDomain.PIPELINE,
            description="Deploy to production",
            skills=["workers", "reporting", "attribution"],
            handoff_to=[],
            escalate_to="tester",
            children=["deployer", "monitor", "rollbacker"],
        )
        
        return nodes


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL AGENT (Runnable with handoffs)
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalAgent(BaseAgent):
    """
    An agent with full hierarchy, skills, and handoff capabilities.
    """
    
    def __init__(self, node: AgentNode, hierarchy: Dict[str, AgentNode]):
        super().__init__(AgentConfig(
            name=node.name,
            description=node.description,
        ))
        self.node = node
        self.hierarchy = hierarchy
        self.skills = node.skills
        self.tools = node.get_mcp_tools()
        self.handoff_to = node.handoff_to
        self.escalate_to = node.escalate_to
        
        # Clients
        self.mcp = MCPClient(MCP_ENDPOINT)
        self.claude = AsyncAnthropic() if ANTHROPIC_AVAILABLE else None
        
        # Handoff manager
        self.handoff_manager = HandoffManager()
        self.metrics = MetricsCollector()
    
    async def _execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute with skill access"""
        prompt = str(input_data) if not isinstance(input_data, dict) else input_data.get("prompt", str(input_data))
        
        # Build context with available tools
        tool_context = f"\n\nYou have access to these MCP tools: {', '.join(self.tools)}"
        
        if self.claude:
            response = await self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=self.node.system_prompt + tool_context if self.node.system_prompt else tool_context,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text
        else:
            result_text = f"[{self.node.name}] Processed: {prompt[:100]}..."
        
        return {
            "status": "complete",
            "agent": self.node.name,
            "category": self.node.category,
            "level": self.node.level.name,
            "skills": self.skills,
            "tools_available": self.tools,
            "response": result_text,
            "can_handoff_to": self.handoff_to,
            "can_escalate_to": self.escalate_to,
        }
    
    async def handoff(self, target_agent: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Hand off task to another agent"""
        if target_agent not in self.handoff_to:
            return {
                "status": "error",
                "error": f"Cannot handoff to {target_agent}. Valid targets: {self.handoff_to}"
            }
        
        target_node = self.hierarchy.get(target_agent)
        if not target_node:
            return {"status": "error", "error": f"Agent {target_agent} not found"}
        
        target = HierarchicalAgent(target_node, self.hierarchy)
        
        # Record handoff
        self.handoff_manager.record(
            from_agent=self.node.name,
            to_agent=target_agent,
            task=task
        )
        
        return await target.run(task)
    
    async def escalate(self, task: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
        """Escalate task to parent agent"""
        if not self.escalate_to:
            return {
                "status": "error",
                "error": f"No escalation path for {self.node.name}"
            }
        
        parent_node = self.hierarchy.get(self.escalate_to)
        if not parent_node:
            return {"status": "error", "error": f"Parent {self.escalate_to} not found"}
        
        parent = HierarchicalAgent(parent_node, self.hierarchy)
        
        # Add escalation context
        task["escalated_from"] = self.node.name
        task["escalation_reason"] = reason
        
        return await parent.run(task)
    
    async def call_tool(self, tool: str, args: Dict = None) -> Dict[str, Any]:
        """Call an MCP tool this agent has access to"""
        if tool not in self.tools:
            return {
                "status": "error",
                "error": f"Tool {tool} not available. Available: {self.tools}"
            }
        
        return await self.mcp.call_tool(tool, args or {})


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AgentHierarchy:
    """
    Manages the complete agent hierarchy.
    
    Usage:
        hierarchy = AgentHierarchy()
        
        # Get any agent
        agent = hierarchy.get("code_reviewer")
        result = await agent.run("Review this code")
        
        # Handoff to another agent
        handoff_result = await agent.handoff("debugger", {"code": "..."})
        
        # Escalate
        escalation = await agent.escalate({"issue": "..."}, reason="Too complex")
        
        # Find agent by task
        agent = hierarchy.route("optimize this SQL query")
    """
    
    def __init__(self):
        self.nodes = HierarchyBuilder.build()
        self.agents: Dict[str, HierarchicalAgent] = {}
        self._build_agents()
    
    def _build_agents(self):
        """Build all hierarchical agents"""
        for name, node in self.nodes.items():
            self.agents[name] = HierarchicalAgent(node, self.nodes)
    
    def get(self, name: str) -> Optional[HierarchicalAgent]:
        """Get agent by name"""
        normalized = name.replace("-", "_").replace(" ", "_").lower()
        return self.agents.get(normalized)
    
    def route(self, task: str) -> HierarchicalAgent:
        """Route task to appropriate agent based on content"""
        task_lower = task.lower()
        
        # Code-related
        if any(w in task_lower for w in ["code", "debug", "test", "review", "refactor", "api"]):
            return self.agents["code_reviewer"]
        
        # Data-related
        if any(w in task_lower for w in ["sql", "query", "data", "database", "etl", "analytics"]):
            return self.agents["data_analyst"]
        
        # Ops-related
        if any(w in task_lower for w in ["deploy", "kubernetes", "k8s", "docker", "devops", "monitor"]):
            return self.agents["sre"]
        
        # Security-related
        if any(w in task_lower for w in ["security", "vulnerability", "audit", "pii", "inject"]):
            return self.agents["security_auditor"]
        
        # Business-related
        if any(w in task_lower for w in ["strategy", "market", "business", "pricing", "okr"]):
            return self.agents["strategist"]
        
        # Communication
        if any(w in task_lower for w in ["email", "presentation", "proposal", "write", "draft"]):
            return self.agents["email_drafter"]
        
        # Creative
        if any(w in task_lower for w in ["name", "brainstorm", "creative", "tagline", "brand"]):
            return self.agents["creative_director"]
        
        # Research
        if any(w in task_lower for w in ["research", "fact", "trend", "compare", "benchmark"]):
            return self.agents["researcher"]
        
        # Default to orchestrator
        return self.agents["orchestrator"]
    
    def get_team(self, category: str) -> List[HierarchicalAgent]:
        """Get all agents in a category"""
        return [a for a in self.agents.values() if a.node.category == category]
    
    def get_by_skill(self, skill: str) -> List[HierarchicalAgent]:
        """Get agents that have a specific skill"""
        return [a for a in self.agents.values() if skill in a.skills]
    
    def list_all(self) -> Dict[str, List[str]]:
        """List all agents by category"""
        result: Dict[str, List[str]] = {}
        for agent in self.agents.values():
            cat = agent.node.category
            if cat not in result:
                result[cat] = []
            result[cat].append(agent.node.name)
        return result
    
    def count(self) -> int:
        """Total agent count"""
        return len(self.agents)
    
    def print_hierarchy(self) -> str:
        """Print hierarchy tree"""
        lines = ["AGENT HIERARCHY", "=" * 50]
        
        for domain in AgentDomain:
            domain_agents = [a for a in self.agents.values() if a.node.domain == domain]
            if domain_agents:
                lines.append(f"\n{domain.value.upper()}")
                for agent in sorted(domain_agents, key=lambda x: x.node.level.value):
                    indent = "  " * agent.node.level.value
                    skills = ", ".join(agent.skills[:3])
                    lines.append(f"{indent}├── {agent.node.name} [{skills}]")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_hierarchy: Optional[AgentHierarchy] = None

def get_hierarchy() -> AgentHierarchy:
    """Get or create global hierarchy"""
    global _hierarchy
    if _hierarchy is None:
        _hierarchy = AgentHierarchy()
    return _hierarchy

def get_hierarchical_agent(name: str) -> Optional[HierarchicalAgent]:
    """Quick access to get an agent"""
    return get_hierarchy().get(name)

def route_task(task: str) -> HierarchicalAgent:
    """Route a task to the right agent"""
    return get_hierarchy().route(task)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "HierarchyLevel",
    "AgentDomain",
    # Classes
    "AgentNode",
    "HierarchyBuilder",
    "HierarchicalAgent",
    "AgentHierarchy",
    # Quick access
    "get_hierarchy",
    "get_hierarchical_agent",
    "route_task",
    # Config
    "CATEGORY_SKILLS",
    "AGENT_SKILLS",
    "HANDOFF_ROUTES",
    "ESCALATION_PATHS",
]
