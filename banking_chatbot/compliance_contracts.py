"""Provider-neutral, validated contracts for the output release gate."""
from typing import Literal, Protocol
from pydantic import BaseModel, ConfigDict, Field

Outcome = Literal['pass', 'fail', 'uncertain', 'error']


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)


class ComponentDescriptor(Contract):
    component_id: str
    version: str
    contract_version: str = '1'
    rule_ids: tuple[str, ...]
    execution: Literal['local', 'external'] = 'local'
    configuration_digest: str = ''
    model: str = ''
    prompt_version: str = ''
    schema_version: str = '1'
    test_only: bool = False


class Candidate(Contract):
    text: str
    finish_reason: str
    origin: Literal['fresh', 'exact_cache', 'semantic_cache'] = 'fresh'


class Reference(Contract):
    id: str
    question: str
    answer: str


class Rule(Contract):
    id: str
    description: str
    required_checks: tuple[str, ...]


class Policy(Contract):
    version: str
    rules: tuple[Rule, ...]
    max_words: int = Field(default=150, gt=0)
    fallback: str


class ComplianceContext(Contract):
    request_id: str
    question: str
    references: tuple[Reference, ...]
    history: tuple[str, ...] = ()
    scope: str = 'banking-demo'


class ValidationRequest(Contract):
    candidate: Candidate
    context: ComplianceContext
    policy: Policy


class ReviewRequest(ValidationRequest):
    pass


class Finding(Contract):
    rule_id: str
    component_id: str
    outcome: Outcome
    source_ids: tuple[str, ...] = ()


class Usage(Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class CheckResult(Contract):
    component_id: str
    findings: tuple[Finding, ...]
    usage: Usage = Usage()
    elapsed_ms: float = Field(default=0, ge=0)


class Claim(Contract):
    text: str
    assessment: Literal['supported', 'contradicted', 'insufficient_evidence']
    source_ids: tuple[str, ...]


class ReviewResult(CheckResult):
    claims: tuple[Claim, ...]
    no_factual_claims: bool


class ComplianceResult(Contract):
    final_answer: str
    decision: Literal['allow', 'block', 'unverified']
    reason_codes: tuple[str, ...]
    configuration_fingerprint: str
    request_id: str
    origin: str
    policy_version: str = ""
    components: tuple[ComponentDescriptor, ...] = ()
    evidence_digest: str = ""
    repair_attempts: int = 0
    initial_reason_codes: tuple[str, ...] = ()
    usage: Usage = Usage()
    elapsed_ms: float = 0


class OutputValidator(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...
    def validate(self, request: ValidationRequest) -> CheckResult: ...


class ComplianceReviewer(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...
    def review(self, request: ReviewRequest) -> ReviewResult: ...


class ComplianceService(Protocol):
    @property
    def fingerprint(self) -> str: ...
    def evaluate(self, candidate: Candidate, context: ComplianceContext) -> ComplianceResult: ...


class RepairRequest(Contract):
    context: ComplianceContext
    policy: Policy
    rule_ids: tuple[str, ...]


class RepairResult(Contract):
    text: str
    finish_reason: str
    usage: Usage = Usage()


class OutputRepairer(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...
    def repair(self, request: RepairRequest) -> RepairResult: ...
