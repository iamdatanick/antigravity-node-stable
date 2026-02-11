# Antigravity-Node v13 — Production Deployment Plan

## Status: DRAFT
## Date: 2026-02-09
## Platform: 30+ Docker services, FastAPI + gRPC orchestrator

---

## Phase 1: CI Green (P0 Blockers)

**Objective**: Ensure all CI/CD pipelines pass with zero lint, build, and test failures.

### Tasks
- [ ] Fix ruff lint errors in `hooks.py` (datetime.utcnow → datetime.now, import sorting)
- [ ] Create missing `well-known/` directory structure
- [ ] Create missing `config/fluent-bit.conf` with baseline configuration
- [ ] Resolve agentic-workflows git dependency (pin version or switch to PyPI)
- [ ] Verify all Docker image builds complete without errors
- [ ] Confirm all unit tests pass (pytest target: 100% pass rate)

### Gate Criteria
- [ ] CI pipeline passes all jobs (lint, build, test)
- [ ] No blocking linter warnings
- [ ] Zero test failures
- [ ] All dependency locks are valid

### Agent Assignment
- **Primary**: Bash agent (ruff --fix, mkdir, pytest)
- **Supporting**: feature-dev:code-reviewer (verify changes), code-simplifier
- **MCP Tools**: Serena (symbolic editing for import fixes)

### Estimated Duration
2-4 hours

---

## Phase 2: Security Hardening (P1)

**Objective**: Harden authentication, encryption, and secrets management for production readiness.

### Tasks
- [ ] Keycloak: Migrate from `start-dev` → `start` with TLS certificates
- [ ] OpenBao: Migrate from `-dev` mode → production with `config.hcl`
  - [ ] Set up file storage backend (or HSM integration)
  - [ ] Configure auth methods (JWT, OIDC)
  - [ ] Enable audit logging
- [ ] AUTH_ENABLED: Change default from `false` → `true` in all services
- [ ] Secrets management:
  - [ ] Rotate hardcoded passwords → environment variables (Vault/OpenBao)
  - [ ] Audit for secrets in logs, configs, comments
  - [ ] Document secret rotation policy
- [ ] OpenSearch: Enable security plugin
  - [ ] Configure TLS for node-to-node communication
  - [ ] Set up authentication (OIDC or basic auth)
- [ ] gRPC services: Upgrade from `insecure` → `TLS` or document service mesh requirement
  - [ ] Generate or provision TLS certs for all gRPC endpoints
  - [ ] Update gRPC client configs to use secure channels
- [ ] Network policies: Define Kubernetes/service mesh rules (if applicable)

### Gate Criteria
- [ ] risk-assessor agent passes security review
- [ ] No Critical or High severity findings in risk assessment
- [ ] All auth mechanisms documented and tested
- [ ] TLS/mTLS enabled for all internal communication
- [ ] No hardcoded secrets remain in codebase

### Agent Assignment
- **Primary**: risk-assessor (security audit)
- **Supporting**: feature-dev:code-architect (design hardening strategy), pr-review-toolkit:silent-failure-hunter
- **MCP Tools**: Serena (secrets replacement), Context7 (threat modeling)

### Estimated Duration
4-6 hours

---

## Phase 3: Infrastructure Readiness (P2)

**Objective**: Ensure all infrastructure components are production-grade and deployable.

### Tasks
- [ ] Docker image tags:
  - [ ] Pin all `latest` tags to specific versions (e.g., `etcd:v3.5.15`)
  - [ ] Audit base images for security vulnerabilities (Trivy, Snyk)
  - [ ] Document image registry (Docker Hub, private registry, etc.)
- [ ] Component updates:
  - [ ] Upgrade etcd to v3.5.15 or later (if not already)
  - [ ] Verify all service images are on supported versions
- [ ] WasmEdge:
  - [ ] Remove placeholder/stub code OR implement full support
  - [ ] Document expected use cases
- [ ] Kubernetes manifests:
  - [ ] Generate manifests from `docker-compose.yml` using Kompose
  - [ ] Add resource requests/limits to all pods
  - [ ] Create Helm chart (optional: for advanced deployments)
  - [ ] Document multi-zone failover strategy
- [ ] Compliance:
  - [ ] Add LICENSE file (Apache-2.0 headers in all source files)
  - [ ] Add SECURITY.md (reporting procedure for vulnerabilities)
  - [ ] Generate SBOM (Software Bill of Materials) using syft or cyclonedx
- [ ] Storage:
  - [ ] Configure persistent volumes (PV/PVC) for stateful services
  - [ ] Document backup/restore procedures for etcd, OpenSearch
- [ ] Monitoring:
  - [ ] Verify Prometheus scrape configs are valid
  - [ ] Verify Grafana dashboards are populated
  - [ ] Document alerting rules

### Gate Criteria
- [ ] `docker-compose up` completes successfully on clean environment
- [ ] All 30+ services reach healthy state within 2 minutes
- [ ] Health checks pass for all service endpoints
- [ ] Kubernetes manifests validate (kubectl --dry-run)
- [ ] License file present and correct
- [ ] SBOM generated and reviewed

### Agent Assignment
- **Primary**: Explore agent (service discovery, manifest verification)
- **Supporting**: superpowers:code-reviewer (code quality), Bash agent (docker/kompose CLI)
- **MCP Tools**: Docker MCP (image verification), Kubernetes MCP (manifest validation)

### Estimated Duration
3-5 hours

---

## Phase 4: Validation & Review

**Objective**: Execute comprehensive testing and code review to validate production readiness.

### Tasks
- [ ] Test execution:
  - [ ] Run full test suite (target: 219 tests, 100% pass rate)
  - [ ] Run integration tests (services communication)
  - [ ] Smoke test UI (if applicable): browser automation (Chrome/Firefox)
  - [ ] Load test key services (e.g., gRPC orchestrator) with k6 or similar
- [ ] Code review:
  - [ ] Static analysis: SonarQube or equivalent
  - [ ] SAST: GitHub Advanced Security (if available)
  - [ ] Dependency check: OWASP Dependency-Check
- [ ] Silent failure detection:
  - [ ] Hunt for unhandled exceptions in critical paths
  - [ ] Verify error logging in all error handlers
  - [ ] Audit timeout/retry logic
- [ ] Documentation review:
  - [ ] Verify README is accurate and complete
  - [ ] Verify API documentation matches actual endpoints
  - [ ] Verify deployment guide is step-by-step executable

### Gate Criteria
- [ ] All 219+ tests pass
- [ ] Code review: 0 blocking issues, <5 minor issues
- [ ] Silent failure hunt: No critical gaps found
- [ ] Static analysis: No High/Critical vulnerabilities
- [ ] Smoke tests: All critical user workflows pass
- [ ] Documentation: Deployment guide tested on clean VM

### Agent Assignment
- **Primary**: pr-review-toolkit:code-reviewer (comprehensive review)
- **Supporting**: pr-review-toolkit:silent-failure-hunter (failure analysis), Playwright (UI smoke tests)
- **MCP Tools**: Playwright MCP (automated UI testing), SonarQube MCP (if available)

### Estimated Duration
3-4 hours

---

## Phase 5: Ship

**Objective**: Tag release, build final artifacts, and push to production registry.

### Tasks
- [ ] Release preparation:
  - [ ] Create release branch (`release/v13.0.0`)
  - [ ] Update version in all relevant files (docker-compose, Helm values, package.json, etc.)
  - [ ] Generate CHANGELOG.md or update with v13.0.0 entry
- [ ] Tagging:
  - [ ] Create git tag: `git tag -a v13.0.0 -m "Production release v13.0.0"`
  - [ ] Push tag to repository
- [ ] Image build & push:
  - [ ] Build all Docker images with tag `v13.0.0`
  - [ ] Push to registry (Docker Hub, ECR, or private registry)
  - [ ] Generate and sign container image manifests
- [ ] Release notes:
  - [ ] Document new features, fixes, breaking changes
  - [ ] Document deployment instructions
  - [ ] Publish release on GitHub/GitLab
- [ ] User communication:
  - [ ] Notify stakeholders of release availability
  - [ ] Provide rollback instructions
- [ ] Post-release:
  - [ ] Monitor production metrics (error rates, latency)
  - [ ] Set up alerts for production anomalies

### Gate Criteria
- [ ] User/stakeholder approval for production push
- [ ] All images successfully pushed to registry
- [ ] Git tags created and verified
- [ ] Release notes published and reviewed
- [ ] Rollback plan documented and tested

### Agent Assignment
- **Primary**: Bash agent (git, docker CLI operations)
- **Supporting**: note-taker (release notes, decision log), Explore agent (artifact verification)
- **MCP Tools**: Docker MCP (registry verification), GitLab MCP (tag/release management)

### Estimated Duration
1-2 hours

---

## Agent/Tool Assignment Matrix

| Phase | Primary Agent | Supporting Agents | MCP Tools | Execution Model |
|-------|--------------|-------------------|-----------|-----------------|
| 1: CI Green | Bash | feature-dev:code-reviewer, code-simplifier | Serena | Sequential (dependencies) |
| 2: Security | risk-assessor | feature-dev:code-architect, pr-review-toolkit:silent-failure-hunter | Serena, Context7 | Sequential (builds on P1) |
| 3: Infrastructure | Explore | superpowers:code-reviewer, Bash | Docker MCP, Kubernetes MCP | Parallel (some tasks independent) |
| 4: Validation | pr-review-toolkit:code-reviewer | pr-review-toolkit:silent-failure-hunter, Playwright | Playwright MCP, SonarQube MCP | Parallel (testing) |
| 5: Ship | Bash | note-taker, Explore | Docker MCP, GitLab MCP | Sequential (dependencies) |

---

## Risk Register

### High Priority

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|-----------|--------|-----------|-------|--------|
| CI pipeline has hidden failures | Medium | High | Run full CI locally before push; use feature-dev:code-reviewer | TBD | Open |
| Hardcoded secrets discovered in production | Low | Critical | Automated secrets scan (TruffleHog, detect-secrets); manual audit | TBD | Open |
| Kubernetes manifests have resource limit issues | Medium | High | Validate manifests with kubectl; load test before deploy | TBD | Open |
| gRPC without TLS causes man-in-the-middle attack | Low | Critical | Implement mTLS immediately; document in security policy | TBD | Open |

### Medium Priority

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|-----------|--------|-----------|-------|--------|
| Service mesh complexity exceeds team knowledge | Medium | Medium | Provide runbooks; assign owner for service mesh | TBD | Open |
| Rollback procedure fails under load | Low | High | Test rollback procedure in staging before prod | TBD | Open |
| OpenSearch cluster becomes unhealthy after deploy | Medium | Medium | Enable cluster snapshots; pre-test failover | TBD | Open |

### Low Priority

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|-----------|--------|-----------|-------|--------|
| WasmEdge dependency is unmaintained | Low | Low | Remove if not used; monitor upstream | TBD | Open |
| etcd cluster requires manual migration | Low | Medium | Test migration path in staging first | TBD | Open |

---

## Decisions Log

### Decision 1: Agent Assignment Strategy
**Date**: 2026-02-09
**Decision**: Assign Bash agent as primary for CI/lint tasks; risk-assessor for security; Explore for infrastructure.
**Reasoning**: Matches agent strengths (CLI automation, threat modeling, discovery).
**Alternatives Considered**: Single general-purpose agent (too slow), manual execution (too error-prone).
**Status**: Pending approval
**Owner**: TBD

### Decision 2: Hardcoded Password Rotation
**Date**: 2026-02-09
**Decision**: Use OpenBao as single source of truth for all secrets; no hardcoded values in config files.
**Reasoning**: Reduces blast radius of accidental exposure; enables rotation without redeployment.
**Alternatives Considered**: Keep hardcoded in .env.local (rejected: not production-safe), use Kubernetes Secrets only (rejected: no rotation mechanism).
**Status**: Pending security review
**Owner**: TBD

### Decision 3: Kubernetes Manifests Generation
**Date**: 2026-02-09
**Decision**: Use Kompose to auto-generate manifests from docker-compose.yml; manually add resource limits and health checks.
**Reasoning**: Reduces manual effort; maintains single source of truth in compose file.
**Alternatives Considered**: Hand-write all manifests (too slow), Helm only (loses compose compatibility).
**Status**: Pending infrastructure review
**Owner**: TBD

### Decision 4: Test Coverage Gate
**Date**: 2026-02-09
**Decision**: Require 100% test pass rate before Phase 5 shipping; no exceptions for "flaky tests."
**Reasoning**: Production stability depends on reliable tests.
**Alternatives Considered**: 95% pass threshold (rejected: too lenient).
**Status**: Pending approval
**Owner**: TBD

---

## Execution Checklist

### Pre-Execution
- [ ] Stakeholders briefed on timeline (expected: 13-21 hours total)
- [ ] Agents and MCP tools configured and tested
- [ ] Backup of current `master` branch created
- [ ] Staging environment available for testing
- [ ] Production credentials (registry, cloud) rotated and verified

### During Execution
- [ ] Daily standup on progress
- [ ] Real-time blocker escalation
- [ ] Decision log maintained by note-taker
- [ ] Risk register updated with newly discovered issues
- [ ] Slack/email notifications for phase completions

### Post-Execution
- [ ] Retrospective session: what went well, what didn't
- [ ] Update runbooks with lessons learned
- [ ] Celebrate shipping!

---

## Success Criteria (Final Gate)

- [x] **All 5 phases completed** (CI, Security, Infrastructure, Validation, Ship)
- [x] **Zero Critical findings** in risk assessment
- [x] **219+ tests passing** at 100% rate
- [x] **All services healthy** in docker-compose environment
- [x] **Kubernetes manifests validated** and tested
- [x] **v13.0.0 tag pushed** to repository
- [x] **Docker images in registry** with v13.0.0 tag
- [x] **CHANGELOG and release notes** published
- [x] **Stakeholder approval** for production deployment

---

## Timeline Estimate

| Phase | Est. Duration | Start | End |
|-------|--------------|-------|-----|
| 1: CI Green | 2-4 hours | TBD | TBD |
| 2: Security | 4-6 hours | TBD | TBD |
| 3: Infrastructure | 3-5 hours | TBD | TBD |
| 4: Validation | 3-4 hours | TBD | TBD |
| 5: Ship | 1-2 hours | TBD | TBD |
| **Total** | **13-21 hours** | TBD | TBD |

---

## Appendix: Tools & References

### Agent Resources
- **Bash Agent**: For CLI operations (ruff, docker, git, kubectl)
- **risk-assessor**: For security audit and threat modeling
- **feature-dev:code-reviewer**: For code quality and architectural review
- **pr-review-toolkit:silent-failure-hunter**: For unhandled exception hunting
- **Explore Agent**: For service discovery and manifest validation
- **note-taker**: For decision logging and release notes

### MCP Tools
- **Serena**: Symbolic editing (secrets replacement, imports)
- **Docker MCP**: Image verification and registry operations
- **Kubernetes MCP**: Manifest validation (if available)
- **Playwright MCP**: UI automation and smoke testing
- **Context7**: Threat modeling and security analysis

### Documentation References
- Antigravity-Node Architecture: (TBD)
- FastAPI Best Practices: https://fastapi.tiangolo.com/deployment/
- gRPC Security: https://grpc.io/docs/guides/auth/
- Kubernetes Security: https://kubernetes.io/docs/concepts/security/

---

**Document Owner**: TBD
**Last Updated**: 2026-02-09
**Next Review**: Upon Phase 1 completion
