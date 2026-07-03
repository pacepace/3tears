# 3tears-agent-knowledge

Agent-side governed-knowledge read stack for LangGraph agents.

Homes the playbook-entry / concept collections (over the platform RBAC proxy), the
`KnowledgeIntegration` wiring, and the `create_agent` `before_model` injection
middleware, on top of the shared three-scope shadow-merge authority in
`threetears.knowledge` (core). It lives in its own package because the collections
depend on `threetears.agent.acl`, which core cannot depend on.
