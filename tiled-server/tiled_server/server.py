"""tiled LSP server — server factory and LSP handler wiring."""

from __future__ import annotations

import logging
from typing import Optional

from lsprotocol import types as lsp

try:
    from pygls.server import LanguageServer
except ImportError:
    from pygls.lsp.server import LanguageServer  # pygls < 2.0

from .completion import build_completions, symbol_to_completion
from .definition import build_definition
from .detection import (
    _RE_IMPORT_T,
    _RE_IMPORT_TILELANG,
    _RE_PRIM_FUNC_NO_KERNEL,
    _RE_T_DOT,
    _RE_TILELANG_DOT,
    is_tilelang_file,
)
from .diagnostics import compute_diagnostics
from .hover import build_hover
from .signature import build_signature_help

logger = logging.getLogger(__name__)

# Backward-compatible aliases so existing tests keep working
_is_tilelang_file = is_tilelang_file
_compute_diagnostics = compute_diagnostics
_symbol_to_completion = symbol_to_completion


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server() -> LanguageServer:
    """Create and configure the tiled language server."""
    server = LanguageServer("tiled", "v0.1.0")

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: lsp.DidOpenTextDocumentParams):
        doc = server.workspace.get_text_document(params.text_document.uri)
        diagnostics = compute_diagnostics(doc.uri, doc.source)
        server.text_document_publish_diagnostics(lsp.PublishDiagnosticsParams(uri=doc.uri, diagnostics=diagnostics))

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    def did_save(params: lsp.DidSaveTextDocumentParams):
        doc = server.workspace.get_text_document(params.text_document.uri)
        diagnostics = compute_diagnostics(doc.uri, doc.source)
        server.text_document_publish_diagnostics(lsp.PublishDiagnosticsParams(uri=doc.uri, diagnostics=diagnostics))

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: lsp.DidChangeTextDocumentParams):
        doc = server.workspace.get_text_document(params.text_document.uri)
        diagnostics = compute_diagnostics(doc.uri, doc.source)
        server.text_document_publish_diagnostics(lsp.PublishDiagnosticsParams(uri=doc.uri, diagnostics=diagnostics))

    @server.feature(
        lsp.TEXT_DOCUMENT_COMPLETION,
        lsp.CompletionOptions(
            trigger_characters=[".", "@"],
            resolve_provider=False,
        ),
    )
    def completions(params: lsp.CompletionParams) -> lsp.CompletionList:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return build_completions(doc, params.position)

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def hover(params: lsp.HoverParams) -> Optional[lsp.Hover]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return build_hover(doc, params.position)

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def definition(params: lsp.DefinitionParams) -> Optional[lsp.Location]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        folders = server.workspace.folders.values() if server.workspace.folders else []
        return build_definition(doc, params.position, list(folders))

    @server.feature(lsp.TEXT_DOCUMENT_SIGNATURE_HELP,
                    lsp.SignatureHelpOptions(trigger_characters=["(", ","]))
    def signature_help(params: lsp.SignatureHelpParams) -> Optional[lsp.SignatureHelp]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return build_signature_help(doc, params.position)

    # ── Code Actions (diagnostic → AI fix suggestions) ────────────────

    # Diagnostic codes that can be auto-fixed by MCP auto_optimize
    _FIXABLE_CODES = {"missing-clear", "deprecated-alloc-buffer", "alloc-outside-kernel", "missing-dtype"}

    @server.feature(
        lsp.TEXT_DOCUMENT_CODE_ACTION,
        lsp.CodeActionOptions(
            code_action_kinds=[lsp.CodeActionKind.QuickFix],
        ),
    )
    def code_action(params: lsp.CodeActionParams) -> list[lsp.CodeAction]:
        actions: list[lsp.CodeAction] = []
        has_fixable = False

        for diag in params.context.diagnostics:
            if diag.source != "tiled":
                continue
            code = diag.code if isinstance(diag.code, str) else ""
            if code in _FIXABLE_CODES:
                has_fixable = True

        if has_fixable:
            actions.append(lsp.CodeAction(
                title="TileLang: Optimize kernel with AI (auto_optimize)",
                kind=lsp.CodeActionKind.QuickFix,
                diagnostics=[d for d in params.context.diagnostics
                             if d.source == "tiled"
                             and (isinstance(d.code, str) and d.code in _FIXABLE_CODES)],
                command=lsp.Command(
                    title="Optimize kernel",
                    command="tiled.optimizeKernel",
                ),
            ))

        return actions

    @server.feature(lsp.INITIALIZE)
    def initialize(params: lsp.InitializeParams) -> None:
        logger.info("tiled language server initialized")

    return server
