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
        server.publish_diagnostics(doc.uri, diagnostics)

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    def did_save(params: lsp.DidSaveTextDocumentParams):
        doc = server.workspace.get_text_document(params.text_document.uri)
        diagnostics = compute_diagnostics(doc.uri, doc.source)
        server.publish_diagnostics(doc.uri, diagnostics)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: lsp.DidChangeTextDocumentParams):
        doc = server.workspace.get_text_document(params.text_document.uri)
        diagnostics = compute_diagnostics(doc.uri, doc.source)
        server.publish_diagnostics(doc.uri, diagnostics)

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

    @server.feature(lsp.TEXT_DOCUMENT_SIGNATURE_HELP,
                    lsp.SignatureHelpOptions(trigger_characters=["(", ","]))
    def signature_help(params: lsp.SignatureHelpParams) -> Optional[lsp.SignatureHelp]:
        doc = server.workspace.get_text_document(params.text_document.uri)
        return build_signature_help(doc, params.position)

    @server.feature(lsp.INITIALIZE)
    def initialize(params: lsp.InitializeParams) -> None:
        logger.info("tiled language server initialized")

    return server
