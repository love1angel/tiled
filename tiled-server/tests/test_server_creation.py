"""Tests for server creation."""

from tiled_server.server import create_server


class TestServerCreation:
    def test_server_name(self):
        server = create_server()
        assert server.name == "tiled"

    def test_server_version(self):
        server = create_server()
        assert server.version == "v0.1.0"
