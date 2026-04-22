"""Tests for FastAPI lifespan context manager.

Validates that the lifespan-based startup/shutdown replaces
the deprecated @app.on_event("startup") handlers correctly.
"""

import pytest
from fastapi.testclient import TestClient


class TestLifespan:
    """Test lifespan context manager integration."""

    def test_lifespan_context_exists(self):
        """Lifespan context must be wired into the app."""
        # Import fresh to avoid cached app instance
        import importlib
        import server as server_module

        importlib.reload(server_module)

        assert server_module.app.router.lifespan_context is not None

    def test_app_boots_without_startup_warnings(self):
        """App should boot cleanly via TestClient without deprecation warnings."""
        import warnings
        import importlib
        import server as server_module

        importlib.reload(server_module)

        # Capture any deprecation warnings during client creation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with TestClient(server_module.app) as client:
                response = client.get("/api/health")
                assert response.status_code == 200

            # Check for FastAPI startup deprecation warnings
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "on_event" in str(warning.message)
            ]
            assert (
                len(deprecation_warnings) == 0
            ), f"Found @app.on_event deprecation warnings: {[str(w.message) for w in deprecation_warnings]}"

    def test_client_enter_exit_cleanly(self):
        """TestClient context manager should enter and exit without errors."""
        import importlib
        import server as server_module

        importlib.reload(server_module)

        # Should be able to enter and exit the client context cleanly
        client = TestClient(server_module.app)
        response = client.get("/api/health")
        assert response.status_code == 200
        client.close()

    def test_no_on_event_startup_handlers_remain(self):
        """Verify no deprecated @app.on_event('startup') handlers exist."""
        import ast
        import inspect
        import importlib
        import server as server_module

        importlib.reload(server_module)

        # Parse the source to find actual decorator usage (not comments/strings)
        source = inspect.getsource(server_module)
        tree = ast.parse(source)

        # Find all decorators that are @app.on_event("startup")
        deprecated_decorators = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if (isinstance(func, ast.Attribute)
                            and func.attr == "on_event"
                            and isinstance(func.value, ast.Name)
                            and func.value.id == "app"):
                            # Check if the argument is "startup"
                            args = decorator.args
                            if args and isinstance(args[0], ast.Constant) and args[0].value == "startup":
                                deprecated_decorators.append(node.name)

        assert len(deprecated_decorators) == 0, (
            f"Found deprecated @app.on_event('startup') handlers: {deprecated_decorators}. "
            "All startup logic should be in the lifespan context manager."
        )
