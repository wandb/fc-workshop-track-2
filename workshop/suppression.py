import logging

# Suppress LiteLLM debug logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.WARNING)

# Suppress Weave trace server logging that causes JSON serialization errors
logging.getLogger("weave").setLevel(logging.CRITICAL)
logging.getLogger("weave.trace_server_bindings").setLevel(logging.CRITICAL)
weave_batch_logger = "weave.trace_server_bindings.async_batch_processor"
logging.getLogger(weave_batch_logger).setLevel(logging.CRITICAL)
weave_http_logger = "weave.trace_server_bindings.remote_http_trace_server"
logging.getLogger(weave_http_logger).setLevel(logging.CRITICAL)

# Suppress Weave trace API publishing messages
logging.getLogger("weave.trace.api").setLevel(logging.WARNING)
# Suppress Weave trace operation links
# logging.getLogger("weave.trace.op").setLevel(logging.WARNING)