"""Drive AI Studio turns through the streaming path the app uses.

Every tab calls ``AIChatWorkflow.stream_*_message``, streams the generator
into the page, then calls ``finalize()``. Tests do the same, with
``AIClientService.stream_tool_loop`` patched to stream a canned response.
"""


def finish(turn):
    """Consume a ``stream_*_message`` result as the UI does; return what
    ``finalize()`` returns."""
    generator, finalize = turn
    for _ in generator:
        pass
    return finalize()


def canned_stream(*responses):
    """A stand-in for ``AIClientService.stream_tool_loop``: call n streams
    ``responses[n]`` (the last one repeats). Each call's keyword arguments
    are kept in ``.calls``."""
    calls = []

    def stream_tool_loop(**kwargs):
        calls.append(kwargs)
        yield responses[min(len(calls), len(responses)) - 1]

    stream_tool_loop.calls = calls
    return stream_tool_loop
