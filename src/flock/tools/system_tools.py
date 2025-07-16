from flock.core.logging.trace_and_logged import traced_and_logged
from flock.core.registry.decorators import flock_tool


@traced_and_logged
@flock_tool
def get_current_time() -> str:
    import datetime

    time = datetime.datetime.now().isoformat()
    return time
