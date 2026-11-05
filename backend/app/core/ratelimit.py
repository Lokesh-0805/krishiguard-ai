from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings

_s = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[_s.rate_limit_default], enabled=_s.rate_limit_enabled)
