import logging

def throw_boolean(ok, info = ""):
    if not ok:
        raise RuntimeError(info)