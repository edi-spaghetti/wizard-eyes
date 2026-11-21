
import logging as _logging

# Module-level loggers (wizard_eyes.login, .camera, .assets, ...) log to the
# console like the engine's per-class loggers do.
_logger = _logging.getLogger('wizard_eyes')
if not _logger.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter('%(message)s'))
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.INFO)
    _logger.propagate = False
