class CarvingError(Exception):
    """Erreur liée au carving JPEG."""


class ValidationError(Exception):
    """Erreur liée à la validation d'image."""


class CorruptionError(Exception):
    """Erreur liée à la corruption d'image."""


class ReconstructionError(Exception):
    """Erreur liée à la reconstruction d'image."""


class EvaluationError(Exception):
    """Erreur liée à l'évaluation d'image."""