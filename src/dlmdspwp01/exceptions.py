class DataValidationError(Exception):
    """Raised when an input dataset does not match the expected structure."""


class MappingNotPossible(Exception):
    """Raised when a test point cannot be mapped to any selected ideal function."""


class DatabaseError(Exception):
    """Raised for database-related issues."""
