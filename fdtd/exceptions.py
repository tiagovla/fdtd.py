"""This module contains all fdtd exceptions."""


class FDTDException(Exception):
    """FDTD base exception."""


class MaterialExists(FDTDException):
    """Material already exists exception."""


class MaterialNotFound(FDTDException):
    """Material not found exception."""


class AlreadyRegistered(FDTDException):
    """Object already registered exception."""


class OutBounds(FDTDException):
    """Object is out bounds exception."""


class WrongBounding(FDTDException):
    """Object's bounding is wrongly defined exception."""
