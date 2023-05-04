from enum import Enum, auto


class PreconditionEnum(Enum):
    JACOBI = auto(),
    GAUSS_SEIDEL = auto(),
    SYMMETRIC_GAUSS_SEIDEL = auto(),


class BuiltInMethodsEnum(Enum):
    GENERAL_SOLVER = auto(),
    LU = auto(),
    # GMRES = auto(),


class ImplementationMethodsEnum(Enum):
    GMRES = auto(),
    CG = auto(),
