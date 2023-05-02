from enum import Enum, auto


class precondition_enum(Enum):
    JACOBI = auto(),
    GAUSS_SEIDEL = auto(),
    SYMMETRIC_GAUSS_SEIDEL = auto(),
