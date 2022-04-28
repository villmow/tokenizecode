from dataclasses import dataclass


@dataclass
class Point:
    """
    Models a point in the text file. Row represents line number and column characters in that line.

    FIXME: Is it really characters or byte??
    """
    row: int
    column: int

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False

        return self.row == other.row and self.column == other.column

    def __lt__(self, other):
        if not isinstance(other, Point):
            return False

        if self.row < other.row:
            return True
        elif self.row == other.row and self.column < other.column:
            return True

        return False

    def __le__(self, other):
        if not isinstance(other, Point):
            return False

        if self.row < other.row:
            return True
        elif self.row == other.row and self.column <= other.column:
            return True

        return False


@dataclass
class Span:
    """ A span in a code file."""
    start_byte: int
    end_byte: int
    start_point: Point
    end_point: Point

    def __post_init__(self):
        if isinstance(self.start_point, (tuple, list)):
            self.start_point = Point(*self.start_point)
        elif isinstance(self.start_point, dict):
            self.start_point = Point(**self.start_point)

        if isinstance(self.end_point, tuple):
            self.end_point = Point(*self.end_point)
        elif isinstance(self.end_point, dict):
            self.end_point = Point(**self.end_point)
