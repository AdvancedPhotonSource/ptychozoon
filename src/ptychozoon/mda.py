# Copyright © 2026 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/ptychozoon/blob/main/LICENSE.TXT
"""Reader for MDA (Multi-Dimensional Array) files.

MDA is a binary file format used at Argonne National Laboratory's synchrotron
beamlines to record multi-dimensional scan data, including motor positions
(positioners), detector counts, and EPICS process variable metadata.

The format uses XDR (External Data Representation) encoding for most fields.
"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Generic, TypeVar
import logging
import sys
import typing
from mda_xdrlib import xdrlib
import yaml

import numpy
import numpy.typing

T = TypeVar("T")

logger = logging.getLogger(__name__)


class EpicsType(IntEnum):
    """EPICS Channel Access data type codes.

    These integer codes identify the data type of a process variable (PV)
    value stored in an MDA extra-PV block.  The naming convention mirrors
    the ``DBR_*`` macros from the EPICS ``db_access.h`` header.
    """

    DBR_STRING = 0
    DBR_SHORT = 1
    DBR_FLOAT = 2
    DBR_ENUM = 3
    DBR_CHAR = 4
    DBR_LONG = 5
    DBR_DOUBLE = 6
    DBR_STS_STRING = 7
    DBR_STS_SHORT = 8
    DBR_STS_FLOAT = 9
    DBR_STS_ENUM = 10
    DBR_STS_CHAR = 11
    DBR_STS_LONG = 12
    DBR_STS_DOUBLE = 13
    DBR_TIME_STRING = 14
    DBR_TIME_SHORT = 15
    DBR_TIME_FLOAT = 16
    DBR_TIME_ENUM = 17
    DBR_TIME_CHAR = 18
    DBR_TIME_LONG = 19
    DBR_TIME_DOUBLE = 20
    DBR_GR_STRING = 21
    DBR_GR_SHORT = 22
    DBR_GR_FLOAT = 23
    DBR_GR_ENUM = 24
    DBR_GR_CHAR = 25
    DBR_GR_LONG = 26
    DBR_GR_DOUBLE = 27
    DBR_CTRL_STRING = 28
    DBR_CTRL_SHORT = 29
    DBR_CTRL_FLOAT = 30
    DBR_CTRL_ENUM = 31
    DBR_CTRL_CHAR = 32
    DBR_CTRL_LONG = 33
    DBR_CTRL_DOUBLE = 34


def read_int_from_buffer(fp: typing.BinaryIO) -> int:
    """Read a 32-bit signed integer from a binary file in XDR format.

    Parameters
    ----------
    fp : BinaryIO
        Open binary file object positioned at the integer to read.

    Returns
    -------
    int
        The decoded integer value.
    """
    unpacker = xdrlib.Unpacker(fp.read(4))
    return unpacker.unpack_int()


def read_float_from_buffer(fp: typing.BinaryIO) -> float:
    """Read a 32-bit IEEE-754 float from a binary file in XDR format.

    Parameters
    ----------
    fp : BinaryIO
        Open binary file object positioned at the float to read.

    Returns
    -------
    float
        The decoded floating-point value.
    """
    unpacker = xdrlib.Unpacker(fp.read(4))
    return unpacker.unpack_float()


def read_counted_string(unpacker: xdrlib.Unpacker) -> str:
    """Read a length-prefixed string from an XDR unpacker.

    The string is preceded by a 32-bit integer giving its byte length.
    An empty string is returned when the length field is zero.

    Parameters
    ----------
    unpacker : xdrlib.Unpacker
        XDR unpacker positioned at the length field.

    Returns
    -------
    str
        The decoded string, or an empty string if the length is zero.
    """
    length = unpacker.unpack_int()
    return unpacker.unpack_string().decode() if length else str()


def read_counted_string_from_buffer(fp: typing.BinaryIO) -> str:
    """Read a length-prefixed string directly from a binary file.

    Reads the 4-byte length header first, then advances the file position
    by the padded string width before decoding.  XDR strings are rounded
    up to a multiple of 4 bytes on disk.

    Parameters
    ----------
    fp : BinaryIO
        Open binary file object positioned at the length field.

    Returns
    -------
    str
        The decoded string, or an empty string if the length is zero.
    """
    length = read_int_from_buffer(fp)

    if length:
        sz = (length + 3) // 4 * 4 + 4
        unpacker = xdrlib.Unpacker(fp.read(sz))
        return unpacker.unpack_string().decode()

    return str()


@dataclass(frozen=True)
class MDAHeader:
    """Top-level header of an MDA file.

    Attributes
    ----------
    version : float
        MDA file format version number.
    scan_number : int
        Unique identifier for this scan.
    dimensions : list[int]
        Requested number of points along each scan axis.
    is_regular : bool
        ``True`` when the scan follows a regular (rectangular) grid.
    extra_pvs_offset : int
        Byte offset within the file at which extra process variables begin.
        Zero when no extra PVs are present.
    """

    version: float
    scan_number: int
    dimensions: list[int]
    is_regular: bool
    extra_pvs_offset: int

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAHeader:
        """Read an :class:`MDAHeader` from the current position in *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the header.

        Returns
        -------
        MDAHeader
            Parsed header.
        """
        unpacker = xdrlib.Unpacker(fp.read(12))
        version = unpacker.unpack_float()
        scan_number = unpacker.unpack_int()
        data_rank = unpacker.unpack_int()

        unpacker.reset(fp.read(4 * data_rank + 8))
        dimensions = unpacker.unpack_farray(data_rank, unpacker.unpack_int)
        is_regular = unpacker.unpack_bool()
        extra_pvs_offset = unpacker.unpack_int()

        return cls(version, scan_number, dimensions, is_regular, extra_pvs_offset)

    @property
    def data_rank(self) -> int:
        """Number of scan dimensions (rank of the data array)."""
        return len(self.dimensions)

    @property
    def has_extra_pvs(self) -> bool:
        """``True`` when the file contains extra process variable records."""
        return self.extra_pvs_offset > 0

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation suitable for YAML serialisation.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all header fields.
        """
        return {
            "version": self.version,
            "scan_number": self.scan_number,
            "dimensions": self.dimensions,
            "is_regular": self.is_regular,
            "extra_pvs_offset": self.extra_pvs_offset,
        }


@dataclass(frozen=True)
class MDAScanHeader:
    """Per-scan header containing dimensional and offset metadata.

    Attributes
    ----------
    rank : int
        Dimensionality of this scan (1 for the inner-most scan).
    num_requested_points : int
        Total number of scan points requested for this dimension.
    current_point : int
        Index of the last point that was actually acquired.
    lower_scan_offsets : list[int]
        File byte offsets to each nested (lower-rank) scan.
        Empty for inner-most (rank-1) scans.
    """

    rank: int
    num_requested_points: int
    current_point: int
    lower_scan_offsets: list[int]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanHeader:
        """Read an :class:`MDAScanHeader` from the current position in *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the scan header.

        Returns
        -------
        MDAScanHeader
            Parsed scan header.
        """
        unpacker = xdrlib.Unpacker(fp.read(12))
        rank = unpacker.unpack_int()
        npts = unpacker.unpack_int()
        cpt = unpacker.unpack_int()
        lower_scan_offsets: list[int] = list()

        if rank > 1:
            unpacker.reset(fp.read(4 * npts))
            lower_scan_offsets = unpacker.unpack_farray(npts, unpacker.unpack_int)

        return cls(rank, npts, cpt, lower_scan_offsets)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation of this scan header.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all scan header fields.
        """
        return {
            "rank": self.rank,
            "num_requested_points": self.num_requested_points,
            "current_point": self.current_point,
            "lower_scan_offsets": self.lower_scan_offsets,
        }


@dataclass(frozen=True)
class MDAScanPositionerInfo:
    """Metadata describing a single scan positioner (motor or PV).

    Attributes
    ----------
    number : int
        Positioner index within the scan.
    name : str
        EPICS PV name of the setpoint.
    description : str
        Human-readable description of the positioner.
    step_mode : str
        Stepping mode string (e.g. ``"LINEAR"``).
    unit : str
        Engineering unit of the setpoint values.
    readback_name : str
        EPICS PV name of the readback.
    readback_description : str
        Human-readable description of the readback.
    readback_unit : str
        Engineering unit of the readback values.
    """

    number: int
    name: str
    description: str
    step_mode: str
    unit: str
    readback_name: str
    readback_description: str
    readback_unit: str

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanPositionerInfo:
        """Read an :class:`MDAScanPositionerInfo` from *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the positioner record.

        Returns
        -------
        MDAScanPositionerInfo
            Parsed positioner metadata.
        """
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        description = read_counted_string_from_buffer(fp)
        step_mode = read_counted_string_from_buffer(fp)
        unit = read_counted_string_from_buffer(fp)
        readback_name = read_counted_string_from_buffer(fp)
        readback_description = read_counted_string_from_buffer(fp)
        readback_unit = read_counted_string_from_buffer(fp)

        return cls(
            number,
            name,
            description,
            step_mode,
            unit,
            readback_name,
            readback_description,
            readback_unit,
        )

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation of this positioner record.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all positioner metadata fields.
        """
        return {
            "number": self.number,
            "name": self.name,
            "description": self.description,
            "step_mode": self.step_mode,
            "unit": self.unit,
            "readback_name": self.readback_name,
            "readback_description": self.readback_description,
            "readback_unit": self.readback_unit,
        }


@dataclass(frozen=True)
class MDAScanDetectorInfo:
    """Metadata describing a single scan detector channel.

    Attributes
    ----------
    number : int
        Detector index within the scan.
    name : str
        EPICS PV name of the detector.
    description : str
        Human-readable description of the detector.
    unit : str
        Engineering unit of the detector values.
    """

    number: int
    name: str
    description: str
    unit: str

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanDetectorInfo:
        """Read an :class:`MDAScanDetectorInfo` from *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the detector record.

        Returns
        -------
        MDAScanDetectorInfo
            Parsed detector metadata.
        """
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        description = read_counted_string_from_buffer(fp)
        unit = read_counted_string_from_buffer(fp)
        return cls(number, name, description, unit)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation of this detector record.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all detector metadata fields.
        """
        return {
            "number": self.number,
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class MDAScanTriggerInfo:
    """Metadata describing a single scan trigger.

    Attributes
    ----------
    number : int
        Trigger index within the scan.
    name : str
        EPICS PV name of the trigger.
    command : float
        Command value sent to the trigger PV at each scan point.
    """

    number: int
    name: str
    command: float

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanTriggerInfo:
        """Read an :class:`MDAScanTriggerInfo` from *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the trigger record.

        Returns
        -------
        MDAScanTriggerInfo
            Parsed trigger metadata.
        """
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        command = read_float_from_buffer(fp)
        return cls(number, name, command)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation of this trigger record.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all trigger metadata fields.
        """
        return {
            "number": self.number,
            "name": self.name,
            "command": self.command,
        }


@dataclass(frozen=True)
class MDAScanInfo:
    """Metadata block for an MDA scan, including positioners and detectors.

    Attributes
    ----------
    scan_name : str
        Name of the scan record (e.g. ``"2xfm:scan1"``).
    time_stamp : str
        ISO-formatted timestamp string recorded at scan start.
    positioner : list[MDAScanPositionerInfo]
        Ordered list of positioner metadata objects.
    detector : list[MDAScanDetectorInfo]
        Ordered list of detector metadata objects.
    trigger : list[MDAScanTriggerInfo]
        Ordered list of trigger metadata objects.
    """

    scan_name: str
    time_stamp: str
    positioner: list[MDAScanPositionerInfo]
    detector: list[MDAScanDetectorInfo]
    trigger: list[MDAScanTriggerInfo]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanInfo:
        """Read an :class:`MDAScanInfo` from *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the scan info block.

        Returns
        -------
        MDAScanInfo
            Parsed scan metadata.
        """
        scan_name = read_counted_string_from_buffer(fp)
        time_stamp = read_counted_string_from_buffer(fp)

        unpacker = xdrlib.Unpacker(fp.read(12))
        np = unpacker.unpack_int()
        nd = unpacker.unpack_int()
        nt = unpacker.unpack_int()

        positioner = [MDAScanPositionerInfo.read(fp) for p in range(np)]
        detector = [MDAScanDetectorInfo.read(fp) for d in range(nd)]
        trigger = [MDAScanTriggerInfo.read(fp) for t in range(nt)]

        return cls(scan_name, time_stamp, positioner, detector, trigger)

    @property
    def num_positioners(self) -> int:
        """Number of positioners in this scan."""
        return len(self.positioner)

    @property
    def num_detectors(self) -> int:
        """Number of detector channels in this scan."""
        return len(self.detector)

    @property
    def num_triggers(self) -> int:
        """Number of triggers in this scan."""
        return len(self.trigger)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation of this scan info block.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all scan metadata fields.
        """
        return {
            "scan_name": self.scan_name,
            "time_stamp": self.time_stamp,
            "positioner": [pos.to_mapping() for pos in self.positioner],
            "detector": [det.to_mapping() for det in self.detector],
            "trigger": [tri.to_mapping() for tri in self.trigger],
        }


@dataclass(frozen=True)
class MDAScanData:
    """Raw numerical data arrays for a single MDA scan.

    Attributes
    ----------
    readback_array : ndarray
        Shape ``(num_positioners, num_requested_points)`` double-precision
        array of readback values recorded at each scan point.
    detector_array : ndarray
        Shape ``(num_detectors, num_requested_points)`` single-precision
        array of detector counts recorded at each scan point.
    """

    readback_array: numpy.typing.NDArray[numpy.floating[Any]]  # double, shape: np x npts
    detector_array: numpy.typing.NDArray[numpy.floating[Any]]  # float, shape: nd x npts

    @classmethod
    def read(
        cls, fp: typing.BinaryIO, scan_header: MDAScanHeader, scan_info: MDAScanInfo
    ) -> MDAScanData:
        """Read the data arrays for a scan from *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the data block.
        scan_header : MDAScanHeader
            Header providing the number of requested points.
        scan_info : MDAScanInfo
            Info block providing the number of positioners and detectors.

        Returns
        -------
        MDAScanData
            Parsed readback and detector arrays.
        """
        npts = scan_header.num_requested_points
        np = scan_info.num_positioners
        nd = scan_info.num_detectors

        unpacker = xdrlib.Unpacker(fp.read(8 * np * npts))
        readback_lol = [unpacker.unpack_farray(npts, unpacker.unpack_double) for p in range(np)]
        readback_array = numpy.array(readback_lol)

        unpacker.reset(fp.read(4 * nd * npts))
        detector_lol = [unpacker.unpack_farray(npts, unpacker.unpack_float) for d in range(nd)]
        detector_array = numpy.array(detector_lol)

        return cls(readback_array, detector_array)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict summary of this data block.

        Returns
        -------
        Mapping[str, Any]
            Dictionary containing dtype and shape strings for each array.
        """
        return {
            "readback_array": f"{self.readback_array.dtype}{self.readback_array.shape}",
            "detector_array": f"{self.detector_array.dtype}{self.detector_array.shape}",
        }


@dataclass(frozen=True)
class MDAScan:
    """Complete representation of one MDA scan dimension.

    For multi-dimensional scans, each outer scan contains a list of inner
    (lower-rank) scans stored in :attr:`lower_scans`.

    Attributes
    ----------
    header : MDAScanHeader
        Dimensional and offset metadata for this scan.
    info : MDAScanInfo
        Positioner, detector, and trigger metadata.
    data : MDAScanData
        Raw numerical data arrays.
    lower_scans : list[MDAScan]
        Nested inner scans (empty for the innermost dimension).
    """

    header: MDAScanHeader
    info: MDAScanInfo
    data: MDAScanData
    lower_scans: list[MDAScan]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScan:
        """Read a complete scan (including nested inner scans) from *fp*.

        Parameters
        ----------
        fp : BinaryIO
            Open binary file object positioned at the start of the scan block.

        Returns
        -------
        MDAScan
            Fully parsed scan including all nested dimensions.
        """
        header = MDAScanHeader.read(fp)
        info = MDAScanInfo.read(fp)
        data = MDAScanData.read(fp, header, info)
        lower_scans: list[MDAScan] = list()

        for offset in header.lower_scan_offsets:
            fp.seek(offset)
            scan = MDAScan.read(fp)
            lower_scans.append(scan)

        return cls(header, info, data, lower_scans)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a nested plain-dict representation of this scan.

        Returns
        -------
        Mapping[str, Any]
            Dictionary containing all scan fields, recursively serialised.
        """
        return {
            "header": self.header.to_mapping(),
            "info": self.info.to_mapping(),
            "data": self.data.to_mapping(),
            "lower_scans": [scan.to_mapping() for scan in self.lower_scans],
        }

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_mapping(), sort_keys=False)


@dataclass(frozen=True)
class MDAProcessVariable(Generic[T]):
    """A single EPICS process variable stored in an MDA extra-PV block.

    The type parameter ``T`` reflects the Python type of :attr:`value`:

    * :class:`str` for ``DBR_STRING`` and ``DBR_CTRL_CHAR`` types.
    * ``list[int]`` for ``DBR_CTRL_SHORT`` / ``DBR_CTRL_LONG``.
    * ``list[float]`` for ``DBR_CTRL_FLOAT`` / ``DBR_CTRL_DOUBLE``.

    Attributes
    ----------
    name : str
        EPICS PV name (e.g. ``"2xfm:m60.VAL"``).
    description : str
        Human-readable description of the PV.
    epics_type : EpicsType
        EPICS Channel Access data type code.
    unit : str
        Engineering unit string (may be empty).
    value : T
        Decoded value of the process variable.
    """

    name: str
    description: str
    epics_type: EpicsType
    unit: str
    value: T

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a plain-dict representation of this process variable.

        Returns
        -------
        Mapping[str, Any]
            Dictionary with all PV fields, including the EPICS type name.
        """
        return {
            "name": self.name,
            "description": self.description,
            "epicsType": self.epics_type.name,
            "unit": self.unit,
            "value": self.value,
        }


@dataclass(frozen=True)
class MDAFile:
    """Complete parsed representation of an MDA file.

    Attributes
    ----------
    header : MDAHeader
        Top-level file header.
    scan : MDAScan
        Root (outermost) scan.
    extra_pvs : list[MDAProcessVariable]
        Extra EPICS process variables recorded alongside the scan.
    """

    header: MDAHeader
    scan: MDAScan
    extra_pvs: list[MDAProcessVariable[Any]]

    @staticmethod
    def _read_pv(unpacker: xdrlib.Unpacker) -> MDAProcessVariable[typing.Any]:
        """Decode a single process variable record from the extra-PV block.

        Parameters
        ----------
        unpacker : xdrlib.Unpacker
            XDR unpacker positioned at the start of the PV record.

        Returns
        -------
        MDAProcessVariable
            Parsed process variable with a typed value.
        """
        pv_name = read_counted_string(unpacker)
        pv_desc = read_counted_string(unpacker)
        pv_type = EpicsType(unpacker.unpack_int())

        if pv_type == EpicsType.DBR_STRING:
            value_str = read_counted_string(unpacker)
            return MDAProcessVariable[str](pv_name, pv_desc, pv_type, str(), value_str)

        count = unpacker.unpack_int()
        pv_unit = read_counted_string(unpacker)

        if pv_type == EpicsType.DBR_CTRL_CHAR:
            value_char = unpacker.unpack_fstring(count).decode()
            value_char = value_char.split("\x00", 1)[0]  # treat as null-terminated string
            return MDAProcessVariable[str](pv_name, pv_desc, pv_type, pv_unit, value_char)
        elif pv_type == EpicsType.DBR_CTRL_SHORT:
            value_short = unpacker.unpack_farray(count, unpacker.unpack_int)
            return MDAProcessVariable[list[int]](pv_name, pv_desc, pv_type, pv_unit, value_short)
        elif pv_type == EpicsType.DBR_CTRL_LONG:
            value_long = unpacker.unpack_farray(count, unpacker.unpack_int)
            return MDAProcessVariable[list[int]](pv_name, pv_desc, pv_type, pv_unit, value_long)
        elif pv_type == EpicsType.DBR_CTRL_FLOAT:
            value_float = unpacker.unpack_farray(count, unpacker.unpack_float)
            return MDAProcessVariable[list[float]](pv_name, pv_desc, pv_type, pv_unit, value_float)
        elif pv_type == EpicsType.DBR_CTRL_DOUBLE:
            value_double = unpacker.unpack_farray(count, unpacker.unpack_double)
            return MDAProcessVariable[list[float]](pv_name, pv_desc, pv_type, pv_unit, value_double)

        return MDAProcessVariable[str](pv_name, pv_desc, pv_type, pv_unit, str())

    @classmethod
    def read(cls, file_path: Path) -> MDAFile:
        """Read and parse an MDA file from disk.

        Parameters
        ----------
        file_path : Path
            Path to the ``.mda`` file.

        Returns
        -------
        MDAFile
            Fully parsed file representation.  If the file cannot be opened,
            the scan and extra PV fields will be empty and the error is logged.
        """
        extra_pvs: list[MDAProcessVariable[Any]] = list()

        try:
            with file_path.open(mode="rb") as fp:
                header = MDAHeader.read(fp)
                scan = MDAScan.read(fp)

                if header.has_extra_pvs:
                    fp.seek(header.extra_pvs_offset)
                    unpacker = xdrlib.Unpacker(fp.read())
                    number_pvs = unpacker.unpack_int()

                    for pvidx in range(number_pvs):
                        pv = cls._read_pv(unpacker)
                        extra_pvs.append(pv)
        except OSError as err:
            logger.exception(err)

        return cls(header, scan, extra_pvs)

    def to_mapping(self) -> Mapping[str, Any]:
        """Return a nested plain-dict representation of the entire file.

        Returns
        -------
        Mapping[str, Any]
            Dictionary containing all file fields, recursively serialised.
        """
        return {
            "header": self.header.to_mapping(),
            "scan": self.scan.to_mapping(),
            "extra_pvs": [pv.to_mapping() for pv in self.extra_pvs],
        }

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_mapping(), sort_keys=False)


def convert_extra_PVs_to_dict(mda_file: MDAFile) -> dict[str, MDAProcessVariable]:
    """Index the extra process variables of an MDA file by PV name.

    Parameters
    ----------
    mda_file : MDAFile
        Parsed MDA file whose extra PVs should be indexed.

    Returns
    -------
    dict[str, MDAProcessVariable]
        Mapping from PV name to :class:`MDAProcessVariable` instance.
    """
    return {pv.name: pv for pv in mda_file.extra_pvs}


if __name__ == "__main__":
    file_path = Path(sys.argv[1])
    mda_file = MDAFile.read(file_path)
    print(f"angle: {convert_extra_PVs_to_dict(mda_file)['2xfm:m60.VAL'].value[0]}")
    print(f"lamino angle: {convert_extra_PVs_to_dict(mda_file)['2xfm:m12.VAL'].value[0]}")
